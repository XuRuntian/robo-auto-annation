import os
import json
import numpy as np
from PIL import Image as PILImage
import copy
import re

# 导入底层组件
from src.core.factory import ReaderFactory
from src.core.types import ArmState
from src.core.kinematics.extractor import ArmExtractor
from src.core.kinematics.calculator import KinematicCalculator
from src.core.physics.gap.segmentor import GAPSegmentor

# 导入重构后的全新语义与调用模块
from src.core.semantics.prompts import build_robotics_pamor_prompt, update_world_state, build_subtask_summary_prompt
from src.core.vlm_caller import QwenVLCaller

# 引入我们定义的强类型 Schema
from src.core.semantics.schema import *

class RoboAnnotationPipeline:
    """
    具身智能自动化标注流水线 (重构版)。
    统筹数据读取、物理切分、特征计算、基于 PDDL 的 VLM 闭环推导及物理强制吸附。
    """
    def __init__(self, config: dict):
        self.config = config
        
        # 1. 初始化运动学模块 (支持单双臂自动推导)
        self.extractor = ArmExtractor(
            config=config.get("robot", {}).get("arms", {}), 
            mimic_gripper=config.get("kinematics", {}).get("mimic_gripper", True)
        )
        self.calculator = KinematicCalculator(
            fps=config.get("kinematics", {}).get("fps", 30),
            num_samples=config.get("kinematics", {}).get("num_samples", 32)
        )
        
        # 2. 初始化物理切分模块
        seg_cfg = config.get("segmentation", {})
        self.use_lstm = seg_cfg.get("use_lstm", True)
        self.segmentor = GAPSegmentor(
            penalty_value=seg_cfg.get("penalty_value", 1.5),
            threshold=seg_cfg.get("threshold", 0.4),
            min_gap=seg_cfg.get("min_gap_frames", 25),
            min_duration=seg_cfg.get("min_duration_frames", 5)
        )
        
        # 3. 初始化带闭环纠错的大模型调用器 (不再需要 Parser)
        self.vlm = QwenVLCaller()

    def _slice_arm_states(self, arm_states: dict[str, ArmState], start_idx: int, end_idx: int) -> dict[str, ArmState]:
        """辅助函数：将完整轨迹的 ArmState 切片为 Chunk 对应的局部片段"""
        sliced_states = {}
        for name, state in arm_states.items():
            sliced_states[name] = ArmState(
                pos=state.pos[start_idx:end_idx+1],
                rot=state.rot[start_idx:end_idx+1],
                gripper=state.gripper[start_idx:end_idx+1],
                arm_type=state.arm_type
            )
        return sliced_states

    def process_episode(self, data_path: str, task_description: str):
        """主处理流程：处理单个 Episode 数据并生成基于物理吸附的 JSON 标签"""
        print(f"\n🚀 [Pipeline] 开始处理数据: {data_path}")
        
        # ==========================================
        # 阶段 1：数据读取与运动学提取
        # ==========================================
        reader = ReaderFactory.get_reader(data_path)
        reader.load(data_path)
        
        traj_len = reader.get_length()
        qpos_list = []
        for i in range(traj_len):
            frame = reader.get_frame(i)
            state = getattr(frame, 'state', None)
            raw_state = state.get('qpos', []) if state is not None else getattr(frame, 'qpos', [])
            qpos_list.append(raw_state)
        qpos_array = np.array(qpos_list)
        
        print("🧠 [Pipeline] 正在进行物理特征提取与阶段切分...")
        arm_states = self.extractor.extract_all(qpos_array)
        
        # ==========================================
        # 阶段 2：动作切分 (Pelt + LSTM)
        # ==========================================
        action_chunks = []
        if self.use_lstm:
            final_phases = self.segmentor.detect_phases(arm_states)
            last_end = 0
            for p_start, p_end in final_phases:
                action_chunks.append({
                    "chunk_start": last_end, "chunk_end": p_end,
                    "stable_start": last_end, "stable_end": max(last_end, p_start - 1)
                })
                last_end = p_end + 1
            if last_end < traj_len:
                action_chunks.append({
                    "chunk_start": last_end, "chunk_end": traj_len - 1,
                    "stable_start": last_end, "stable_end": traj_len - 1
                })
        else:
            action_chunks.append({
                "chunk_start": 0, "chunk_end": traj_len - 1,
                "stable_start": 0, "stable_end": traj_len - 1
            })

        # ==========================================
        # 阶段 3：语义闭环与物理强制吸附 (Snapping)
        # ==========================================
        # 初始化新的精简版世界状态机
        current_wsm = {
            "temporal_context": {"last_action": "none", "last_target": "none"},
            "robot_interaction_state": {
                "right_end_effector": {"contact_target": "none", "grasp_type": "none", "is_constrained": False},
                "left_end_effector": {"contact_target": "none", "grasp_type": "none", "is_constrained": False}
            }
        }
        
        global_dataset_annotations = []
        wsm_trajectory = [] 
        
        print(f"🤖 [Pipeline] 开始基于 PDDL 的 VLM 闭环语义标注 (共 {len(action_chunks)} 个物理切片)...")
        
        for idx, chunk in enumerate(action_chunks):
            # 获取当前物理切片的绝对边界
            chunk_start, chunk_end = chunk["stable_start"], chunk["chunk_end"]
            chunk_states = self._slice_arm_states(arm_states, chunk_start, chunk_end)
            
            kinematic_json, local_indices = self.calculator.compute(chunk_states)
            global_sample_indices = chunk_start + local_indices
            
            # 提取视觉图像送给 VLM
            pil_images = []
            for frame_idx in global_sample_indices:
                frame_data = reader.get_frame(int(frame_idx))
                if not frame_data: continue
                # 优先选用前置头部相机
                cam_name = 'cam_front_head_rgb' if 'cam_front_head_rgb' in frame_data.images else list(frame_data.images.keys())[0]
                img_array = frame_data.images[cam_name]
                
                try:
                    if not isinstance(img_array, np.ndarray): img_array = np.array(img_array)
                    if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]: img_array = np.transpose(img_array, (1, 2, 0))
                    if img_array.dtype in [np.float32, np.float64]: img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    elif img_array.dtype != np.uint8: img_array = img_array.astype(np.uint8)
                    pil_images.append(PILImage.fromarray(img_array))
                except Exception:
                    pass
            
            if not pil_images: continue
            # 👇 ========================================== 👇
            # 🐛 新增：DEBUG 视觉输入，把喂给 VLM 的图存下来看看
            # ==========================================
            debug_dir = os.path.join(os.path.dirname(data_path), "debug_vlm_inputs", f"chunk_{idx}")
            os.makedirs(debug_dir, exist_ok=True)
            for img_idx, img in enumerate(pil_images):
                # 将采样出来的每张图保存，文件名带上真实的物理帧号
                real_frame_id = int(global_sample_indices[img_idx])
                img.save(os.path.join(debug_dir, f"idx_{img_idx:02d}_frame_{real_frame_id}.jpg"))
            print(f"   🐛 [Debug] Chunk {idx} 的 {len(pil_images)} 张采样图已保存至: {debug_dir}")
            # 👆 ========================================== 👆
            # 构造强约束 PDDL Prompt
            prompt = build_robotics_pamor_prompt(
                kinematic_json=kinematic_json, 
                task_description=task_description, 
                world_state_dict=current_wsm,
                allowed_verbs=VOCAB["verbs"],           # 新增传参
                allowed_predicates=VOCAB["predicates"]  # 新增传参
            )
            
            try:
                # 🌟🌟 核心 1：使用带闭环纠错的 Pydantic 生成器
                pddl_trajectory = self.vlm.generate_with_validation(prompt=prompt, images=pil_images, max_retries=3)
                
                if pddl_trajectory is None:
                    print(f"   🛑 [Chunk {idx+1}/{len(action_chunks)}] VLM 校验熔断，标记为需要人工 Review。")
                    continue
                    
                # 🌟🌟 核心 2：物理强制吸附 (Snapping)
                # 彻底抛弃 VLM 生成的幻觉时间戳，直接将解析出的算子吸附到当前底层物理计算出的区间
                for op in pddl_trajectory.operators:
                    annotation = {
                        "global_start_frame": int(chunk_start),
                        "global_end_frame": int(chunk_end),
                        "action_verb": op.action_verb,
                        "subject": op.subject,
                        "object": op.target_object,
                        "effects": [eff.model_dump() for eff in op.effects] # 保留执行效果以便后续训练查阅
                    }
                    global_dataset_annotations.append(annotation)
                
                # 🌟🌟 核心 3：更新全局状态，用于下一次 Chunk 推理的上下文
                current_wsm = update_world_state(current_wsm, pddl_trajectory)
                wsm_trajectory.append({
                    "chunk_idx": idx,
                    "global_frame_range": [int(chunk_start), int(chunk_end)], 
                    "world_state": copy.deepcopy(current_wsm),
                    "thought": pddl_trajectory.thought # 记录下 VLM 每一步为何这么做的思考过程
                })
                
                verbs = [op.action_verb for op in pddl_trajectory.operators]
                print(f"   ✅ [Chunk {idx+1}/{len(action_chunks)}] 吸附成功！解析出动作: {verbs}")
                
            except Exception as e:
                print(f"   ❌ [Chunk {idx+1}/{len(action_chunks)}] 流水线异常: {e}")
        
        # ==========================================
        # 阶段 4：保存底层的细粒度结构化结果
        # ==========================================
        output_path = os.path.join(os.path.dirname(data_path), "auto_annotations.json")
        if global_dataset_annotations:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(global_dataset_annotations, f, indent=4, ensure_ascii=False)
            print(f"\n🎉 结构化动作标签已保存至:\n{output_path}")
        else:
            print("\n⚠️ 警告：未生成任何有效标注。")
            
        output_wsm_path = os.path.join(os.path.dirname(data_path), "auto_annotations_wsm.json")
        if wsm_trajectory:
            with open(output_wsm_path, 'w', encoding='utf-8') as f:
                json.dump(wsm_trajectory, f, indent=4, ensure_ascii=False)
            print(f"🎉 宏观世界状态机(WSM)轨迹已保存至:\n{output_wsm_path}")

        # ==========================================
        # 阶段 5：高级子任务标签生成 (Subtask Segmentation)
        # ==========================================
        print("\n📝 [Pipeline] 开始生成人类可读的子任务指令...")
        
        total_frames = traj_len  
        video_id = "video_001"
        subtask_prompt = build_subtask_summary_prompt(
            task_description=task_description,
            wsm_trajectory=wsm_trajectory, 
            kpm_annotations=global_dataset_annotations, 
            total_frames=total_frames,
            video_id=video_id
        )
        try:
            # 调用纯文本总结，剥离图片传输成本
            subtask_response = self.vlm.generate(prompt=subtask_prompt, images=[]) 
            json_match = re.search(r'```json\s*(.*?)\s*```', subtask_response, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                subtask_json = json.loads(json_match.group(1))
                output_subtask_path = os.path.join(os.path.dirname(data_path), "subtask_instructions.json")
                with open(output_subtask_path, 'w', encoding='utf-8') as f:
                    json.dump(subtask_json, f, indent=4, ensure_ascii=False)
                print(f"🎉 子任务标签已保存至:\n{output_subtask_path}")
            else:
                print("⚠️ 无法解析子任务 JSON。")
                
        except Exception as e:
            print(f"❌ 生成子任务指令失败: {e}")