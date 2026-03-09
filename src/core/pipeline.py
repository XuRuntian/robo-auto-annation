import os
import json
import numpy as np
from PIL import Image as PILImage
import copy
import re
# 导入我们之前拆分好的模块
from src.core.factory import ReaderFactory
from src.core.types import ArmState
from src.core.kinematics.extractor import ArmExtractor
from src.core.kinematics.calculator import KinematicCalculator
from src.core.physics.gap.segmentor import GAPSegmentor
from src.core.semantics.prompts import build_robotics_pamor_prompt, update_world_state, build_subtask_summary_prompt
from src.core.semantics.parser import VLMOutputParser
from src.core.vlm_caller import QwenVLCaller

class RoboAnnotationPipeline:
    """
    具身智能自动化标注流水线。
    统筹数据读取、物理切分、特征计算、VLM 标注及结果解析。
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
        
        # 3. 初始化语义解析与大模型
        self.parser = VLMOutputParser()
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
        """主处理流程：处理单个 Episode 数据并生成 JSON 标签"""
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
        # 阶段 3：语义标注与图像对齐
        # ==========================================
        current_wsm = {
            "temporal_context": {"macro_skill_id": "initialization", "execution_phase": "idle"},
            "robot_interaction_state": {
                "right_end_effector": {"contact_target": "none", "grasp_type": "none", "is_constrained": False},
                "left_end_effector": {"contact_target": "none", "grasp_type": "none", "is_constrained": False}
            },
            "environment_topology_state": [],
            "object_physical_state": []
        }
        print(f"🤖 [Pipeline] 开始 VLM 语义标注 (共 {len(action_chunks)} 个 Chunks)...")
        global_dataset_annotations = []
        wsm_trajectory = []  # 【新增】用来保存随时间变化的世界状态机轨迹
        for idx, chunk in enumerate(action_chunks):
            # 获取 Chunk 的绝对边界
            chunk_start, chunk_end = chunk["stable_start"], chunk["chunk_end"]
            chunk_states = self._slice_arm_states(arm_states, chunk_start, chunk_end)
            
            # 计算 VLM 需要的物理特征 JSON 与 32 帧局部索引
            kinematic_json, local_indices = self.calculator.compute(chunk_states)
            global_sample_indices = chunk_start + local_indices
            
            # 提取视觉图像
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
            num_actual = len(pil_images)
            # 构造 Prompt 并请求大模型
            prompt = build_robotics_pamor_prompt(kinematic_json, task_description, world_state_dict=current_wsm, num_actual=num_actual)
            try:
                semantic_label = self.vlm.generate(prompt=prompt, images=pil_images)
                # 动态更新历史因果
                current_wsm = update_world_state(current_wsm, semantic_label)
                wsm_trajectory.append({
                    "chunk_idx": idx,
                    "global_frame_range": [int(chunk_start), int(chunk_end)], 
                    "world_state": copy.deepcopy(current_wsm) 
                })
                # 正则解析并将 32 帧相对位置映射为 800 帧级绝对位置
                mapped_actions = self.parser.parse_and_map(semantic_label, global_sample_indices)
                global_dataset_annotations.extend(mapped_actions)
                
                print(f"   ✅ [Chunk {idx+1}/{len(action_chunks)}] 成功映射了 {len(mapped_actions)} 个子动作。")
            except Exception as e:
                print(f"   ❌ [Chunk {idx+1}/{len(action_chunks)}] VLM 调用或解析失败: {e}")
        
        # ==========================================
        # 阶段 4：保存结果
        # ==========================================
        output_path = os.path.join(os.path.dirname(data_path), "auto_annotations.json")
        if global_dataset_annotations:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(global_dataset_annotations, f, indent=4, ensure_ascii=False)
            print(f"\n🎉 完美！高质量结构化动作标签已保存至:\n{output_path}")
        else:
            print("\n⚠️ 警告：未生成任何有效标注。")
            
        output_wsm_path = os.path.join(os.path.dirname(data_path), "auto_annotations_wsm.json")
        if wsm_trajectory:
            with open(output_wsm_path, 'w', encoding='utf-8') as f:
                json.dump(wsm_trajectory, f, indent=4, ensure_ascii=False)
            print(f"🎉 完美！宏观世界状态机(WSM)轨迹已保存至:\n{output_wsm_path}")

        # ==========================================
        # 阶段 5：高级子任务标签生成 (Subtask Segmentation)
        # ==========================================
        print("\n📝 [Pipeline] 开始生成人类可读的子任务指令...")
        
        # 2. 调用纯文本大模型进行翻译
        total_frames = 801  
        video_id = "video_001"
        subtask_prompt = build_subtask_summary_prompt(
            task_description=task_description,
            wsm_trajectory=wsm_trajectory, 
            kpm_annotations=global_dataset_annotations, 
            total_frames=total_frames,
            video_id=video_id
        )
        
        try:
            # 注意：这里你可以直接调用纯文本接口 (比如 gpt-4o-mini)，速度飞快且极度便宜
            subtask_response = self.vlm.generate(prompt=subtask_prompt, images=[]) # 传空列表不传图
            
            # 正则提取最终 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', subtask_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                subtask_json = json.loads(json_match.group(1))
                
                # 保存最终的子任务标注
                output_subtask_path = os.path.join(os.path.dirname(data_path), "subtask_instructions.json")
                with open(output_subtask_path, 'w', encoding='utf-8') as f:
                    json.dump(subtask_json, f, indent=4, ensure_ascii=False)
                print(f"🎉 完美！人类可读的高级子任务标签已保存至:\n{output_subtask_path}")
            else:
                print("⚠️ 无法解析子任务 JSON。")
                
        except Exception as e:
            print(f"❌ 生成子任务指令失败: {e}")
        