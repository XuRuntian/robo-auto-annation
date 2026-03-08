import os
import json
import numpy as np
from PIL import Image as PILImage

# 导入我们之前拆分好的模块
from src.core.factory import ReaderFactory
from src.core.types import ArmState
from src.core.kinematics.extractor import ArmExtractor
from src.core.kinematics.calculator import KinematicCalculator
from src.core.physics.gap.segmentor import GAPSegmentor
from src.core.semantics.prompts import build_robotics_pamor_prompt, update_task_history
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
        print(f"🤖 [Pipeline] 开始 VLM 语义标注 (共 {len(action_chunks)} 个 Chunks)...")
        global_dataset_annotations = []
        task_history = "Task started."
        
        for idx, chunk in enumerate(action_chunks):
            # 获取 Chunk 的绝对边界
            chunk_start, chunk_end = chunk["stable_start"], chunk["chunk_end"]
            chunk_states = self._slice_arm_states(arm_states, chunk_start, chunk_end)
            
            # 计算 VLM 需要的物理特征 JSON 与 32 帧局部索引
            kinematic_json, local_indices = self.calculator.compute(chunk_states)
            global_sample_indices = chunk_start + local_indices
            
            # 提取 32 帧视觉图像
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
            
            # 构造 Prompt 并请求大模型
            prompt = build_robotics_pamor_prompt(kinematic_json, task_description, task_history)
            print(prompt)
            try:
                semantic_label = self.vlm.generate(prompt=prompt, images=pil_images)
                print(f"\n   🎯 [Chunk {idx+1}/{len(action_chunks)}] VLM 输出:\n   {semantic_label}")
                # 动态更新历史因果
                task_history = update_task_history(task_history, semantic_label)
                
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
            
        return global_dataset_annotations