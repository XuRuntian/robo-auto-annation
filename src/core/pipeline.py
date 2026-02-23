# src/core/pipeline.py
import numpy as np
import time
from src.core.factory import ReaderFactory
from src.core.kinematics import KinematicScreener, align_and_segment

class RoboETLPipeline:
    def __init__(self, dataset_path, robot_config):
        self.dataset_path = dataset_path
        self.robot_config = robot_config
        self.reader = ReaderFactory.get_reader(dataset_path)
        if not self.reader.load(dataset_path):
            raise ValueError("数据集加载失败")

    def extract_qpos(self, ep_length):
        qpos_list = []
        for idx in range(ep_length):
            frame = self.reader.get_frame(idx)
            state = getattr(frame, 'state', {})
            val = state.get("qpos") if state.get("qpos") is not None else state.get("action")
            qpos_list.append(val if val is not None else np.zeros(6))
        return np.array(qpos_list)

    def process_episode(self, episode_idx, progress_callback=None):
        """处理单条轨迹的完整流水线"""
        self.reader.set_episode(episode_idx)
        ep_length = self.reader.get_length()
        
        # 步骤 1：底层物理数据提取
        if progress_callback: progress_callback("🔄 正在提取底层运动学数据...")
        qpos_data = self.extract_qpos(ep_length)
        
        # 步骤 2 & 3：K-Means 活跃区筛选与九宫格关键帧定位
        if progress_callback: progress_callback("🧠 正在进行 K-Means 物理特征聚类...")
        screener = KinematicScreener(fps=30)
        active_start, active_end = screener.get_active_window(qpos_data)
        indices_rel = screener.select_key_frames_kmeans(qpos_data, active_start, active_end)
        
        # 步骤 4：生成九宫格图片并请求 VLM (此处用 Mock 代替)
        if progress_callback: progress_callback("🌐 正在请求 VLM 大模型进行宏观拆解...")
        time.sleep(1.5) # 模拟网络延迟
        mock_vlm_json = [
            {"subtask_id": 1, "instruction": "Approach phone", "start_image": 1, "end_image": 3},
            {"subtask_id": 2, "instruction": "Grasp phone", "start_image": 3, "end_image": 5},
            {"subtask_id": 3, "instruction": "Place on base", "start_image": 5, "end_image": 9}
        ]
        
        # 步骤 5：微观物理边界对齐
        if progress_callback: progress_callback("⚙️ 正在执行微观物理状态缝合...")
        final_labels = align_and_segment(
            vlm_json=mock_vlm_json,
            indices_rel=indices_rel,
            qpos_data=qpos_data,
            dataset_start_offset=0,
            gripper_dim_indices=self.robot_config["gripper_dim_indices"],
            gripper_threshold=self.robot_config["gripper_threshold"]
        )
        
        # 返回标准的格式
        return [
            {
                "subtask_id": r["subtask_id"], 
                "instruction": r["instruction"], 
                "start_frame": r["global_start_frame"], 
                "end_frame": r["global_end_frame"]
            } for r in final_labels
        ]

    def close(self):
        self.reader.close()