from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Optional

from src.core.physics.base import BasePhysicsDetector
from src.core.types import CutPoint

class KMeansPhysicsDetector(BasePhysicsDetector):
    def __init__(self, fps=30):
        self.fps = fps

    def compute_energy(self, qpos_data):
        velocity = np.diff(qpos_data, axis=0, prepend=qpos_data[0:1])
        return np.sum(velocity ** 2, axis=1)

    def propose_cut_points(
        self, 
        qpos_data: np.ndarray, 
        action_data: Optional[np.ndarray] = None
    ) -> List[CutPoint]:
        if len(qpos_data) == 0:
            return []
            
        # 1. 计算能量并确定活动窗口
        raw_energy = self.compute_energy(qpos_data)
        window_size = 15
        window = np.ones(window_size) / window_size
        smooth_energy = np.convolve(raw_energy, window, mode='same')
        
        noise_floor = np.mean(np.sort(smooth_energy)[:int(max(1, len(smooth_energy)*0.05))]) 
        noise_floor = max(noise_floor, 1e-6) 
        threshold = noise_floor * 2.0
        
        active_indices = np.where(smooth_energy > threshold)[0]
        if len(active_indices) == 0:
            return []

        active_start = active_indices[0]
        active_end = active_indices[-1]
        
        padding = int(0.5 * self.fps)
        start = max(0, active_start - padding)
        end = min(len(qpos_data) - 1, active_end + padding)
        
        # 2. 特征工程
        active_qpos = qpos_data[start:end+1]
        if len(active_qpos) < 9:  # 最少需要9帧才能保证关键帧数量
            return [CutPoint(
                frame_idx=int(i), 
                energy_score=0.0
            ) for i in np.linspace(start, end, 9, dtype=int)]
            
        # 特征标准化
        scaler_pos = StandardScaler()
        qpos_scaled = scaler_pos.fit_transform(active_qpos)
        
        velocities = np.diff(active_qpos, axis=0, prepend=active_qpos[0:1])
        scaler_vel = StandardScaler()
        vel_scaled = scaler_vel.fit_transform(velocities)
        
        time_steps = np.arange(len(active_qpos)).reshape(-1, 1)
        scaler_time = StandardScaler()
        time_scaled = scaler_time.fit_transform(time_steps)
        
        # 特征加权组合
        W_pos, W_vel, W_time = 1.0, 2.0, 1.5  
        features = np.hstack([
            qpos_scaled * W_pos, 
            vel_scaled * W_vel, 
            time_scaled * W_time
        ])
        
        # 3. KMeans聚类
        kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
        kmeans.fit(features)
        
        key_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features - center, axis=1)
            key_idx = np.argmin(distances) + start
            key_indices.append(key_idx)
            
        # 4. 保证唯一性和数量
        key_indices = sorted(list(set(key_indices)))
        while len(key_indices) < 9:
            fallback = np.linspace(start, end, 9, dtype=int).tolist()
            key_indices = sorted(list(set(key_indices + fallback)))[:9]
            
        # 5. 构建CutPoint列表
        cut_points = []
        for idx in key_indices:
            velocity = np.diff(qpos_data[max(0, idx-1):idx+1], axis=0)
            energy_score = float(np.sum(velocity ** 2)) if idx > 0 else 0.0
            cut_points.append(CutPoint(
                frame_idx=int(idx),
                energy_score=energy_score
            ))
            
        return cut_points
