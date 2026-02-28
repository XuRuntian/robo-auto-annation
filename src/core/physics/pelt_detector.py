import numpy as np
import ruptures as rpt
from typing import List, Optional
from src.core.physics.base import BasePhysicsDetector
from src.core.types import CutPoint

class PeltPhysicsDetector(BasePhysicsDetector):
    def __init__(self, fps: int = 30, min_dist: int = 30, penalty: int = 10, n_points: int = 9):
        super().__init__()
        self.fps = fps
        self.min_dist = min_dist
        self.penalty = penalty
        self.n_points = n_points  # 🎯 目标帧数，默认为 9
        self.model = rpt.Pelt(model="rbf", min_size=5, jump=1)
    
    def compute_energy(self, qpos_data: np.ndarray):
        velocity = np.diff(qpos_data, axis=0, prepend=qpos_data[0:1])
        return np.sum(velocity ** 2, axis=1)

    def propose_cut_points(
        self, 
        qpos_data: np.ndarray, 
        action_data: Optional[np.ndarray] = None
    ) -> List[CutPoint]:
        if len(qpos_data) == 0:
            return []

        # 1. PELT 物理变点检测
        raw_energy = self.compute_energy(qpos_data)
        window_size = 5
        window = np.ones(window_size) / window_size
        smooth_energy = np.convolve(raw_energy, window, mode='same').reshape(-1, 1)
        
        self.model.fit(smooth_energy)
        try:
            raw_cps = self.model.predict(pen=self.penalty)
        except:
            raw_cps = [len(qpos_data) - 1]

        # 2. 移除越界点
        detected_indices = sorted(list(set([cp for cp in raw_cps if cp < len(qpos_data)])))

        # 3. 🎯 直接返回真实找到的点，不要补齐到 9 个！
        # 如果它只找到了 3 个，那就是 3 个。
        cut_points = []
        for idx in detected_indices:
            score = float(raw_energy[idx]) if idx < len(raw_energy) else 0.0
            cut_points.append(CutPoint(frame_idx=int(idx), energy_score=score))
            
        return cut_points