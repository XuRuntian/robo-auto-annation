import numpy as np
import ruptures as rpt

from src.core.physics.base import BasePhysicsDetector


class PeltPhysicsDetector(BasePhysicsDetector):
    def __init__(self, min_dist: int = 30):
        self.min_dist = min_dist
        self.model = rpt.Pelt(model="rbf", min_size=5, jump=1)
    
    def detect(self, qpos: np.ndarray) -> list[int]:
        # 计算速度特征 (差分)
        velocity = np.diff(qpos, axis=0)
        energy = np.sum(velocity**2, axis=1)
        
        # 使用 Pelt 算法检测变化点
        self.model.fit(energy)
        raw_cps = self.model.predict(pen=10)
        
        # 应用 min_dist 约束
        filtered_cps = []
        last_cp = -self.min_dist
        
        for cp in raw_cps:
            if cp - last_cp >= self.min_dist:
                filtered_cps.append(cp)
                last_cp = cp
        
        return filtered_cps
