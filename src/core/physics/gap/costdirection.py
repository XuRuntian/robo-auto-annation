r"""CostDirection (find change point for different directions sub-trajectory)"""
import numpy as np
from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints
from scipy.spatial.distance import pdist, squareform
import quaternion


class CostDirection(BaseCost):
    r"""Find change point for different directions sub-trajectory."""

    model = "direction"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 10

    def fit(self, signal) -> "CostDirection":
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        return self


    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        def delta_vector(point1, point2):
            move_vector = point2[: 3] - point1[: 3]
            rotate_vector = point2[3: -2] - point1[3: -2]
            gripper_vector = point2[-2: ] - point1[-2: ]
            return move_vector, rotate_vector, gripper_vector
        
        def direction_distance(v1, v2):
            norm_v1_move = np.linalg.norm(v1[0])
            norm_v2_move = np.linalg.norm(v2[0])
            
            # --- 修复 1：平移死区 (Translation Deadband) ---
            # 设定阈值（比如 5e-4，根据你的实际单位 mm 或 m 调整）
            # 如果当前步的平移极小，说明是传感器底噪或静止，强制赋予完美连贯的 cost (-1.0)
            if norm_v2_move < 5e-4 or norm_v1_move < 1e-8: 
                move_distance = 0.0
            else:
                move_distance = -1 * np.dot(v1[0], v2[0]) / (norm_v1_move * norm_v2_move)
                
            norm_v1_rot = np.linalg.norm(v1[1])
            norm_v2_rot = np.linalg.norm(v2[1])
            
            # --- 修复 2：旋转死区 (Rotation Deadband) ---
            # 忽略极微小的手腕抖动噪声
            if norm_v2_rot < 1e-3 or norm_v1_rot < 1e-8:
                rotate_distance = 0.0
            else:
                rotate_distance = -1 * np.dot(v1[1], v2[1]) / (norm_v1_rot * norm_v2_rot)
                
            if np.array_equal(np.sign(v1[2]), np.sign(v2[2])):
                gripper_distance = 0.0
            else:
                gripper_distance = 1.0
            
            # 同样加上你之前修复的 1.0 * rotate_distance
            return move_distance + 1.0 * rotate_distance - 2 * np.exp(-3) * gripper_distance
        
        total_direction = delta_vector(self.signal[start-1], self.signal[end-1])

        error = []
        for i in range(start, end-1):
            delta = delta_vector(self.signal[i], self.signal[i+1])
            error.append(direction_distance(total_direction, delta))

        val = np.mean(error)
        return val
