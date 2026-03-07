from dataclasses import dataclass
import numpy as np

@dataclass
class ArmState:
    """机械臂状态的标准协议"""
    pos: np.ndarray      # 笛卡尔坐标 (T, 3)
    rot: np.ndarray      # 旋转欧拉角 (T, 3)
    gripper: np.ndarray  # 夹爪状态 (T, 1) 或 (T, 2)
    name: str            # "right" 或 "left"