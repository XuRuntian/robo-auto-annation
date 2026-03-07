import numpy as np
from src.core.types import ArmState

class ArmExtractor:
    """
    负责根据配置从原始 qpos 序列中提取单臂或双臂的运动学状态。
    """
    def __init__(self, config: dict, mimic_gripper: bool = True):
        """
        config 示例: 
        {
            "right": {"move": [20, 23], "rotate": [23, 26], "gripper": 19},
            "left": {"move": [7, 10], "rotate": [10, 13], "gripper": 6}
        }
        如果是单臂，config 只需包含一个 key 即可。
        """
        self.mimic_gripper = mimic_gripper
        # 预处理配置，将列表转为 slice 方便使用
        self.indices = {}
        for arm, cfg in config.items():
            self.indices[arm] = {
                "move": slice(*cfg["move"]),
                "rotate": slice(*cfg["rotate"]),
                "gripper": cfg["gripper"]
            }

    def _mimic_gripper(self, val: float) -> np.ndarray:
        return np.array([val, val], dtype=np.float32)

    def extract_arm(self, qpos_array: np.ndarray, arm_type: str) -> ArmState:
        """提取指定手臂的状态"""
        idx_cfg = self.indices.get(arm_type)
        if not idx_cfg:
            raise ValueError(f"Config 中未找到 {arm_type} 的索引配置")

        # 利用 numpy 的高级切片一次性提取（比循环快得多）
        pos = qpos_array[:, idx_cfg["move"]]
        rot = qpos_array[:, idx_cfg["rotate"]]
        gripper_raw = qpos_array[:, idx_cfg["gripper"]]

        if self.mimic_gripper:
            # (T,) -> (T, 2)
            gripper = np.stack([gripper_raw, gripper_raw], axis=-1)
        else:
            # (T,) -> (T, 1)
            gripper = gripper_raw[:, np.newaxis]

        return ArmState(pos=pos, rot=rot, gripper=gripper, arm_type=arm_type)

    def extract_all(self, qpos_array: np.ndarray) -> dict[str, ArmState]:
        """
        自动根据配置提取所有可用的手臂（支持单臂或双臂）
        返回: {"right": ArmState, "left": ArmState} 或 {"right": ArmState}
        """
        results = {}
        for arm_type in self.indices.keys():
            results[arm_type] = self.extract_arm(qpos_array, arm_type)
        return results