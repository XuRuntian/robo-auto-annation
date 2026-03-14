import numpy as np
from src.core.types import ArmState

class DynamicKeyframeSampler:
    """
    动态关键帧采样器
    基于运动学波动和夹爪状态变化，进行自适应非均匀采样。
    """
    def __init__(self, num_samples: int = 32):
        self.num_samples = num_samples

    def sample(self, arm_states: dict[str, ArmState], metrics_dict: dict) -> np.ndarray:
        """
        :param arm_states: 包含原始轨迹 (pos, rot, gripper) 的状态字典
        :param metrics_dict: 由 Calculator 预先计算好的运动学指标字典 (vel, ang_vel 等)
        :return: 采样后的帧索引数组
        """
        first_arm = next(iter(arm_states.values()))
        T = len(first_arm.pos)
        
        # 如果总帧数不足以抽样，直接返回全序列
        if T <= self.num_samples:
            return np.arange(T)

        # 1. 初始化基础重要度 (保证最基本的均匀采样底座)
        importance_scores = np.ones(T) * 0.1 

        # 2. 叠加各手臂的动态特征权重
        for name, state in arm_states.items():
            metrics = metrics_dict[name]
            # --- a. 考虑夹爪开闭突变 (赋予最高权重) ---
            # 计算夹爪的一阶差分
            gripper_diff = np.abs(np.diff(state.gripper, axis=0))
            # 补齐首帧，维持长度为T
            gripper_diff = np.vstack([np.zeros((1, state.gripper.shape[1])), gripper_diff])
            gripper_change = np.sum(gripper_diff, axis=1)

            # 归一化并放大权重
            if np.max(gripper_change) > 1e-5:
                norm_gripper = gripper_change / np.max(gripper_change)
                importance_scores += norm_gripper * 3.0  # 权重系数可以根据实际效果微调
            
            # --- b. 考虑运动学数据波动 (速度/角速度) ---
            vel = metrics["vel"]
            ang_vel = metrics["ang_vel"]
            
            if np.max(vel) > 1e-5:
                importance_scores += (vel / np.max(vel)) * 1.0
            if np.max(ang_vel) > 1e-5:
                importance_scores += (ang_vel / np.max(ang_vel)) * 1.0

        # 3. 基于 CDF (累积分布函数) 进行逆变换采样
        # 这一步能保证重要度高的地方被分配更多的样本点
        cdf = np.cumsum(importance_scores)
        cdf_normalized = cdf / cdf[-1]
        
        # 在 Y 轴 (0~1) 上均匀取点
        target_cdf = np.linspace(0, 1, self.num_samples)
        
        # 映射回 X 轴寻找对应的帧索引
        sample_indices = np.searchsorted(cdf_normalized, target_cdf)
        
        # 强制包含首尾帧
        sample_indices[0] = 0
        sample_indices[-1] = T - 1
        
        # 4. 去重与单调性修正 (确保 VLM 接收到的时序是严格递增的)
        for i in range(1, self.num_samples - 1):
            if sample_indices[i] <= sample_indices[i-1]:
                sample_indices[i] = sample_indices[i-1] + 1
                
        # 越界保护
        sample_indices = np.clip(sample_indices, 0, T - 1)
        
        return sample_indices