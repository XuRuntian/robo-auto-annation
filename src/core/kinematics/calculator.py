import numpy as np
from src.core.types import ArmState

class KinematicCalculator:
    """
    负责计算动作块（Chunk）的运动学特征。
    支持单臂或双臂数据的自动聚合。
    """
    def __init__(self, fps: int = 30, num_samples: int = 32):
        self.fps = fps
        self.dt = 1.0 / fps
        self.num_samples = num_samples

    def _compute_arm_metrics(self, state: ArmState):
        """原子计算：计算单只手臂的速度和频域特征"""
        T = len(state.pos)
        
        # 1. 时域：计算线速度和角速度 (L2 Norm)
        vel = np.linalg.norm(np.diff(state.pos, axis=0), axis=1) / self.dt
        ang_vel = np.linalg.norm(np.diff(state.rot, axis=0), axis=1) / self.dt
        
        # 补齐首帧，保持长度为 T
        vel = np.insert(vel, 0, 0.0)
        ang_vel = np.insert(ang_vel, 0, 0.0)

        # 2. 频域：计算 FFT 得分
        v_score, v_fft = self._compute_fft(vel)
        a_score, a_fft = self._compute_fft(ang_vel)

        return {
            "vel": vel,
            "ang_vel": ang_vel,
            "v_score": v_score,
            "a_score": a_score,
            "v_fft": v_fft,
            "a_fft": a_fft
        }

    def _compute_fft(self, signal: np.ndarray):
        """内部方法：执行信号的 FFT 处理"""
        T = len(signal)
        if T < 2: return 0.0, [0.0] * 20
        fft_raw = np.abs(np.fft.rfft(signal))
        score = np.std(fft_raw)
        fft_list = (fft_raw / T).tolist()[:20]
        return float(score), [round(x, 3) for x in fft_list]

    def compute(self, arm_states: dict[str, ArmState]):
        """
        主接口：接收提取出的手臂状态字典，返回 VLM 所需的 JSON。
        :param arm_states: {"right": ArmState, "left": ArmState} 或单臂字典
        """
        # 1. 验证数据长度（取第一个可用手臂的长度）
        first_arm = next(iter(arm_states.values()))
        T = len(first_arm.pos)
        if T < 2:
            return {}, np.array([0] * self.num_samples)

        # 2. 遍历计算每只手臂的物理特征
        all_metrics = {}
        for name, state in arm_states.items():
            all_metrics[name] = self._compute_arm_metrics(state)

        # 3. 全局统计值聚合（支持单双臂平滑切换）
        # 如果是双臂，则取平均值（对应原逻辑的 CM）；如果是单臂，则直接使用该臂数值
        avg_vel = np.mean([m["vel"] for m in all_metrics.values()], axis=0)
        avg_ang = np.mean([m["ang_vel"] for m in all_metrics.values()], axis=0)
        avg_v_score = np.mean([m["v_score"] for m in all_metrics.values()])
        avg_a_score = np.mean([m["a_score"] for m in all_metrics.values()])

        # 4. 关键帧抽样
        sample_indices = np.linspace(0, T - 1, num=self.num_samples, dtype=int)
        frame_angles = {}
        
        for out_idx, f_idx in enumerate(sample_indices):
            # 动态组装当前帧中所有可用手臂的姿态
            current_frame_data = {}
            for name, state in arm_states.items():
                prefix = 'r' if name == "right" else 'l'
                current_frame_data.update({
                    f'{prefix}_arm_rx': round(float(state.rot[f_idx, 0]), 2),
                    f'{prefix}_arm_ry': round(float(state.rot[f_idx, 1]), 2),
                    f'{prefix}_arm_rz': round(float(state.rot[f_idx, 2]), 2),
                    f'{prefix}_gripper': round(float(state.gripper[f_idx, 0]), 2),
                })
            frame_angles[str(out_idx)] = [current_frame_data]

        # 5. 生成最终 JSON
        # 聚合后的 FFT 列表也可以取平均
        v_fft_avg = np.mean([m["v_fft"] for m in all_metrics.values()], axis=0).round(3).tolist()
        a_fft_avg = np.mean([m["a_fft"] for m in all_metrics.values()], axis=0).round(3).tolist()

        kinematic_json = {
            'vel': round(float(np.mean(avg_vel)), 2),
            'angle': round(float(np.mean(avg_ang)), 2),
            'vel_score': round(float(avg_v_score), 2),
            'angle_score': round(float(avg_a_score), 2),
            'frame_angles': frame_angles,
            'vel-fft': v_fft_avg,
            'angle_fft': a_fft_avg
        }
        
        return kinematic_json, sample_indices