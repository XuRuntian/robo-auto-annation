import numpy as np
from src.core.types import ArmState
from .sampler import DynamicKeyframeSampler
class KinematicCalculator:
    """
    负责计算动作块（Chunk）的运动学特征。
    支持单臂或双臂数据的自动聚合。
    """
    def __init__(self, fps: int = 30, num_samples: int = 32):
        self.fps = fps
        self.dt = 1.0 / fps
        self.num_samples = num_samples
        self.sampler = DynamicKeyframeSampler(num_samples=num_samples)
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
        """
        # 1. 验证数据长度
        if not arm_states:
            return {}, np.array([0] * self.num_samples)
        
        first_arm = next(iter(arm_states.values()))
        T = len(first_arm.pos)
        if T < 2:
            return {}, np.array([0] * self.num_samples)

        # 2. 计算每只手臂的物理特征（保留原始数据，不求全局平均）
        kinematic_json = {}
        all_metrics = {}
        
        for name, state in arm_states.items():
            metrics = self._compute_arm_metrics(state)
            all_metrics[name] = metrics
            
            # 将统计值按手臂名称（left/right）存入 JSON
            kinematic_json[name] = {
                'vel': round(float(np.mean(metrics["vel"])), 2),
                'angle': round(float(np.mean(metrics["ang_vel"])), 2),
                'vel_score': round(float(metrics["v_score"]), 2),
                'angle_score': round(float(metrics["a_score"]), 2),
                'vel_fft': np.round(metrics["v_fft"], 3).tolist(),
                'angle_fft': np.round(metrics["a_fft"], 3).tolist(),
            }

        # 3. 关键帧抽样 (frame_angles 逻辑保持不变，因为它已经区分了 r_ 和 l_ 前缀)
        # actual_num_samples = min(T, self.num_samples) 
        sample_indices = self.sampler.sample(arm_states=arm_states, metrics_dict=all_metrics)

        frame_angles = {}
        
        for out_idx, f_idx in enumerate(sample_indices):
            current_frame_data = {}
            for name, state in arm_states.items():
                prefix = 'r' if name == "right" else 'l'
                current_frame_data.update({
                    f'{prefix}_arm_rx': round(float(state.rot[f_idx, 0]), 2),
                    f'{prefix}_arm_ry': round(float(state.rot[f_idx, 1]), 2),
                    f'{prefix}_arm_rz': round(float(state.rot[f_idx, 2]), 2),
                    f'{prefix}_gripper': round(float(state.gripper[f_idx, 0]), 2),
                })
            frame_angles[str(out_idx)] = current_frame_data

        # 4. 装载抽样后的姿态序列
        kinematic_json['frame_angles'] = frame_angles
        return kinematic_json, sample_indices