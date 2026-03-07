import numpy as np
import torch
import ruptures as rpt
from src.core.types import ArmState
from src.core.physics.gap.costdirection import CostDirection
from src.core.physics.gap.models import LSTMModel, CustomLoss

class GAPSegmentor:
    """
    基于 Pelt 和 LSTM 的动作轨迹切分器。
    自动适应单臂或双臂数据。
    """
    def __init__(self, penalty_value=1.5, threshold=0.4, min_gap=25, min_duration=5, device="cuda"):
        self.penalty_value = penalty_value
        self.threshold = threshold
        self.min_gap = min_gap
        self.min_duration = min_duration
        self.device = device if torch.cuda.is_available() else "cpu"

    def detect_phases(self, arm_states: dict[str, ArmState], epochs=500):
        """
        核心切分逻辑，返回连续动作的帧区间列表 final_phases
        """
        if not arm_states:
            raise ValueError("传入的 arm_states 字典为空！")

        T = len(next(iter(arm_states.values())).pos)
        
        # 1. 动态遍历，收集所有手臂特征并运行 Pelt
        arm_features = []
        all_breakpoints = set()

        for arm_name, state in arm_states.items():
            # 拼接 3+3+2 维特征
            feature = np.concatenate([state.pos, state.rot, state.gripper], axis=-1)
            arm_features.append(feature)
            
            # 单臂 Pelt 检测
            algo = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(feature)
            bp = algo.predict(pen=self.penalty_value)
            all_breakpoints.update(bp)

        combined_breakpoints = sorted(list(all_breakpoints))
        
        # 2. 合并所有手臂特征用于 LSTM 输入
        # 单臂 shape为 (T, 8), 双臂 shape为 (T, 16)
        combined_features = np.concatenate(arm_features, axis=-1)
        input_dim = combined_features.shape[-1]

        # 3. 构造 LSTM 训练数据
        discrete_label = np.zeros(T)
        for cp in combined_breakpoints:
            if cp <= T: discrete_label[cp - 1] = 1.0
                
        cp_for_dist = [1] + [cp for cp in combined_breakpoints if cp < T] + [T]
        traj_weights = np.array([min([abs(i + 1 - point) for point in cp_for_dist]) for i in range(T)], dtype=np.float32)
        traj_weights = 0.001 * (traj_weights ** 2)
        
        pro_diff = np.diff(combined_features, axis=0)
        td_pro = np.vstack([np.zeros((1, input_dim)), pro_diff]) 
        
        x_tensor = torch.from_numpy(td_pro).unsqueeze(0).float().to(self.device) * 1e4 
        y_tensor = torch.from_numpy(discrete_label).unsqueeze(0).float().to(self.device)
        w_tensor = torch.from_numpy(traj_weights).unsqueeze(0).float().to(self.device)
        len_tensor = torch.tensor([T]).int()

        # 4. 极速训练 LSTM
        model = LSTMModel(input_size=input_dim, hidden_size=64).to(self.device)
        criterion = CustomLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_tensor, len_tensor)
            loss = criterion(outputs, y_tensor, w_tensor)
            loss.backward()
            optimizer.step()

        # 5. 预测与平滑合并
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor, len_tensor).squeeze(0).cpu().numpy()

        window_size = 5
        smoothed_logits = np.convolve(logits, np.ones(window_size)/window_size, mode='same')
        phase_frames = np.where(smoothed_logits > self.threshold)[0]
        
        if len(phase_frames) == 0:
            return [] # 未检测到变点
            
        raw_phases = []
        current_phase = [phase_frames[0]]
        for i in range(1, len(phase_frames)):
            if phase_frames[i] == phase_frames[i-1] + 1:
                current_phase.append(phase_frames[i])
            else:
                raw_phases.append((current_phase[0], current_phase[-1]))
                current_phase = [phase_frames[i]]
        raw_phases.append((current_phase[0], current_phase[-1]))

        # 合并近距离段落并过滤短毛刺
        merged_phases = []
        for p in raw_phases:
            if not merged_phases:
                merged_phases.append(p)
            else:
                last_p = merged_phases[-1]
                if p[0] - last_p[1] <= self.min_gap:
                    merged_phases[-1] = (last_p[0], p[1])
                else:
                    merged_phases.append(p)
                    
        final_phases = [p for p in merged_phases if (p[1] - p[0]) >= self.min_duration]
        
        return final_phases