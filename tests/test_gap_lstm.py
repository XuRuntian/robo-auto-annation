import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import ruptures as rpt
import rerun as rr
# 确保能导入你的 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factory import ReaderFactory
from src.core.physics.gap.costdirection import CostDirection
import matplotlib.pyplot as plt

def plot_phase_detection(logits, breakpoints, final_phases, threshold=0.4, title="Action Phase Detection"):
    plt.figure(figsize=(15, 6))
    
    # 1. 绘制概率曲线 (蓝色)
    plt.plot(logits, label='LSTM Probability (ρ)', color='#1f77b4', linewidth=1.5)
    
    # 2. 填充合并/过滤后的 Phase 区域 (橙色)
    for i, (start, end) in enumerate(final_phases):
        plt.axvspan(start, end, color='orange', alpha=0.3, 
                    label='Merged Phase Segments' if i == 0 else "")
    
    # 3. 绘制阈值线 (红色虚线)
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.6, label=f'Threshold ({threshold})')
    
    # 4. 绘制原始 Pelt 变点 (灰色垂直点线)
    for i, cp in enumerate(breakpoints):
        plt.axvline(x=cp, color='gray', linestyle=':', alpha=0.4, 
                    label='Pelt Breakpoints' if i == 0 else "")

    plt.title(title, fontsize=14)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
# ==========================================
# 1. 搬运自原论文的 LSTM 模型和 Loss
# ==========================================
class LSTMModel(nn.Module):
    # input_size 改为 16 (左臂8维 + 右臂8维)
    def __init__(self, input_size=16, hidden_size=64, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        max_len = x.shape[1]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        output = self.fc(output)
        logits = self.sigmoid(output)
        return logits.squeeze(-1)

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, penalty_factor):
        # 纵容变点附近的假阳性，惩罚远处的假阳性
        adjusted_loss = - (target * torch.log(pred + 1e-8) + penalty_factor * (1 - target) * torch.log(1 - pred + 1e-8))
        return adjusted_loss.mean()

# ==========================================
# 2. 数据处理工具
# ==========================================
def extract_single_arm(qpos_array, arm_type="right"):
    arm_data = []
    for point in qpos_array:
        if arm_type == "right":
            move = point[20:23]
            rotate = point[23:26]
            gripper_val = point[19]
        else:
            move = point[7:10]
            rotate = point[10:13]
            gripper_val = point[6]
        gripper = np.array([gripper_val, gripper_val])
        arm_data.append(np.concatenate([move, rotate, gripper]))
    return np.array(arm_data)

# ==========================================
# 3. 主测试逻辑
# ==========================================
def test_lstm_smoothing():
    DATA_PATH = "/home/user/test_data/lerobot/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink" 
    
    print(f"🚀 加载数据: {DATA_PATH}")
    reader = ReaderFactory.get_reader(DATA_PATH)
    reader.load(DATA_PATH)
    
    qpos_list = []
    for i in range(reader.get_length()):
        frame = reader.get_frame(i)
        state = getattr(frame, 'state', None)
        raw_state = state.get('qpos', []) if state is not None else getattr(frame, 'qpos', [])
        qpos_list.append(raw_state)
    qpos_array = np.array(qpos_list)
    traj_len = len(qpos_array)

    # ================= 1. 运行双臂底层变点检测 (System 1) =================
    penalty_value = 1.5
    
    # 检测右臂
    right_arm_data = extract_single_arm(qpos_array, "right")
    algo_r = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(right_arm_data)
    breakpoints_r = algo_r.predict(pen=penalty_value)
    
    # 检测左臂
    left_arm_data = extract_single_arm(qpos_array, "left")
    algo_l = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(left_arm_data)
    breakpoints_l = algo_l.predict(pen=penalty_value)
    
    # 合并双臂变点（取并集并排序）
    combined_breakpoints = sorted(list(set(breakpoints_r) | set(breakpoints_l)))
    print(f"✅ [Pelt] 右臂变点数: {len(breakpoints_r)}, 左臂变点数: {len(breakpoints_l)}")
    print(f"✅ [Pelt] 合并后总变点: {combined_breakpoints}")

    # ================= 2. 准备 LSTM 训练数据 =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2.1 构造弱标签 Target
    discrete_label = np.zeros(traj_len)
    for cp in combined_breakpoints:
        if cp <= traj_len:
            discrete_label[cp - 1] = 1.0
            
    # 2.2 构造抛物线权重 Penalty
    cp_for_dist = [1] + [cp for cp in combined_breakpoints if cp < traj_len] + [traj_len]
    traj_weights = np.array([min([abs(i + 1 - point) for point in cp_for_dist]) for i in range(traj_len)], dtype=np.float32)
    traj_weights = 0.001 * (traj_weights ** 2)
    
    # 2.3 融合双臂输入特征 (8维 + 8维 = 16维)
    both_arms_data = np.concatenate([right_arm_data, left_arm_data], axis=-1)
    pro_diff = np.diff(both_arms_data, axis=0)
    td_pro = np.vstack([np.zeros((1, both_arms_data.shape[1])), pro_diff]) 
    
    x_tensor = torch.from_numpy(td_pro).unsqueeze(0).float().to(device) * 1e4 
    y_tensor = torch.from_numpy(discrete_label).unsqueeze(0).float().to(device)
    w_tensor = torch.from_numpy(traj_weights).unsqueeze(0).float().to(device)
    len_tensor = torch.tensor([traj_len]).int()

    # ================= 3. 极速训练 LSTM =================
    print(f"🧠 开始训练双臂 LSTM (Input Size: 16, 设备: {device})...")
    model = LSTMModel(input_size=16, hidden_size=64).to(device)
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(500): # 增加到 150 轮以适应更复杂的双臂特征
        optimizer.zero_grad()
        outputs = model(x_tensor, len_tensor)
        loss = criterion(outputs, y_tensor, w_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}/150, Loss: {loss.item():.4f}")

    # ================= 4. 预测与分析 =================
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, len_tensor).squeeze(0).cpu().numpy()

    print("\n================ 最终平滑效果 (双臂融合) ================")
    threshold = 0.4 
    
    # 1. 可选：对概率曲线做一次轻微的滑动平均平滑 (减少横跳)
    window_size = 5
    smoothed_logits = np.convolve(logits, np.ones(window_size)/window_size, mode='same')
    
    phase_frames = np.where(smoothed_logits > threshold)[0]
    
    if len(phase_frames) == 0:
        print(f"未检测到显著的动作转换期 (当前阈值: {threshold})")
        return
        
    # 2. 提取初始的散碎段落
    raw_phases = []
    current_phase = [phase_frames[0]]
    for i in range(1, len(phase_frames)):
        if phase_frames[i] == phase_frames[i-1] + 1:
            current_phase.append(phase_frames[i])
        else:
            raw_phases.append((current_phase[0], current_phase[-1]))
            current_phase = [phase_frames[i]]
    raw_phases.append((current_phase[0], current_phase[-1]))

    # 3. 核心修复 1：合并距离过近的段落 (Merge)
    min_gap = 10  # ⚡ 关键参数：间隔少于 25 帧的波峰强制合并
    merged_phases = []
    for p in raw_phases:
        if not merged_phases:
            merged_phases.append(p)
        else:
            last_p = merged_phases[-1]
            if p[0] - last_p[1] <= min_gap:
                # 合并：修改最后一个 phase 的结束时间
                merged_phases[-1] = (last_p[0], p[1])
            else:
                merged_phases.append(p)
                
    # 4. 核心修复 2：滤除持续时间极短的孤立毛刺 (Filter)
    min_duration = 5 # ⚡ 关键参数：抛弃持续少于 5 帧的孤立假阳性
    final_phases = [p for p in merged_phases if (p[1] - p[0]) >= min_duration]

    for idx, (start, end) in enumerate(final_phases):
        print(f"🎯 [Merged Phase {idx+1}] Frame {start} -> {end} (持续 {end-start} 帧)")
        
    # 调用更新后的画图函数
    # plot_phase_detection(smoothed_logits, combined_breakpoints, final_phases, threshold=threshold)
    # ================= 5. Rerun 同步可视化 (集成相机) =================
    import rerun as rr

    print("\n📺 正在启动 Rerun 播放器并同步相机数据...")
    rr.init("Robot_Action_Annotation_Viewer", spawn=True)
    
    # [修正] 使用 colors (复数) 且记录为静态样式
    rr.log("analysis/probability", rr.SeriesLines(colors=[[31, 119, 180]], names=["Prob"]), static=True)
    rr.log("analysis/threshold", rr.SeriesLines(colors=[[255, 0, 0]], names=["Threshold"], widths=[2.0]), static=True)

    for i in range(traj_len):
        # [优化] 根据 Rerun 警告使用更现代的时间轴记录方式
        rr.set_time("frame_idx", sequence=i)        
        # 获取当前帧数据 (LeRobotAdapter.get_frame)
        frame_data = reader.get_frame(i)
        if frame_data is None:
            continue

        # 1. 记录标量曲线
        rr.log("analysis/probability", rr.Scalars(smoothed_logits[i]))
        rr.log("analysis/threshold", rr.Scalars(threshold))

        # 2. 核心：记录相机图像数据
        # frame_data.images 是一个 {相机名: RGB数组} 的字典
        for cam_name, img_array in frame_data.images.items():
            if img_array is not None:
                # LeRobotAdapter 已将图像转为 RGB，可以直接记录
                # 路径结构设为 camera/名称，方便在 Rerun 界面排列
                rr.log(f"camera/{cam_name}", rr.Image(img_array))

        # 3. 记录物理变点事件 (可选)
        if (i + 1) in combined_breakpoints:
            rr.log("logs/events", rr.TextLog(f"Breakpoint detected", level="INFO"))

        # 4. 记录当前 Phase 状态文字
        active_phase = "Steady"
        for idx, (p_start, p_end) in enumerate(final_phases):
            if p_start <= i <= p_end:
                active_phase = f"Phase {idx+1}"
                break
        rr.log("logs/phase", rr.TextLog(active_phase))

    print("✅ Rerun 同步完成。请在 Rerun UI 中查看多视角图像与曲线。")

    print("✅ Rerun 数据流传输完成。")

# ================= 5. 结合 VLM 提取高级语义 (Physics-First Labeling) =================
    print("\n🤖 正在启动 VLM 进行语义打标...")
    
    # 统一导入，避免作用域问题
    from PIL import Image as PILImage
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.core.vlm_caller import QwenVLCaller 
    
    # 初始化 VLM
    vlm_caller = QwenVLCaller() 
    
    # 1. 根据物理 Phase 边界，生成连续的 Action Chunks
    # 这里我们认为：[上一个变点结束, 当前变点结束] 为一个完整动作块
    action_chunks = []
    last_end = 0
    for p_start, p_end in final_phases:
        action_chunks.append({
            "chunk_start": last_end,
            "chunk_end": p_end,
            # 💡 策略：在动作转换期（p_start-p_end）之前的“平稳期”抽帧，识别更准
            "stable_start": last_end,
            "stable_end": max(last_end, p_start - 1)
        })
        last_end = p_end + 1
        
    # 处理最后一个收尾的 Chunk
    if last_end < traj_len:
        action_chunks.append({
            "chunk_start": last_end,
            "chunk_end": traj_len - 1,
            "stable_start": last_end,
            "stable_end": traj_len - 1
        })
    previous_label = "None (task just started)"
    all_final_labels = []
    # 2. 遍历每个 Chunk，智能抽帧并让 VLM 命名
    for idx, chunk in enumerate(action_chunks):
        c_start, c_end = chunk["chunk_start"], chunk["chunk_end"]
        s_start, s_end = chunk["stable_start"], chunk["stable_end"]
        
        # 即使平稳期很短，也确保能抽出帧
        if s_end < s_start:
            s_end = s_start
            
        # 均匀抽取 3 个关键帧
        sample_indices = np.linspace(s_start, s_end, num=3, dtype=int)
        
        # ✅ 核心修复：这个列表只存放转换后的 PIL 对象
        pil_images_for_vlm = []
        
        for frame_idx in sample_indices:
            frame_data = reader.get_frame(frame_idx)
            if frame_data is None:
                continue

            # 1. 选择相机 (优先选头部/前视相机)
            cam_name = 'cam_front_head_rgb' if 'cam_front_head_rgb' in frame_data.images else list(frame_data.images.keys())[0]
            img_array = frame_data.images[cam_name]
            
            # 2. 稳健的图像转换流程
            try:
                # 确保是 Numpy 数组
                if not isinstance(img_array, np.ndarray):
                    img_array = np.array(img_array)
                
                # 处理维度顺序 (C, H, W) -> (H, W, C)
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                    img_array = np.transpose(img_array, (1, 2, 0))
                
                # 处理数据类型和值域 (0.0-1.0 -> 0-255)
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
                
                # 转换为 PIL 并添加到列表
                pil_img = PILImage.fromarray(img_array)
                pil_images_for_vlm.append(pil_img)
                
            except Exception as e:
                print(f"⚠️ 无法处理第 {frame_idx} 帧图像: {e}")

        if not pil_images_for_vlm:
            print(f"⏩ Chunk {idx+1} 无有效图像，跳过。")
            continue

        # 3. 构造给 VLM 的 Prompt
        # 1. 定义动态的任务描述变量（可以从数据集中读取）
        
# ==================== 3. 构造给 VLM 的进度感知 Prompt (泛用型) ====================
        
        # 1. 任务全局描述 (支持任意任务)
        task_description = "Pour water or beverage into the cup using one hand."
        
        # 2. 核心创新：计算动态进度条并转化为语言提示 (Temporal Grounding)
        # 1. 语义化位置描述（非硬性阈值）
        total_chunks = len(action_chunks)
        if idx == 0:
            position_context = "This is the initial interaction with the environment."
        elif idx == total_chunks - 1:
            position_context = "This is the final segment of the recorded sequence."
        else:
            # 描述它在序列中的相对位置，但不下定义
            position_context = f"This is an intermediate segment (Chunk {idx+1} of {total_chunks})."

        # 2. 引入物理动能描述 (利用 LSTM 的输出或物理差异)
        # 如果该 Chunk 覆盖了 LSTM 的概率波峰，说明它是“动作剧烈转换期”
        in_phase = any(p_start <= c_start <= p_end for p_start, p_end in final_phases)
        dynamic_hint = "Noticeable state transition detected via physical sensors." if in_phase else "The physical state is relatively stable."

        # 3. 整理全局历史，让 VLM 拥有时序记忆
        history_str = " -> ".join(all_final_labels) if all_final_labels else "None (Task just started)"
        
        # 4. 构造强逻辑约束的 Prompt，纯靠提示工程与进度防跳变
        prompt = f"""You are an expert roboticist.
        TASK GOAL: {task_description}

        SCENE CONTEXT:
        - Sequence Position: {position_context}
        - Physical Dynamics: {dynamic_hint}
        - Previous Action: {all_final_labels[-1] if all_final_labels else "None (Start)"}

        OBSERVATION STRATEGY:
        1. Compare the 3 images to identify the movement direction and state change.
        2. If the 'Physical Dynamics' hint is 'stable', the robot is likely continuing the previous action or performing a steady-state subtask (e.g., constant pouring).
        3. If the 'Physical Dynamics' hint is 'transition', look for a change in contact, grasp, or trajectory.
        4. Align your description with the overall goal: Is the robot approaching, interacting, or withdrawing?

        OUTPUT REQUIREMENT:
        Provide a concise imperative action label (e.g., 'move to cup', 'pour water', 'retract arm'). 
        NO punctuation. NO chatter.
        """
        
        print(f"\n📦 [Chunk {idx+1}/{len(action_chunks)}] Frame {c_start} -> {c_end} (抽帧: {sample_indices})")
        print(f"   📖 历史记忆: [{history_str}]")
        
        # 4. 调用 VLM
        try:
            semantic_label = vlm_caller.generate(prompt=prompt, images=pil_images_for_vlm)
            
            # 清洗标签（去除多余标点和换行）
            semantic_label = semantic_label.strip().replace('.', '').replace('"', '').replace("'", "")

            print(f"   💬 VLM 语义标签: {semantic_label}")

            # 更新历史
            previous_label = semantic_label
            chunk["instruction"] = semantic_label
            all_final_labels.append(semantic_label)
            
        except Exception as e:
            print(f"   ❌ VLM 调用失败: {e}")

    print("\n✅ 所有 Action Chunks 标注完成。")
if __name__ == "__main__":
    test_lstm_smoothing()