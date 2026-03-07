import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import ruptures as rpt
import rerun as rr
import re
import json

# 确保能导入你的 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factory import ReaderFactory
from src.core.physics.gap.costdirection import CostDirection
import matplotlib.pyplot as plt

def build_robotics_pamor_prompt(kinematic_json, task_description="Pour water or beverage into the cup using one hand"):
    kinematics_str = json.dumps(kinematic_json, ensure_ascii=False)
    
    prompt = f"""You are an expert in describing robotic motion content for embodied AI. I will give you 32 frames of "video frames" uniformly extracted from a robot manipulation trajectory, input them in chronological order, and provide you with kinematic posture information corresponding to this sequence.
Please analyze the visual content based on the video frames and posture information, and output the motion description of the robot in the video.

TASK GOAL OF THIS EPISODE: {task_description}

The format of the kinematic information is as follows: "{{\"vel\": overall_velocity, \"angle\": overall_angular_velocity, \"vel_score\": ..., \"frame_angles\": {{\"frame_idx\": [{{\"joint_name\": joint_angle}}]}}}}".
The posture information analyzes motion from the perspectives of end-effector translation and joint rotation. 'vel' corresponds to the movement rate of the end-effectors, while 'angle' and Euler angles ('rx', 'ry', 'rz') represent the rotational transformation of the arms. You need to pay special attention to the Euler angles for orientation changes (like pouring) and the 'gripper' value for grasping actions.

The specific description rules are as follows:
1. Please accurately identify all the subjects (e.g., right arm, left arm, right gripper) and objects/backgrounds in the video, and refer to them with specific words.
2. The description of the robot's action needs to be fine-grained. Use precise robotic manipulation verbs (e.g., reach, grasp, retract, rotate, pour, hold) and reflect the intensity and direction of the action.
3. Then output "[{{0, 31}}]" at the beginning of the first line, which means that this sequence description starts from the 0th frame and ends at the 31st frame.
4. We stipulate that movement is divided into robot-level (movement of the overall robot base/torso, if any), arm-level (movement of the main robotic arms in 3D space), and gripper-level (movement of the end-effectors/grippers, such as opening, closing, or holding). Please output "robot-level" in the second line, then output all robot-level information, output "arm-level" in a new line, output all arm-level information, and then output "gripper-level" in a new line, and output all gripper-level information.
5. Output all the moving subjects you can observe by line, using the format we call motion-unit, which is "[{{begin_frame, end_frame}}, (motion_subject, motion, motion_object, motion_adverbial, motion_amplitude), (motion_subject, modifiers_subject), (motion_object, modifiers_object)]", where the first unit indicates the start and end frame of the motion. The second unit represents the subject of the action, the action description, the receptor of the action, the adverbial of the action, and the amplitude of the action. The third unit represents the modifier of the subject, and the fourth unit represents the modifier of the receptor. Each action is output in one line.
6. For the description of direction, please use the camera-centered or robot-centered perspective (e.g., "toward the cup", "downward").
7. If a robotic arm or gripper remains motionless in the video, please use the same format to describe its state (e.g., holding still).
8. Please use English to answer, no need to worry about the length limit.
9. This is an explanation of each specific element in motion-unit:
   - motion_subject: the agent of motion (e.g., right_arm, left_gripper)
   - motion_object: the object being manipulated (e.g., cup, water_pitcher, none)
   - motion: the specific manipulation verb (e.g., rotate, pour, move)
   - motion_adverbial: the direction or spatial relation (e.g., downward, towards the table)
   - motion_amplitude: the speed or intensity (e.g., slow, moderate, steady, fast)
   - modifiers_subject: feature description of the robot part (e.g., right manipulator)
   - modifiers_object: feature description of the object (e.g., red cup, clear bottle, none)
10. All kinematic values in the posture information correspond to specific fine-grained motions. Combined with the posture information and the video frame content, accurately define the start and end frames of each motion unit. Pay special attention to sudden jumps in Euler angles (which often indicate actions like pouring/flipping) and stabilize the boundaries.

The kinematic posture information provided is as follows:
{kinematics_str}

Your English description is:
"""
    return prompt

def compute_chunk_kinematics(chunk_data, fps=30, num_samples=32):
    T = len(chunk_data)
    dt = 1.0 / fps
    if T < 2:
        return {}, np.array([0]*num_samples)
        
    pos_r, rot_r, grip_r = chunk_data[:, 0:3], chunk_data[:, 3:6], chunk_data[:, 6]
    pos_l, rot_l, grip_l = chunk_data[:, 8:11], chunk_data[:, 11:14], chunk_data[:, 14]

    vel_r = np.linalg.norm(np.diff(pos_r, axis=0), axis=1) / dt
    vel_l = np.linalg.norm(np.diff(pos_l, axis=0), axis=1) / dt
    vel_cm = (vel_r + vel_l) / 2.0  
    vel_cm = np.insert(vel_cm, 0, 0.0) 

    ang_vel_r = np.linalg.norm(np.diff(rot_r, axis=0), axis=1) / dt
    ang_vel_l = np.linalg.norm(np.diff(rot_l, axis=0), axis=1) / dt
    ang_vel_cm = (ang_vel_r + ang_vel_l) / 2.0
    ang_vel_cm = np.insert(ang_vel_cm, 0, 0.0)

    vel_fft_raw = np.abs(np.fft.rfft(vel_cm))
    ang_fft_raw = np.abs(np.fft.rfft(ang_vel_cm))
    
    vel_score = np.std(vel_fft_raw) if len(vel_fft_raw) > 1 else 0.0
    angle_score = np.std(ang_fft_raw) if len(ang_fft_raw) > 1 else 0.0

    vel_fft_list = (vel_fft_raw / T).tolist()[:20] 
    angle_fft_list = (ang_fft_raw / T).tolist()[:20]

    sample_indices = np.linspace(0, T - 1, num=num_samples, dtype=int)
    
    frame_angles = {}
    for out_idx, frame_idx in enumerate(sample_indices):
        frame_data = {
            'r_arm_rx': round(float(rot_r[frame_idx, 0]), 2),
            'r_arm_ry': round(float(rot_r[frame_idx, 1]), 2),
            'r_arm_rz': round(float(rot_r[frame_idx, 2]), 2),
            'r_gripper': round(float(grip_r[frame_idx]), 2),
            'l_arm_rx': round(float(rot_l[frame_idx, 0]), 2),
            'l_arm_ry': round(float(rot_l[frame_idx, 1]), 2),
            'l_arm_rz': round(float(rot_l[frame_idx, 2]), 2),
            'l_gripper': round(float(grip_l[frame_idx]), 2)
        }
        frame_angles[str(out_idx)] = [frame_data]

    kinematic_json = {
        'vel': round(float(np.mean(vel_cm)), 2),
        'angle': round(float(np.mean(ang_vel_cm)), 2),
        'vel_score': round(float(vel_score), 2),
        'angle_score': round(float(angle_score), 2),
        'frame_angles': frame_angles,
        'vel-fft': [round(x, 3) for x in vel_fft_list],
        'angle_fft': [round(x, 3) for x in angle_fft_list]
    }
    
    return kinematic_json, sample_indices

# ==========================================
# 新增：VLM 文本解析与绝对帧映射工具
# ==========================================
def parse_and_map_vlm_output(vlm_text, global_indices):
    parsed_actions = []
    current_level = "unknown"
    
    # 改进后的正则表达式：
    # 1. \[ \{? (\d+) , \s* (\d+) \}? \]  --> 匹配 [{0, 31}] 或 [0, 31]，花括号设为可选
    # 2. (?: \s* , \s* )?                --> 匹配中间可能存在的逗号和任意空格
    # 3. \( ([^)]+) \)                   --> 抓取括号内直到右括号的所有内容
    pattern = r'\[\{(\d+),\s*(\d+)\},\s*\(([^)]+)\)'
    
    for line in vlm_text.split('\n'):
        line = line.strip()
        if not line: continue
            
        # 识别层级
        if "robot-level" in line.lower(): current_level = "robot-level"; continue
        if "arm-level" in line.lower(): current_level = "arm-level"; continue
        if "gripper-level" in line.lower(): current_level = "gripper-level"; continue
            
        # 执行匹配
        match = re.search(pattern, line)
        if match:
            local_start = int(match.group(1))
            local_end = int(match.group(2))
            
            # 这里的 group(3) 就是 (robot_base, hold_still, ...) 括号里的内容
            content_str = match.group(3)
            
            # 统一将冒号替换为逗号，防止 VLM 乱用标点，然后按逗号分割
            elements = [e.strip() for e in content_str.replace(':', ',').split(',')]
            
            # 只要元素够 5 个（subject, action, object, direction, amplitude）就抓取
            if len(elements) >= 5:
                # 索引映射
                local_start = max(0, min(local_start, len(global_indices) - 1))
                local_end = max(0, min(local_end, len(global_indices) - 1))
                global_start = int(global_indices[local_start])
                global_end = int(global_indices[local_end])
                
                parsed_actions.append({
                    "level": current_level,
                    "global_start_frame": global_start,
                    "global_end_frame": global_end,
                    "subject": elements[0],
                    "action_verb": elements[1],
                    "object": elements[2],
                    "direction": elements[3],
                    "amplitude": elements[4]
                })
                
    return parsed_actions

def plot_phase_detection(logits, breakpoints, final_phases, threshold=0.4, title="Action Phase Detection"):
    plt.figure(figsize=(15, 6))
    plt.plot(logits, label='LSTM Probability (ρ)', color='#1f77b4', linewidth=1.5)
    for i, (start, end) in enumerate(final_phases):
        plt.axvspan(start, end, color='orange', alpha=0.3, label='Merged Phase Segments' if i == 0 else "")
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.6, label=f'Threshold ({threshold})')
    for i, cp in enumerate(breakpoints):
        plt.axvline(x=cp, color='gray', linestyle=':', alpha=0.4, label='Pelt Breakpoints' if i == 0 else "")

    plt.title(title, fontsize=14)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

class LSTMModel(nn.Module):
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
        adjusted_loss = - (target * torch.log(pred + 1e-8) + penalty_factor * (1 - target) * torch.log(1 - pred + 1e-8))
        return adjusted_loss.mean()

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

def test_lstm_smoothing():
    DATA_PATH = "/home/user/test_data/lerobot/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink" 
    use_lstm = False
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

    penalty_value = 1.5
    right_arm_data = extract_single_arm(qpos_array, "right")
    algo_r = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(right_arm_data)
    breakpoints_r = algo_r.predict(pen=penalty_value)
    
    left_arm_data = extract_single_arm(qpos_array, "left")
    algo_l = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(left_arm_data)
    breakpoints_l = algo_l.predict(pen=penalty_value)
    
    combined_breakpoints = sorted(list(set(breakpoints_r) | set(breakpoints_l)))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discrete_label = np.zeros(traj_len)
    for cp in combined_breakpoints:
        if cp <= traj_len:
            discrete_label[cp - 1] = 1.0
            
    cp_for_dist = [1] + [cp for cp in combined_breakpoints if cp < traj_len] + [traj_len]
    traj_weights = np.array([min([abs(i + 1 - point) for point in cp_for_dist]) for i in range(traj_len)], dtype=np.float32)
    traj_weights = 0.001 * (traj_weights ** 2)
    
    both_arms_data = np.concatenate([right_arm_data, left_arm_data], axis=-1)
    pro_diff = np.diff(both_arms_data, axis=0)
    td_pro = np.vstack([np.zeros((1, both_arms_data.shape[1])), pro_diff]) 
    
    x_tensor = torch.from_numpy(td_pro).unsqueeze(0).float().to(device) * 1e4 
    y_tensor = torch.from_numpy(discrete_label).unsqueeze(0).float().to(device)
    w_tensor = torch.from_numpy(traj_weights).unsqueeze(0).float().to(device)
    len_tensor = torch.tensor([traj_len]).int()

    print(f"🧠 开始训练双臂 LSTM (Input Size: 16, 设备: {device})...")
    model = LSTMModel(input_size=16, hidden_size=64).to(device)
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(x_tensor, len_tensor)
        loss = criterion(outputs, y_tensor, w_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, len_tensor).squeeze(0).cpu().numpy()

    threshold = 0.4 
    window_size = 5
    smoothed_logits = np.convolve(logits, np.ones(window_size)/window_size, mode='same')
    phase_frames = np.where(smoothed_logits > threshold)[0]
    
    if len(phase_frames) == 0:
        print(f"未检测到显著的动作转换期 (当前阈值: {threshold})")
        return
        
    raw_phases = []
    current_phase = [phase_frames[0]]
    for i in range(1, len(phase_frames)):
        if phase_frames[i] == phase_frames[i-1] + 1:
            current_phase.append(phase_frames[i])
        else:
            raw_phases.append((current_phase[0], current_phase[-1]))
            current_phase = [phase_frames[i]]
    raw_phases.append((current_phase[0], current_phase[-1]))

    min_gap = 25
    merged_phases = []
    for p in raw_phases:
        if not merged_phases:
            merged_phases.append(p)
        else:
            last_p = merged_phases[-1]
            if p[0] - last_p[1] <= min_gap:
                merged_phases[-1] = (last_p[0], p[1])
            else:
                merged_phases.append(p)
                
    min_duration = 5
    final_phases = [p for p in merged_phases if (p[1] - p[0]) >= min_duration]

    print("\n🤖 正在启动 VLM 进行语义打标...")
    
    from PIL import Image as PILImage
    from src.core.vlm_caller import QwenVLCaller 
    
    vlm_caller = QwenVLCaller() 
    
    action_chunks = []
    if use_lstm:
        last_end = 0
        for p_start, p_end in final_phases:
            action_chunks.append({
                "chunk_start": last_end,
                "chunk_end": p_end,
                "stable_start": last_end,
                "stable_end": max(last_end, p_start - 1)
            })
            last_end = p_end + 1
            
        if last_end < traj_len:
            action_chunks.append({
                "chunk_start": last_end,
                "chunk_end": traj_len - 1,
                "stable_start": last_end,
                "stable_end": traj_len - 1
            })
    else:
        # 模式 B：不使用 LSTM (适用于 <30s 的短视频)
        # 直接将整个视频作为一个整体发送给 VLM
        action_chunks.append({
            "chunk_start": 0,
            "chunk_end": traj_len - 1,
            "stable_start": 0,
            "stable_end": traj_len - 1
        })
        
    # ===============================================
    # 核心新增：存储全局映射后的动作结果
    # ===============================================
    global_dataset_annotations = []
    
    task_description = "Pour water or beverage into the cup using one hand."

    for idx, chunk in enumerate(action_chunks):
        s_start, s_end = chunk["stable_start"], chunk["stable_end"]
        c_start, c_end = chunk["chunk_start"], chunk["chunk_end"]
        
        chunk_absolute_start = s_start
        chunk_absolute_end = c_end
        chunk_dense_data = both_arms_data[chunk_absolute_start : chunk_absolute_end + 1]
        
        kinematic_json, local_sample_indices = compute_chunk_kinematics(chunk_dense_data, fps=30, num_samples=32)
        global_sample_indices = chunk_absolute_start + local_sample_indices
        
        pil_images_for_vlm = []
        for frame_idx in global_sample_indices:
            frame_data = reader.get_frame(int(frame_idx))
            if frame_data is None: continue
            cam_name = 'cam_front_head_rgb' if 'cam_front_head_rgb' in frame_data.images else list(frame_data.images.keys())[0]
            img_array = frame_data.images[cam_name]
            
            try:
                if not isinstance(img_array, np.ndarray): img_array = np.array(img_array)
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]: img_array = np.transpose(img_array, (1, 2, 0))
                if img_array.dtype in [np.float32, np.float64]: img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                elif img_array.dtype != np.uint8: img_array = img_array.astype(np.uint8)
                pil_images_for_vlm.append(PILImage.fromarray(img_array))
            except Exception:
                pass

        if not pil_images_for_vlm: continue
        print(f"\n📦 [Chunk {idx+1}/{len(action_chunks)}] 物理数据生成成功！提取了 {len(pil_images_for_vlm)} 帧。")
        
        prompt = build_robotics_pamor_prompt(kinematic_json, task_description=task_description)
        
        try:
            semantic_label = vlm_caller.generate(prompt=prompt, images=pil_images_for_vlm)
            
            # 打印 VLM 原始输出以供检查
            print("   💬 VLM 原始输出:")
            print(semantic_label)

            # ===============================================
            # 核心新增：调用解析器，将局部索引转换为全局绝对帧号
            # ===============================================
            mapped_actions = parse_and_map_vlm_output(
                vlm_text=semantic_label, 
                global_indices=global_sample_indices
            )
            
            # 将该 Chunk 解析出来的动作合并到全局列表
            global_dataset_annotations.extend(mapped_actions)
            
            print(f"   ✅ 成功解析并映射了 {len(mapped_actions)} 个绝对帧动作：")
            for act in mapped_actions:
                print(f"      [{act['global_start_frame']:>4} -> {act['global_end_frame']:>4}] {act['subject']:>15} | {act['action_verb']:>12} | {act['object']}")

        except Exception as e:
            print(f"   ❌ VLM 调用或解析出错: {e}")

    print("\n✅ 所有 Action Chunks 标注与映射完成。")
    
    # ===============================================
    # 核心新增：将整个轨迹的精细化标注保存为 JSON
    # ===============================================
    if global_dataset_annotations:
        output_path = os.path.join(os.path.dirname(DATA_PATH), "auto_annotations.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(global_dataset_annotations, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 完美！含有绝对物理帧的高质量动作标签已保存至: {output_path}")

if __name__ == "__main__":
    test_lstm_smoothing()