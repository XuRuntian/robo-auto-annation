import sys
import os
import numpy as np
import ruptures as rpt

# 确保能导入你的 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factory import ReaderFactory
# 假设你把 costdirection.py 放在了这里
from src.core.physics.gap.costdirection import CostDirection

def extract_single_arm(qpos_array, arm_type="right"):
    """
    把 26 维的混合数据，抽取成单臂的标准 8 维位姿 (3平移 + 3旋转 + 2夹爪)
    """
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
            
        # ⚠️ 关键技巧：CostDirection 源码是用 point[-2:] 提取夹爪
        # 为了不改源码且保持维度对齐，我们将 1 维的夹爪复制成 2 维 [gripper_val, gripper_val]
        gripper = np.array([gripper_val, gripper_val])
        
        # 拼成单臂的 8 维数组：前3平移，中间3旋转，最后2夹爪
        arm_data.append(np.concatenate([move, rotate, gripper]))
        
    return np.array(arm_data)


def test_gap_cpd():
    DATA_PATH = "/home/user/test_data/lerobot/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink" 
    
    print(f"🚀 加载数据: {DATA_PATH}")
    reader = ReaderFactory.get_reader(DATA_PATH)
    if not reader.load(DATA_PATH):
        print("❌ 加载失败")
        return

    length = reader.get_length()
    qpos_list = []
    
    # 提取所有帧的物理状态数据
    for i in range(length):
        frame = reader.get_frame(i)
        state = getattr(frame, 'state', None)
        if state is not None:
            raw_state = state.get('qpos', [])
        else:
            raw_state = getattr(frame, 'qpos', [])
        qpos_list.append(raw_state)
        
    qpos_array = np.array(qpos_list)
    print(f"📊 提取原始数据形状: {qpos_array.shape}")

    penalty_value = 1.7
    print(f"⚙️  使用 PELT 算法，penalty={penalty_value}")
    # ================= 1. 检测右臂 (Right Arm) =================
    print("\n🧠 开始运行【右臂】 CostDirection 变点检测...")
    right_arm_data = extract_single_arm(qpos_array, arm_type="right")
    
    algo_right = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(right_arm_data)
    breakpoints_right = algo_right.predict(pen=penalty_value)
    print(f"✅ 【右臂】检测完成！变点索引: {breakpoints_right}")

    # ================= 2. 检测左臂 (Left Arm) =================
    print("\n🧠 开始运行【左臂】 CostDirection 变点检测...")
    left_arm_data = extract_single_arm(qpos_array, arm_type="left")
    
    algo_left = rpt.Pelt(custom_cost=CostDirection(), min_size=10).fit(left_arm_data)
    breakpoints_left = algo_left.predict(pen=penalty_value)
    print(f"✅ 【左臂】检测完成！变点索引: {breakpoints_left}")

if __name__ == "__main__":
    test_gap_cpd()