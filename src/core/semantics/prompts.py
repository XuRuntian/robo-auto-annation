import json
import re
import copy

def build_subtask_summary_prompt(task_description, wsm_trajectory, kpm_annotations, total_frames, video_id="video_001"):
    wsm_str = json.dumps(wsm_trajectory, ensure_ascii=False)
    kpm_str = json.dumps(kpm_annotations, ensure_ascii=False)
    
    prompt = f"""
    You are an embodied AI specialist. Your goal is to synthesize low-level kinematic data and high-level world states into a coherent, causal narrative of a robot's task.

    ### 1. Analysis Framework (Causal Logic)
    To understand the "Why" and "How", follow this logic:
    - **Trigger**: Look at `environment_topology_state` and `object_physical_state` in WSM. When a state changes (e.g., "on table" -> "being lifted"), a subtask boundary exists.
    - **Mechanism**: Look at KPM to find the specific `action_verb` (e.g., "rotate_clockwise") and `subject` (e.g., "right_arm") that caused that state change.
    - **Consistency**: Ensure the `macro_skill_id` (WSM) aligns with the physical motion (KPM).

    ### 2. Inputs
    - **Task Description**: {task_description}
    - **WSM (World State Machine)**: {wsm_str}
    - **KPM (Kinematic Physical Motions)**: {kpm_str}
    - **Metadata**: Video ID: {video_id}, Total Frames: {total_frames}

    ### 3. Segmentation & Instruction Rules
    1. **Identify Pivot Frames**: Use WSM's `chunk_idx` and `execution_phase` as the primary segmentation guide.
    2. **Synthesize Descriptions**:
       - **Action**: Use a precise verb from KPM (e.g., "Tilt" instead of "Move").
       - **Object**: Use specific modifiers from KPM (e.g., "clear plastic bottle with red label").
       - **Causality**: The instruction must reflect the *intent*. (e.g., if the state changes to "partially filled", the instruction should be "Pour the liquid into the cup").
    3. **Merge Policy**: If consecutive WSM chunks have the same `macro_skill_id` and the KPM shows continuous motion toward the same goal, merge them into one meaningful subtask.

    ### 4. Output Format
    Return ONLY a JSON object:
    ```json
    {{
      "video_id": "{video_id}",
      "nframes": {total_frames},
      "logical_flow": "Briefly describe the causal chain of the task (e.g., Reach -> Grasp -> Pour -> Reset)",
      "segments": [
        {{
          "seg_id": 0,
          "start_frame": 0,
          "end_frame": 160,
          "instruction": "...",
          "causal_intent": "The robot moves the right arm toward the bottle to prepare for grasping."
        }}
      ]
    }}
    ```
    """
    return prompt
def build_robotics_pamor_prompt(kinematic_json, task_description, world_state_dict):
    """
    注意：这里的 task_history 参数被替换为了 world_state_dict (即当前的世界状态机图)
    """
    kinematics_str = json.dumps(kinematic_json, ensure_ascii=False)
    state_str = json.dumps(world_state_dict, indent=2) # 将状态机转为漂亮格式的字符串
    
    active_arms = [k for k in kinematic_json.keys() if k != 'frame_angles']
    arm_names_str = " and ".join([f"'{arm}'" for arm in active_arms])
    json_format_example = {
        "arm_name": {"vel": "...", "angle": "...", "vel_score": "..."},
        "frame_angles": {"frame_idx": {"r_arm_rx": "...", "l_arm_rx": "..."}}
    }
    json_example_str = json.dumps(json_format_example)
    
    # 将 JSON 模板提供给 VLM，让它知道需要输出什么格式
    wsm_template = """
    ```json
    {
      "temporal_context": {"macro_skill_id": "...", "execution_phase": "..."},
      "robot_interaction_state": {
        "right_end_effector": {"contact_target": "...", "grasp_type": "...", "is_constrained": false},
        "left_end_effector": {"contact_target": "...", "grasp_type": "...", "is_constrained": false}
      },
      "environment_topology_state": [
        {"subject": "...", "relation": "...", "target": "..."} 
      ],
      "object_physical_state": [
        {"object": "...", "property_changed": "...", "current_value": "..."}
      ]
    }
    ```
    """
    prompt = f"""You are an expert in describing robotic motion content for embodied AI. I will give you 32 frames of "video frames" uniformly extracted from a robot manipulation trajectory, input them in chronological order, and provide you with kinematic posture information corresponding to this sequence.
    Please analyze the visual content based on the video frames and posture information, and output the motion description of the robot in the video.

    TASK GOAL OF THIS EPISODE: {task_description}
    WORLD STATE: {state_str}
    The format of the kinematic information is as follows: The format of the kinematic information is as follows: 
    The provided data contains information for {len(active_arms)} arm(s): {arm_names_str}.
    The format follows this structure: {json_example_str}

    The posture information analyzes motion for the right and left arms independently. 
    'vel' and 'angle' represent the movement intensity of each arm. 
    In 'frame_angles', 'r_' prefixes denote the right arm/gripper and 'l_' prefixes denote the left. 
    Euler angles ('rx', 'ry', 'rz') represent rotational transformations, and 'gripper' values (0 to 1) indicate grasping status. 
    Use the per-arm 'vel' and 'vel_score' to distinguish which arm is active versus holding still.

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
    11. CRITICAL NEW RULE: Before outputting the line-by-line motion-units, you MUST act as a State Tracker. You must output a JSON block summarizing the new macroscopic World State Graph at the END of these 32 frames. Use the `environment_topology_state` array to act as a scene graph. Here is the template you must strictly follow:
    {wsm_template}
    The kinematic posture information provided is as follows:
    {kinematics_str}

    Your English description is:
    """
    return prompt

def update_world_state(current_state: dict, latest_vlm_output: str) -> dict:
    """
    不再拼接字符串历史，而是提取 VLM 生成的 JSON 状态机，进行增量更新。
    返回更新后的字典。
    """
    updated_state = copy.deepcopy(current_state) # 避免意外修改原字典
    
    # 使用正则捕获 markdown 格式的 json 块
    json_match = re.search(r'```json\s*(.*?)\s*```', latest_vlm_output, re.DOTALL | re.IGNORECASE)
    
    if json_match:
        try:
            new_state_str = json_match.group(1)
            new_state_dict = json.loads(new_state_str)
            
            # 使用 VLM 推理出的新状态覆盖旧状态
            updated_state.update(new_state_dict)
            print(f"   🧠 [状态更新] 当前宏观技能迁移至: {updated_state.get('temporal_context', {}).get('macro_skill_id', 'unknown')}")
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ [警告] VLM 输出了 JSON 但格式错误，维持上一状态。错误: {e}")
    else:
        print("   ⚠️ [警告] 无法从 VLM 输出中提取 JSON 状态机，维持上一状态。")
        
    return updated_state