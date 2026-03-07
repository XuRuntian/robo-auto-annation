import json
import re

def build_robotics_pamor_prompt(kinematic_json, task_description, task_history):
    kinematics_str = json.dumps(kinematic_json, ensure_ascii=False)
    
    prompt = f"""You are an expert in describing robotic motion content for embodied AI. I will give you 32 frames of "video frames" uniformly extracted from a robot manipulation trajectory, input them in chronological order, and provide you with kinematic posture information corresponding to this sequence.
    Please analyze the visual content based on the video frames and posture information, and output the motion description of the robot in the video.

    TASK GOAL OF THIS EPISODE: {task_description}
    TASK HISTORY: {task_history}
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

def update_task_history(current_history: str, latest_vlm_output: str) -> str:
    """提取最新的动作元组更新历史记忆，保持因果连贯性"""
    important_lines = re.findall(r'\[\{.*\}\].*', latest_vlm_output)
    new_segment = "\n".join(important_lines)
    
    updated_history = current_history + f"\n--- Previous Chunk Actions ---\n" + new_segment
    
    # 限制历史长度，防止超出 Token 限制（保留最近 3 个 Chunk 的记录）
    history_chunks = updated_history.split('--- Previous Chunk Actions ---')
    if len(history_chunks) > 4:
        updated_history = "Task started.\n--- Previous Chunk Actions ---" + "--- Previous Chunk Actions ---".join(history_chunks[-3:])
        
    return updated_history