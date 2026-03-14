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
    3. Merge & Isolation Policy:
    - DO NOT merge 'Reach' (arm moving towards object) with 'Grasp/Release' (gripper closing/opening). They MUST be separate segments.
    - Any change in the gripper state (e.g., from open to closed) indicates a critical interaction and must form its own distinct segment.
    - Post-Task Phase: If the verb is 'retract', 'place', or 'release' near the end of the video, describe it accurately as the robot returning to its rest position or concluding the task, NOT as "maintaining position".
    - Post-Task Phase: If the physical motion action is 'idle' or 'retract' at the end of the trajectory, describe it simply as "The robot retracts the arm and concludes the task." DO NOT hallucinate reasons like "maintaining position".
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
def build_robotics_pamor_prompt(kinematic_json, task_description, world_state_dict, allowed_verbs, allowed_predicates):
    """
    全新版本的 Prompt：强制 VLM 进行 PDDL 逻辑推导，不输出任何冗余文本和帧号
    """
    import json
    kinematics_str = json.dumps(kinematic_json, ensure_ascii=False)
    state_str = json.dumps(world_state_dict, indent=2)
    
    # 告诉 VLM 我们的 Schema 结构
    schema_template = """
    {
      "thought": "Analyze the physical kinematics and images here. E.g., The gripper velocity spikes, indicating a grasp...",
      "operators": [
        {
          "action_verb": "grasp",
          "subject": "right_arm",
          "target_object": "apple",
          "preconditions": [
            {"predicate": "hand_free", "objects": ["right_arm"]},
            {"predicate": "is_open", "objects": ["right_gripper"]}
          ],
          "effects": [
            {"predicate": "holding", "objects": ["right_arm", "apple"]},
            {"predicate": "is_closed", "objects": ["right_gripper"]}
          ]
        }
      ]
    }
    """

    prompt = f"""You are an Embodied AI planner and logical validator.
    I will provide you with images of a robotic action chunk and its physical kinematics.
    Your task is to act as a strict PDDL (Planning Domain Definition Language) compiler.

    [GLOBAL TASK]: {task_description}
    [CURRENT WORLD STATE]: {state_str}
    
    [KINEMATICS DATA]: 
    {kinematics_str}
    (Note: 'gripper' near 1.0 means OPEN, near 0.0 means CLOSED. A sudden change means grasp or release).

    [STRICT RULES]:
    1. DO NOT output any markdown text outside the JSON block.
    2. DO NOT output any frame numbers or timestamps. The system will align the time physically.
    3. You must ONLY output a JSON object exactly matching this structure:
    ```json
    {schema_template}
    ```
    4. Allowed Action Verbs: {allowed_verbs}
    5. Allowed Predicates: {allowed_predicates}
    6. Logical Strictness: 
       - If you output 'grasp', the precondition MUST include 'hand_free', and effect MUST include 'holding'.
       - Physical sensors (Kinematics) have the highest authority. If the gripper value doesn't change to closed, DO NOT output 'grasp'.
    7. Retract vs. Idle Strategy (End of Task):
       - Look at the overall 'vel' (velocity) and 'angle' in the kinematics data.
       - If the arm is moving out of the camera view or away from the objects, AND the 'vel' is significantly greater than 0, you MUST output 'retract' (the robot is actively returning to home position).
       - If the arm is out of view OR completely motionless, AND the 'vel' drops to near 0, you MUST output 'idle' and use 'at_rest' for the effect predicate. Do not invent task-related actions if the kinematics show no movement.
    """
    return prompt

def update_world_state(current_state: dict, pddl_trajectory) -> dict:
    """
    新版状态更新：直接读取 Pydantic 对象中的 Effects 进行全局状态覆盖
    """
    updated_state = copy.deepcopy(current_state)
    
    if not pddl_trajectory or not pddl_trajectory.operators:
        return updated_state
        
    # 根据这一帧的所有动作 Effects 更新全局状态
    for op in pddl_trajectory.operators:
        for eff in op.effects:
            # 这里可以做一个简单的映射，把 effect 写入到 object_physical_state
            # 为了简化，我们直接把动作记录到 temporal_context 中
            updated_state["temporal_context"]["last_action"] = op.action_verb
            updated_state["temporal_context"]["last_target"] = op.target_object
            
            # 如果是 holding，更新末端执行器状态
            if eff.predicate == "holding":
                updated_state["robot_interaction_state"]["right_end_effector"]["grasp_type"] = "holding"
                updated_state["robot_interaction_state"]["right_end_effector"]["contact_target"] = op.target_object
            elif eff.predicate == "hand_free":
                updated_state["robot_interaction_state"]["right_end_effector"]["grasp_type"] = "none"
                updated_state["robot_interaction_state"]["right_end_effector"]["contact_target"] = "none"
                
    return updated_state