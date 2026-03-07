import re

class VLMOutputParser:
    """将 VLM 的文本输出解析为含绝对物理帧的结构化字典列表"""
    
    def __init__(self):
        # 匹配行首的时间区间，兼容 [0, 12] 或 [{0, 12}] 或 [0, 12],
        self.interval_pattern = r'^\[\s*\{?\s*(\d+)\s*,\s*(\d+)\s*\}?\s*\]?'
        # 匹配该行内所有的动作元组 (subject, action, ...)
        self.tuple_pattern = r'\(([^)]+)\)'

    def parse_and_map(self, vlm_text: str, global_indices: list) -> list[dict]:
        parsed_actions = []
        current_level = "unknown"
        
        for line in vlm_text.split('\n'):
            line = line.strip()
            if not line: continue
                
            # 1. 识别并更新当前所处的层级
            low_line = line.lower()
            if "robot-level" in low_line: current_level = "robot-level"; continue
            if "arm-level" in low_line: current_level = "arm-level"; continue
            if "gripper-level" in low_line: current_level = "gripper-level"; continue
                
            # 2. 尝试在行首匹配帧区间
            interval_match = re.search(self.interval_pattern, line)
            if interval_match:
                local_start = int(interval_match.group(1))
                local_end = int(interval_match.group(2))
                
                # 映射局部索引到全局绝对帧号 (加入越界保护)
                local_start = max(0, min(local_start, len(global_indices) - 1))
                local_end = max(0, min(local_end, len(global_indices) - 1))
                global_start = int(global_indices[local_start])
                global_end = int(global_indices[local_end])
                
                # 3. 查找该行内【所有】的元组
                tuples = re.findall(self.tuple_pattern, line)
                
                if tuples:
                    # ✅ 核心修复：永远只把第一个 tuple 当作 Core Action
                    core_tuple = tuples[0]
                    core_elements = [e.strip() if e.strip() else "none" for e in core_tuple.replace(':', ',').split(',')]
                    
                    if len(core_elements) >= 2:
                        action_dict = {
                            "level": current_level,
                            "global_start_frame": global_start,
                            "global_end_frame": global_end,
                            "subject": core_elements[0],
                            "action_verb": core_elements[1],
                            "object": core_elements[2] if len(core_elements) > 2 else "none",
                            "direction": core_elements[3] if len(core_elements) > 3 else "none",
                            "amplitude": core_elements[4] if len(core_elements) > 4 else "none",
                            "subject_modifier": "none",
                            "object_modifier": "none"
                        }
                        
                        # ✅ 解析第二个 tuple (Subject Modifier)
                        if len(tuples) > 1:
                            subj_mod = [e.strip() for e in tuples[1].replace(':', ',').split(',')]
                            if len(subj_mod) > 1:
                                action_dict["subject_modifier"] = subj_mod[1]
                                
                        # ✅ 解析第三个 tuple (Object Modifier)
                        if len(tuples) > 2:
                            obj_mod = [e.strip() for e in tuples[2].replace(':', ',').split(',')]
                            if len(obj_mod) > 1:
                                action_dict["object_modifier"] = obj_mod[1]
                                
                        parsed_actions.append(action_dict)
                        
        return parsed_actions