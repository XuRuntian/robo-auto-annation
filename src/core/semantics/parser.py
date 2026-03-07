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
                
                # 3. 查找该行内【所有】的元组 (VLM 可能在一行输出多个)
                tuples = re.findall(self.tuple_pattern, line)
                
                for content_str in tuples:
                    # 分割、去除两端空格，如果为空则强制替换为 "none" (处理 ", , ," 的情况)
                    elements = [e.strip() if e.strip() else "none" for e in content_str.replace(':', ',').split(',')]
                    
                    if len(elements) >= 2: # 至少要有 Subject 和 Action
                        parsed_actions.append({
                            "level": current_level,
                            "global_start_frame": global_start,
                            "global_end_frame": global_end,
                            "subject": elements[0],
                            "action_verb": elements[1],
                            "object": elements[2] if len(elements) > 2 else "none",
                            "direction": elements[3] if len(elements) > 3 else "none",
                            "amplitude": elements[4] if len(elements) > 4 else "none"
                        })
                        
        return parsed_actions