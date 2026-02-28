import os
import numpy as np
from typing import List, Dict, Optional, Any
from src.core.physics.base import BasePhysicsDetector
from src.core.windows.generator import LocalWindowGenerator
from src.core.semantics.base import BaseSemanticValidator
from src.core.types import CutPoint, ValidationResult
from src.core.image_utils import GridImageGenerator
import logging

logger = logging.getLogger(__name__)

class RoboETLPipeline:
    def __init__(
        self,
        reader: Any,
        physics_detector: BasePhysicsDetector,
        window_generator: LocalWindowGenerator,
        semantic_validator: BaseSemanticValidator
    ):
        self.reader = reader
        self.physics_detector = physics_detector
        self.window_generator = window_generator
        self.semantic_validator = semantic_validator
        self.global_sop = None  # 初始化全局SOP变量

    def extract_qpos(self, ep_length: int) -> np.ndarray:
        qpos_list = []
        for idx in range(ep_length):
            frame = self.reader.get_frame(idx)
            state = getattr(frame, 'state', {})
            val = state.get("qpos") if state.get("qpos") is not None else state.get("action")
            qpos_list.append(val if val is not None else np.zeros(6))
        return np.array(qpos_list)

    def process_episode(self, episode_idx: int, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        处理单个 episode，生成动作片段
        
        参数:
            episode_idx: episode 的索引
            confidence_threshold: 置信度阈值
            
        返回:
            List[Dict]: 动作片段列表
        """
        # 新逻辑实现
        # A. 读取轨迹
        self.reader.set_episode(episode_idx)
        ep_length = self.reader.get_length()
        qpos_data = self.extract_qpos(ep_length)
        # B. 提取全局 SOP（如果为空）
        if not self.global_sop:
            self.extract_global_sop()
        
        # C. 获取嫌疑切点
        cut_points = self.physics_detector.propose_cut_points(qpos_data)

        # D. 动态对齐逻辑
        valid_cut_points = []
        current_sop_index = 0
        
        # D.1 验证切点与SOP对齐
        for cut_point in cut_points:
            # 提前终止条件：SOP已完全匹配
            if current_sop_index >= len(self.global_sop):
                break
                
            # 生成局部窗口
            window_data = self.window_generator.generate_windows(self.reader, cut_point)
            
            # 验证当前切点是否匹配当前SOP步骤
            result = self.semantic_validator.validate_point(
                window_data=window_data,
                physics_energy=cut_point.energy_score,
                # expected_instruction=self.global_sop[current_sop_index]
            )
            print(f"{result=}")
            
            # 如果验证通过
            if result.is_true_switch and result.confidence_score > confidence_threshold:
                valid_cut_points.append(cut_point)
                current_sop_index += 1
        
        # E. 处理边缘情况
        # E.1 所有切点都被拒绝或SOP未完全走完
        if current_sop_index == 0 or not valid_cut_points:
            return [{
                "subtask_id": 1,
                "instruction": "⚠️ [未检测到有效切换点] 请复核",
                "start_frame": 0,
                "end_frame": ep_length - 1
            }]
        
        # E.2 将剩余帧划归最后一步
        final_segments = []
        valid_cut_points.insert(0, CutPoint(frame_idx=0, energy_score=0))  # 插入起始点
        valid_cut_points.append(CutPoint(frame_idx=ep_length-1, energy_score=0))  # 插入结束点
        
        # E.3 生成最终片段
        for i in range(len(valid_cut_points) - 1):
            start_frame = valid_cut_points[i].frame_idx
            end_frame = valid_cut_points[i+1].frame_idx - 1
            
            # 使用SOP中的指令
            instruction = self.global_sop[i] if i < len(self.global_sop) else f"Final step {i+1}"
            
            # 防止由于相邻帧过近导致的逻辑反转
            if start_frame <= end_frame:
                final_segments.append({
                    "subtask_id": i + 1,
                    "instruction": instruction,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "confidence": 1.0  # 由于使用了全局SOP，置信度设为1
                })
        
        # E.4 处理SOP剩余步骤（如果有的话）
        if len(self.global_sop) > len(final_segments):
            last_segment = final_segments[-1]
            last_segment["instruction"] = self.global_sop[-1]
            last_segment["subtask_id"] = len(self.global_sop)
            last_segment["end_frame"] = ep_length - 1
        
        return final_segments

    def extract_global_sop(self, sample_size: int = 3, task_desc: str = '', uniform_sampling: bool = True) -> List[str]:
        """
        提取全局SOP
        """
        import json
        import re

        total_eps = self.reader.get_total_episodes()
        sample_indices = np.linspace(0, total_eps - 1, sample_size, dtype=int)
        
        # 1. 收集采样配置
        sample_configs = []
        for ep_idx in sample_indices:
            self.reader.set_episode(ep_idx)
            ep_length = self.reader.get_length()

            if uniform_sampling:
                # 均匀采样直接生成帧索引列表
                frame_indices = np.linspace(0, ep_length - 1, 9, dtype=int).tolist()
            else:
                # 物理探测并统一转为帧索引
                qpos = self.extract_qpos(ep_length)
                raw_points = self.physics_detector.propose_cut_points(qpos)
                frame_indices = [cp.frame_idx for cp in raw_points] if raw_points else []

            if frame_indices:
                sample_configs.append((ep_idx, frame_indices))
        
        if not sample_configs:
            logger.error("未能获取任何有效的采样点")
            return []

        # 2. 生成图片
        mega_path = "mega_sop_grid.jpg"
        print(f"Generating mega grid for SOP extraction with {len(sample_configs)} samples...")
        GridImageGenerator.generate_mega_grid(self.reader, sample_configs, mega_path)
        
        # 3. 专家级 Prompt
        sop_prompt = f"""You are an expert robotic data annotator. You are given a grid image containing {len(sample_configs)} separate execution instances of the same task. Each instance is shown as a 3x3 sequential keyframe grid.

        [Global Context]
        Task Description: "{task_desc}"

        [Goal]
        Identify the CONSISTENT macro-level steps (SOP) across all instances.

        [Action Rules]
        1. Merge micro-actions: Combine "approach" and "grasp" into one step; combine "move" and "release" into one step.
        2. Language: MUST be in English.

        [Output Format]
        Return ONLY a strict JSON object. No conversational text.
        Structure: 
        {{
          "sop_steps": ["Step 1...", "Step 2..."]
        }}
        """

        try:
            # 4. 调用 API
            response = self.semantic_validator._call_qwen_api(mega_path, sop_prompt)
            print(f"Qwen API raw response: {response}")
            
            # --- 关键改进：鲁棒性解析逻辑 ---
            extracted_data = None
            
            if isinstance(response, str):
                # 使用正则提取第一个 { ... } 块
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        extracted_data = json.loads(match.group())
                    except json.JSONDecodeError:
                        logger.error("JSON 解码失败，请检查 API 返回内容内容格式")
                else:
                    # 如果没找到 {}，尝试寻找 []
                    match_list = re.search(r'\[.*\]', response, re.DOTALL)
                    if match_list:
                        extracted_data = json.loads(match_list.group())
            elif isinstance(response, (dict, list)):
                extracted_data = response

            # --- 结果分发 ---
            if isinstance(extracted_data, dict) and 'sop_steps' in extracted_data:
                self.global_sop = extracted_data['sop_steps']
            elif isinstance(extracted_data, list):
                self.global_sop = extracted_data
            else:
                self.global_sop = []
                logger.warning(f"Qwen API 返回格式不符合预期，原始响应内容为: {response}")
                
            return self.global_sop
            
        except Exception as e:
            logger.error(f"SOP 提取失败: {str(e)}")
            self.global_sop = []
            return []
    
    def generate_global_template(
        self, 
        task_desc: str, 
        sample_size: int = 3, 
        progress_callback: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        生成全局任务模板
        1. 使用物理探测器获取关键帧
        2. 使用语义验证器验证关键帧
        3. 返回结构化模板
        """
        total_eps = self.reader.get_total_episodes()
        sample_indices = np.linspace(0, total_eps - 1, sample_size, dtype=int)
        
        # 收集采样配置
        sample_configs = []
        if progress_callback:
            progress_callback(f"🧪 正在采集 {len(sample_indices)} 条参考轨迹的关键帧...")

        for ep_idx in sample_indices:
            self.reader.set_episode(ep_idx)
            qpos = self.extract_qpos(self.reader.get_length())
            cut_points = self.physics_detector.propose_cut_points(qpos)
            if cut_points:
                indices = [cp.frame_idx for cp in cut_points]
                sample_configs.append((ep_idx, indices))

        # 1. 物理拼接成一张超级大图
        mega_path = "mega_template_grid.jpg"
        GridImageGenerator.generate_mega_grid(self.reader, sample_configs, mega_path)

        # 2. 修改 Prompt，告诉 AI 同时看这三组数据
        mega_prompt = (
            f"{task_desc}\n"
            f"图中垂直排列了 {sample_size} 组相同任务的执行过程。每一组都是 3x3 九宫格。\n"
            "请观察这些不同实例的共性，给出最通用的子任务拆解逻辑（Subtask JSON）。"
        )

        try:
            if progress_callback: 
                progress_callback("🌐 正在发送超级大图，请求全局任务标准 (Only 1 API Call)...")
            
            # 3. 使用语义验证器验证模板
            temp_path = "temp_template_grid.jpg"
            GridImageGenerator.generate_template_grid(sample_configs, temp_path)
            window_data = {
                'pre': [self.reader.get_frame(i) for i in range(0, 5)],
                'mid': [self.reader.get_frame(i) for i in range(5, 10)],
                'post': [self.reader.get_frame(i) for i in range(10, 15)]
            }
            
            validation_results = []
            for i in range(len(sample_configs)):
                result = self.semantic_validator.validate_point(
                    window_data=window_data,
                    physics_energy=0.8  # 使用默认物理能量值
                )
                validation_results.append(result)
            
            # 4. 组装最终模板
            master_template = []
            for i, result in enumerate(validation_results):
                if result.is_true_switch:
                    master_template.append({
                        "subtask_id": i + 1,
                        "instruction": result.instruction,
                        "start_image": 1,
                        "end_image": 9
                    })
            
            return master_template
            
        finally:
            if os.path.exists(mega_path): os.remove(mega_path)
            if os.path.exists("temp_template_grid.jpg"): os.remove("temp_template_grid.jpg")

    def process_with_template(
        self, 
        episode_idx: int, 
        template: List[Dict[str, Any]],
        is_suspect: bool = False
    ) -> List[Dict[str, Any]]:
        """
        使用全局模板处理具体 episode
        1. 提取物理数据
        2. 探测切点
        3. 生成局部窗口
        4. 验证切点
        5. 组装最终标注
        """
        self.reader.set_episode(episode_idx)
        ep_length = self.reader.get_length()
        
        if is_suspect:
            return [{
                "subtask_id": 1, 
                "instruction": "⚠️ [异常废片] 请复核", 
                "start_frame": 0, 
                "end_frame": ep_length - 1
            }]

        qpos_data = self.extract_qpos(ep_length)
        cut_points = self.physics_detector.propose_cut_points(qpos_data)
        
        # 确保至少有3个切点
        if len(cut_points) < 3:
            return [{
                "subtask_id": 1, 
                "instruction": "⚠️ [关键帧不足] 请复核", 
                "start_frame": 0, 
                "end_frame": ep_length - 1
            }]
        
        # 生成验证结果
        validation_results = []
        for cut_point in cut_points:
            window_data = self.window_generator.generate_windows(self.reader, cut_point)
            result = self.semantic_validator.validate_point(
                window_data=window_data,
                physics_energy=cut_point.energy_score
            )
            validation_results.append(result)
        
        # 组装最终标注
        final_annotations = []
        valid_results = [r for r in validation_results if r.is_true_switch]
        
        if not valid_results:
            return [{
                "subtask_id": 1, 
                "instruction": "⚠️ [未检测到有效切换点] 请复核", 
                "start_frame": 0, 
                "end_frame": ep_length - 1
            }]
        
        # 将验证结果映射到模板
        for i, result in enumerate(valid_results[:len(template)]):
            start_idx = cut_points[i].frame_idx if i < len(cut_points) else 0
            end_idx = cut_points[i+1].frame_idx if i+1 < len(cut_points) else ep_length - 1
            
            final_annotations.append({
                "subtask_id": template[i]["subtask_id"],
                "instruction": result.instruction,
                "start_frame": int(start_idx),
                "end_frame": int(end_idx)
            })
            
        return final_annotations
    
    def close(self):
        self.reader.close()
