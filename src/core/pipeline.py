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
        try:
            # A. 读取轨迹
            self.reader.set_episode(episode_idx)
            ep_length = self.reader.get_length()
            qpos_data = self.extract_qpos(ep_length)
            
            # B. 获取嫌疑切点
            cut_points = self.physics_detector.propose_cut_points(qpos_data)
            print(f"Episode {episode_idx}: Detected {len(cut_points)} cut points.,{cut_points=}")
            if not cut_points:
                return [{
                    "subtask_id": 1,
                    "instruction": "⚠️ [未检测到切点] 请复核",
                    "start_frame": 0,
                    "end_frame": ep_length - 1
                }]
            
            # C. 验证切点
            valid_segments = []
            for cut_point in cut_points:
                # 生成局部窗口
                window_data = self.window_generator.generate_windows(self.reader, cut_point)
                # 验证切点
                result = self.semantic_validator.validate_point(
                    window_data=window_data,
                    physics_energy=cut_point.energy_score
                )
                print(f"{result=}")
                # 满足置信度阈值
                if result.is_true_switch and result.confidence_score > confidence_threshold:
                    valid_segments.append({
                        "frame_idx": cut_point.frame_idx,
                        "instruction": result.instruction,
                        "confidence": result.confidence_score
                    })
            
            # D. 无有效切点
            if not valid_segments:
                return [{
                    "subtask_id": 1,
                    "instruction": "⚠️ [未检测到有效切换点] 请复核",
                    "start_frame": 0,
                    "end_frame": ep_length - 1
                }]
            
            # E. 转换为动作片段
            total_frames = ep_length
            valid_segments.sort(key=lambda x: x["frame_idx"])  # 按帧索引排序
            
            final_segments = []
            
            for i, seg in enumerate(valid_segments):
                # 1. 填补第一段（从 0 帧到第一个切点之前）
                if i == 0 and seg["frame_idx"] > 0:
                    final_segments.append({
                        "subtask_id": 1,
                        "instruction": "Initial approach / Task setup", # 默认启动动作
                        "start_frame": 0,
                        "end_frame": seg["frame_idx"] - 1,
                        "confidence": 1.0
                    })
                
                # 2. 当前切点到下一个切点（或视频结尾）
                start_frame = seg["frame_idx"]
                end_frame = valid_segments[i+1]["frame_idx"] - 1 if i + 1 < len(valid_segments) else total_frames - 1
                
                # 防止由于相邻帧过近导致的逻辑反转
                if start_frame <= end_frame:
                    final_segments.append({
                        "subtask_id": len(final_segments) + 1,
                        "instruction": seg["instruction"],
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "confidence": float(seg["confidence"])
                    })
                
            return final_segments
            
        except Exception as e:
            logger.error(f"Episode processing failed: {str(e)}", exc_info=True)
            return [{
                "subtask_id": 1,
                "instruction": f"❌ [处理失败] {str(e)}",
                "start_frame": 0,
                "end_frame": ep_length - 1
            }]

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
