from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np
from src.core.types import CutPoint, ValidationResult

class BaseSemanticValidator(ABC):
    @abstractmethod
    def validate_point(
        self,
        window_data: Dict[str, List[np.ndarray]],
        physics_energy: float,
        previous_instruction: Optional[str] = None
    ) -> ValidationResult:
        """验证切点是否为真实任务切换点"""
        pass

    def compute_fused_confidence(
        self,
        physics_energy: float,
        window_results: List[bool]
    ) -> float:
        """
        计算融合置信度
        接收3个重叠窗口的VLM布尔结果，应用离散汉宁窗权重[0.25, 1.0, 0.25]计算加权VLM得分
        与归一化到0-1的physics_energy进行4:6加权融合
        """
        # 计算VLM加权得分
        vlm_weights = [0.25, 1.0, 0.25]
        vlm_score = sum(w * (1.0 if result else 0.0) for w, result in zip(vlm_weights, window_results))
        vlm_score /= sum(vlm_weights)  # 归一化到0-1范围
        
        # 融合物理能量和VLM得分（4:6比例）
        return 0.4 * physics_energy + 0.6 * vlm_score
