from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np
from src.core.types import CutPoint, ValidationResult

class BaseSemanticValidator(ABC):
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
        vlm_certainty: float
    ) -> float:
        """
        计算融合置信度
        简单加权逻辑：物理能量占60%，VLM确定性占40%
        """
        return 0.6 * physics_energy + 0.4 * vlm_certainty
