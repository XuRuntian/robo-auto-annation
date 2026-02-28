from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class CutPoint:
    frame_idx: int
    energy_score: float
    confidence: float = 1.0

@dataclass
class ValidationResult:
    is_true_switch: bool
    instruction: str
    confidence_score: float
    reasoning: str
