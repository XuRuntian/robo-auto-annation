from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from src.core.types import CutPoint

class BasePhysicsDetector(ABC):
    @abstractmethod
    def propose_cut_points(
        self, 
        qpos_data: np.ndarray, 
        action_data: Optional[np.ndarray] = None
    ) -> List[CutPoint]:
        pass
