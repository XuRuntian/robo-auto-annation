from typing import Dict, List
import numpy as np
from src.core.types import CutPoint

class LocalWindowGenerator:
    """
    本地窗口生成器，用于生成围绕切点帧的前后窗口。
    
    参数:
        fps (int): 帧率，每秒帧数。
        window_sec (float): 窗口时长，默认为1.0秒。
    """
    def __init__(self, fps: int, window_sec: float = 1.0):
        self.fps = fps
        self.window_sec = window_sec
        self.window_frames = int(fps * window_sec)
    
    def generate_windows(self, reader, cut_point: CutPoint) -> Dict[str, List[np.ndarray]]:
        """
        生成包含三个窗口的字典，每个窗口包含3张均匀抽取的图片。
        
        参数:
            reader: 视频读取器，用于读取帧。
            cut_point (CutPoint): 切点，包含帧索引。
        
        返回:
            Dict[str, List[np.ndarray]]: 包含三个窗口的字典，每个窗口包含3张图片。
        """
        T = cut_point.frame_idx
        total_frames = reader.get_length()
        half_w = self.window_frames // 2
        
        # 定义三个窗口的帧区间
        ranges = {
            'before': (max(0, T - self.window_frames), T),
            'center': (max(0, T - half_w), min(total_frames - 1, T + half_w)),
            'after': (T, min(total_frames - 1, T + self.window_frames))
        }
        
        result = {}
        for key, (start, end) in ranges.items():
            num_frames = end - start + 1
            indices = np.linspace(0, num_frames - 1, 3, dtype=int)
            frames = [reader.read_frame(start + i) for i in indices]
            result[key] = frames
        
        return result
