from typing import List, Dict
import numpy as np

class LocalWindowGenerator:
    def __init__(self, window_size_seconds: float = 1.0, fps: int = 30):
        self.window_size_seconds = window_size_seconds
        self.fps = fps
        self.window_size_frames = int(window_size_seconds * fps)

    def generate_windows(
        self, 
        reader, 
        cut_point: 'CutPoint'
    ) -> Dict[str, List[np.ndarray]]:
        """
        生成切点周围的局部窗口序列
        返回包含 pre, mid, post 三个窗口的字典
        """
        frame_idx = cut_point.frame_idx
        total_frames = reader.get_total_frames()
        
        # 计算每个窗口的起止帧
        windows = {
            'pre': self._get_window_range(frame_idx - self.window_size_frames * 2, frame_idx - self.window_size_frames, total_frames),
            'mid': self._get_window_range(frame_idx - self.window_size_frames // 2, frame_idx + self.window_size_frames // 2, total_frames),
            'post': self._get_window_range(frame_idx + self.window_size_frames, frame_idx + self.window_size_frames * 2, total_frames)
        }
        
        # 读取图像数据
        return {
            'pre': [reader.get_frame(i) for i in windows['pre']],
            'mid': [reader.get_frame(i) for i in windows['mid']],
            'post': [reader.get_frame(i) for i in windows['post']]
        }
    
    def _get_window_range(self, start: int, end: int, total_frames: int) -> range:
        """获取调整后的窗口范围"""
        # 确保范围不超出视频边界
        adjusted_start = max(0, min(start, total_frames - 1))
        adjusted_end = max(0, min(end, total_frames - 1))
        
        # 如果请求范围无效（开始>=结束），返回单帧
        if adjusted_start >= adjusted_end:
            return range(adjusted_start, adjusted_start + 1)
            
        return range(adjusted_start, adjusted_end + 1)
