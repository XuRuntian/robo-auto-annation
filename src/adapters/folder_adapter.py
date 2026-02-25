# src/adapters/folder_adapter.py
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class FolderAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.frames = [] 
        self.sensors = []
        
        # [新增] 轨迹管理
        self.episode_dirs = []
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        if not self.root_path.is_dir(): return False
        
        self.episode_dirs = []
        
        # 判断当前目录是否就是一条轨迹
        def has_images(d):
            return bool(list(d.glob("*.jpg")) or list(d.glob("*.png")) or list((d/"colors").glob("*.jpg")))
            
        if has_images(self.root_path):
            self.episode_dirs.append(self.root_path)
        else:
            # 扫描子目录
            for d in sorted(self.root_path.iterdir()):
                if d.is_dir() and has_images(d):
                    self.episode_dirs.append(d)
                    
        if not self.episode_dirs:
            print("❌ [Folder] 未发现包含图片的目录")
            return False
            
        print(f"✅ [Folder] 扫描到 {len(self.episode_dirs)} 个图片序列文件夹")
        self.set_episode(0)
        return True

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_dirs): return
        self.current_episode_idx = episode_idx
        
        target_dir = self.episode_dirs[episode_idx]
        print(f"🔄 [Folder] 切换至 Episode {episode_idx} ({target_dir.name})")
        
        search_dirs = [target_dir, target_dir / "colors"]
        img_files = []
        for d in search_dirs:
            if d.exists(): img_files.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.png"))))

        frame_dict = {} 
        detected_sensors = set()

        for p in img_files:
            match = re.match(r"(\d+)_+(.*)\.(jpg|png)", p.name)
            if match:
                idx = int(match.group(1))
                sensor = match.group(2)
                if idx not in frame_dict: frame_dict[idx] = {'images': {}}
                frame_dict[idx]['images'][sensor] = str(p)
                detected_sensors.add(sensor)

        sorted_indices = sorted(frame_dict.keys())
        self.frames = [frame_dict[i] for i in sorted_indices]
        self.sensors = list(detected_sensors)

    def get_total_episodes(self) -> int:
        return len(self.episode_dirs)

    def get_length(self) -> int:
        return len(self.frames)

    def get_all_sensors(self) -> List[str]:
        return self.sensors

    def get_frame(self, index: int) -> FrameData:
        if index < 0 or index >= len(self.frames): return None
        frame_info = self.frames[index]
        images = {}
        for sensor, path in frame_info['images'].items():
            img = cv2.imread(path)
            if img is not None: images[sensor] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return FrameData(timestamp=float(index)/30.0, images=images, state={})

    def close(self):
        pass