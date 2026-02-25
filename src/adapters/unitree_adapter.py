# src/adapters/unitree_adapter.py
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class UnitreeAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.current_dir = None
        self.data_list = [] 
        self.fps = 30.0
        self.image_keys = []
        
        # [新增] 轨迹管理
        self.episode_files = []
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episode_files = []
        
        if (self.root_path / "data.json").exists():
            self.episode_files.append(self.root_path / "data.json")
        else:
            self.episode_files = sorted(list(self.root_path.rglob("data.json")))

        if not self.episode_files:
            print(f"❌ [Unitree] 找不到 data.json")
            return False

        print(f"✅ [Unitree] 扫描到 {len(self.episode_files)} 条轨迹")
        try:
            self.set_episode(0)
            return True
        except Exception as e:
            print(f"❌ [Unitree] 初始化失败: {e}")
            return False

    def set_episode(self, episode_idx: int):
        if episode_idx < 0 or episode_idx >= len(self.episode_files): return
        self.current_episode_idx = episode_idx
        
        json_file = self.episode_files[episode_idx]
        self.current_dir = json_file.parent
        self.data_list = []
        
        print(f"🔄 [Unitree] 切换至 Episode {episode_idx} ({self.current_dir.name})")
        with open(json_file, 'r') as f: content = json.load(f)
        
        if isinstance(content, dict) and "info" in content:
            if "image" in content["info"]: self.fps = float(content["info"]["image"].get("fps", 30.0))

        if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
            self.data_list = content["data"]
        elif isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and ("colors" in v[0] or "states" in v[0]):
                    self.data_list = v; break
        elif isinstance(content, list): self.data_list = content

        if not self.data_list: raise ValueError("JSON 中未找到有效的数据列表")
            
        first = self.data_list[0]
        self.image_keys = list(first["colors"].keys()) if "colors" in first else ["color_0", "color_1"]

    def get_total_episodes(self) -> int:
        return len(self.episode_files)

    def get_length(self) -> int:
        return len(self.data_list)

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_frame(self, index: int) -> FrameData:
        if index >= len(self.data_list): return None
        frame_dict = self.data_list[index]
        images = {}
        
        if "colors" in frame_dict:
            for cam_name, rel_path in frame_dict["colors"].items():
                if rel_path:
                    fp = self.current_dir / rel_path
                    if fp.exists():
                        img = cv2.imread(str(fp))
                        if img is not None: images[cam_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        state = {}
        try:
            if "states" in frame_dict:
                source = frame_dict["states"]
                qpos_list = []
                for part in ["left_arm", "right_arm", "left_ee", "right_ee", "head", "body"]:
                    if part in source and "qpos" in source[part]: qpos_list.extend(source[part]["qpos"])
                if qpos_list: state['qpos'] = np.array(qpos_list)

            if "tactiles" in frame_dict:
                for t_name, t_path in frame_dict["tactiles"].items():
                    if t_path and (self.current_dir / t_path).exists():
                        state[t_name] = np.load(self.current_dir / t_path)
        except: pass

        return FrameData(timestamp=frame_dict.get("idx", index)/self.fps, images=images, state=state)

    def close(self):
        pass