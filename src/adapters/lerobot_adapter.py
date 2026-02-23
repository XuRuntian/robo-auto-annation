import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import imageio
from typing import List, Dict, Any
from src.core.interface import BaseDatasetReader, FrameData

class LeRobotAdapter(BaseDatasetReader):
    def __init__(self):
        self.root_path = None
        self.df = None
        self.fps = 30.0
        self.image_keys = []
        self.full_feature_keys = {}
        self.image_path_tpl = ""
        self.video_path_tpl = ""  
        self.cap_cache = {} 
        self.version = ""
        
        # [新增] 轨迹管理属性
        self.episode_files = [] 
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        
        # 获取所有 parquet 文件并按文件名排序
        parquet_files = sorted(list(self.root_path.rglob("data/**/*.parquet")))
        if not parquet_files:
            parquet_files = sorted(list(self.root_path.glob("*.parquet")))

        if not parquet_files:
            print(f"❌ [LeRobot] 找不到 parquet 数据")
            return False
            
        # [修改] 存储文件路径列表，不立刻 concat 读取
        self.episode_files = parquet_files

        try:
            meta_path = self.root_path / "meta" / "info.json"
            if not meta_path.exists():
                print("❌ [LeRobot] 缺少 meta/info.json")
                return False

            with open(meta_path, 'r') as f:
                info = json.load(f)
                self.version = info.get("codebase_version", "v2.1")
                self.fps = info.get("fps", 30.0)
                self.image_path_tpl = info.get("image_path", "")
                self.video_path_tpl = info.get("video_path", "")
                
                features = info.get("features", {})
                self.image_keys = []
                self.full_feature_keys = {}
                
                for key, val in features.items():
                    if isinstance(val, dict) and val.get("dtype") in ["image", "video"]:
                        short_name = key.split(".")[-1]
                        self.image_keys.append(short_name)
                        self.full_feature_keys[short_name] = key
            
            # [新增] 默认加载第一条轨迹用于初始化
            self.set_episode(0)
            print(f"✅ [LeRobot] 加载成功: 共扫描到 {len(self.episode_files)} 条轨迹, 相机: {self.image_keys}")
            return True

        except Exception as e:
            print(f"❌ [LeRobot] 加载异常: {e}")
            return False

    # ==========================================
    # [新增] 对外统一的管理接口
    # ==========================================
    def set_episode(self, episode_idx: int):
        """按需加载单条轨迹，避免内存爆炸"""
        if episode_idx < 0 or episode_idx >= len(self.episode_files):
            return
            
        self.current_episode_idx = episode_idx
        
        # 释放旧轨迹的视频缓存
        self.close() 
        
        # 仅读取当前选中的单条 parquet
        parquet_path = self.episode_files[episode_idx]
        self.df = pd.read_parquet(parquet_path)
        print(f"🔄 [LeRobot] 已切换至 Episode {episode_idx} ({parquet_path.name}), 当前轨迹帧数: {len(self.df)}")

    def get_total_episodes(self) -> int:
        return len(self.episode_files)

    # ==========================================
    # 继承的接口实现
    # ==========================================
    def get_length(self) -> int:
        return len(self.df) if self.df is not None else 0

    def get_all_sensors(self) -> List[str]:
        return self.image_keys

    def get_frame(self, index: int) -> FrameData:
        if self.df is None or index >= len(self.df): return None
        
        row = self.df.iloc[index]
        images = {}
        
        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])
        
        for short_name in self.image_keys:
            full_key = self.full_feature_keys[short_name]
            img_loaded = False
            
            # 1. 优先尝试静态图片
            if self.image_path_tpl:
                rel_path = self.image_path_tpl.format(
                    image_key=full_key,
                    episode_index=ep_idx,
                    frame_index=frame_idx
                )
                full_path = self.root_path / rel_path
                if full_path.exists():
                    img = cv2.imread(str(full_path))
                    if img is not None:
                        images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_loaded = True
            
            # 2. 图片不存在，尝试视频抽帧
            if not img_loaded:
                video_files = list(self.root_path.rglob(f"**/{full_key}/*episode_{ep_idx:06d}.mp4"))
                if video_files:
                    video_path = video_files[0]
                    
                    try:
                        reader = self.cap_cache.get(str(video_path))
                        if not reader:
                            # 使用 ffmpeg 插件，完美兼容 AV1 等高压缩格式
                            reader = imageio.get_reader(str(video_path), 'ffmpeg')
                            self.cap_cache[str(video_path)] = reader
                        
                        # get_data 直接返回当前帧的 RGB numpy array
                        frame = reader.get_data(frame_idx)
                        images[short_name] = frame
                    except Exception as e:
                        print(f"❌ [LeRobot] 视频读取失败或越界: {video_path} 帧 {frame_idx}, 报错: {str(e)}")

        state = {
            "action": np.array(row["action"]) if "action" in row else None,
            "qpos": np.array(row["observation.state"]) if "observation.state" in row else None
        }

        return FrameData(
            timestamp=float(row.get("timestamp", index / self.fps)),
            images=images,
            state={k: v for k, v in state.items() if v is not None}
        )

    def close(self):
        # 释放 imageio 视频流资源
        for reader in self.cap_cache.values():
            try:
                reader.close()
            except:
                pass
        self.cap_cache.clear()