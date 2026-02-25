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
        self.current_dataset_root = None # 用于记录当前激活 Episode 的根目录
        self.df = None
        self.fps = 30.0
        self.image_keys = []
        self.full_feature_keys = {}
        self.image_path_tpl = ""
        self.video_path_tpl = ""  
        self.cap_cache = {} 
        self.version = ""
        self.dorobot_version = "" # [新增] dorobot 专属版本标识
        
        # [修改] 存储每条轨迹的完整元信息列表
        self.episodes_meta = [] 
        self.current_episode_idx = 0

    def load(self, file_path: str) -> bool:
        self.root_path = Path(file_path)
        self.episodes_meta = []
        
        # 兼容两种目录结构：
        # 1. 标准 LeRobot: 传入的是单个数据集根目录 (包含 meta/info.json)
        # 2. Dorobot 格式: 传入的是父目录，包含多个子目录 (每个子目录包含自己的 meta/info.json)
        meta_paths = []
        if (self.root_path / "meta" / "info.json").exists():
            meta_paths.append(self.root_path / "meta" / "info.json")
        else:
            # 扫描二级子目录中的 meta
            meta_paths = sorted(list(self.root_path.glob("*/meta/info.json")))
            
        if not meta_paths:
            print(f"❌ [LeRobot] 找不到任何 meta/info.json，格式不匹配")
            return False

        # 遍历解析所有找到的 Dataset (每个子目录视为独立的数据集分片)
        for meta_path in meta_paths:
            dataset_root = meta_path.parent.parent
            try:
                with open(meta_path, 'r') as f:
                    info = json.load(f)
                    
                # 查找该 dataset 下的 parquet 数据文件
                parquet_files = sorted(list(dataset_root.rglob("data/**/*.parquet")))
                if not parquet_files:
                    parquet_files = sorted(list(dataset_root.glob("*.parquet")))
                
                # 将该目录下的所有 parquet 注册为独立的 episode
                for pq in parquet_files:
                    self.episodes_meta.append({
                        "root": dataset_root,
                        "parquet": pq,
                        "info": info
                    })
            except Exception as e:
                print(f"⚠️ [LeRobot] 读取 {meta_path} 失败: {e}")

        if not self.episodes_meta:
            print(f"❌ [LeRobot] 找到了 meta，但未找到对应的 parquet 数据文件")
            return False
            
        # 默认加载第一条轨迹用于初始化状态
        self.set_episode(0)
        
        print(f"✅ [LeRobot] 加载成功: 共扫描到 {len(self.episodes_meta)} 条轨迹, 包含相机: {self.image_keys}")
        if self.dorobot_version:
            print(f"🤖 [Dorobot] 识别到基于 LeRobot 修改的 Dorobot 格式 (版本: {self.dorobot_version})")
            
        return True

    def set_episode(self, episode_idx: int):
        """按需加载单条轨迹，隔离上下文，避免内存爆炸"""
        if episode_idx < 0 or episode_idx >= len(self.episodes_meta):
            return
            
        self.current_episode_idx = episode_idx
        
        # 释放旧轨迹的视频缓存
        self.close() 
        
        ep_meta = self.episodes_meta[episode_idx]
        self.current_dataset_root = ep_meta["root"]
        parquet_path = ep_meta["parquet"]
        info = ep_meta["info"]
        
        # 更新当前 Episode 专属的配置上下文
        self.version = info.get("codebase_version", "v2.1")
        self.dorobot_version = info.get("dorobot_dataset_version", "")
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
                
        # 加载单条 parquet 数据
        self.df = pd.read_parquet(parquet_path)
        print(f"🔄 [LeRobot] 已切换至 Episode {episode_idx} ({self.current_dataset_root.name}/{parquet_path.name}), 轨迹帧数: {len(self.df)}")

    def get_total_episodes(self) -> int:
        return len(self.episodes_meta)

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
            
            # 1. 优先尝试静态图片 (同时兼容 {image_key} 为短命名或全命名的情况)
            if self.image_path_tpl:
                possible_keys = [short_name, full_key]
                for key_variant in possible_keys:
                    rel_path = self.image_path_tpl.format(
                        image_key=key_variant,
                        episode_index=ep_idx,
                        frame_index=frame_idx
                    )
                    # 注意: 使用当前轨迹专属的 dataset_root
                    full_path = self.current_dataset_root / rel_path
                    if full_path.exists():
                        img = cv2.imread(str(full_path))
                        if img is not None:
                            images[short_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_loaded = True
                            break # 成功找到图片则跳出 key_variant 循环
            
            # 2. 图片不存在，尝试视频抽帧
            if not img_loaded:
                video_files = list(self.current_dataset_root.rglob(f"**/{full_key}/*episode_{ep_idx:06d}.mp4"))
                if video_files:
                    video_path = video_files[0]
                    try:
                        reader = self.cap_cache.get(str(video_path))
                        if not reader:
                            reader = imageio.get_reader(str(video_path), 'ffmpeg')
                            self.cap_cache[str(video_path)] = reader
                        
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
        for reader in self.cap_cache.values():
            try:
                reader.close()
            except:
                pass
        self.cap_cache.clear()