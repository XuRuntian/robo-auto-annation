# src/core/factory.py
from pathlib import Path
from typing import Optional
from src.core.interface import BaseDatasetReader
from src.adapters.hdf5_adapter import HDF5Adapter
from src.adapters.ros_adapter import RosAdapter
from src.adapters.unitree_adapter import UnitreeAdapter
from src.adapters.folder_adapter import FolderAdapter
from src.adapters.lerobot_adapter import LeRobotAdapter

class ReaderFactory:
    @staticmethod
    def detect_type(path: Path) -> str:
        """
        只检测类型，不返回 Reader 实例（轻量级）
        """
        if path.is_dir():
            # Unitree 特征
            if (path / "data.json").exists():
                return "Unitree"
                
            # [Fix] LeRobot / Dorobot 特征增强
            # 1. 检查标准元数据文件 (原生 LeRobot，单条轨迹直接包含 meta 文件夹)
            if (path / "meta" / "info.json").exists():
                return "LeRobot"
                
            # 2. [新增] 检查子目录中是否包含 meta/info.json (Dorobot 格式，多轨迹被包裹在父文件夹中)
            # 只要任意一个子文件夹里有 meta/info.json，就认为是 LeRobot/Dorobot 格式集
            if list(path.glob("*/meta/info.json")):
                return "LeRobot"
                
            # 3. 检查 data 目录下是否有 parquet (支持递归查找 chunk-xxx)
            if (path / "data").is_dir():
                try:
                    # 使用 rglob 查找任意深度的 parquet 文件，找到一个即止
                    next((path / "data").rglob("*.parquet"))
                    return "LeRobot"
                except StopIteration:
                    pass
                    
            # Folder 特征 (包含图片)
            if list(path.glob("*.jpg")) or list(path.glob("*.png")) or \
               list(path.glob("colors/*.jpg")):
                return "RawFolder"
        else:
            ext = path.suffix.lower()
            if ext in ['.h5', '.hdf5']: return "HDF5"
            if ext in ['.bag', '.mcap']: return "ROS"
            if ext == '.parquet': return "LeRobot"
        
        return "Unknown"

    @staticmethod
    def get_reader(file_path: str) -> BaseDatasetReader:
        path = Path(file_path)
        dtype = ReaderFactory.detect_type(path)
        
        if dtype == "Unitree": return UnitreeAdapter()
        if dtype == "LeRobot": return LeRobotAdapter()
        if dtype == "RawFolder": return FolderAdapter()
        if dtype == "HDF5": return HDF5Adapter()
        if dtype == "ROS": return RosAdapter()
        
        raise ValueError(f"无法识别的数据格式: {path.name} (绝对路径: {path.absolute()})")