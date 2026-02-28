# tests/test_new_architecture.py
import sys
import os
import json
import logging

# 确保能找到 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.factory import ReaderFactory
from src.core.physics.kmeans_detector import KMeansPhysicsDetector
from src.core.windows.generator import LocalWindowGenerator
from src.core.semantics.qwen_validator import QwenSemanticValidator
from src.core.pipeline import RoboETLPipeline

# 配置日志输出，方便看报错
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    # ⚠️ 1. 请替换为你的真实 LeRobot 或 HDF5 纯净测试数据路径
    TEST_DATA_PATH = "/home/xuruntian/下载/AIRBOT_MMK2_mobile_phone_storage" 
    
    # ⚠️ 2. 请替换为你的 Qwen API Key
    API_KEY = os.getenv("key")

    print("🛠️ 正在初始化 Robo-ETL 核心组件...")
    
    try:
        # A. 加载数据源
        reader = ReaderFactory.get_reader(TEST_DATA_PATH)
        if not reader.load(TEST_DATA_PATH):
            print("❌ 数据集加载失败，请检查路径。")
            return

        # B. 实例化四大金刚 (策略模式)
        physics_detector = KMeansPhysicsDetector(fps=30)
        window_generator = LocalWindowGenerator(fps=30, window_sec=1.0) # 前后各切1秒
        semantic_validator = QwenSemanticValidator({"api_key": API_KEY})
        
        # C. 装配流水线
        pipeline = RoboETLPipeline(
            reader=reader,
            physics_detector=physics_detector,
            window_generator=window_generator,
            semantic_validator=semantic_validator
        )
        
        print("🚀 启动端到端测试，处理 Episode 0 ...")
        # 运行核心流
        segments = pipeline.process_episode(episode_idx=0, confidence_threshold=0.3)
        
        print("\n🎉 处理完成！最终输出的时序 JSON 如下：")
        print(json.dumps(segments, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"\n💥 测试崩溃: {e}")
    finally:
        if 'reader' in locals():
            reader.close()

if __name__ == "__main__":
    run_test()