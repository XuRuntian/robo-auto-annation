# tests/test_pipeline.py
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

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    # ⚠️ 1. 请替换为你的真实数据路径
    TEST_DATA_PATH = "/home/xuruntian/下载/AIRBOT_MMK2_mobile_phone_storage" 
    
    # ⚠️ 2. 请替换为你的 API Key (如果没有配置环境变量的话)
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的API_KEY填这里")

    # ⚠️ 3. 提供一个简短的全局任务描述，帮助大模型更好地生成 SOP 剧本
    TASK_DESC = "机器人正在执行一个桌面整理任务，它需要抓取计算器和遥控器并放到床上，还会抓取蓝色方块。"

    print("🛠️ 正在初始化 Robo-ETL (混合架构版) ...")
    
    try:
        # A. 加载数据源
        reader = ReaderFactory.get_reader(TEST_DATA_PATH)
        if not reader.load(TEST_DATA_PATH):
            print("❌ 数据集加载失败，请检查路径。")
            return

        # B. 实例化组件 (策略模式)
        # 物理层：使用 Pelt 算法，最小间隔 30 帧(1秒)，惩罚项 penalty 可根据需要微调
        physics_detector = KMeansPhysicsDetector(fps=30, n_clusters=9) 
        
        # 窗口层：前后各切1秒，共抽取3帧
        window_generator = LocalWindowGenerator(fps=30, window_sec=1.0) 
        
        # 语义层：带上你的 API_KEY
        semantic_validator = QwenSemanticValidator({"api_key": API_KEY})
        
        # C. 装配流水线
        pipeline = RoboETLPipeline(
            reader=reader,
            physics_detector=physics_detector,
            window_generator=window_generator,
            semantic_validator=semantic_validator
        )
        
        print("\n==================================================")
        print("🎬 Step 1: 触发冷启动，生成全局 SOP 剧本...")
        print("==================================================")
        # 抽 3 条轨迹，拼成超级九宫格，让大模型输出宏观 SOP
        sop_list = pipeline.extract_global_sop(sample_size=3, task_desc=TASK_DESC, uniform_sampling=True)
        print(f"\n✅ 全局 SOP 生成完毕:\n{json.dumps(sop_list, indent=2, ensure_ascii=False)}")
        print(len(sop_list))
        print("\n==================================================")
        print("🔍 Step 2: 物理层提取切点 & 局部大模型对齐...")
        print("==================================================")
        # 处理 Episode 0 (这个时候内部会用提取好的 SOP 去做对齐)
        segments = pipeline.process_episode(episode_idx=0, confidence_threshold=0.3)
        
        print("\n==================================================")
        print("🎉 最终输出的结构化时序 JSON 如下：")
        print("==================================================")
        print(json.dumps(segments, indent=4, ensure_ascii=False))

    except Exception as e:
        logger.error(f"\n💥 测试崩溃: {e}", exc_info=True)
    finally:
        if 'reader' in locals():
            reader.close()

if __name__ == "__main__":
    run_test()