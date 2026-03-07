import yaml
from src.core.pipeline import RoboAnnotationPipeline

def main():
    # 1. 读取配置文件 (上一步我们约定的 yaml 格式)
    with open("configs/robot_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. 实例化黑盒流水线
    pipeline = RoboAnnotationPipeline(config=config)
    
    # 3. 传入数据集和人类指令，一键启动
    data_path = "/home/user/test_data/lerobot/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink"
    task_desc = "Pour water or beverage into the cup using one hand."
    
    pipeline.process_episode(data_path, task_desc)

if __name__ == "__main__":
    main()