import yaml
from src.core.pipeline import RoboAnnotationPipeline

def main():
    # 1. 读取配置文件 (上一步我们约定的 yaml 格式)
    with open("configs/robot_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. 实例化黑盒流水线
    pipeline = RoboAnnotationPipeline(config=config)
    
    # 3. 传入数据集和人类指令，一键启动
    data_path = "/home/xuruntian/下载/RMC-AIDA-L_desktop_organization"
    task_desc = "put the orange in the fruit plate, put the paper ball in the trash can, stand the bottle upright, put the art knife in the pen holder, put the glue in the pen holder, and put the eraser in the pen holder."
    
    pipeline.process_episode(data_path, task_desc)

if __name__ == "__main__":
    main()