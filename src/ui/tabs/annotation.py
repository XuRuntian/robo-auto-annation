import streamlit as st
import yaml
import os
from src.core.pipeline import RoboAnnotationPipeline

def render_annotation_tab():
    st.markdown("#### 配置纯净数据源与全局先验")
    
    data_path = st.text_input(
        "数据集路径 (需指向具体的任务文件夹):", 
        value=st.session_state.get('data_path', ""),
        key="input_data_path"
    )
    
    task_desc = st.text_area(
        "📝 设定全局任务描述 (Prior Knowledge):", 
        value="Pour water or beverage into the cup using one hand", 
        height=80
    )

    if st.button("▶️ 启动自动化标注", type="primary", use_container_width=True):
        if not os.path.exists(data_path):
            st.error("❌ 路径不存在，请检查后重试！")
            return
            
        with st.status("🚀 正在执行自动化标注管线...", expanded=True) as status:
            try:
                # 读取配置
                st.write("📂 加载配置文件...")
                with open("configs/robot_config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                
                # 初始化 Pipeline
                pipeline = RoboAnnotationPipeline(config=config)
                
                # 执行核心流程
                st.write("🧠 正在进行物理切分与 VLM 语义标注 (这可能需要一些时间)...")
                pipeline.process_episode(data_path, task_desc)
                
                cost_report = pipeline.vlm.get_cost_report()
                
                st.session_state.data_path = data_path
                st.session_state.data_loaded = True
                
                status.update(label="🎉 自动化标注完成！", state="complete", expanded=False)
                
                # 用 st.metric 画三个漂亮的数据指标卡片
                st.divider()
                st.markdown("### 💰 本次标注消费账单")
                col1, col2, col3 = st.columns(3)
                col1.metric("图像+提示词 Tokens (输入)", f"{cost_report['prompt_tokens']:,}")
                col2.metric("生成文本 Tokens (输出)", f"{cost_report['completion_tokens']:,}")
                col3.metric("预估花费 (RMB)", f"¥ {cost_report['estimated_cost_rmb']:.4f}")

            except Exception as e:
                status.update(label=f"❌ 流水线崩溃: {str(e)}", state="error")