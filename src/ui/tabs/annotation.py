import streamlit as st
import yaml
import os
from src.core.pipeline import RoboAnnotationPipeline

def render_annotation_tab():
    st.markdown("#### 配置纯净数据源与全局先验")
    
    # 1. 基础数据路径
    data_path = st.text_input(
        "数据集路径 (需指向具体的任务文件夹):", 
        value=st.session_state.get('data_path', ""),
        key="input_data_path"
    )
    
    # 2. PDDL 先验配置区
    with st.expander("⚙️ 任务先验与硬件配置 (PDDL Context)", expanded=True):
        st.markdown("填入准确的物理与任务先验，帮助大模型框定动作生成空间，防止幻觉。")
        
        task_desc = st.text_input(
            "📝 全局任务描述 (Task Description):", 
            value="Tidy up the desk. put the orange in the fruit plate, put the paper ball in the trash can, stand the bottle upright...",
            help="描述机器人的最终目的和核心动作"
        )
        
        interacting_objects = st.text_input(
            "📦 交互物体 (Interacting Objects, 逗号分隔):", 
            value="fruit plate, orange, paper ball, trash can, bottle, art knife, pen holder, glue, eraser",
            help="限制 VLM 只能针对这些物体生成空间谓词 (Predicates)"
        )
        
        robot_info = st.text_input(
            "🤖 机器人硬件信息 (Hardware Setup):", 
            value="7-DOF Dual-Arm Manipulator with Parallel Grippers",
            help="极其关键：限制 VLM 能生成的动作上限。如没有夹爪就不能生成 grasp"
        )
        
        st.divider()
        
        # 左右分栏：左边生成控制，右边状态预览与编辑
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### 词汇表生成")
            if st.button("🪄 动态生成 task_vocab.yaml", use_container_width=True):
                if not os.path.exists(data_path):
                    st.error("❌ 请先在上方输入正确的数据集路径，我们需要从中抽取视觉帧！")
                else:
                    with st.spinner("正在从轨迹中抽取 9 帧视觉序列，并请求 VLM 推导动作空间，请稍候..."):
                        try:
                            # 初始化一个仅用于调用的 Pipeline 实例
                            temp_pipeline = RoboAnnotationPipeline(config={})
                            vocab = temp_pipeline.auto_generate_vocab(
                                data_path=data_path,
                                task_desc=task_desc,
                                objects=interacting_objects,
                                robot_info=robot_info
                            )
                            if vocab:
                                st.success("🎉 词汇表生成成功！右侧已刷新。")
                                # 强制重载页面，让右侧的 YAML 预览框读取最新数据
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ 生成失败: {str(e)}")
                    
        with col2:
            st.markdown("##### 词汇表人工审核与编辑")
            vocab_path = "configs/task_vocab.yaml"
            
            # 读取现有词汇表，如果不存在则提供基础模板
            current_yaml = "verbs:\n  - reach\n  - grasp\npredicates:\n  - hand_free\n  - holding\n"
            if os.path.exists(vocab_path):
                try:
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        current_yaml = f.read()
                except Exception as e:
                    st.error(f"读取配置失败: {e}")
            
            # 使用 text_area 替换静态的 st.code，允许用户修改
            edited_yaml = st.text_area(
                "确认无误后点击下方保存：", 
                value=current_yaml, 
                height=250,
                help="请确保格式为合法的 YAML，且包含 'verbs' 和 'predicates' 两个根节点。"
            )
            
            # 增加手动保存按钮
            if st.button("💾 保存词汇表修改", type="secondary", use_container_width=True):
                try:
                    # 尝试解析 YAML，防止用户写错缩进
                    parsed_yaml = yaml.safe_load(edited_yaml)
                    
                    if not isinstance(parsed_yaml, dict) or "verbs" not in parsed_yaml or "predicates" not in parsed_yaml:
                        st.error("❌ 格式错误：YAML 必须包含 'verbs' 和 'predicates' 两个列表节点！")
                    else:
                        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
                        with open(vocab_path, "w", encoding="utf-8") as f:
                            f.write(edited_yaml)
                        st.success("✅ 词汇表保存成功！接下来的流水线将使用最新规则。")
                except yaml.YAMLError as e:
                    st.error(f"❌ YAML 语法错误，请检查缩进或冒号: {e}")

    st.divider()

    # 3. 核心流水线启动区
    if st.button("▶️ 启动自动化标注", type="primary", use_container_width=True):
        if not os.path.exists(data_path):
            st.error("❌ 路径不存在，请检查后重试！")
            return
            
        with st.status("🚀 正在执行自动化标注管线...", expanded=True) as status:
            try:
                st.write("📂 加载配置文件...")
                with open("configs/robot_config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                
                pipeline = RoboAnnotationPipeline(config=config)
                
                st.write("🧠 正在进行物理切分与 VLM 语义标注...")
                pipeline.process_episode(data_path, task_desc)
                
                cost_report = pipeline.vlm.get_cost_report()
                
                st.session_state.data_path = data_path
                st.session_state.data_loaded = True
                
                status.update(label="🎉 自动化标注完成！", state="complete", expanded=False)
                
                st.divider()
                st.markdown("### 💰 本次标注消费账单")
                c1, c2, c3 = st.columns(3)
                c1.metric("图像+提示词 Tokens", f"{cost_report['prompt_tokens']:,}")
                c2.metric("生成文本 Tokens", f"{cost_report['completion_tokens']:,}")
                c3.metric("预估花费 (RMB)", f"¥ {cost_report['estimated_cost_rmb']:.4f}")

            except Exception as e:
                status.update(label=f"❌ 流水线崩溃: {str(e)}", state="error")