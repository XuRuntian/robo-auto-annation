import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

from src.core.factory import ReaderFactory
from src.core.pipeline import RoboETLPipeline
from src.core.physics.kmeans_detector import KMeansPhysicsDetector
from src.core.windows.generator import LocalWindowGenerator
from src.core.semantics.qwen_validator import QwenSemanticValidator
from src.core.ai_screener import AIScreener
from src.core.organizer import DatasetOrganizer

# 全局配置
ROBOT_CONFIG = {
    "gripper_dim_indices": list(range(12, 36)),
    "gripper_threshold": 0.05,
    "fps": 30  # 添加帧率配置
}

def extract_kinematic_waveform(reader, total_frames):
    waveform = []
    for i in range(total_frames):
        try:
            frame_data = reader.get_frame(i)
            val = 0.0
            if frame_data and frame_data.state is not None:
                for key in ['action', 'qpos']:
                    if key in frame_data.state and frame_data.state[key] is not None:
                        val = np.linalg.norm(frame_data.state[key])
                        break
            waveform.append(val)
        except: waveform.append(0.0)
    return np.array(waveform)

def main():
    st.set_page_config(page_title="Robo-ETL 具身智能工作台", layout="wide")
    st.title("🤖 Robo-ETL 具身智能数据流水线")

    if 'suspect_episodes' not in st.session_state:
        st.session_state.suspect_episodes = []
    if 'all_annotations' not in st.session_state:
        st.session_state.all_annotations = {}

    tab_pipeline, tab_align = st.tabs(["🚀 Phase 1: 自动管线 & 质检", "✍️ Phase 2: 人工微调校验"])

    with tab_pipeline:
        # ---------------------------------------------------------
        # Step 0: 物理数据分类 (大杂烩分拣)
        # ---------------------------------------------------------
        with st.expander("🗂️ 前置工具：混合数据一键分类 (Organizer)", expanded=False):
            st.markdown("如果你的文件夹里混杂了多种格式的数据，请先在这里进行物理分拣。")
            mix_path = st.text_input("大杂烩目录路径:", value="/home/user/test_data/mix_data", key="mix_path")
            if st.button("🪄 一键物理分类"):
                try:
                    org = DatasetOrganizer(mix_path)
                    res = org.auto_organize()
                    st.success("✅ 分类完成！请在下方输入生成的 grouped_XXX 路径进行标注。")
                    st.json(res)
                except Exception as e:
                    st.error(f"分类失败: {e}")
        
        st.divider()
        
        # ---------------------------------------------------------
        # Step 1: 核心处理管线
        # ---------------------------------------------------------
        st.markdown("#### 1. 配置纯净数据源与全局先验")
        
        data_path = st.text_input(
            "纯净数据集路径 (⚠️ 请具体到单任务子文件夹):", 
            value="/home/user/test_data/mix_data/grouped_LeRobot/整理线缆与USB插入...", 
            help="请不要直接输入 grouped_XXX 根目录！必须进入一层，选择具体的任务文件夹，以保证这批数据都是同一个动作。"
        )
        
        from pathlib import Path
        if Path(data_path).name.startswith("grouped_"):
            st.warning("✋ 停！你当前选中了分类大汇总文件夹。为了防止大模型任务提示词混乱，请在路径后面加上具体的任务子文件夹名称（例如：.../grouped_LeRobot/整理线缆）。")
            return
            
        global_task_desc = st.text_area("📝 设定全局任务描述 (Prior Knowledge):", value="The overall task is: Mobile phone storage...", height=80)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👁️ 运行 AI 视觉异常筛查 (推荐)", use_container_width=True):
                if os.path.exists(data_path):
                    with st.spinner("CLIP 正在扫描特征..."):
                        reader = ReaderFactory.get_reader(data_path)
                        if reader.load(data_path):
                            screener = AIScreener()
                            suspects = screener.detect_outliers(reader, outlier_ratio=0.1, similarity_threshold=0.90)
                            st.session_state.suspect_episodes = suspects
                            if suspects: st.warning(f"🚨 发现疑似废片: {suspects}")
                            else: st.success("✨ 数据集很健康！")
                        reader.close()
                else: st.error("路径不存在！")

        with col2:
            if st.button("▶️ 启动全局标准拆解", type="primary", width='stretch'):
                if not os.path.exists(data_path):
                    st.error("路径不存在！")
                    return
                    
                with st.status("🚀 Robo-ETL 正在执行批处理...", expanded=True) as status:
                    try:
                        # 初始化核心组件
                        physics_detector = KMeansPhysicsDetector(fps=ROBOT_CONFIG["fps"])
                        window_generator = LocalWindowGenerator(window_size_seconds=1.0, fps=ROBOT_CONFIG["fps"])
                        semantic_validator = QwenSemanticValidator()
                        
                        # 创建流水线
                        pipeline = RoboETLPipeline(
                            data_path, 
                            ROBOT_CONFIG,
                            physics_detector,
                            window_generator,
                            semantic_validator
                        )
                        
                        # 第一阶段：生成全局模板
                        st.write("🔍 **第一阶段：抽取采样点并生成全局标注标准 (Only 1 API Call)...**")
                        master_template = pipeline.generate_global_template(global_task_desc, progress_callback=st.write)
                        
                        # 第二阶段：本地秒级对齐
                        st.write("⚡ **第二阶段：基于标准进行本地物理对齐处理...**")
                        total_eps = pipeline.reader.get_total_episodes()
                        
                        # 处理所有 Episode
                        for ep_idx in range(total_eps):
                            is_suspect = ep_idx in st.session_state.suspect_episodes
                            labels = pipeline.process_with_template(ep_idx, master_template, is_suspect=is_suspect)
                            st.session_state.all_annotations[str(ep_idx)] = labels
                            st.write(f"✅ Episode {ep_idx} 处理完成 (本地算法对齐)")
                        
                        st.session_state.data_path = data_path
                        st.session_state.data_loaded = True
                        pipeline.close()
                        status.update(label="🎉 数据集拆解完成！子任务指令已全局对齐。", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"❌ 流水线崩溃: {e}", state="error")
    
    # ==========================================
    # Tab 2: 人工微调校验 (保持原有逻辑不变)
    # ==========================================
    with tab_align:
        if not st.session_state.get('data_loaded', False):
            st.info("👈 请先在 Phase 1 中运行流水线。")
            return
            
        if 'reader' not in st.session_state:
            reader = ReaderFactory.get_reader(st.session_state.data_path)
            reader.load(st.session_state.data_path)
            st.session_state.reader = reader
            
        # 侧边栏控制区
        with st.sidebar:
            st.header("🎯 微调控制区")
            
            available_eps = [int(k) for k in st.session_state.all_annotations.keys()]
            if not available_eps:
                st.warning("所有轨迹已被剔除！")
                return
                
            available_eps.sort()

            def format_label(x):
                return f"Episode {x} 🚨 [废片]" if x in st.session_state.suspect_episodes else f"Episode {x}"

            selected_ep = st.selectbox("选择轨迹", options=available_eps, format_func=format_label)
            
            if selected_ep in st.session_state.suspect_episodes:
                st.error("⚠️ AI 警告：画面严重偏离！建议直接剔除。")
                
            if st.button("🗑️ 彻底剔除此轨迹 (丢弃)", type="primary", use_container_width=True):
                ep_key = str(selected_ep)
                if ep_key in st.session_state.all_annotations:
                    del st.session_state.all_annotations[ep_key]
                    st.success("已剔除！")
                    st.rerun()

            if selected_ep != st.session_state.get('current_episode_idx', -1):
                st.session_state.reader.set_episode(selected_ep)
                total_frames = st.session_state.reader.get_length()
                waveform = extract_kinematic_waveform(st.session_state.reader, total_frames)
                st.session_state.update({'current_episode_idx': selected_ep, 'total_frames': total_frames, 'waveform': waveform, 'current_frame': 0})

        # 主界面可视化
        ep_key = str(st.session_state.current_episode_idx)
        current_subtasks = st.session_state.all_annotations.get(ep_key, [])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"📹 Episode {st.session_state.current_episode_idx}")
            curr_frame = st.slider("拖拽时间轴", 0, max(0, st.session_state.total_frames - 1), key='frame_slider')
            try:
                frame_data = st.session_state.reader.get_frame(curr_frame)
                if frame_data and frame_data.images:
                    cam_name = list(frame_data.images.keys())[0]
                    st.image(frame_data.images[cam_name], use_container_width=True)
            except: pass

        with col2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7])
            colors = ['#EF553B', '#00CC96', '#AB63FA']
            for i, task in enumerate(current_subtasks):
                fig.add_shape(
                    type="rect", 
                    x0=task.get('start_frame', 0), 
                    x1=task.get('end_frame', 0), 
                    y0=0, y1=1, 
                    fillcolor=colors[i % len(colors)], 
                    opacity=0.6, 
                    row=1, 
                    col=1
                )
                fig.add_annotation(
                    x=(task.get('start_frame', 0) + task.get('end_frame', 0)) / 2,
                    y=0.5,
                    text=task.get('instruction', ''),
                    showarrow=False,
                    font=dict(size=12),
                    row=1,
                    col=1
                )
            
            fig.add_trace(go.Scatter(
                x=np.arange(st.session_state.total_frames), 
                y=st.session_state.waveform, 
                mode='lines'
            ), row=2, col=1)
            
            fig.add_vline(x=curr_frame, line_width=2, line_dash="dash", line_color="red")
            fig.update_layout(height=400, hovermode="x unified", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        df_subtasks = pd.DataFrame(current_subtasks) if current_subtasks else pd.DataFrame(
            columns=["subtask_id", "instruction", "start_frame", "end_frame"]
        )
        edited_df = st.data_editor(df_subtasks, use_container_width=True, hide_index=True, num_rows="dynamic")
        
        if not edited_df.equals(df_subtasks):
            st.session_state.all_annotations[ep_key] = edited_df.to_dict('records')
            st.rerun()

        st.download_button(
            "💾 导出最终 JSON", 
            data=json.dumps(st.session_state.all_annotations, indent=4, ensure_ascii=False), 
            file_name="final_dataset.json", 
            use_container_width=True
        )

if __name__ == "__main__":
    main()
