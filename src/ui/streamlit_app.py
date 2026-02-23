import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from src.core.factory import ReaderFactory
from src.core.pipeline import RoboETLPipeline

# --- 硬件配置 ---
ROBOT_CONFIG = {
    "gripper_dim_indices": list(range(12, 36)),
    "gripper_threshold": 0.05        
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
        except Exception:
            waveform.append(0.0)
    return np.array(waveform)

def main():
    st.set_page_config(page_title="Robo-ETL 具身智能工作台", layout="wide")
    st.title("🤖 Robo-ETL 具身智能数据流水线")

    # 双模工作台布局
    tab_pipeline, tab_align = st.tabs(["🚀 Phase 1: AI 自动管线", "✍️ Phase 2: 人工微调校验"])

    # ==========================================
    # Tab 1: Pipeline 流水线控制中心
    # ==========================================
    with tab_pipeline:
        st.markdown("#### 1. 配置数据源与离群检测")
        data_path = st.text_input("数据集路径:", value="/home/user/test_data/hdf5")
        
        col_btn1, col_btn2 = st.columns([1, 4])
        if col_btn1.button("▶️ 启动端到端处理", type="primary"):
            if not os.path.exists(data_path):
                st.error("路径不存在！")
                return

            with st.status("🚀 Robo-ETL 流水线执行中...", expanded=True) as status:
                try:
                    # 初始化后端管线
                    pipeline = RoboETLPipeline(data_path, ROBOT_CONFIG)
                    total_eps = pipeline.reader.get_total_episodes()
                    
                    # 准备全局字典格式
                    all_annotations = {}
                    
                    # 演示：为了防止时间过长，只处理前 3 条轨迹
                    process_limit = min(3, total_eps)
                    for ep_idx in range(process_limit):
                        st.write(f"**处理轨迹 Episode {ep_idx}/{total_eps}**")
                        
                        # 核心：调用后端处理该条轨迹，传入 progress_callback 动态更新 UI
                        def ui_logger(msg): st.write(msg)
                        ep_labels = pipeline.process_episode(ep_idx, progress_callback=ui_logger)
                        
                        # 存入大字典
                        all_annotations[str(ep_idx)] = ep_labels
                        st.write(f"✅ Episode {ep_idx} 处理完成！")
                    
                    # 收尾并将结果存入全局 session，无缝移交给 Tab 2
                    st.session_state.all_annotations = all_annotations
                    st.session_state.data_path = data_path
                    st.session_state.data_loaded = True
                    
                    pipeline.close()
                    status.update(label="🎉 全部处理完成！请切换至【人工微调校验】面板。", state="complete", expanded=False)
                    st.balloons()
                except Exception as e:
                    status.update(label=f"❌ 流水线崩溃: {e}", state="error")

    # ==========================================
    # Tab 2: 人工微调校验 (沿用之前的精细打磨代码)
    # ==========================================
    with tab_align:
        if not st.session_state.get('data_loaded', False):
            st.info("👈 请先在 Phase 1 中运行流水线，生成初始预标注数据。")
            return
            
        # 侧边栏注入阅读器用于渲染
        if 'reader' not in st.session_state:
            reader = ReaderFactory.get_reader(st.session_state.data_path)
            reader.load(st.session_state.data_path)
            st.session_state.reader = reader
            
        total_eps = st.session_state.reader.get_total_episodes()
        
        # 侧边栏控制器
        with st.sidebar:
            st.header("🎯 微调控制区")
            ep_options = list(range(total_eps))
            selected_ep = st.selectbox(
                "选择需要微调的轨迹", 
                options=ep_options, 
                index=st.session_state.get('current_episode_idx', 0),
                format_func=lambda x: f"Episode {x}"
            )
            
            # 如果轨迹切换，重新读取波形
            if selected_ep != st.session_state.get('current_episode_idx', -1):
                st.session_state.reader.set_episode(selected_ep)
                total_frames = st.session_state.reader.get_length()
                waveform = extract_kinematic_waveform(st.session_state.reader, total_frames)
                st.session_state.update({
                    'current_episode_idx': selected_ep,
                    'total_frames': total_frames,
                    'waveform': waveform,
                    'current_frame': 0
                })

        # --- 以下完全复用之前写的联动波形图和 DataEditor 逻辑 ---
        ep_key = str(st.session_state.current_episode_idx)
        current_subtasks = st.session_state.all_annotations.get(ep_key, [])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"📹 Episode {selected_ep} 预览")
            curr_frame = st.slider("拖拽时间轴", 0, max(0, st.session_state.total_frames - 1), key='frame_slider')
            try:
                frame_data = st.session_state.reader.get_frame(curr_frame)
                if frame_data and frame_data.images:
                    cam_name = list(frame_data.images.keys())[0]
                    st.image(frame_data.images[cam_name], use_container_width=True)
            except Exception: pass

        with col2:
            st.subheader("📊 宏观-微观物理对齐")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7])
            colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
            
            for i, task in enumerate(current_subtasks):
                fig.add_shape(type="rect", x0=task.get('start_frame', 0), x1=task.get('end_frame', 0), y0=0, y1=1, fillcolor=colors[i % len(colors)], opacity=0.6, row=1, col=1)
            
            fig.add_trace(go.Scatter(x=np.arange(st.session_state.total_frames), y=st.session_state.waveform, mode='lines', name='L2 Norm'), row=2, col=1)
            fig.add_vline(x=curr_frame, line_width=2, line_dash="dash", line_color="red")
            fig.update_layout(height=400, hovermode="x unified", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        df_subtasks = pd.DataFrame(current_subtasks) if current_subtasks else pd.DataFrame(columns=["subtask_id", "instruction", "start_frame", "end_frame"])
        edited_df = st.data_editor(df_subtasks, use_container_width=True, hide_index=True, num_rows="dynamic")
        
        if not edited_df.equals(df_subtasks):
            st.session_state.all_annotations[ep_key] = edited_df.to_dict('records')
            st.rerun()

        # 全局导出
        json_str = json.dumps(st.session_state.all_annotations, indent=4, ensure_ascii=False)
        st.download_button("💾 保存并导出最终数据集 JSON", data=json_str, file_name="final_robo_etl_dataset.json", use_container_width=True)

if __name__ == "__main__":
    main()