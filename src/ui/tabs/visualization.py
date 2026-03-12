import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.core.factory import ReaderFactory
from src.ui.utils import load_local_annotations, generate_preview_video

def render_visualization_tab():
    if not st.session_state.get('data_loaded', False):
        st.info("👈 请先在【自动化标注管线】中完成数据处理，或者确保路径已加载。")
        return

    data_path = st.session_state.data_path
    
    # 初始化/加载数据
    if 'reader' not in st.session_state or st.session_state.reader is None:
        reader = ReaderFactory.get_reader(data_path)
        reader.load(data_path)
        st.session_state.reader = reader
        
        total_frames = reader.get_length()
        st.session_state.total_frames = total_frames
        st.session_state.annotations = load_local_annotations(data_path)

    total_frames = st.session_state.total_frames
    annotations = st.session_state.annotations

    # ==========================================
    # 第一层：视频播放区 (带 HUD)
    # ==========================================
    st.markdown("### 📹 轨迹视频与任务同步回放")
    with st.spinner("正在合成带任务遥测 HUD 的视频，请稍候..."):
        video_path = generate_preview_video(
            st.session_state.reader, 
            total_frames, 
            annotations, # 把标签传进去烧录
            fps=30
        )
    
    if video_path:
        # 控制视频居中并限制最大宽度，视觉效果更好
        _, col_vid, _ = st.columns([1, 4, 1])
        with col_vid:
            st.video(video_path)
    else:
        st.error("视频生成失败，请检查数据格式。")

    # ==========================================
    # 第二层：全局时间轴概览
    # ==========================================
    st.markdown("### ⏱️ 全局动作时序区间")
    
    fig = go.Figure()
    colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    
    for i, task in enumerate(annotations):
        start = task.get('global_start_frame', 0)
        end = task.get('global_end_frame', 0)
        verb = task.get('action_verb', 'Action')
        obj = task.get('object', '')
        hover_text = f"<b>动作:</b> {verb} {obj}<br><b>区间:</b> [{start} - {end}]"
        
        fig.add_shape(
            type="rect", x0=start, x1=end, y0=0, y1=1, 
            fillcolor=colors[i % len(colors)], opacity=0.7, 
            line=dict(width=0)
        )
        
        fig.add_trace(go.Scatter(
            x=[(start + end) / 2], y=[0.5],
            mode="markers", marker=dict(color="rgba(0,0,0,0)"),
            hoverinfo="text", hovertext=hover_text,
            showlegend=False
        ))
        
    fig.update_layout(
        height=100, # 高度压扁，作为纯粹的进度条总览
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        xaxis=dict(title="Frame Index", showgrid=True, range=[0, total_frames]),
        margin=dict(l=0, r=0, t=0, b=30),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 第三层：数据详情表
    # ==========================================
    st.markdown("### 📝 结构化动作列表")
    if annotations:
        df = pd.DataFrame(annotations)
        display_cols = [c for c in df.columns if c not in ["level", "subject_modifier", "object_modifier"]]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("当前轨迹暂无标注数据。")