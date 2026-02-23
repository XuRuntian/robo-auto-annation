"""
# Robo-ETL 交互式数据对齐工作台 (全局轨迹适配版)
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.core.factory import ReaderFactory
import numpy as np
import json
import os
import pandas as pd

def extract_kinematic_waveform(reader, total_frames):
    """提取运动学波形 (例如：夹爪动作、关节速度的 L2 Norm)"""
    waveform = []
    progress_bar = st.sidebar.progress(0, text="正在解析运动学特征...")
    
    for i in range(total_frames):
        try:
            frame_data = reader.get_frame(i)
            val = 0.0
            if frame_data and frame_data.state is not None:
                for key in ['action', 'qpos', 'qvel', 'velocity', 'effort']:
                    if key in frame_data.state and frame_data.state[key] is not None:
                        # 计算 L2 范数
                        val = np.linalg.norm(frame_data.state[key])
                        break
            waveform.append(val)
        except Exception as e:
            waveform.append(0.0)
            
        if i % 10 == 0 or i == total_frames - 1:
            progress_bar.progress((i + 1) / total_frames, text=f"提取特征: {i+1}/{total_frames}")
            
    progress_bar.empty()
    return np.array(waveform)

def main():
    st.set_page_config(page_title="Robo-ETL 数据对齐工作台", layout="wide")
    
    # ==========================================
    # 1. 数据源与顶层控制 (侧边栏)
    # ==========================================
    with st.sidebar:
        st.header("📂 导入全局数据")
        
        data_path = st.text_input("数据集路径 (如 .hdf5 / .mcap / parquet 根目录)", 
                                  value="/home/user/test_data/")
        
        uploaded_json = st.file_uploader("上传 AI 全局预标注 JSON (包含多条轨迹字典)", type=['json','jsonl'])
        
        if st.button("加载并初始化工作台", type="primary"):
            if not os.path.exists(data_path):
                st.error(f"找不到数据集文件：{data_path}")
                return
            if uploaded_json is None:
                st.warning("请上传包含 AI 预标注信息的 JSON 文件。")
                return
                
            try:
                # 解析上传的全局 JSON 字典
                all_annotations = json.load(uploaded_json)
                
                # 初始化 Reader
                reader = ReaderFactory.get_reader(data_path)
                if not reader.load(data_path):
                    st.error("数据集加载失败，请检查文件格式。")
                    return
                
                # 尝试获取总轨迹数 (兼容无 set_episode 的旧适配器)
                total_episodes = getattr(reader, "get_total_episodes", lambda: 1)()
                if hasattr(reader, "set_episode"):
                    reader.set_episode(0)
                    
                total_frames = reader.get_length()
                waveform = extract_kinematic_waveform(reader, total_frames)
                
                # 保存所有必需的状态到全局
                st.session_state.update({
                    'reader': reader,
                    'total_episodes': total_episodes,
                    'all_annotations': all_annotations,
                    'current_episode_idx': 0,
                    'total_frames': total_frames,
                    'waveform': waveform,
                    'current_frame': 0,
                    'data_loaded': True
                })
                st.rerun() # 加载完直接刷新页面
            except Exception as e:
                st.error(f"初始化失败：{str(e)}")

        # --- 轨迹切换控制区 ---
        if st.session_state.get('data_loaded', False):
            st.divider()
            st.header("🔄 轨迹切换")
            
            # 使用下拉框选择当前标注的 Episode
            ep_options = list(range(st.session_state.total_episodes))
            selected_ep = st.selectbox(
                "当前作业轨迹 (Episode)", 
                options=ep_options, 
                index=st.session_state.current_episode_idx,
                format_func=lambda x: f"Episode {x}"
            )
            
            # 侦测到用户切换了轨迹
            if selected_ep != st.session_state.current_episode_idx:
                reader = st.session_state.reader
                if hasattr(reader, "set_episode"):
                    reader.set_episode(selected_ep)
                
                total_frames = reader.get_length()
                waveform = extract_kinematic_waveform(reader, total_frames)
                
                # 更新状态并重置播放头
                st.session_state.update({
                    'current_episode_idx': selected_ep,
                    'total_frames': total_frames,
                    'waveform': waveform,
                    'current_frame': 0
                })
                st.rerun()

    if not st.session_state.get('data_loaded', False):
        st.info("👈 请在左侧边栏配置数据集路径并上传标注 JSON 文件。")
        return

    # 从全局字典中获取当前 episode 的 subtasks 列表
    ep_key = str(st.session_state.current_episode_idx)
    current_subtasks = st.session_state.all_annotations.get(ep_key, [])

    # ==========================================
    # 2. 视频/图像预览区 (左侧布局)
    # ==========================================
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"📹 图像预览 - Episode {st.session_state.current_episode_idx}")
        
        current_frame = st.slider(
            "拖拽时间轴查看具体帧", 
            0, 
            max(0, st.session_state.total_frames - 1),
            value=st.session_state.get('current_frame', 0),
            key='frame_slider'
        )
        st.session_state.current_frame = current_frame
        
        try:
            frame_data = st.session_state.reader.get_frame(current_frame)
            if frame_data and frame_data.images:
                cam_name = list(frame_data.images.keys())[0]
                img_array = frame_data.images[cam_name]
                st.image(img_array, caption=f"帧号: {current_frame} | 视角: {cam_name}", use_container_width=True)
                
                if len(frame_data.images) > 1:
                    tabs = st.tabs(list(frame_data.images.keys()))
                    for idx, (c_name, c_img) in enumerate(frame_data.images.items()):
                        with tabs[idx]:
                            st.image(c_img, use_container_width=True)
            else:
                st.warning("该帧未包含图像数据")
        except Exception as e:
            st.error(f"图像读取失败：{str(e)}")

    # ==========================================
    # 3. 多模态时间轴与物理波形图 (右侧布局)
    # ==========================================
    with col2:
        st.subheader("📊 多模态时间轴 (AI预测 vs 物理状态)")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.7])
        
        # Row 1: 时间轴色块
        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
        for i, task in enumerate(current_subtasks):
            fig.add_shape(
                type="rect",
                x0=task.get('start_frame', 0), x1=task.get('end_frame', 0),
                y0=0, y1=1,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                line_width=0,
                row=1, col=1
            )
            fig.add_trace(go.Scatter(
                x=[(task.get('start_frame', 0) + task.get('end_frame', 0)) / 2],
                y=[0.5],
                mode="text",
                text=[f"Task {task.get('subtask_id', i+1)}"],
                hoverinfo="text",
                hovertext=f"<b>指令:</b> {task.get('instruction', 'N/A')}<br><b>区间:</b> {task.get('start_frame')}-{task.get('end_frame')}",
                showlegend=False
            ), row=1, col=1)
        
        # Row 2: 物理波形图
        fig.add_trace(
            go.Scatter(
                x=np.arange(st.session_state.total_frames),
                y=st.session_state.waveform,
                mode='lines',
                name='Kinematic Feature',
                line=dict(color='#377eb8', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_vline(x=st.session_state.current_frame, line_width=2, line_dash="dash", line_color="red")
        
        fig.update_layout(
            height=450, margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified", xaxis2_title="Frame Number",
            yaxis=dict(showticklabels=False, showgrid=False),
            yaxis2=dict(title="Kinematic Feature")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 4. 交互式数据对齐编辑器 (底部布局)
    # ==========================================
    st.divider()
    st.subheader(f"✍️ Episode {st.session_state.current_episode_idx} 边界微调")
    
    # 构建 DataFrame，处理当前 episode 为空列表的情况
    df_subtasks = pd.DataFrame(current_subtasks)
    if df_subtasks.empty:
        df_subtasks = pd.DataFrame(columns=["subtask_id", "instruction", "start_frame", "end_frame"])
        
    edited_df = st.data_editor(
        df_subtasks,
        column_config={
            "subtask_id": st.column_config.NumberColumn("任务 ID", disabled=True),
            "instruction": st.column_config.TextColumn("自然语言指令 (Instruction)"),
            "start_frame": st.column_config.NumberColumn("起始帧", min_value=0, max_value=st.session_state.total_frames, step=1),
            "end_frame": st.column_config.NumberColumn("结束帧", min_value=0, max_value=st.session_state.total_frames, step=1)
        },
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic", # 允许增删行
        key='data_editor'
    )
    
    # 实时检测数据变更并回写到全局字典
    if not edited_df.equals(df_subtasks):
        st.session_state.all_annotations[ep_key] = edited_df.to_dict('records')
        st.rerun()

    # ==========================================
    # 5. 全局导出功能
    # ==========================================
    col3, col4 = st.columns([8, 2])
    with col4:
        # 注意：这里导出的是全集 all_annotations，而非单条轨迹
        json_str = json.dumps(st.session_state.all_annotations, indent=4, ensure_ascii=False)
        st.download_button(
            label="💾 保存并导出全局 JSON",
            data=json_str,
            file_name="all_aligned_annotations.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()