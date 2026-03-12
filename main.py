import streamlit as st
from src.ui.tabs.annotation import render_annotation_tab
from src.ui.tabs.visualization import render_visualization_tab

def main():
    st.set_page_config(page_title="Robo-ETL 具身智能工作台", layout="wide")
    st.title("🤖 Robo-ETL 具身智能数据流水线")

    # 1. 全局状态初始化
    if 'data_path' not in st.session_state:
        st.session_state.data_path = ""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []

    # 2. 划分功能模块 (Tabs)
    tab_pipeline, tab_visual = st.tabs(["🚀 自动化标注管线", "👁️ 标注结果可视化"])

    with tab_pipeline:
        render_annotation_tab()

    with tab_visual:
        render_visualization_tab()

if __name__ == "__main__":
    main()