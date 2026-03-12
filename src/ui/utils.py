import json
import os
import tempfile
import imageio
import cv2
import streamlit as st

@st.cache_data(show_spinner=False)
def generate_preview_video(_reader, total_frames, annotations, fps=30):
    """
    合成视频，并将当前的子任务语义实时“烧录”到画面上(HUD)，支持字体自动缩放
    """
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "robo_preview_video_with_hud.mp4")
    
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', macro_block_size=None)
    
    for i in range(total_frames):
        try:
            frame_data = _reader.get_frame(i)
            if frame_data and frame_data.images:
                cam_name = list(frame_data.images.keys())[0]
                img = frame_data.images[cam_name].copy() 
                
                # --- 1. 寻找当前帧任务 ---
                current_task = None
                for task in annotations:
                    start = task.get('start_frame', task.get('global_start_frame', 0))
                    end = task.get('end_frame', task.get('global_end_frame', 0))
                    if start <= i <= end:
                        current_task = task.get('instruction', '')
                        break
                
                # --- 2. 绘制半透明背景框 ---
                overlay = img.copy()
                # 矩形框坐标 (x1, y1), (x2, y2), 颜色, -1 表示填充
                cv2.rectangle(overlay, (10, 10), (810, 65), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
                # --- 3. 动态计算字体大小 ---
                display_info = f"Frame: {i:04d} | Task: {current_task if current_task else 'None'}"
                
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                target_scale = 0.6  # 理想初始大小
                thickness = 2
                max_width = 780     # 允许的最大像素宽度 (留点 padding)
                
                # 核心逻辑：自动缩小字体直到能放下
                while target_scale > 0.35:
                    (w, h), _ = cv2.getTextSize(display_info, font_face, target_scale, thickness)
                    if w <= max_width:
                        break
                    target_scale -= 0.15
                
                # 如果缩到 0.35 还是放不下，说明句子长得离谱，强制截断
                if target_scale <= 0.35:
                    display_info = display_info[:80] + "..."

                # --- 4. 渲染文字 ---
                color = (255, 255, 255) if current_task else (160, 160, 160)
                # (25, 45) 是起始坐标，根据框的高度做了微调
                cv2.putText(img, display_info, (25, 45), font_face, target_scale, color, thickness, cv2.LINE_AA)

                writer.append_data(img)
        except Exception as e:
            # st.error(f"Error at frame {i}: {e}") # 调试用
            continue
            
    writer.close()
    return video_path

def load_local_annotations(data_path):
    """
    加载并解析本地生成的 JSON。
    优先读取 subtask_instructions.json（包含人类可读的 instruction），
    并提取其中的 'segments' 列表。
    """
    # 优先查找你提供的新版 JSON (之前管线中保存的名字应该是 subtask_instructions.json)
    json_path = os.path.join(os.path.dirname(data_path), "subtask_instructions.json")
    
    # 兜底：如果没有高级指令，找底层的动作元组
    if not os.path.exists(json_path):
        json_path = os.path.join(os.path.dirname(data_path), "auto_annotations.json")
        
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 判断如果是新的结构 {"segments": [...]}，则提取列表
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            return data # 兼容直接是列表的旧结构
    return []