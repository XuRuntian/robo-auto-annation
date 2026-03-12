import json
import os
import tempfile
import imageio
import cv2
import streamlit as st

def wrap_text_opencv(text, font_face, font_scale, thickness, max_width):
    """
    辅助函数：根据最大像素宽度，将一段纯文本拆分成多行列表
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        # 测试加上当前单词后的宽度
        test_line = current_line + word + " "
        (w, h), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
        
        # 如果超宽了，且当前行不为空，就把当前行塞入列表，新单词作为下一行的开头
        if w > max_width and current_line != "":
            lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line = test_line
            
    if current_line:
        lines.append(current_line.strip())
        
    return lines

@st.cache_data(show_spinner=False)
def generate_preview_video(_reader, total_frames, annotations, fps=30):
    """
    合成视频，并将当前的子任务语义实时“烧录”到画面上(HUD)，支持固定字号+自动多行换行
    """
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "robo_preview_video_with_hud.mp4")
    
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', macro_block_size=None)
    
    # 设定固定的字体样式参数
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65  # 固定字号，保证清晰度
    thickness = 2
    
    for i in range(total_frames):
        try:
            frame_data = _reader.get_frame(i)
            if frame_data and frame_data.images:
                cam_name = list(frame_data.images.keys())[0]
                img = frame_data.images[cam_name].copy() 
                
                # 获取画面的宽度，动态决定文字框的最大宽度 (留出左右各 20 像素的 padding)
                img_h, img_w = img.shape[:2]
                max_text_width = img_w - 60 
                
                # --- 1. 寻找当前帧任务 ---
                current_task = None
                for task in annotations:
                    start = task.get('start_frame', task.get('global_start_frame', 0))
                    end = task.get('end_frame', task.get('global_end_frame', 0))
                    if start <= i <= end:
                        current_task = task.get('instruction', '')
                        break
                
                # --- 2. 准备要渲染的文本行 ---
                display_info = f"Task: {current_task if current_task else 'None'}"
                # 将文本折行
                wrapped_lines = wrap_text_opencv(display_info, font_face, font_scale, thickness, max_text_width)
                # 把帧号作为独立的第一行
                all_lines = [f"Frame: {i:04d}"] + wrapped_lines
                
                # --- 3. 动态计算背景框的高度 ---
                # 获取单行文字的高度
                (test_w, test_h), baseline = cv2.getTextSize("Test", font_face, font_scale, thickness)
                line_spacing = test_h + baseline + 8  # 行距
                
                box_x1, box_y1 = 10, 10
                box_x2 = img_w - 10
                # 背景框高度 = 上边距 + (行数 * 行高) + 下边距
                box_y2 = box_y1 + 10 + (len(all_lines) * line_spacing) + 5
                
                # --- 4. 绘制半透明背景框 ---
                overlay = img.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
                
                # --- 5. 逐行渲染文字 ---
                color = (255, 255, 255) if current_task else (160, 160, 160)
                
                # 初始 Y 坐标，加上文字自身的高度
                current_y = box_y1 + 15 + test_h 
                
                for idx, line in enumerate(all_lines):
                    # 帧号可以用不同颜色区分一下，比如亮黄色 (0, 255, 255) 在 BGR 里
                    line_color = (0, 255, 255) if idx == 0 else color
                    cv2.putText(img, line, (box_x1 + 15, current_y), font_face, font_scale, line_color, thickness, cv2.LINE_AA)
                    current_y += line_spacing

                writer.append_data(img)
        except Exception as e:
            continue
            
    writer.close()
    return video_path

def load_local_annotations(data_path):
    """
    加载并解析本地生成的 JSON。
    """
    json_path = os.path.join(os.path.dirname(data_path), "subtask_instructions.json")
    
    if not os.path.exists(json_path):
        json_path = os.path.join(os.path.dirname(data_path), "auto_annotations.json")
        
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "segments" in data:
                return data["segments"]
            return data
    return []