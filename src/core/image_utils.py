# src/core/image_utils.py
import cv2
import numpy as np
from PIL import Image
import os

class GridImageGenerator:
    """负责将底层数据帧渲染并拼接成发给 VLM 的九宫格图像"""

    @staticmethod
    def ensure_3d_rgb(image: np.ndarray) -> np.ndarray:
        """万能转换器：确保输入图像严格是 (H, W, 3) 的 RGB 格式"""
        if image is None:
            return None
            
        # 1. 如果是二维的 (H, W) -> 变成 (H, W, 3)
        if image.ndim == 2:
            # 如果是 16 位深度图或灰度图，先归一化到 8 位
            if image.dtype == np.uint16:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # 2. 如果是 (H, W, 1) 的三维图 -> 抽掉多余维度再变三通道
        if image.ndim == 3 and image.shape[2] == 1:
            img_2d = image[:, :, 0]
            if img_2d.dtype == np.uint16:
                img_2d = cv2.normalize(img_2d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            elif img_2d.dtype != np.uint8:
                img_2d = img_2d.astype(np.uint8)
            return cv2.cvtColor(img_2d, cv2.COLOR_GRAY2RGB)
            
        # 3. 如果是 (H, W, 4) 带透明度的 RGBA 图 -> 丢弃透明度
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        return image
    
    @staticmethod
    def draw_text_cv2(img_array, text, position=(20, 50), font_scale=1.5, thickness=3, text_color=(255, 255, 255)):
        """绘制带黑色背景框的高对比度文本"""
        img = img_array.copy()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = position
        cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        return img

    @staticmethod
    def create_optimal_composite_frame(frame_images, step_num):
        """提取 Global 和 Wrist 两个核心视角，上下拼接"""
        # 提取全局视角
        global_cam = None
        for key in ['cam_high_rgb', 'cam_third_view', 'front']:
            if key in frame_images and frame_images[key] is not None:
                global_cam = frame_images[key]
                break
        if global_cam is None:
            global_cam = next(iter(frame_images.values()))

        # 提取末端视角
        local_cam = None
        for key in ['cam_right_wrist_rgb', 'cam_left_wrist_rgb', 'wrist']:
            if key in frame_images and frame_images[key] is not None:
                local_cam = frame_images[key]
                break
        if local_cam is None:
            local_cam = global_cam  

        # 👇 【关键修复】在进行 resize 和 vstack 前，强制转换所有图片为 3D RGB
        global_cam = GridImageGenerator.ensure_3d_rgb(global_cam)
        local_cam = GridImageGenerator.ensure_3d_rgb(local_cam)

        # 统一缩放尺寸
        target_size = (640, 480)
        global_cam = cv2.resize(global_cam, target_size)
        local_cam = cv2.resize(local_cam, target_size)

        # 打上标签
        global_cam = GridImageGenerator.draw_text_cv2(global_cam, f"[{step_num}] Global", position=(20, 60), text_color=(255, 50, 50))
        local_cam = GridImageGenerator.draw_text_cv2(local_cam, f"Wrist", position=(20, 60), text_color=(255, 255, 255))

        return np.vstack((global_cam, local_cam))

    @staticmethod
    def generate_3x3_grid(reader, indices, output_path, max_dim_size=2000):
        """
        根据给定的关键帧索引，从 reader 中抽帧并生成 3x3 九宫格。
        返回: 成功与否的布尔值。
        """
        combo_images = []
        for i, idx in enumerate(indices):
            frame = reader.get_frame(idx)
            if frame and hasattr(frame, 'images') and frame.images:
                combo_arr = GridImageGenerator.create_optimal_composite_frame(frame.images, step_num=i+1)
                combo_images.append(combo_arr)
        
        if len(combo_images) != 9:
            print(f"❌ 警告: 无法提取完整的 9 帧图像，当前只有 {len(combo_images)} 帧。")
            return False

        # 拼装九宫格
        row1 = np.hstack(combo_images[0:3])
        row2 = np.hstack(combo_images[3:6])
        row3 = np.hstack(combo_images[6:9])
        master_grid = np.vstack((row1, row2, row3))
        
        # 尺寸保护：防止图片太大导致 API 超时
        max_dim = max(master_grid.shape[0], master_grid.shape[1])
        if max_dim > max_dim_size:
            scale = max_dim_size / max_dim
            master_grid = cv2.resize(master_grid, (0,0), fx=scale, fy=scale)
            
        Image.fromarray(master_grid).save(output_path, quality=85)
        return True