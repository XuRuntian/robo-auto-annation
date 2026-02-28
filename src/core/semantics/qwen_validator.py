import os
import json
import base64
import re
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
import httpx
from PIL import Image, ImageDraw, ImageFont

from src.core.semantics.base import BaseSemanticValidator
from src.core.types import ValidationResult
from src.core.image_utils import GridImageGenerator

def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_json_from_response(text: str) -> dict:
    """健壮的 JSON 提取器"""
    json_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
    match = json_pattern.search(text)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
        else:
            json_str = text
            
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法从 VLM 响应中解析 JSON。原始响应:\n{text}\n错误: {e}")

class QwenSemanticValidator(BaseSemanticValidator):
    def __init__(self):
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请先设置！")
            
        # 强制阿里云域名不走系统 VPN/代理（国内服务器挂代理必断连）
        os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,aliyuncs.com"

    def validate_point(
        self,
        window_data: Dict[str, List[np.ndarray]],
        physics_energy: float,
        previous_instruction: Optional[str] = None
    ) -> ValidationResult:
        """
        验证切点是否为真实任务切换点
        1. 将局部窗口图像拼接成九宫格
        2. 调用 Qwen-VL 模型进行语义验证
        3. 解析响应并生成 ValidationResult
        """
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量，请先设置！")
            
        # 1. 生成临时窗口图像
        temp_path = "temp_window_grid.jpg"
        GridImageGenerator.generate_window_grid(window_data, temp_path)
        
        try:
            # 2. 调用 Qwen-VL API
            result = self._call_qwen_vl_api(temp_path)
            
            # 3. 计算融合置信度
            fused_confidence = self.compute_fused_confidence(physics_energy, result["certainty"])
            
            return ValidationResult(
                is_true_switch=result["is_true_switch"],
                instruction=result["instruction"],
                confidence_score=fused_confidence,
                reasoning=result["reasoning"]
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _call_qwen_vl_api(self, image_path: str) -> dict:
        """调用阿里云 Qwen-VL 模型并返回解析好的结果"""
        # 修复了 URL 格式，并配置底层的 http 客户端（设置 120 秒超时）
        http_client = httpx.Client(
            timeout=httpx.Timeout(120.0), # 给大图片上传留足时间
        )

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 👈 这里的网址必须是纯净的！
            http_client=http_client
        )

        prompt_text = """请分析图像中的机器人动作序列，判断是否为真实任务切换点。

        [图像说明]
        图像包含三个时间窗口：
        - 左侧（pre）：切点前1秒的帧序列
        - 中间（mid）：切点附近0.5秒的帧序列
        - 右侧（post）：切点后1秒的帧序列

        [任务要求]
        请判断画面中是真实的任务切换（True）还是中途停顿（False）：
        1. 如果是真实切换（True），请给出指令描述和置信度
        2. 如果是中途停顿（False），请说明理由

        [输出格式]
        {
            "is_true_switch": true,
            "instruction": "Robotic arm grasps the object",
            "certainty": 0.92,
            "reasoning": "检测到明显的动作模式变化和目标交互"
        }
        """

        base64_image = encode_image_to_base64(image_path)
        print(f"📸 图片转 Base64 成功，体积大小约为: {len(base64_image) / 1024 / 1024:.2f} MB")
        print(f"🚀 正在建立与阿里云百炼的连接，请耐心等待 10-30 秒...")

        try:
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1, 
            )
        except Exception as e:
            print(f"❌ 网络请求阶段抛出异常: {e}")
            raise e

        print(f"✅ 已收到 VLM 模型的响应，正在解析...")
        raw_output = response.choices[0].message.content
        print(f"\n[VLM 原始返回]\n{raw_output}\n")
        
        result = extract_json_from_response(raw_output)
        return result
