"""
QwenVLCaller 类实现用于机器人自动化标注流的多模态大模型调用器

该模块提供了一个高可用、高扩展性的类，用于调用阿里云 Qwen-VL 或其他兼容 OpenAI 格式的多模态大模型。
"""

import os
import json
import base64
import re
import logging
import io
from typing import List, Optional, Union, Dict, Any
from PIL import Image
from openai import OpenAI
import httpx

# 配置日志记录
logger = logging.getLogger(__name__)

class QwenVLCaller:
    """
    QwenVLCaller 类用于调用阿里云 Qwen-VL 或其他兼容 OpenAI 格式的多模态大模型
    
    Attributes:
        model_name: 使用的模型名称
        temperature: 生成文本的温度参数
        max_tokens: 最大生成 token 数量
        timeout: 请求超时时间
        api_key: API 访问密钥
        client: OpenAI 客户端实例
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None
    ):
        """
        初始化 QwenVLCaller 实例
        
        Args:
            model_name: 使用的模型名称，默认为 "qwen-vl-max"
            temperature: 生成文本的温度参数，默认为 0.1
            max_tokens: 最大生成 token 数量，默认为 4096
            timeout: 请求超时时间，默认为 120.0 秒
            base_url: API 基础 URL，默认为阿里云百炼兼容版URL
            api_key: API 访问密钥，默认从环境变量获取
            
        Raises:
            ValueError: 如果未提供 api_key 且环境变量中也未找到
        """
        # 设置环境变量防止代理断连
        os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,aliyuncs.com"
        
        # 获取 API 密钥
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and DASHSCOPE_API_KEY environment variable not found")
        
        # 配置 HTTP 客户端
        http_client = httpx.Client(timeout=httpx.Timeout(timeout))
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=http_client
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
    def _encode_pil_image_to_base64(self, image: Image.Image, max_size: int = 1024) -> str:
        """
        将 PIL 图像编码为 base64 格式
        
        Args:
            image: 需要编码的 PIL 图像对象
            max_size: 图像最大尺寸，默认为 1024
            
        Returns:
            str: base64 编码的图像字符串
        """
        # 转换为 RGB 模式（丢弃透明通道）
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        # 计算缩放比例
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        # 将图像保存为 JPEG 格式
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        
        # 返回 base64 编码
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_json(self, text: str) -> Union[dict, list]:
        """
        从文本中提取 JSON 数据
        
        Args:
            text: 包含 JSON 的原始文本
            
        Returns:
            Union[dict, list]: 提取的 JSON 对象或数组
            
        Raises:
            ValueError: 如果无法提取到有效的 JSON
        """
        # 优先级 1: 查找 Markdown 格式的 JSON（增强兼容性）
        markdown_match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if markdown_match:
            try:
                return json.loads(markdown_match.group(1))
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from markdown block: %s. Error: %s", 
                            markdown_match.group(1), str(e))
                raise ValueError(f"Failed to extract JSON from markdown block. Original text: {text}") from None
                
        # 优先级 2: 查找最外层的 JSON 结构
        try:
            # 先尝试整个文本解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 寻找最外层的 JSON 结构
            start_brace = text.find('{')
            start_bracket = text.find('[')
            
            if start_brace == -1 and start_bracket == -1:
                logger.error("No JSON structure found in text: %s", text)
                raise ValueError(f"No JSON structure found. Original text: {text}")
                
            # 确定起始位置
            start_pos = min((pos for pos in [start_brace, start_bracket] if pos != -1))
            
            # 寻找结束位置
            end_brace = text.rfind('}')
            end_bracket = text.rfind(']')
            
            if end_brace == -1 and end_bracket == -1:
                logger.error("No closing bracket/brace found in text: %s", text)
                raise ValueError(f"No closing bracket/brace found. Original text: {text}")
                
            end_pos = max((pos for pos in [end_brace, end_bracket] if pos != -1))
            
            try:
                return json.loads(text[start_pos:end_pos+1])
            except json.JSONDecodeError as e:
                logger.error("Failed to parse extracted JSON: %s. Error: %s", 
                            text[start_pos:end_pos+1], str(e))
                raise ValueError(f"Failed to extract JSON. Original text: {text}") from e

    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        system_prompt: Optional[str] = "You are a helpful and precise robotic expert."
    ) -> str:
        # 1. 构建基础消息结构
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 2. 构造多模态用户内容
        user_content = [{"type": "text", "text": prompt}]
        
        # 处理图像
        if images:
            for img in images:
                try:
                    # 确保编码过程不会因为单张图损坏而中断整个流程
                    img_base64 = self._encode_pil_image_to_base64(img)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to encode image: {e}")
                    continue
        
        messages.append({"role": "user", "content": user_content})
        
        # 3. 发起请求并增加安全校验
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if hasattr(response, 'usage') and response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                # 你也可以在这里加个 logger 打印单次消耗
                logger.info(f"单次请求消耗: Prompt {response.usage.prompt_tokens}, Completion {response.usage.completion_tokens}")

            # 检查响应有效性
            if not response.choices or len(response.choices) == 0:
                logger.warning("VLM returned an empty choice list.")
                return ""
                
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            logger.error(f"VLM API request failed: {e}")
            raise  # 或者返回空字符串，取决于你的重试策略

    def get_cost_report(self):
            """
            返回累计的 token 和估算费用。
            注意：这里的价格是按 qwen-vl-max 目前大概的单价估算的（例如：输入 0.02元/千token，输出 0.02元/千token）
            具体费率可以根据阿里云百炼最新的计费文档调整。
            """
            # 假设单价 (RMB / 1000 tokens)
            price_per_1k_prompt = 0.003
            price_per_1k_completion = 0.009
            
            cost = (self.total_prompt_tokens / 1000.0) * price_per_1k_prompt + \
                (self.total_completion_tokens / 1000.0) * price_per_1k_completion
                
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "estimated_cost_rmb": round(cost, 4)
            }