from typing import Dict, List, Optional, Union
import numpy as np
import cv2
import json
import re
import base64
import logging
from src.core.types import ValidationResult, CutPoint
from src.core.image_utils import encode_image_to_base64
from src.core.semantics.base import BaseSemanticValidator

logger = logging.getLogger(__name__)

class QwenSemanticValidator(BaseSemanticValidator):
    """
    基于Qwen-VL大模型的语义验证器，用于判断切点是否为子任务动作切换点。
    
    继承自 BaseSemanticValidator
    """
    def __init__(self, model_config: Dict = None):
        """
        初始化验证器
        
        参数:
            model_config (Dict): 模型配置参数，包含 API 密钥等信息
        """
        super().__init__()
        self.model_config = model_config or {}
        self.api_key = self.model_config.get('api_key', 'default_key')
        self.api_endpoint = self.model_config.get('api_endpoint', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation')
    
    def validate_point(self, window_data: Dict[str, List[np.ndarray]], 
                      physics_energy: float, 
                      previous_instruction: Optional[str] = None) -> ValidationResult:
        try:
            window_results = []
            instructions = []  # 存3个窗口分别的指令
            required_windows = ['before', 'center', 'after']
            
            for window in required_windows:
                if window not in window_data or len(window_data[window]) != 3:
                    raise ValueError(f"Invalid window data in '{window}'")
                
                # 调用Qwen-VL API (每次发送当前窗口的 3 张图)
                prompt = self._build_validation_prompt(previous_instruction)
                api_response = self._call_qwen_api(window_data[window], prompt)
                
                # 解析响应 (返回的是字典 Dict)
                vlm_dict = self._parse_api_response(api_response)
                
                # 收集结果
                window_results.append(vlm_dict.get('is_switch', False))
                instructions.append(vlm_dict.get('instruction', ""))
            
            # 2. 指令提取：优先使用center窗口，其次使用其他判定为True的窗口
            final_instruction = ""
            if window_results[1]:  # center 窗口判断为 True
                final_instruction = instructions[1]
            elif window_results[0]: # before 窗口判断为 True
                final_instruction = instructions[0]
            elif window_results[2]: # after 窗口判断为 True
                final_instruction = instructions[2]
            
            # 3. 融合物理能量和窗口结果 (调用父类的汉宁窗加权)
            final_confidence = self.compute_fused_confidence(
                physics_energy=physics_energy,
                window_results=window_results
            )
            
            # 4. 返回标准化的 ValidationResult
            return ValidationResult(
                is_true_switch=any(window_results), # 只要有一个窗口认定是切换即可（得分低会被UI标黄）
                instruction=final_instruction,
                confidence_score=final_confidence,
                reasoning=json.dumps({"window_votes": window_results}) # 原来的 raw_response 字段在基类里叫 reasoning
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            return ValidationResult(
                is_true_switch=False,
                instruction="",
                confidence_score=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    
    def _build_validation_prompt(self, previous_instruction: Optional[str] = None) -> str:
        # 【关键修改】Prompt 里要说是 3 帧
        prompt = """这是一段连续截取的3帧短视频（按时间顺序排列）。底层物理传感器提示这段时间内存在动作突变。请严格判断：
        这是全新的子任务动作切换（返回 {"is_switch": true}），还是同一个动作中的停顿/过程（返回 {"is_switch": false}）？

        如果是新的动作切换，请提供：
        - instruction: 纯英文的动作指令描述（如 "Pick up the red block"）
        - confidence: 你的判断确信度(0.0-1.0)

        请严格返回JSON格式，不要包含任何多余字符或Markdown标记：
        {
        "is_switch": boolean,
        "instruction": string,
        "confidence": number
        }"""
        
        if previous_instruction:
            prompt += f"\n\n注意：上一个已完成动作的指令是：{previous_instruction}"
        return prompt
    
    def _call_qwen_api(self, images: List[np.ndarray], prompt: str) -> str:
        """调用Qwen-VL API进行图像分析"""
        # 构建图像内容数组
        content_array = []
        
        # 将每张图像转换为base64格式并添加到content数组
        for image in images:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            content_array.append({
                "image": f"data:image/jpeg;base64,{image_base64}"
            })
        
        # 添加文本提示
        content_array.append({"text": prompt})
        
        # 构建请求体
        payload = {
            "model": "qwen-vl",
            "input": {
                "content": content_array
            }
        }
        
        # 发送请求（此处为伪代码，实际需要根据API文档实现）
        # response = requests.post(
        #     self.api_endpoint,
        #     headers={
        #         "Authorization": f"Bearer {self.api_key}",
        #         "Content-Type": "application/json"
        #     },
        #     json=payload
        # )
        
        # 返回模拟响应（用于测试）
        return json.dumps({
            "is_switch": True,
            "instruction": "Pick up the red block",
            "confidence": 0.85
        })
    
    def _parse_api_response(self, response: str) -> Dict:
        """解析API返回的JSON响应，支持带Markdown格式的响应"""
        try:
            # 使用正则表达式提取JSON内容（匹配第一个完整的JSON对象）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            # 解析提取的JSON字符串
            result = json.loads(json_match.group())
            
            # 验证返回格式
            if not all(key in result for key in ['is_switch', 'confidence']):
                raise ValueError("Invalid response format: missing required fields")
            
            # 确保置信度在有效范围内
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
            
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse API response: {str(e)}")
