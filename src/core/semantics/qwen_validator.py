from typing import Dict, List, Optional, Union
import numpy as np
import cv2
import json
import re
import base64
import os
import logging
import httpx
from openai import OpenAI

from src.core.types import ValidationResult, CutPoint
from src.core.semantics.base import BaseSemanticValidator

logger = logging.getLogger(__name__)

class QwenSemanticValidator(BaseSemanticValidator):
    """
    基于 Qwen-VL 大模型的语义验证器，结合了强大的 OpenAI 兼容接口。
    """
    def __init__(self, model_config: Dict = None):
        super().__init__()
        self.model_config = model_config or {}
        # 从环境变量或配置中读取 API Key
        self.api_key = self.model_config.get('api_key') or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 API Key，请在配置中传入或设置 DASHSCOPE_API_KEY 环境变量！")
            
        # 强制使用兼容模式端点
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = self.model_config.get('model_name', 'qwen-vl-max')
        
        # 强制阿里云域名不走系统代理
        os.environ["NO_PROXY"] = "dashscope.aliyuncs.com,aliyuncs.com"
        
        # 配置底层 HTTP 客户端，留足超时时间
        self.http_client = httpx.Client(timeout=httpx.Timeout(120.0))
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client
        )
    
    def validate_point(self, window_data: Dict[str, List[np.ndarray]], 
                      physics_energy: float, 
                      previous_instruction: Optional[str] = None,
                      expected_next_step: str = "") -> ValidationResult:
        try:
            window_results = []
            instructions = []
            required_windows = ['before', 'center', 'after']
            
            for window in required_windows:
                if window not in window_data or len(window_data[window]) != 3:
                    raise ValueError(f"Invalid window data in '{window}'")
                
                # 提取图像
                images_for_api = []
                for frame_obj in window_data[window]:
                    # 1. 兼容直接传入的 numpy 数组
                    if isinstance(frame_obj, np.ndarray):
                        images_for_api.append(frame_obj)
                    # 2. 提取 FrameData 中的 images 字典 (修复这里的属性名和字典解析)
                    elif hasattr(frame_obj, 'images') and frame_obj.images:
                        # 你可以默认取第一个视角，也可以写死取全局视角 'cam_high_rgb'
                        cam_name = list(frame_obj.images.keys())[0] 
                        images_for_api.append(frame_obj.images[cam_name])
                    else:
                        logger.warning(f"⚠️ 无法从帧对象提取图像: {type(frame_obj)}")

                # 安全拦截：如果当前窗口没提取出 3 张图，直接跳过
                if len(images_for_api) != 3:
                    logger.warning(f"⚠️ 窗口 {window} 提取的有效图像数量为 {len(images_for_api)}，期望为 3")
                    continue

                # 构建 Prompt 并调用
                prompt = self._build_validation_prompt(previous_instruction)
                api_response = self._call_qwen_api(images_for_api, prompt)
                
                # 解析响应
                vlm_dict = self._parse_api_response(api_response)
                window_results.append(vlm_dict.get('is_switch', False))
                instructions.append(vlm_dict.get('instruction', ""))
            
            # 提取最终指令
            final_instruction = ""
            if len(window_results) == 3:
                if window_results[1]:  # center 窗口判断为 True
                    final_instruction = instructions[1]
                elif window_results[0]:
                    final_instruction = instructions[0]
                elif window_results[2]:
                    final_instruction = instructions[2]
            
            # 融合打分
            final_confidence = self.compute_fused_confidence(
                physics_energy=physics_energy,
                window_results=window_results if len(window_results) == 3 else [False, False, False]
            )
            
            return ValidationResult(
                is_true_switch=any(window_results) if window_results else False,
                instruction=final_instruction,
                confidence_score=final_confidence,
                reasoning=json.dumps({"window_votes": window_results})
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}", exc_info=True)
            return ValidationResult(
                is_true_switch=False,
                instruction="",
                confidence_score=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def _build_validation_prompt(self, expected_next_step: str) -> str:
        prompt = """You are a highly precise robotic data annotator.
        You are given a sequence of 3 keyframes (representing a local time window).
        A physical change-point detection algorithm (Pelt) has flagged this window as a potential transition due to energy/velocity fluctuations.

        [Your Mission]
        Your job is to determine if this window represents a genuine SEMANTIC SWITCH to a new macro-level subtask.

        [Action Persistence Principle - READ CAREFULLY]
        - DO NOT over-segment. Minor speed adjustments, jitter, or pauses within the SAME logical action are NOT switches.
        - Return "is_switch": false if the robot is still pursuing the SAME goal as the previous subtask.
        - Return "is_switch": true ONLY if there is a fundamental change in the nature of interaction, the target object, or the intended sub-goal (e.g., from 'approaching' to 'grasping', or 'moving' to 'releasing').

        [Output Format]
        Output ONLY a strict JSON object. No conversational text.
        {
            "is_switch": boolean,
            "instruction": "If is_switch is true, write a short goal-oriented English instruction for the NEW subtask. Otherwise, leave it empty.",
            "confidence": float (0.0 to 1.0),
            "reasoning": "A very brief explanation of why this is or isn't a new subtask (e.g., 'Still moving toward the cube' or 'Grasp initiated')"
        }
        """
        # 使用标准流程步骤进行判断
        prompt += f"\n[任务标准流程]\n当前阶段应执行的操作是: '{expected_next_step}'.\n"
        prompt += "请严格依据上述标准流程判断，当前帧序列是否标志着机器人已成功切入该步骤。"

        return prompt
    
    def _call_qwen_api(self, images: List[np.ndarray], prompt: str) -> str:
        """使用 OpenAI SDK 兼容模式调用 Qwen-VL"""
        # 构建消息体：先放文本
        content_list = [{"type": "text", "text": prompt}]
        
        # 依次放入图像序列
        for idx, image in enumerate(images):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, buffer = cv2.imencode('.jpg', image, encode_param)
            if success:
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })

        try:
            # 采用你旧代码里极其稳定的调用方式
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content_list
                    }
                ],
                temperature=0.1,  # 👈 核心关键：禁止模型发散思维
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Client 请求抛出异常: {e}")
            raise e
    
    def _parse_api_response(self, text: str) -> Dict:
        """移植你旧代码中健壮的 JSON 提取器"""
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
            result = json.loads(json_str)
            # 确保置信度字段存在
            if 'confidence' not in result:
                result['confidence'] = 0.5
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"无法从 VLM 响应中解析 JSON。原始响应:\n{text}\n错误: {e}")
