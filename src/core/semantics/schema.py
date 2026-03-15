import os
import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List

# 1. 动态获取当前最新词汇表的函数
def get_current_vocab():
    vocab_path = "configs/task_vocab.yaml"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        # 默认保底
        return {
            "verbs": ["reach", "grasp", "lift", "move", "place", "release"],
            "predicates": ["hand_free", "holding", "is_open", "is_closed"]
        }

class StatePredicate(BaseModel):
    predicate: str
    objects: List[str] = Field(..., description="参与该谓词的实体列表")

    @model_validator(mode='after')
    def check_predicate_dynamic(self):
        vocab = get_current_vocab()
        if self.predicate not in vocab.get("predicates", []):
            raise ValueError(f"幻觉拦截: 谓词 '{self.predicate}' 不在当前任务允许的词汇表中 {vocab.get('predicates', [])}")
        return self

class Operator(BaseModel):
    action_verb: str
    subject: str = Field(default="robot_arm", description="动作执行者")
    target_object: str = Field(..., description="主要操作对象")
    preconditions: List[StatePredicate] = Field(...)
    effects: List[StatePredicate] = Field(...)

    @model_validator(mode='after')
    def check_verb_dynamic(self):
        vocab = get_current_vocab()
        if self.action_verb not in vocab.get("verbs", []):
            raise ValueError(f"幻觉拦截: 动词 '{self.action_verb}' 不在当前任务允许的词汇表中 {vocab.get('verbs', [])}")
        return self

    @model_validator(mode='after')
    def check_logical_conflicts(self):
        """本地逻辑校验器：基础物理规则强校验"""
        pre_preds = [p.predicate for p in self.preconditions]
        eff_preds = [p.predicate for p in self.effects]
        
        # 1. 互斥状态校验
        if "hand_free" in eff_preds and "holding" in eff_preds:
            raise ValueError(f"逻辑冲突: {self.action_verb} 的 Effect 不能同时包含 hand_free 和 holding")
            
        if "is_open" in eff_preds and "is_closed" in eff_preds:
            raise ValueError(f"逻辑冲突: {self.action_verb} 的 Effect 不能同时包含 is_open 和 is_closed")
            
        # 注意：这里我们放宽了对 grasp 和 release 的硬编码检查，
        # 因为不同构型的机器人（如针筒、吸盘）不一定适用 is_open/is_closed，交由 VLM 自由推导。
        return self

class PDDLTrajectory(BaseModel):
    thought: str = Field(..., description="分析物理和拓扑关系的思考过程")
    operators: List[Operator] = Field(..., description="当前视频片段内的动作序列")