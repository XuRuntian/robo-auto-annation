# src/core/semantics/schema.py
import os
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

# 1. 动态加载词汇表
def load_vocabulary():
    vocab_path = "configs/task_vocab.yaml"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        print("⚠️ 警告: 未找到 configs/task_vocab.yaml，使用默认词汇表")
        return {
            "verbs": ["reach", "grasp", "lift", "move", "place", "release", "pour", "insert"],
            "predicates": ["hand_free", "holding", "is_open", "is_closed", "on_table", "in_target"]
        }

VOCAB = load_vocabulary()

class StatePredicate(BaseModel):
    predicate: str
    objects: List[str] = Field(..., description="参与该谓词的实体列表")

    @field_validator('predicate')
    def check_predicate(cls, v):
        if v not in VOCAB["predicates"]:
            raise ValueError(f"幻觉拦截: 谓词 '{v}' 不在当前任务允许的词汇表中 {VOCAB['predicates']}")
        return v

class Operator(BaseModel):
    action_verb: str
    subject: str = Field(default="robot_arm", description="动作执行者")
    target_object: str = Field(..., description="主要操作对象")
    preconditions: List[StatePredicate] = Field(...)
    effects: List[StatePredicate] = Field(...)

    @field_validator('action_verb')
    def check_verb(cls, v):
        if v not in VOCAB["verbs"]:
            raise ValueError(f"幻觉拦截: 动词 '{v}' 不在当前任务允许的词汇表中 {VOCAB['verbs']}")
        return v

    @model_validator(mode='after')
    def check_logical_conflicts(self):
        """🌟 本地逻辑校验器：基于动态词汇表做基础规则校验"""
        pre_preds = [p.predicate for p in self.preconditions]
        eff_preds = [p.predicate for p in self.effects]
        
        # 1. 互斥状态校验
        if "hand_free" in eff_preds and "holding" in eff_preds:
            raise ValueError(f"逻辑冲突: {self.action_verb} 的 Effect 不能同时包含 hand_free 和 holding")
            
        if "is_open" in eff_preds and "is_closed" in eff_preds:
            raise ValueError(f"逻辑冲突: {self.action_verb} 的 Effect 不能同时包含 is_open 和 is_closed")
            
        # 2. 关键动作的前置/后置强校验
        if self.action_verb == "grasp":
            if "hand_free" not in pre_preds and "is_open" not in pre_preds:
                raise ValueError("逻辑错误: 执行 grasp 的 Precondition 必须说明手是空的或夹爪张开")
            if "holding" not in eff_preds:
                raise ValueError("逻辑错误: 执行 grasp 的 Effect 必须包含 holding")

        if self.action_verb in ["release", "place"]:
            if "holding" not in pre_preds:
                raise ValueError(f"逻辑错误: 执行 {self.action_verb} 的 Precondition 必须包含 holding")
            if "hand_free" not in eff_preds:
                raise ValueError(f"逻辑错误: 执行 {self.action_verb} 的 Effect 必须包含 hand_free")

        # 3. 针对靠近和撤回的放宽 (防止之前 Chunk 1,2 熔断)
        if self.action_verb == "approach":
            if "near" not in eff_preds and "contact" not in eff_preds:
                raise ValueError("建议: approach 的 Effect 通常应该包含 'near' 或 'contact'")

        return self

class PDDLTrajectory(BaseModel):
    thought: str = Field(..., description="分析物理和拓扑关系的思考过程")
    operators: List[Operator] = Field(..., description="当前视频片段内的动作序列")