"""
验收结果数据模型 - Verification Result Data Model
===============================================
定义验收检测的结果数据结构
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from models.enums import TaskState, RepeatType, Severity, TerminationCondition


@dataclass
class VerificationResult:
    """验收结果数据类
    
    表示任务验收检测的结果
    
    Attributes:
        completed: 是否完成
        confidence: 置信度 (0-1)
        evidence: 证据字典 (视觉、逻辑、决策)
        reason: 原因说明
        suggestions: 建议列表
        task_state: 任务状态
    """
    completed: bool
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    suggestions: List[str] = field(default_factory=list)
    task_state: TaskState = TaskState.UNKNOWN
    
    def __post_init__(self):
        """验证置信度范围"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"置信度必须在0-1之间, 当前: {self.confidence}")
    
    def is_in_progress(self) -> bool:
        """检查任务是否进行中"""
        return self.task_state == TaskState.IN_PROGRESS
    
    def is_stuck(self) -> bool:
        """检查任务是否卡住"""
        return self.task_state == TaskState.STUCK
    
    def should_continue(self) -> bool:
        """检查是否应该继续执行"""
        return self.task_state in [TaskState.IN_PROGRESS, TaskState.UNKNOWN]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "completed": self.completed,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reason": self.reason,
            "suggestions": self.suggestions,
            "task_state": str(self.task_state)
        }


@dataclass
class RepeatResult:
    """重复操作检测结果数据类
    
    Attributes:
        has_repeat: 是否检测到重复
        repeat_count: 重复次数
        repeat_type: 重复类型
        severity: 严重程度
        pattern: 重复模式
        is_valid: 是否有效重复 (参数不同)
    """
    has_repeat: bool
    repeat_count: int = 0
    repeat_type: RepeatType = RepeatType.NONE
    severity: Severity = Severity.LOW
    pattern: List[str] = field(default_factory=list)
    is_valid: bool = True
    
    def should_terminate(self) -> bool:
        """判断是否应该终止
        
        Returns:
            是否应该终止
        """
        if not self.has_repeat:
            return False
        
        if self.is_valid:
            return False
        
        if self.severity == Severity.HIGH:
            return True
        
        if self.repeat_count >= 4 and self.severity == Severity.MEDIUM:
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "has_repeat": self.has_repeat,
            "repeat_count": self.repeat_count,
            "repeat_type": str(self.repeat_type),
            "severity": str(self.severity),
            "pattern": self.pattern,
            "is_valid": self.is_valid
        }


@dataclass
class TerminationDecision:
    """终止决策数据类
    
    Attributes:
        should_terminate: 是否应该终止
        condition: 终止条件
        reason: 原因说明
        priority: 优先级 (0-3, 数字越大优先级越高)
    """
    should_terminate: bool
    condition: Optional[TerminationCondition] = None
    reason: str = ""
    priority: int = 0
    
    def __post_init__(self):
        """验证优先级范围"""
        if self.priority < 0 or self.priority > 3:
            raise ValueError(f"优先级必须在0-3之间, 当前: {self.priority}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "should_terminate": self.should_terminate,
            "condition": str(self.condition) if self.condition else None,
            "reason": self.reason,
            "priority": self.priority
        }


@dataclass
class VerificationEvidence:
    """验收证据数据类
    
    Attributes:
        visual_score: 视觉证据分数 (0-1)
        logic_score: 逻辑证据分数 (0-1)
        decision_score: 决策证据分数 (0-1)
        visual_details: 视觉证据详情
        logic_details: 逻辑证据详情
        decision_details: 决策证据详情
    """
    visual_score: float = 0.0
    logic_score: float = 0.0
    decision_score: float = 0.0
    visual_details: Dict[str, Any] = field(default_factory=dict)
    logic_details: Dict[str, Any] = field(default_factory=dict)
    decision_details: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_confidence(self) -> float:
        """计算综合置信度
        
        权重: 视觉0.4 + 逻辑0.3 + 决策0.3
        
        Returns:
            综合置信度
        """
        return (
            self.visual_score * 0.4 +
            self.logic_score * 0.3 +
            self.decision_score * 0.3
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "visual_score": self.visual_score,
            "logic_score": self.logic_score,
            "decision_score": self.decision_score,
            "visual_details": self.visual_details,
            "logic_details": self.logic_details,
            "decision_details": self.decision_details,
            "confidence": self.calculate_confidence()
        }