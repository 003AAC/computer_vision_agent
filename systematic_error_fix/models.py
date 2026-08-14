"""
系统性错误修复 - 数据模型与枚举定义
==================================
定义错误诊断、置信度评估、路径规划、弹窗处理、里程碑管理所需的数据结构
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


class ErrorType(Enum):
    """错误类型枚举（按优先级排序）"""
    PROGRAM_NOT_FOUND = "program_not_found"
    PATH_NOT_FOUND = "path_not_found"
    ENCODING_ERROR = "encoding_error"
    PERMISSION_DENIED = "permission_denied"
    COMMAND_TIMEOUT = "command_timeout"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def priority(self) -> int:
        priorities = {
            ErrorType.PROGRAM_NOT_FOUND: 1,
            ErrorType.PATH_NOT_FOUND: 2,
            ErrorType.ENCODING_ERROR: 3,
            ErrorType.PERMISSION_DENIED: 4,
            ErrorType.COMMAND_TIMEOUT: 5,
            ErrorType.NETWORK_ERROR: 6,
            ErrorType.UNKNOWN_ERROR: 7
        }
        return priorities.get(self, 7)


class QualityLevel(Enum):
    """质量等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    
    def __str__(self) -> str:
        return self.value


class PathStatus(Enum):
    """路径状态枚举"""
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    SUCCESS = "success"
    FAILED = "failed"
    ABANDONED = "abandoned"
    
    def __str__(self) -> str:
        return self.value


class RepeatType(Enum):
    """重复类型枚举"""
    SINGLE = "single"
    DUAL = "dual"
    MULTI = "multi"
    NONE = "none"
    
    def __str__(self) -> str:
        return self.value


class PopupType(Enum):
    """弹窗类型枚举"""
    ERROR_DIALOG = "error_dialog"
    WARNING_DIALOG = "warning_dialog"
    INFO_DIALOG = "info_dialog"
    CONFIRMATION_DIALOG = "confirmation_dialog"
    FILE_NOT_FOUND = "file_not_found"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class PopupAction(Enum):
    """弹窗动作枚举"""
    CLICK_OK = "click_ok"
    CLICK_CANCEL = "click_cancel"
    CLICK_YES = "click_yes"
    CLICK_NO = "click_no"
    PRESS_ESCAPE = "press_escape"
    CLOSE_WINDOW = "close_window"
    
    def __str__(self) -> str:
        return self.value


class MilestoneStatus(Enum):
    """里程碑状态枚举"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class FixAction:
    """修复动作数据类"""
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    expected_outcome: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "description": self.description,
            "parameters": self.parameters,
            "priority": self.priority,
            "expected_outcome": self.expected_outcome
        }


@dataclass
class ErrorDiagnosisResult:
    """错误诊断结果数据类"""
    error_type: ErrorType
    confidence: float
    features: Dict[str, Any]
    fix_actions: List[FixAction] = field(default_factory=list)
    matched_rule: str = ""
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": str(self.error_type),
            "confidence": self.confidence,
            "features": self.features,
            "fix_actions": [fa.to_dict() for fa in self.fix_actions],
            "matched_rule": self.matched_rule,
            "reasoning": self.reasoning
        }


@dataclass
class ConfidenceEvaluationResult:
    """置信度评估结果数据类"""
    confidence: float
    quality_level: QualityLevel
    is_pseudo_success: bool
    evidence: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    should_degrade: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "quality_level": str(self.quality_level),
            "is_pseudo_success": self.is_pseudo_success,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
            "should_degrade": self.should_degrade
        }


@dataclass
class PathSuggestion:
    """路径建议数据类"""
    path: str
    method: str
    confidence: float
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "confidence": self.confidence,
            "description": self.description
        }


@dataclass
class NegativeExperience:
    """负面经验数据类"""
    path: str
    error_type: ErrorType
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "error_type": str(self.error_type),
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason
        }


@dataclass
class PathPlanningResult:
    """路径规划结果数据类"""
    has_loop: bool
    loop_type: RepeatType
    suggested_paths: List[PathSuggestion] = field(default_factory=list)
    negative_experiences: List[NegativeExperience] = field(default_factory=list)
    should_switch_path: bool = False
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_loop": self.has_loop,
            "loop_type": str(self.loop_type),
            "suggested_paths": [p.to_dict() for p in self.suggested_paths],
            "negative_experiences": [n.to_dict() for n in self.negative_experiences],
            "should_switch_path": self.should_switch_path,
            "reason": self.reason
        }


@dataclass
class PopupInfo:
    """弹窗信息数据类"""
    popup_type: PopupType
    title: str
    text: str
    buttons: List[str] = field(default_factory=list)
    rect: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "popup_type": str(self.popup_type),
            "title": self.title,
            "text": self.text,
            "buttons": self.buttons,
            "rect": self.rect
        }


@dataclass
class PopupHandlingResult:
    """弹窗处理结果数据类"""
    success: bool
    action_taken: PopupAction
    button_clicked: str = ""
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_taken": str(self.action_taken),
            "button_clicked": self.button_clicked,
            "error": self.error
        }


@dataclass
class Milestone:
    """里程碑数据类"""
    milestone_id: str
    name: str
    target_step: int
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    current_step: int = 0
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "target_step": self.target_step,
            "status": str(self.status),
            "current_step": self.current_step,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class StrategyAdjustment:
    """策略调整数据类"""
    adjustment_type: str
    reason: str
    new_max_steps: Optional[int] = None
    new_strategy: Optional[str] = None
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjustment_type": self.adjustment_type,
            "reason": self.reason,
            "new_max_steps": self.new_max_steps,
            "new_strategy": self.new_strategy,
            "priority": self.priority
        }


@dataclass
class MilestoneProgressReport:
    """里程碑进度报告数据类"""
    current_milestone: Optional[Milestone]
    overall_progress: float
    adjustments: List[StrategyAdjustment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_milestone": self.current_milestone.to_dict() if self.current_milestone else None,
            "overall_progress": self.overall_progress,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "warnings": self.warnings
        }