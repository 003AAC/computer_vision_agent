"""
枚举类型定义 - Enum Types
=======================
定义复杂任务处理所需的所有枚举类型
"""
from enum import Enum


class SubTaskStatus(Enum):
    """子任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __str__(self) -> str:
        return self.value


class TaskState(Enum):
    """任务状态枚举"""
    IN_PROGRESS = "in_progress"
    STUCK = "stuck"
    COMPLETED = "completed"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value


class RepeatType(Enum):
    """重复类型枚举"""
    SINGLE = "single"
    SEQUENCE = "sequence"
    CYCLE = "cycle"
    NONE = "none"
    
    def __str__(self) -> str:
        return self.value


class Severity(Enum):
    """严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    def __str__(self) -> str:
        return self.value


class DecompositionSource(Enum):
    """分解来源枚举"""
    KNOWLEDGE_BASE = "knowledge_base"
    RULE_BASED = "rule_based"
    FALLBACK = "fallback"
    
    def __str__(self) -> str:
        return self.value


class TerminationCondition(Enum):
    """终止条件枚举"""
    MAX_STEPS = "max_steps"
    TASK_COMPLETED = "task_completed"
    REPEATED_OPERATIONS = "repeated_operations"
    TIER_FAILED = "tier_failed"
    LOW_CONFIDENCE = "low_confidence"
    
    def __str__(self) -> str:
        return self.value


class TaskTier(Enum):
    """任务执行层级枚举"""
    TIER_1_CLI = 1
    TIER_2_HOTKEY = 2
    TIER_3_UI_AUTOMATION = 3
    TIER_4_VISION = 4
    
    def __str__(self) -> str:
        return f"Tier {self.value}"