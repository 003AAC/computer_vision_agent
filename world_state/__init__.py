"""
World State Understanding Module - 世界状态理解模块
====================================================
让 agent 拥有"持续更新的世界模型"，而不是每步重新观察。

核心模块：
  - base.py: Evidence（证据）/ Belief（信念）模型
  - state.py: 5 个模块化 State 类（environment/ui/application/file/task）
  - extractor.py: 工具结果 → evidence updates
  - manager.py: 编排 + 生命周期 + 压缩摘要
  - diff.py: 状态差异 + semantic_summary
  - validator.py: 矛盾检测 + capability 检查（Phase C）
  - capability.py: 能力系统（Phase C）
"""
from world_state.base import (
    Belief, Evidence, StateField, StateBase,
    SOURCE_CONFIDENCE, source_reliability,
)
from world_state.state import (
    EnvironmentState, UIState, ApplicationState, FileState, TaskState,
)
from world_state.extractor import StateExtractor, EvidenceUpdate
from world_state.manager import WorldStateManager
from world_state.diff import StateDiff
from world_state.validator import StateValidator, ConflictRecord
from world_state.capability import Capability, CapabilityRegistry
from world_state.lesson import Lesson, LessonExtractor

__all__ = [
    "Belief", "Evidence", "StateField", "StateBase",
    "SOURCE_CONFIDENCE", "source_reliability",
    "EnvironmentState", "UIState", "ApplicationState", "FileState", "TaskState",
    "StateExtractor", "EvidenceUpdate",
    "WorldStateManager",
    "StateDiff",
    "StateValidator", "ConflictRecord",
    "Capability", "CapabilityRegistry",
    "Lesson", "LessonExtractor",
]
