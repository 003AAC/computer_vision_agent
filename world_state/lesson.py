"""
Lesson Extraction - 任务经验沉淀
================================
任务 WorldState 销毁前，将失败/成功经验沉淀到 Skill Memory 或结构化记录。

设计：
  - 从 WorldState 中提取：
    - 高置信确认的事实（confirmed facts）
    - 矛盾/不确定状态（未解决的状态）
    - capability 缺失（能力缺口）
  - 输出为可注入 Historical/Skill Memory 的结构化 Lesson
"""
from typing import Any, Dict, List, Optional

from world_state.manager import WorldStateManager
from world_state.validator import StateValidator
from world_state.capability import CapabilityRegistry


class Lesson:
    """一条任务经验"""

    def __init__(self, task: str = "", succeeded: bool = False):
        self.task = task
        self.succeeded = succeeded
        self.confirmed_facts: List[Dict[str, Any]] = []   # 高置信确认
        self.uncertain_states: List[Dict[str, Any]] = []  # 低置信/不确定
        self.capability_gaps: List[str] = []              # 缺失能力
        self.consistency_issues: List[str] = []           # 逻辑矛盾

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "succeeded": self.succeeded,
            "confirmed_facts": self.confirmed_facts,
            "uncertain_states": self.uncertain_states,
            "capability_gaps": self.capability_gaps,
            "consistency_issues": self.consistency_issues,
        }

    def __repr__(self) -> str:
        return f"Lesson({self.task}, success={self.succeeded}, facts={len(self.confirmed_facts)})"


class LessonExtractor:
    """经验提取器"""

    def __init__(self):
        self.validator = StateValidator()
        self.capabilities = CapabilityRegistry()

    def extract(self, manager: WorldStateManager,
                succeeded: bool = False) -> Lesson:
        """从 manager 当前状态提取经验

        Args:
            manager: WorldStateManager（尚未销毁）
            succeeded: 任务是否成功

        Returns:
            Lesson
        """
        lesson = Lesson(task=manager.task.goal.value or "",
                        succeeded=succeeded)

        # 1. 高置信确认的事实
        self._extract_confirmed_facts(manager, lesson)

        # 2. 不确定状态
        self._extract_uncertain_states(manager, lesson)

        # 3. capability 缺失
        self._extract_capability_gaps(manager, lesson)

        # 4. 一致性矛盾
        lesson.consistency_issues = self.validator.check_consistency(manager)

        return lesson

    # ============================================================

    def _extract_confirmed_facts(self, manager: WorldStateManager,
                                 lesson: Lesson):
        """收集所有高置信（>=0.8）状态"""
        def collect_fields(state_obj, domain):
            for attr, val in vars(state_obj).items():
                belief = getattr(val, "belief", None)
                if belief is None or belief.confidence < 0.8:
                    continue
                lesson.confirmed_facts.append({
                    "domain": domain,
                    "field": attr,
                    "value": belief.value,
                    "confidence": belief.confidence,
                    "source": (belief.supporting[-1].source
                               if belief.supporting else "unknown"),
                })

        collect_fields(manager.environment, "environment")
        collect_fields(manager.ui, "ui")
        for name, app in manager.applications.items():
            for attr, val in vars(app).items():
                belief = getattr(val, "belief", None)
                if belief is None or belief.confidence < 0.8:
                    continue
                lesson.confirmed_facts.append({
                    "domain": "applications",
                    "field": f"{name}.{attr}",
                    "value": belief.value,
                    "confidence": belief.confidence,
                    "source": (belief.supporting[-1].source
                               if belief.supporting else "unknown"),
                })

    def _extract_uncertain_states(self, manager: WorldStateManager,
                                  lesson: Lesson):
        """收集不确定状态（未知或低置信）"""
        def collect_uncertain(state_obj, domain):
            for attr, val in vars(state_obj).items():
                belief = getattr(val, "belief", None)
                if belief is None:
                    continue
                if belief.value is None or belief.confidence < 0.5:
                    lesson.uncertain_states.append({
                        "domain": domain,
                        "field": attr,
                        "value": belief.value,
                        "confidence": belief.confidence,
                        "status": belief.status,
                    })

        collect_uncertain(manager.environment, "environment")
        collect_uncertain(manager.ui, "ui")
        for name, app in manager.applications.items():
            for attr, val in vars(app).items():
                belief = getattr(val, "belief", None)
                if belief and (belief.value is None or belief.confidence < 0.5):
                    lesson.uncertain_states.append({
                        "domain": "applications",
                        "field": f"{name}.{attr}",
                        "value": belief.value,
                        "confidence": belief.confidence,
                        "status": belief.status,
                    })

    def _extract_capability_gaps(self, manager: WorldStateManager,
                                 lesson: Lesson):
        """检查哪些能力当前不可用"""
        for cap_name in self.capabilities.all_capabilities():
            result = self.capabilities.check(cap_name, manager)
            if not result["available"] and result["missing"]:
                # 只记录有实质缺失的（非"未注册"）
                if result["missing"][0]["path"] != "<capability未注册>":
                    lesson.capability_gaps.append(cap_name)