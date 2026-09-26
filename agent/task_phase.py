"""
Task Phase Machine - 任务阶段状态机
====================================
显式跟踪任务执行阶段，不让 LLM 每一步重新判断世界。

与 AgentStateMachine（运行健康状态）互补：
  - AgentStateMachine: INIT → RUNNING → STUCK → COMPLETED/FAILED（健康状态）
  - TaskPhaseMachine: DESKTOP → GAME_STARTED → MAIN_MENU → ...（任务阶段）

每个阶段定义：
  1. entry_conditions    - 进入条件（证据驱动的关键词/元素）
  2. allowed_actions     - 该阶段允许执行的动作
  3. expected_next       - 预期下一阶段
  4. verification_method - 如何验证当前处于此阶段

状态转换由**观察证据**驱动（OCR 找到关键词 → 进入下一阶段），
不由 LLM 自述，防止"伪完成"。
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PhaseDefinition:
    """阶段定义模型"""

    def __init__(
        self,
        name: str,
        entry_conditions: List[str] = None,
        allowed_actions: List[str] = None,
        expected_next: List[str] = None,
        verification_method: List[str] = None,
        description: str = "",
        est_cost: float = 0.0,
        alternatives: List[str] = None,
    ):
        """
        Args:
            name: 阶段名称（如 "MAIN_MENU"）
            entry_conditions: 进入条件（屏幕应出现的文字/元素关键词）
            allowed_actions: 该阶段允许的动作工具名
            expected_next: 预期的下一阶段列表
            verification_method: 验证方法描述
            description: 阶段说明
            est_cost: 该阶段预估成本（成本点，规划器给出）
            alternatives: 该阶段的备选做法（用于规避失效路径）
        """
        self.name = name
        self.entry_conditions = entry_conditions or []
        self.allowed_actions = allowed_actions or []
        self.expected_next = expected_next or []
        self.verification_method = verification_method or []
        self.description = description
        self.est_cost = float(est_cost or 0.0)
        self.alternatives = alternatives or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entry_conditions": self.entry_conditions,
            "allowed_actions": self.allowed_actions,
            "expected_next": self.expected_next,
            "verification_method": self.verification_method,
            "description": self.description,
            "est_cost": self.est_cost,
            "alternatives": self.alternatives,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseDefinition":
        return cls(
            name=d.get("name", ""),
            entry_conditions=d.get("entry_conditions", []),
            allowed_actions=d.get("allowed_actions", []),
            expected_next=d.get("expected_next", []),
            verification_method=d.get("verification_method", []),
            description=d.get("description", ""),
            est_cost=d.get("est_cost", 0.0),
            alternatives=d.get("alternatives", []),
        )


# 内置默认阶段定义（供无规划结果时兜底）
DEFAULT_PHASE = PhaseDefinition(
    name="EXPLORING",
    description="探索阶段：LLM 自主决策，无显式阶段定义时使用",
    allowed_actions=[
        "visual_scan", "visual_scan_region", "visual_scan_grid",
        "visual_locate", "visual_locate_region",
        "visual_read_text", "visual_read_region", "visual_find_text",
        "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
        "run_powershell",
    ],
)


class TaskPhaseMachine:
    """任务阶段状态机"""

    def __init__(self, phases: List[PhaseDefinition] = None,
                 start_phase: str = "EXPLORING"):
        """
        Args:
            phases: 阶段定义列表
            start_phase: 起始阶段名
        """
        self._phases: Dict[str, PhaseDefinition] = {}
        if phases:
            for p in phases:
                self._phases[p.name] = p
        # 始终保证兜底阶段存在
        if "EXPLORING" not in self._phases:
            self._phases["EXPLORING"] = DEFAULT_PHASE

        self.current_phase = start_phase if start_phase in self._phases else "EXPLORING"
        self._transition_log: List[Dict[str, Any]] = []
        self._max_transition_log = 20

    # ============================================================
    # 阶段管理
    # ============================================================

    def add_phase(self, phase: PhaseDefinition):
        self._phases[phase.name] = phase

    def add_phases(self, phases: List[PhaseDefinition]):
        for p in phases:
            self.add_phase(p)

    def get_phase(self, name: str) -> Optional[PhaseDefinition]:
        return self._phases.get(name)

    def all_phase_names(self) -> List[str]:
        return list(self._phases.keys())

    def has_phase(self, name: str) -> bool:
        return name in self._phases

    def get_expected_next(self) -> List[str]:
        """当前阶段的预期下一状态"""
        phase = self._phases.get(self.current_phase)
        return phase.expected_next if phase else []

    def build_instruction(self) -> str:
        """构建阶段状态指令（注入提示词）"""
        phase = self._phases.get(self.current_phase)
        lines = ["## [任务阶段] 当前阶段"]
        lines.append(f"- 当前阶段: {self.current_phase}")
        if phase and phase.description:
            lines.append(f"- 阶段说明: {phase.description}")
        if phase and phase.expected_next:
            lines.append(f"- 预期下一阶段: {', '.join(phase.expected_next)}")
        if phase and phase.allowed_actions:
            lines.append(f"- 本阶段允许操作: {', '.join(phase.allowed_actions[:8])}")
        if phase and phase.entry_conditions:
            lines.append(f"- 本阶段屏幕特征: {', '.join(phase.entry_conditions[:6])}")
        return "\n".join(lines)

    # ============================================================
    # 状态转换（证据驱动）
    # ============================================================

    def try_transition(self, observed_texts: List[str]) -> Optional[str]:
        """根据观察证据尝试状态转换

        若下一阶段的 entry_conditions 中的关键词出现在 observed_texts 中，
        则执行转换。

        Args:
            observed_texts: 最新观察到的文字/元素列表（OCR/视觉结果）

        Returns:
            转换后的阶段名；若无转换返回 None
        """
        if not observed_texts:
            return None

        current = self._phases.get(self.current_phase)
        candidates = current.expected_next if current else []

        for next_name in candidates:
            next_phase = self._phases.get(next_name)
            if next_phase is None:
                continue
            # 检查进入条件是否满足
            if self._check_conditions(next_phase.entry_conditions, observed_texts):
                self.transition_to(next_name, matched_evidence=observed_texts)
                return next_name

        # 如果当前阶段无 expected_next 或未匹配，尝试全阶段扫描
        # （用于 LLM 已声明完成但阶段未推进的情况）
        for name, phase in self._phases.items():
            if name == self.current_phase or name == "EXPLORING":
                continue
            if self._check_conditions(phase.entry_conditions, observed_texts):
                self.transition_to(name, matched_evidence=observed_texts)
                return name

        return None

    def _check_conditions(self, conditions: List[str],
                          observed_texts: List[str]) -> bool:
        """检查进入条件（关键词匹配）"""
        if not conditions:
            return False
        # 观察文本归一化
        obs = [t.lower().strip() for t in observed_texts]
        for cond in conditions:
            cond_l = cond.lower().strip()
            # 条件本身可能包含多个词（如 "苏联 国旗"），全部出现才算匹配
            parts = [p.strip() for p in cond_l.replace("，", " ").replace(",", " ").split() if p.strip()]
            if not parts:
                continue
            if all(any(p in o for o in obs) for p in parts):
                return True
        return False

    def transition_to(self, new_phase: str, matched_evidence: List[str] = None) -> bool:
        """显式转换到新阶段

        Args:
            new_phase: 目标阶段名
            matched_evidence: 匹配的证据（观察文本）

        Returns:
            是否成功转换
        """
        if new_phase not in self._phases:
            logger.warning(f"阶段 {new_phase} 未定义，忽略转换")
            return False
        if new_phase == self.current_phase:
            return False

        old = self.current_phase
        self.current_phase = new_phase
        self._transition_log.append({
            "from": old,
            "to": new_phase,
            "evidence": matched_evidence or [],
        })
        if len(self._transition_log) > self._max_transition_log:
            self._transition_log = self._transition_log[-self._max_transition_log:]
        logger.info(f"任务阶段转换: {old} → {new_phase}")
        return True

    def force_phase(self, phase: str) -> bool:
        """强制设置阶段（用于 LLM 明确声明 / 规划结果应用）"""
        if phase not in self._phases:
            return False
        self.current_phase = phase
        return True

    # ============================================================
    # 工具校验
    # ============================================================

    def is_action_allowed(self, tool_name: str) -> bool:
        """检查工具是否在当前阶段允许使用

        若当前阶段为 EXPLORING 或未定义 allowed_actions，则全部允许。
        """
        phase = self._phases.get(self.current_phase)
        if phase is None or not phase.allowed_actions:
            return True
        return tool_name in phase.allowed_actions

    def get_disallowed_hint(self, tool_name: str) -> str:
        """构建工具越权提示（注入 LLM）"""
        phase = self._phases.get(self.current_phase)
        allowed = phase.allowed_actions if phase else []
        return (
            f"【阶段限制】当前处于 [{self.current_phase}] 阶段，"
            f"工具 {tool_name} 不在允许列表。"
            f"请先执行允许的操作完成本阶段，"
            f"或使用: {', '.join(allowed[:8])}"
        )

    # ============================================================
    # 序列化 / 信息
    # ============================================================

    def get_status_payload(self) -> Dict[str, Any]:
        phase = self._phases.get(self.current_phase)
        return {
            "current_phase": self.current_phase,
            "phase_description": phase.description if phase else "",
            "expected_next": phase.expected_next if phase else [],
            "transition_count": len(self._transition_log),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "phases": [p.to_dict() for p in self._phases.values()],
            "transition_log": self._transition_log,
        }