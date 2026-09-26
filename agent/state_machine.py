"""
Agent 状态感知机 - State Machine
==============================
统一管理 Agent 执行过程中的状态转换、卡住检测、死循环检测、进展追踪。

状态流转：
  INIT → RUNNING → STUCK → RECOVERING → RUNNING → COMPLETED
                                     ↘ FAILED
           RUNNING → FAILED（达到最大步数/致命错误）

核心功能：
  - 状态定义与合法性转换校验
  - 连续无进展检测（卡住判定）
  - 死循环检测（复用 systematic_error_fix.LoopDetector）
  - 状态注入提示词（让 LLM 感知自身处境）
"""
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from collections import Counter

from models.enums import TaskState, TerminationCondition
from systematic_error_fix.path_planner import LoopDetector

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent 运行状态枚举"""

    INIT = "init"
    RUNNING = "running"
    STUCK = "stuck"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


# 合法的状态转换表：{当前状态: 允许的下一个状态集合}
_VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.INIT: {AgentState.RUNNING, AgentState.FAILED},
    AgentState.RUNNING: {
        AgentState.RUNNING, AgentState.STUCK,
        AgentState.COMPLETED, AgentState.FAILED,
    },
    AgentState.STUCK: {AgentState.RECOVERING, AgentState.FAILED},
    AgentState.RECOVERING: {AgentState.RUNNING, AgentState.FAILED},
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
}


class StateTransitionError(Exception):
    """非法状态转换异常"""


class AgentStateMachine:
    """Agent 状态感知机

    Attributes:
        state: 当前状态
        max_stuck_steps: 连续无进展步数阈值（超过即判定 STUCK）
        max_same_tool_steps: 连续相同工具调用阈值
        task_state: 任务级状态（复用 models.enums.TaskState）
    """

    # 动作类工具（执行操作，产生实质进展）
    ACTION_TOOL_HINTS = {
        "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
        "run_powershell",
    }
    # 纯感知工具（只观察不操作）
    PERCEPTION_TOOLS = {
        "visual_scan", "visual_scan_region", "visual_scan_grid",
        "visual_locate", "visual_locate_region",
        "visual_read_text", "visual_read_region", "visual_find_text",
    }

    def __init__(
        self,
        max_stuck_steps: int = 4,
        max_same_tool_steps: int = 3,
        max_steps: int = 50,
    ):
        """初始化状态感知机

        Args:
            max_stuck_steps: 连续无进展步数阈值
            max_same_tool_steps: 连续相同工具调用阈值
            max_steps: 最大允许总步数
        """
        self.state = AgentState.INIT
        self.task_state = TaskState.UNKNOWN
        self.max_stuck_steps = max_stuck_steps
        self.max_same_tool_steps = max_same_tool_steps
        self.max_steps = max_steps

        # 步数追踪
        self.current_step = 0
        self.steps_without_progress = 0   # 连续无实质进展步数
        self.continuous_scan_count = 0    # 连续纯感知步数
        self.consecutive_failures = 0
        self.total_failures = 0

        # 工具调用历史
        self._tool_history: List[str] = []
        self._tool_count: Counter = Counter()

        # 重复检测器（复用系统性错误修复模块）
        self.loop_detector = LoopDetector(history_size=20)

        # 终止条件记录
        self.termination_condition: Optional[TerminationCondition] = None
        self.termination_reason: str = ""

        # 事件记录（供调试/提示词注入）
        self.events: List[Dict[str, Any]] = []

    def transition(self, new_state: AgentState, reason: str = "") -> bool:
        """执行状态转换（带合法性校验）"""
        if new_state == self.state:
            return False

        allowed = _VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            logger.warning(
                f"非法状态转换: {self.state} → {new_state}（原因: {reason}）"
            )
            return False

        old_state = self.state
        self.state = new_state
        self.events.append({
            "step": self.current_step,
            "from": str(old_state),
            "to": str(new_state),
            "reason": reason,
        })
        logger.info(f"状态转换: {old_state} → {new_state}（{reason}）")

        if new_state == AgentState.COMPLETED:
            self.task_state = TaskState.COMPLETED
        elif new_state == AgentState.FAILED:
            self.task_state = TaskState.UNKNOWN

        return True

    def start(self):
        """开始执行：INIT → RUNNING"""
        self.transition(AgentState.RUNNING, "任务开始执行")

    def mark_completed(self, reason: str = "任务完成"):
        """标记完成：RUNNING → COMPLETED"""
        if self.transition(AgentState.COMPLETED, reason):
            self.termination_condition = TerminationCondition.TASK_COMPLETED
            self.termination_reason = reason
            self.steps_without_progress = 0

    def mark_failed(self, reason: str, condition: Optional[TerminationCondition] = None):
        """标记失败：任意 → FAILED"""
        if self.transition(AgentState.FAILED, reason):
            self.termination_condition = condition or TerminationCondition.MAX_STEPS
            self.termination_reason = reason

    def record_step(
        self,
        tool_names: List[str],
        has_progress: bool = True,
        is_failure: bool = False,
    ) -> Dict[str, Any]:
        """记录一步执行，更新内部计数器并返回状态情报"""
        self.current_step += 1

        for name in tool_names:
            self._tool_history.append(name)
            self._tool_count[name] += 1

        if is_failure:
            self.total_failures += 1

        if has_progress:
            self.consecutive_failures = 0
        elif is_failure:
            self.consecutive_failures += 1

        if has_progress:
            self.steps_without_progress = 0
            self.continuous_scan_count = 0
        else:
            self.steps_without_progress += 1
            if tool_names and all(n in self.PERCEPTION_TOOLS for n in tool_names):
                self.continuous_scan_count += 1
            else:
                self.continuous_scan_count = 0

        stuck_reason = self._detect_stuck()
        if stuck_reason and self.state == AgentState.RUNNING:
            self.transition(AgentState.STUCK, stuck_reason)

        if self.current_step >= self.max_steps and self.state in (
            AgentState.RUNNING, AgentState.STUCK, AgentState.RECOVERING,
        ):
            self.mark_failed(
                f"达到最大步数限制 {self.max_steps}",
                TerminationCondition.MAX_STEPS,
            )

        return self.get_status_payload()

    def _detect_stuck(self) -> str:
        """检测是否卡住，返回卡住原因（空串表示未卡住）"""
        if self.steps_without_progress >= self.max_stuck_steps:
            return (
                f"连续 {self.steps_without_progress} 步无实质进展，"
                "可能陷入循环或不了解当前界面状态"
            )

        if self.continuous_scan_count >= 2:
            return (
                f"已连续 {self.continuous_scan_count} 步只调用视觉扫描"
                "但没有执行任何操作"
            )

        if len(self._tool_history) >= self.max_same_tool_steps:
            recent = self._tool_history[-self.max_same_tool_steps:]
            if len(set(recent)) == 1:
                return (
                    f"连续 {self.max_same_tool_steps} 步调用相同工具 "
                    f"'{recent[0]}'，疑似死循环"
                )

        if self._tool_history:
            self.loop_detector.record(self._tool_history[-1])
            has_loop, loop_type, count = self.loop_detector.detect()
            if has_loop:
                return f"检测到{loop_type}循环模式（重复{count}次），建议切换策略"

        return ""

    def recover(self, action_summary: str = ""):
        """恢复：STUCK/RECOVERING → RUNNING"""
        if self.state in (AgentState.STUCK, AgentState.RECOVERING):
            self.transition(AgentState.RUNNING, f"已切换策略恢复执行: {action_summary}")
            self.steps_without_progress = 0
            self.continuous_scan_count = 0
            self.consecutive_failures = 0

    def enter_recovering(self, action_summary: str):
        """进入恢复状态：STUCK → RECOVERING"""
        if self.state == AgentState.STUCK:
            self.transition(AgentState.RECOVERING, f"开始执行修复: {action_summary}")

    def get_status_payload(self) -> Dict[str, Any]:
        """获取状态情报（用于提示词注入和日志）"""
        recent_tools = self._tool_history[-5:]
        return {
            "state": str(self.state),
            "task_state": str(self.task_state),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "steps_without_progress": self.steps_without_progress,
            "continuous_scan_count": self.continuous_scan_count,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "recent_tools": recent_tools,
            "tool_usage_counts": {
                k: v for k, v in self._tool_count.most_common(8)
            },
        }

    def build_state_instruction(self) -> str:
        """构建状态指令文本（插入系统提示词，让 LLM 感知处境）"""
        payload = self.get_status_payload()
        lines = [
            f"## [状态感知] 当前状态信息",
            f"- 执行阶段: {payload['state']}",
            f"- 当前步数: {payload['current_step']}/{payload['max_steps']}",
            f"- 连续无进展步数: {payload['steps_without_progress']}",
            f"- 连续失败步数: {payload['consecutive_failures']}",
            f"- 最近工具: {', '.join(payload['recent_tools']) or '无'}",
        ]

        if self.state == AgentState.STUCK:
            lines.append(
                "- ⚠️ **你已处于【卡住】状态！** 不要重复尝试相同方法。"
                "请立即切换策略：尝试 visual_find_text() 定位、换 PowerShell 命令、"
                "或更换操作路径。"
            )
        elif self.state == AgentState.RECOVERING:
            lines.append(
                "- 🔧 **你正在【恢复】状态**：正在执行修复动作。"
                "修复完成后应回到正常执行。"
            )

        if payload["steps_without_progress"] >= 2:
            lines.append(
                f"- ⚠️ 已连续 {payload['steps_without_progress']} 步无实质进展，"
                "请立即执行操作或改变策略。"
            )

        if payload["continuous_scan_count"] >= 1:
            lines.append(
                f"- 👀 你已连续 {payload['continuous_scan_count']} 步只扫描未操作。"
                "请根据已有视觉信息直接执行 click_at / type_text / press_key。"
            )

        return "\n".join(lines)

    @classmethod
    def is_action_tool(cls, tool_name: str) -> bool:
        """判断工具是否为动作类（产生实质操作）"""
        return tool_name in cls.ACTION_TOOL_HINTS

    @classmethod
    def is_perception_tool(cls, tool_name: str) -> bool:
        """判断工具是否为感知类（只观察）"""
        return tool_name in cls.PERCEPTION_TOOLS

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": str(self.state),
            "task_state": str(self.task_state),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "steps_without_progress": self.steps_without_progress,
            "termination_condition": (
                str(self.termination_condition) if self.termination_condition else None
            ),
            "termination_reason": self.termination_reason,
            "events": self.events[-20:],
        }