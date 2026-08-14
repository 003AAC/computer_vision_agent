"""
State Manager - 全局状态管理器
==============================
在 WorkingMemory（短期工作记忆）之上增加**结构化世界状态**管理。

与 WorkingMemory 的关系：
  - WorkingMemory: 单任务生命周期内的轻量记忆（task/last_action/expected_transition）
  - StateManager:  在其上增加分层世界状态 + 失败计数 + 观察历史 + 持久化

分层状态：
  {
    "task": "打开 Beholder 并游玩",
    "current_stage": "MAIN_MENU",          # 对应 TaskPhaseMachine.current_phase
    "world_state": {                        # 结构化世界理解
      "environment": {"desktop_visible": true, "foreground_window": "Beholder"},
      "ui_state": {"has_popup": false, "popup_type": null},
      "task_state": "in_progress"
    },
    "last_action": {...},                   # 复用 WorkingMemory
    "expected_transition": {...},           # 复用 WorkingMemory
    "failure_count": 2,                     # 连续失败次数
    "retry_count": 1,                       # 当前动作重试次数
    "observation_history": [...]            # 最近观察记录
  }

设计原则：
  - **完全向后兼容**：WorkingMemory 的所有旧接口保持不变
  - StateManager 持有 WorkingMemory 引用（组合而非继承），所有旧方法透传
  - 新增能力都是"附加层"，现有代码（core.py）无需改动即可继续工作
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from agent.memory.working_memory import WorkingMemory


class StateManager:
    """全局状态管理器（WorkingMemory 的增强层）"""

    # 分层状态键
    ENVIRONMENT = "environment"
    UI = "ui_state"
    TASK = "task_state"

    def __init__(self, working_memory: WorkingMemory = None):
        """
        Args:
            working_memory: 已有的 WorkingMemory 实例；若 None 则新建
        """
        # 核心：持有工作记忆引用（组合）
        self.wm = working_memory or WorkingMemory()

        # ============================================================
        # 分层世界状态
        # ============================================================
        self.world_state: Dict[str, Any] = {
            self.ENVIRONMENT: {},
            self.UI: {},
            self.TASK: "unknown",
        }

        # ============================================================
        # 执行追踪
        # ============================================================
        self._failure_count: int = 0
        self._retry_count: int = 0
        self._max_retries: int = 3
        self._observation_history: List[Dict[str, Any]] = []
        self._max_observation_history: int = 10

        # ============================================================
        # 持久化
        # ============================================================
        self._snapshots: List[Dict[str, Any]] = []

    # ============================================================
    # WorkingMemory 透传（保持旧接口完全兼容）
    # ============================================================

    @property
    def data(self) -> Dict[str, Any]:
        """直接访问底层工作记忆数据"""
        return self.wm._data

    def init_task(self, task: str, application: str = "",
                  initial_state: str = "", confidence: float = 0.5):
        """初始化任务（透传 WorkingMemory + 重置 StateManager 附加状态）"""
        self.wm.init_task(task, application, initial_state, confidence)
        self.world_state = {
            self.ENVIRONMENT: {"application": application},
            self.UI: {},
            self.TASK: "unknown",
        }
        self._failure_count = 0
        self._retry_count = 0
        self._observation_history = []
        self._snapshots = []

    def update(self, **kwargs):
        """透传 WorkingMemory.update"""
        self.wm.update(**kwargs)

    def record_action(self, tool_name: str, args: Dict[str, Any],
                      result_str: str = "") -> Dict[str, Any]:
        """透传 WorkingMemory.record_action"""
        return self.wm.record_action(tool_name, args, result_str)

    def update_after_action(self, tool_name: str, args: Dict[str, Any],
                            result_str: str, success: bool):
        """透传 + 更新失败/重试计数"""
        self.wm.update_after_action(tool_name, args, result_str, success)
        if success:
            # 成功 → 重置计数
            self._failure_count = 0
            self._retry_count = 0
        else:
            self._failure_count += 1
            self._retry_count += 1

    def set_expected_transition(self, after_state: str,
                                should_appear: List[str],
                                timeout: float = 3.0):
        """透传 WorkingMemory.set_expected_transition"""
        self.wm.set_expected_transition(after_state, should_appear, timeout)

    def set_prediction(self, predicted_state: str, timeout: float = 3.0):
        """透传 WorkingMemory.set_prediction"""
        self.wm.set_prediction(predicted_state, timeout)

    def update_state_after_observation(self, observed_state: str,
                                       confidence: float):
        """透传 WorkingMemory.update_state_after_observation"""
        self.wm.update_state_after_observation(observed_state, confidence)

    def clear_prediction(self):
        """透传 WorkingMemory.clear_prediction"""
        self.wm.clear_prediction()

    def get(self, key: str, default: Any = None) -> Any:
        """透传 WorkingMemory.get"""
        return self.wm.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """透传 WorkingMemory.to_dict"""
        return self.wm.to_dict()

    def build_instruction(self) -> str:
        """透传 WorkingMemory.build_instruction"""
        return self.wm.build_instruction()

    def get_short_summary(self) -> str:
        """透传 WorkingMemory.get_short_summary"""
        return self.wm.get_short_summary()

    def has_pending_prediction(self) -> bool:
        """透传 WorkingMemory.has_pending_prediction"""
        return self.wm.has_pending_prediction()

    def prediction_expired(self, now: float = None) -> bool:
        """透传 WorkingMemory.prediction_expired"""
        return self.wm.prediction_expired(now)

    def get_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """透传 WorkingMemory.get_history"""
        return self.wm.get_history(n)

    def snapshot(self) -> Dict[str, Any]:
        """透传 WorkingMemory.snapshot"""
        return self.wm.snapshot()

    def clear(self):
        """透传 WorkingMemory.clear + 清理附加状态"""
        self.wm.clear()
        self.world_state = {
            self.ENVIRONMENT: {},
            self.UI: {},
            self.TASK: "unknown",
        }
        self._failure_count = 0
        self._retry_count = 0
        self._observation_history = []
        self._snapshots = []

    # ============================================================
    # 分层世界状态管理（新增能力）
    # ============================================================

    def update_environment(self, **kwargs):
        """更新环境层状态（current_window / desktop / resolution 等）"""
        self.world_state[self.ENVIRONMENT].update(kwargs)

    def update_ui(self, **kwargs):
        """更新 UI 层状态（has_popup / popup_type / focus 等）"""
        self.world_state[self.UI].update(kwargs)

    def update_task_state(self, stage: str):
        """更新任务状态（current_stage + task_state）"""
        self.world_state[self.TASK] = stage
        # 同步到 WorkingMemory 的 current_state（保持旧字段一致）
        self.wm.update(current_state=stage)

    def set_stage(self, stage: str):
        """设置当前阶段（同步 TaskPhaseMachine）"""
        self.update_task_state(stage)

    def get_world_state(self) -> Dict[str, Any]:
        """获取完整世界状态"""
        return {
            "current_stage": self.world_state.get(self.TASK, ""),
            "environment": self.world_state.get(self.ENVIRONMENT, {}),
            "ui_state": self.world_state.get(self.UI, {}),
            "last_action": self.wm.get("last_action"),
            "expected_transition": self.wm.get("expected_transition"),
            "confidence": self.wm.get("confidence", 0.0),
        }

    # ============================================================
    # 失败计数 / 重试
    # ============================================================

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int):
        self._max_retries = value

    def record_success(self):
        """记录一次成功（重置失败/重试计数）"""
        self._failure_count = 0
        self._retry_count = 0

    def record_failure(self):
        """记录一次失败（递增失败/重试计数）"""
        self._failure_count += 1
        self._retry_count += 1

    def reset_retry(self):
        """重置当前动作的重试计数（动作已更换时调用）"""
        self._retry_count = 0

    def should_retry(self) -> bool:
        """是否还应重试当前动作"""
        return self._retry_count < self._max_retries

    def can_recover(self) -> bool:
        """是否仍可自主恢复（未超过失败上限）"""
        return self._failure_count < self._max_retries * 2

    def get_retry_hint(self) -> str:
        """构建重试提示（注入 LLM）"""
        return (
            f"【状态】已连续失败 {self._failure_count} 次，"
            f"当前动作已重试 {self._retry_count}/{self._max_retries} 次。"
            f"请更换策略，不要重复相同操作。"
        )

    # ============================================================
    # 观察历史
    # ============================================================

    def record_observation(self, observation: Dict[str, Any]):
        """记录一次观察结果

        Args:
            observation: 观察到的内容（OCR文字 / 视觉扫描 / 工具结果摘要）
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content": observation,
        }
        self._observation_history.append(entry)
        if len(self._observation_history) > self._max_observation_history:
            self._observation_history = self._observation_history[
                -self._max_observation_history:
            ]

    def get_recent_observations(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近 n 条观察"""
        return self._observation_history[-n:]

    # ============================================================
    # 持久化 / 恢复
    # ============================================================

    def save_snapshot(self, reason: str = "") -> str:
        """保存当前状态快照（用于调试/异常恢复）

        Returns:
            快照 ID
        """
        snap_id = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot = {
            "id": snap_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "working_memory": self.wm.to_dict(),
            "world_state": dict(self.world_state),
            "failure_count": self._failure_count,
            "retry_count": self._retry_count,
            "observation_history": list(self._observation_history),
            "action_history": self.wm.get_history(8),
        }
        self._snapshots.append(snapshot)
        return snap_id

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """获取所有快照"""
        return self._snapshots

    def to_json(self) -> str:
        """完整序列化为 JSON（供日志/调试）"""
        return json.dumps(
            {
                "working_memory": self.wm.to_dict(),
                "world_state": self.world_state,
                "failure_count": self._failure_count,
                "retry_count": self._retry_count,
                "observation_history": self._observation_history,
                "action_history": self.wm.get_history(8),
            },
            ensure_ascii=False,
            indent=2,
        )

    # ============================================================
    # 综合指令构建（扩展版）
    # ============================================================

    def build_extended_instruction(self) -> str:
        """构建扩展状态指令（注入 LLM）

        在 WorkingMemory 基础上增加：
          - 世界状态摘要
          - 失败/重试计数
        """
        parts = []

        # 基础工作记忆指令（透传）
        base = self.wm.build_instruction()
        if base:
            parts.append(base)

        # 世界状态附加段
        ws_parts = []
        env = self.world_state.get(self.ENVIRONMENT, {})
        ui = self.world_state.get(self.UI, {})
        stage = self.world_state.get(self.TASK, "")

        if env:
            ws_parts.append(
                f"- 环境: {', '.join(f'{k}={v}' for k, v in env.items() if v)}"
            )
        if ui:
            ws_parts.append(
                f"- UI: {', '.join(f'{k}={v}' for k, v in ui.items() if v)}"
            )
        if stage and stage not in ("unknown", ""):
            ws_parts.append(f"- 任务阶段: {stage}")

        if ws_parts:
            parts.append("## [世界状态] 当前环境理解\n" + "\n".join(ws_parts))

        # 失败计数
        if self._failure_count > 0:
            parts.append(
                f"## [状态告警] 连续失败 {self._failure_count} 次，"
                f"当前动作重试 {self._retry_count}/{self._max_retries}"
            )

        return "\n\n".join(parts)