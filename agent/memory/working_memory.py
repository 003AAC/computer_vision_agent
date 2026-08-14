"""
Working Memory - 短期工作记忆
=============================
模拟人类执行任务时的短期记忆。在每次工具调用前后更新，
让 Agent 知道自己"刚做了什么"和"下一步应该发生什么"。

结构：
{
  "task": 任务描述,
  "current_application": 当前应用,
  "current_state": 当前界面状态,
  "last_action": {action, target, coordinate},
  "expected_transition": {after, should_appear[]},
  "prediction": {timeout, resolution},
  "confidence": 置信度,
  "history": 最近动作历史（回溯用）
}

设计原则：
  - 只保留最近 1-2 步关键信息注入提示词，避免上下文爆炸
  - 每次工具调用前后都由 AgentLoop 统一更新
  - 任务结束后 clear()，不影响跨任务记忆
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class WorkingMemory:
    """短期工作记忆（单任务生命周期）"""

    # 关键动作类型（非感知动作，会影响世界状态）
    ACTION_TOOLS = {
        "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
        "run_powershell",
    }
    # 感知/定位类工具（不改变世界，只获取信息）
    PERCEPTION_TOOLS = {
        "visual_scan", "visual_scan_region", "visual_scan_grid",
        "visual_locate", "visual_locate_region",
        "visual_read_text", "visual_read_region", "visual_find_text",
    }

    def __init__(self):
        self._data: Dict[str, Any] = self._fresh()
        # 历史记录（最近 N 步，供回溯）
        self._history: List[Dict[str, Any]] = []
        self._max_history = 8

    def _fresh(self) -> Dict[str, Any]:
        return {
            "task": "",
            "current_application": "",
            "current_state": "",
            "last_action": None,           # {action, target, coordinate, result}
            "expected_transition": None,   # {after, should_appear[]}
            "prediction": None,            # {predicted_state, timeout, timestamp}
            "confidence": 0.0,
            "note": "",
        }

    # ============================================================
    # 初始化
    # ============================================================

    def init_task(self, task: str, application: str = "",
                  initial_state: str = "", confidence: float = 0.5):
        """任务开始时初始化工作记忆"""
        self._data["task"] = task
        self._data["current_application"] = application
        self._data["current_state"] = initial_state
        self._data["confidence"] = confidence
        self._history = []

    # ============================================================
    # 更新
    # ============================================================

    def update(self, **kwargs):
        """通用字段更新"""
        for key, value in kwargs.items():
            if value is not None and key in self._data:
                self._data[key] = value

    def record_action(self, tool_name: str, args: Dict[str, Any],
                      result_str: str = "") -> Dict[str, Any]:
        """记录一次工具调用（动作前调用，用于预测更新）

        Args:
            tool_name: 工具名
            args: 工具参数
            result_str: 工具返回结果（如已有）

        Returns:
            更新后的 last_action
        """
        last_action = {
            "action": tool_name,
            "target": self._extract_target(tool_name, args),
            "coordinate": self._extract_coordinate(tool_name, args),
            "timestamp": time.time(),
        }
        if result_str:
            last_action["result"] = result_str[:200]

        self._data["last_action"] = last_action
        self._push_history(last_action)
        return last_action

    def update_after_action(self, tool_name: str, args: Dict[str, Any],
                            result_str: str, success: bool):
        """动作执行后更新（更新 last_action.result + 记忆置信度）

        Args:
            tool_name: 工具名
            args: 工具参数
            result_str: 工具返回结果
            success: 是否执行成功
        """
        if self._data["last_action"] is None:
            self.record_action(tool_name, args, result_str)
        else:
            self._data["last_action"]["result"] = result_str[:200]
            self._data["last_action"]["success"] = success
            self._data["last_action"]["timestamp"] = time.time()

        # 动作成功 → 置信度小幅提升；失败 → 降低
        if success:
            self._data["confidence"] = min(1.0, self._data["confidence"] + 0.05)
        else:
            self._data["confidence"] = max(0.0, self._data["confidence"] - 0.1)

    def set_expected_transition(self, after_state: str,
                                should_appear: List[str],
                                timeout: float = 3.0):
        """设定预期转换（动作后应出现的状态/元素）

        Args:
            after_state: 预期进入的下一阶段
            should_appear: 应该出现的元素/文字列表（用于观察验证）
            timeout: 等待超时秒数
        """
        self._data["expected_transition"] = {
            "after": after_state,
            "should_appear": should_appear,
            "timeout": timeout,
            "timestamp": time.time(),
        }

    def set_prediction(self, predicted_state: str, timeout: float = 3.0):
        """设置动作后预测（供对比验证）"""
        self._data["prediction"] = {
            "predicted_state": predicted_state,
            "timeout": timeout,
            "timestamp": time.time(),
        }

    def update_state_after_observation(self, observed_state: str,
                                       confidence: float):
        """观察后更新当前状态"""
        self._data["current_state"] = observed_state
        self._data["confidence"] = confidence
        # 观察确认了新状态，清理过期预测
        if self._data["prediction"] is not None:
            pred = self._data["prediction"]
            if pred.get("predicted_state") == observed_state:
                self._data["prediction"]["confirmed"] = True
                self._data["confidence"] = max(
                    self._data["confidence"], 0.6
                )

    def clear_prediction(self):
        """清除当前预测"""
        self._data["prediction"] = None

    # ============================================================
    # 读取 / 指令构建
    # ============================================================

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def build_instruction(self) -> str:
        """构建当前记忆摘要（注入系统提示词）

        只注入最近 1-2 步关键信息，避免上下文爆炸。
        """
        lines = ["## [工作记忆] 当前状态"]
        d = self._data

        if d["task"]:
            lines.append(f"- 任务: {d['task'][:100]}")
        if d["current_application"]:
            lines.append(f"- 当前应用: {d['current_application']}")
        if d["current_state"]:
            lines.append(f"- 当前状态: {d['current_state']}")
        if d["last_action"]:
            la = d["last_action"]
            coord = f" 坐标{la['coordinate']}" if la.get("coordinate") else ""
            target = f" 目标[{la['target']}]" if la.get("target") else ""
            lines.append(
                f"- 刚执行: {la['action']}{target}{coord}"
            )
        if d["expected_transition"]:
            et = d["expected_transition"]
            lines.append(
                f"- 预期下一步: → {et['after']}"
                f" (应出现: {', '.join(et['should_appear'])})"
            )
        if d["prediction"]:
            pred = d["prediction"]
            status = "[✓已确认]" if pred.get("confirmed") else "[等待验证]"
            lines.append(
                f"- 动作预测: {status} {pred['predicted_state']}"
            )
        if d["note"]:
            lines.append(f"- 备注: {d['note']}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def get_short_summary(self) -> str:
        """极简摘要（用于日志/调试）"""
        d = self._data
        parts = []
        if d["current_state"]:
            parts.append(f"状态={d['current_state']}")
        if d["last_action"]:
            la = d["last_action"]
            parts.append(f"最近={la['action']}")
        if d["expected_transition"]:
            parts.append(f"预期→{d['expected_transition']['after']}")
        parts.append(f"置信度={d['confidence']:.2f}")
        return "; ".join(parts)

    # ============================================================
    # 工具
    # ============================================================

    def _extract_target(self, tool_name: str, args: Dict[str, Any]) -> str:
        """从工具参数提取目标描述"""
        if tool_name in ("visual_locate", "visual_locate_region"):
            return args.get("target_description", "")
        if tool_name == "visual_find_text":
            return args.get("target_text", "")
        if tool_name == "click_at":
            return ""
        if tool_name == "type_text":
            return f"输入: {str(args.get('text', ''))[:30]}"
        if tool_name == "press_key":
            return args.get("key", "")
        if tool_name == "hotkey":
            keys = args.get("keys", [])
            return "+".join(keys)
        if tool_name == "run_powershell":
            return f"PS: {str(args.get('command', ''))[:50]}"
        return ""

    def _extract_coordinate(self, tool_name: str, args: Dict[str, Any]) -> List[int]:
        """从工具参数提取坐标"""
        if tool_name == "click_at":
            x, y = args.get("x"), args.get("y")
            if x is not None and y is not None:
                return [int(x), int(y)]
        if tool_name == "drag_mouse":
            sx, sy = args.get("start_x"), args.get("start_y")
            ex, ey = args.get("end_x"), args.get("end_y")
            if all(v is not None for v in (sx, sy, ex, ey)):
                return [int(sx), int(sy), int(ex), int(ey)]
        return []

    def _push_history(self, entry: Dict[str, Any]):
        """推入历史（限制数量）"""
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近 n 条动作历史"""
        return self._history[-n:]

    def has_pending_prediction(self) -> bool:
        """是否有待验证的预测"""
        pred = self._data.get("prediction")
        return pred is not None and not pred.get("confirmed", False)

    def prediction_expired(self, now: float = None) -> bool:
        """预测是否已超时"""
        pred = self._data.get("prediction")
        if pred is None or pred.get("confirmed"):
            return False
        now = now or time.time()
        return now - pred.get("timestamp", now) > pred.get("timeout", 3.0)

    # ============================================================
    # 清理
    # ============================================================

    def clear(self):
        """任务结束清除"""
        self._data = self._fresh()
        self._history = []

    # ============================================================
    # 序列化（供日志/快照）
    # ============================================================

    def to_json(self) -> str:
        return json.dumps(self._data, ensure_ascii=False, indent=2)

    def snapshot(self) -> Dict[str, Any]:
        """生成快照（供异常处理/日志）"""
        return {
            "timestamp": datetime.now().isoformat(),
            "data": dict(self._data),
            "history": list(self._history),
        }