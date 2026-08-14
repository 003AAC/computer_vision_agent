"""
Runtime Memory - 运行时记忆（当前任务状态）
============================================
只保存当前任务：当前窗口、状态、下一步目标。
任务结束清除。
"""
from typing import Dict, Any, Optional


class RuntimeMemory:
    """运行时记忆：仅当前任务有效"""

    def __init__(self):
        self._data: Dict[str, Any] = {
            "current_window": "",
            "state": "",
            "next_goal": "",
        }

    def update(self, **kwargs):
        """更新运行时状态"""
        for key, value in kwargs.items():
            if value is not None:
                self._data[key] = value

    def set(self, key: str, value: Any):
        """设置单个字段"""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取字段值"""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（供提示词注入）"""
        return dict(self._data)

    def build_instruction(self) -> str:
        """构建当前状态摘要（注入提示词）"""
        lines = ["## [当前状态] 运行时记忆"]
        if self._data["current_window"]:
            lines.append(f"- 当前窗口: {self._data['current_window']}")
        if self._data["state"]:
            lines.append(f"- 当前界面状态: {self._data['state']}")
        if self._data["next_goal"]:
            lines.append(f"- 下一步目标: {self._data['next_goal']}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def clear(self):
        """任务结束清除"""
        self._data = {
            "current_window": "",
            "state": "",
            "next_goal": "",
        }

    def sample_state(self, window_title: str = "", scan_summary: str = "",
                     next_goal: str = ""):
        """从执行过程采样状态（供 extractor 提取经验卡）"""
        self.update(
            current_window=window_title,
            state=scan_summary,
            next_goal=next_goal,
        )
        return self.to_dict()