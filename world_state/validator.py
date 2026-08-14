"""
State Validator - 矛盾检测 + 状态一致性
=======================================
检测世界状态中的矛盾，防止一次视觉误判推翻已确认状态。

设计：
  - Belief 本身已内置 evidence 融合逻辑（高置信证据胜出）
  - Validator 负责**跨状态一致性检查**：
    - 矛盾检测：同一状态收到冲突证据时的处置
    - 一致性规则：应用打开但未安装等逻辑矛盾
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from world_state.base import Belief
from world_state.manager import WorldStateManager


class ConflictRecord:
    """一条冲突记录"""

    def __init__(self, path: str, old_value: Any, new_value: Any,
                 old_confidence: float, new_confidence: float,
                 resolution: str):
        self.path = path
        self.old_value = old_value
        self.new_value = new_value
        self.old_confidence = old_confidence
        self.new_confidence = new_confidence
        self.resolution = resolution
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "resolution": self.resolution,
            "timestamp": self.timestamp,
        }


class StateValidator:
    """世界状态一致性验证器"""

    def __init__(self):
        self._conflicts: List[ConflictRecord] = []

    def get_conflicts(self) -> List[Dict[str, Any]]:
        """获取冲突记录"""
        return [c.to_dict() for c in self._conflicts]

    def reset(self):
        self._conflicts = []

    # ============================================================
    # 一致性规则
    # ============================================================

    @staticmethod
    def check_consistency(manager: WorldStateManager) -> List[str]:
        """检查跨状态逻辑一致性

        规则：
          1. 应用"打开" 但无"安装"证据 → 警告
          2. 应用"状态=crashed" 但 "opened=true" → 矛盾
          3. 弹窗存在但无标题 → 警告
        """
        issues: List[str] = []

        # 规则1+2：应用
        for name, app in manager.applications.items():
            opened = app.opened.belief
            installed = app.installed.belief
            status = app.status.belief

            if opened.value is True and installed.value is False:
                issues.append(
                    f"规则冲突: 应用[{name}] 打开=true 但 安装=false"
                )
            if status.value == "crashed" and opened.value is True:
                issues.append(
                    f"规则冲突: 应用[{name}] 崩溃(crashed) 但 打开=true"
                )

        # 规则3：弹窗
        ui = manager.ui
        popup_present = ui.popup_present.belief
        popup_title = ui.popup_title.belief
        if popup_present.value is True and not popup_title.value:
            issues.append("规则冲突: 弹窗存在但无标题（可能为误判）")

        return issues

    # ============================================================
    # 矛盾检测（belief 融合辅助）
    # ============================================================

    def record_conflict(self, path: str, belief: Belief,
                        conflicting_value: Any,
                        conflicting_source: str) -> ConflictRecord:
        """记录一次证据冲突（供调试/lesson extraction）"""
        conflict = ConflictRecord(
            path=path,
            old_value=belief.value,
            new_value=conflicting_value,
            old_confidence=belief.confidence,
            new_confidence=0.0,
            resolution="已按置信度融合规则处理",
        )
        self._conflicts.append(conflict)
        return conflict