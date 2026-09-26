"""
Tool Manager - 工具调用策略管理器
================================
解决"LLM 直接选工具导致选错"的问题。

职责：
  1. analyze_task(task) → 推荐任务分类 + 建议工具集
  2. execute_with_strategy(tool, args, task_type) → 执行 + 记录统计
  3. 记录每个工具在不同任务类型下的成功率，供后续推荐参考
  4. 提示并减少"用 DINO 做 OCR / 用 CLIP 做定位"等错配

设计原则：
  - 保持现有 ALL_TOOLS / TOOL_REGISTRY 不变（LLM 仍可直接调用）
  - ToolManager 是**建议与统计层**，不替代执行结果验证，也不强制拦截调用
  - 统计持久化到 agent/memories/tool_stats.json
"""
import json
import logging
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional

from vision.router import classify_task, VisionTaskType

logger = logging.getLogger(__name__)


# ============================================================
# 任务类型 → 推荐工具映射
# ============================================================

# 各类任务的推荐工具集（优先级从高到低）
TASK_TOOL_RECOMMENDATIONS = {
    "ocr": ["visual_find_text", "visual_read_region", "visual_read_text"],
    "detect": ["visual_locate", "visual_locate_region", "visual_scan_region"],
    "classify": ["visual_scan", "visual_scan_region"],
    "scene": ["visual_scan", "visual_scan_grid", "visual_scan_region", "visual_read_region"],
    "action": ["visual_find_text", "click_at", "type_text", "press_key", "hotkey", "wait"],
    "powershell": ["run_powershell", "visual_scan"],
    "unknown": [],
}

# 视觉任务类型
_VISION_TYPES = {
    VisionTaskType.OCR, VisionTaskType.DETECT,
    VisionTaskType.CLASSIFY, VisionTaskType.SCENE,
}


# 动作动词（出现即视为动作任务，优先于视觉按钮文字）
_ACTION_VERBS = [
    "点击", "双击", "输入", "键入", "按键", "按下", "打开", "关闭",
    "运行", "启动", "拖拽", "选择", "切换到",
    "click", "double click", "type", "input", "press", "open",
    "close", "run", "launch", "drag", "select", "switch",
]

# 动作任务关键词（描述性任务）
_ACTION_HINTS = ["帮我", "请", "然后", "并", "再", "之后"]


def infer_task_type(task: str, tool_name: str = "") -> str:
    """推断任务类型

    Args:
        task: 用户任务描述（或目标描述）
        tool_name: 当前工具名（若已知）

    Returns:
        任务类型：ocr / detect / classify / scene / action / powershell / unknown
    """
    t = (task or "").strip().lower()

    # PowerShell 类（工具名或明确命令关键词）
    if tool_name == "run_powershell" or any(
        kw in t for kw in ["命令", "执行", "查询", "创建", "删除", "复制", "移动", "列出"]
    ):
        return "powershell"

    # 动作类：动作动词优先（"点击确定按钮" → action，而非 OCR）
    if any(kw in t for kw in _ACTION_VERBS):
        return "action"

    # 视觉类任务（走 Vision Router）
    vt = classify_task(task)
    if vt in _VISION_TYPES:
        return vt

    # 有动作提示词但无明确动词 → 仍视为动作
    if any(kw in t for kw in _ACTION_HINTS):
        return "action"

    return "unknown"


class ToolStats:
    """工具统计（按任务类型）"""

    def __init__(self):
        # {tool_name: {task_type: {"success": n, "total": m}}}
        self._data: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0, "total": 0})
        )

    def record(self, tool_name: str, task_type: str, success: bool):
        """记录一次工具执行结果"""
        entry = self._data[tool_name][task_type]
        entry["total"] += 1
        if success:
            entry["success"] += 1

    def get_success_rate(self, tool_name: str, task_type: str) -> float:
        """获取工具在某任务类型下的成功率"""
        entry = self._data.get(tool_name, {}).get(task_type, {})
        total = entry.get("total", 0)
        if total == 0:
            return 0.0
        return entry.get("success", 0) / total

    def to_dict(self) -> Dict[str, Any]:
        return {k: dict(v) for k, v in self._data.items()}

    def from_dict(self, data: Dict[str, Any]):
        for tool, tasks in (data or {}).items():
            for task_type, stats in tasks.items():
                self._data[tool][task_type] = stats


class ToolManager:
    """工具管理器"""

    def __init__(self, stats_path: str = None):
        if stats_path is None:
            stats_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memories", "tool_stats.json"
            )
        self.stats_path = stats_path
        self.stats = ToolStats()
        self._load_stats()

    # ============================================================
    # 任务分析
    # ============================================================

    def analyze_task(self, task: str) -> Dict[str, Any]:
        """分析任务，返回推荐工具集

        Args:
            task: 用户任务描述

        Returns:
            {
              "task_type": str,
              "recommended_tools": [...],
              "avoid_tools": [...],      # 应避免的工具
              "hint": str
            }
        """
        task_type = infer_task_type(task, "")

        if task_type in _VISION_TYPES:
            return self._analyze_vision_task(task, task_type)

        if task_type == "action":
            return {
                "task_type": task_type,
                "recommended_tools": TASK_TOOL_RECOMMENDATIONS["action"],
                "avoid_tools": [],
                "hint": "动作类任务：先用 OCR/视觉定位目标，再执行操作",
            }

        if task_type == "powershell":
            return {
                "task_type": task_type,
                "recommended_tools": TASK_TOOL_RECOMMENDATIONS["powershell"],
                "avoid_tools": [],
                "hint": "PowerShell 任务：直接执行命令，必要时用视觉验证结果",
            }

        return {
            "task_type": "unknown",
            "recommended_tools": [],
            "avoid_tools": [],
            "hint": "",
        }

    def _analyze_vision_task(self, target: str, task_type: str) -> Dict[str, Any]:
        """视觉任务的细化分析"""
        recs = TASK_TOOL_RECOMMENDATIONS.get(task_type, [])

        # 复杂语义任务：避免直接使用 DINO 定位
        avoid = []
        hint = ""
        if task_type == VisionTaskType.SCENE:
            avoid = ["visual_locate", "visual_locate_region"]
            hint = (
                "场景理解任务：先用 visual_scan() 全局扫描，"
                "再对候选区域用 visual_scan_region() 放大，最后可结合 OCR 读文字。"
                "避免直接 visual_locate 复杂描述。"
            )
        elif task_type == VisionTaskType.OCR:
            avoid = ["visual_locate", "visual_locate_region"]
            hint = "文字任务：优先用 visual_find_text() 直接定位文字，或用 visual_read_region() 区域 OCR。"
        elif task_type == VisionTaskType.DETECT:
            hint = "已知目标：用 visual_locate() 或 visual_locate_region() 精确定位。"

        return {
            "task_type": task_type,
            "recommended_tools": recs,
            "avoid_tools": avoid,
            "hint": hint,
        }

    # ============================================================
    # 工具执行 + 统计
    # ============================================================

    def execute_with_strategy(
        self,
        tool_fn,
        args: Dict[str, Any],
        task: str,
        tool_name: str = "",
    ) -> Dict[str, Any]:
        """执行工具并记录统计

        Args:
            tool_fn: 可调用对象（工具函数或 StructuredTool）
            args: 工具参数
            task: 当前任务描述
            tool_name: 工具名（自动从 fn.name 获取）

        Returns:
            {"success": bool, "result": str, "tool_name": str}
        """
        if not tool_name:
            tool_name = getattr(tool_fn, "name", "unknown")

        task_type = infer_task_type(task, tool_name)

        try:
            # 调用工具（StructuredTool 用 invoke，普通函数直接调用）
            if hasattr(tool_fn, "invoke"):
                result = tool_fn.invoke(args)
            else:
                result = tool_fn(**args)
            result_str = str(result)

            # 简单失败判断
            success = not _is_failure_string(result_str)

            self.stats.record(tool_name, task_type, success)
            self._save_stats()
            return {"success": success, "result": result_str, "tool_name": tool_name}
        except Exception as e:
            self.stats.record(tool_name, task_type, False)
            self._save_stats()
            return {
                "success": False,
                "result": f"工具执行异常: {e}",
                "tool_name": tool_name,
            }

    def get_tool_advice(self, target_tool: str, task: str) -> str:
        """获取使用某工具的建议（提示 LLM）

        Returns:
            建议文本（空串则无需建议）
        """
        task_type = infer_task_type(task, target_tool)

        # 视觉工具与任务类型匹配检查
        if task_type == VisionTaskType.SCENE and target_tool in ("visual_locate", "visual_locate_region"):
            return (
                f"⚠️ 当前任务 '{task_type}' 不适合 use {target_tool}（复杂语义描述）。"
                f"建议改用 visual_scan() + visual_scan_region() 逐步定位。"
            )
        if task_type == VisionTaskType.OCR and target_tool == "visual_locate":
            return "⚠️ 文字任务建议用 visual_find_text() / visual_read_region()，而不是视觉定位。"

        # 成功率统计建议
        rate = self.stats.get_success_rate(target_tool, task_type)
        if rate > 0 and rate < 0.3:
            return (
                f"⚠️ 工具 {target_tool} 在 {task_type} 任务中成功率仅 {rate:.0%}。"
                f"建议更换工具或改用其他策略。"
            )

        return ""

    # ============================================================
    # 持久化
    # ============================================================

    def _load_stats(self):
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    self.stats.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"工具统计加载失败: {e}")

    def _save_stats(self):
        try:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"工具统计保存失败: {e}")

    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {"path": self.stats_path, "data": self.stats.to_dict()}


# 简单失败字符串检测（避免循环依赖）
_FAILURE_STRINGS = [
    "失败", "错误", "异常", "未找到", "找不到", "not found", "error",
    "failed", "timeout", "无法访问", "拒绝",
]


def _is_failure_string(s: str) -> bool:
    s_l = s.lower()
    return any(f in s_l for f in _FAILURE_STRINGS)