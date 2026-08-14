"""
State Extractor - 工具结果 → 结构化状态
=======================================
将工具返回的原始字符串，解析为对 5 个 State 类的 **evidence 更新**。

核心设计：
  - 不存储原始数据，只提取"理解了什么"
  - 按工具类型分发到对应解析器
  - 返回 EvidenceUpdate 列表（domain + field + source + value + note）
  - 无法解析的结果 → 返回空列表（不崩溃）

EvidenceUpdate 结构：
  {
    "domain": "environment" | "ui" | "applications" | "files" | "task",
    "field": "字段名"（如 active_window / installed / exists）,
    "value": 状态值,
    "source": "system_api" | "code_check" | "powershell" | "ocr" | "vision",
    "note": "描述"
  }
"""
import json
import re
from typing import Dict, Any, List

from world_state.state import (
    EnvironmentState, UIState, ApplicationState, FileState, TaskState,
)


class EvidenceUpdate:
    """一条状态证据更新（交给 manager 应用到对应 State）"""

    def __init__(self, domain: str, field: str, value: Any,
                 source: str, note: str = ""):
        self.domain = domain
        self.field = field
        self.value = value
        self.source = source
        self.note = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "field": self.field,
            "value": self.value,
            "source": self.source,
            "note": self.note,
        }


class StateExtractor:
    """状态提取器：工具结果 → EvidenceUpdate 列表"""

    # ============================================================
    # 主入口
    # ============================================================

    def extract(self, tool_name: str, result_str: str,
                tool_args: Dict[str, Any] = None) -> List[EvidenceUpdate]:
        """从工具结果提取证据更新

        Args:
            tool_name: 工具名
            result_str: 工具返回的原始字符串
            tool_args: 工具参数（可能含上下文线索）
        """
        tool_args = tool_args or {}
        try:
            if tool_name == "run_powershell":
                return self._extract_powershell(result_str)
            if tool_name in ("visual_scan", "visual_scan_region",
                             "visual_scan_grid"):
                return self._extract_scan(result_str)
            if tool_name in ("visual_read_text", "visual_read_region",
                             "visual_find_text"):
                return self._extract_vision_text(result_str, tool_name)
            if tool_name == "click_at":
                return self._extract_click(tool_args)
        except Exception as e:
            print(f"  [WorldState] 提取 {tool_name} 结果失败: {e}")
        return []

    # ============================================================
    # PowerShell 解析
    # ============================================================

    def _extract_powershell(self, result_str: str) -> List[EvidenceUpdate]:
        """从 PowerShell 输出提取状态

        识别模式：
          - 路径（C:\\...）→ 文件/目录存在
          - "命令执行成功" / "✅" → 命令已确认执行
          - 版本号 → 应用版本
          - 目录名 → 应用/项目存在性
        """
        updates: List[EvidenceUpdate] = []
        text = result_str

        # 命令成功 → 记为 code_check 确认
        if "命令执行成功" in text or "✅" in text:
            updates.append(EvidenceUpdate(
                "environment", "last_powershell_success", True, "code_check",
                "PowerShell 命令执行成功"
            ))

        # 提取 Windows 路径 → FileState
        paths = re.findall(r'[A-Za-z]:\\[^\s"\']+', text)
        seen_paths = set()
        for path in paths[:10]:
            norm = path.rstrip('\\/')
            if norm in seen_paths:
                continue
            seen_paths.add(norm)
            updates.append(EvidenceUpdate(
                "files", "path_detected", norm, "powershell",
                f"PowerShell 输出中出现路径"
            ))

        # 提取版本号（如 2022 / 17.8）→ ApplicationState
        versions = re.findall(r'(?:2022|2024|17\.\d+|1[0-9]\.\d+(?:\.\d+)?)', text)
        if versions:
            updates.append(EvidenceUpdate(
                "applications", "version_detected", versions[0], "powershell",
                f"检测到版本号 {versions[0]}"
            ))

        return updates

    # ============================================================
    # 视觉扫描解析
    # ============================================================

    def _extract_scan(self, result_str: str) -> List[EvidenceUpdate]:
        """从 visual_scan 结果提取状态

        返回 JSON: {scene, elements: [{label, cx, cy, confidence}]}
        """
        updates: List[EvidenceUpdate] = []
        try:
            data = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            return updates

        if not isinstance(data, dict):
            return updates

        # 场景描述 → UI 状态（vision 来源，低置信度）
        scene = data.get("scene")
        if scene:
            updates.append(EvidenceUpdate(
                "ui", "scene", scene, "vision", "视觉场景理解"
            ))

        # 元素标签 → UI 可见元素（vision 来源）
        elements = data.get("elements", [])
        labels = [e.get("label") for e in elements
                  if e.get("label") and e.get("confidence", 0) >= 0.5]
        if labels:
            updates.append(EvidenceUpdate(
                "ui", "visible_elements", labels[:10], "vision",
                "视觉检测到的元素"
            ))

        # 文本场景提示 → 建议 OCR
        note = data.get("note", "")
        if "文本" in note:
            updates.append(EvidenceUpdate(
                "ui", "text_dominated", True, "vision",
                "屏幕以文本为主，建议使用 OCR"
            ))

        return updates

    # ============================================================
    # OCR 文本解析
    # ============================================================

    def _extract_vision_text(self, result_str: str,
                             tool_name: str) -> List[EvidenceUpdate]:
        """从 OCR 结果提取状态

        返回 JSON: {texts: [{text, cx, cy, confidence}]}
        或文本: "✅ OCR 找到文字[开始] -> 坐标(...)"
        """
        updates: List[EvidenceUpdate] = []
        texts = []

        # 尝试 JSON 解析
        try:
            data = json.loads(result_str)
            if isinstance(data, dict):
                texts = [t.get("text", "") for t in data.get("texts", [])
                         if t.get("text")]
        except (json.JSONDecodeError, TypeError):
            pass

        # 非 JSON（如 visual_find_text 的文本返回）
        if not texts:
            m = re.search(r'找到文字\[(.+?)\]', result_str)
            if m:
                texts.append(m.group(1))
        if not texts:
            m = re.search(r'未找到文字\[(.+?)\]', result_str)
            if m:
                updates.append(EvidenceUpdate(
                    "ui", "text_not_found", m.group(1), "ocr",
                    f"OCR 未找到: {m.group(1)}"
                ))

        if texts:
            updates.append(EvidenceUpdate(
                "ui", "visible_texts", texts[:10], "ocr",
                f"OCR 识别到 {len(texts)} 个文字块"
            ))

        # 尝试从文字推断活动窗口标题（OCR 置信来源）
        # （简化版：首个文字可能为窗口标题；更精确需待窗口 API）
        return updates

    # ============================================================
    # 点击解析
    # ============================================================

    def _extract_click(self, tool_args: Dict[str, Any]) -> List[EvidenceUpdate]:
        """点击动作 → UI 交互记录（不污染世界状态，仅记录 action）"""
        updates = []
        x, y = tool_args.get("x"), tool_args.get("y")
        if x is not None and y is not None:
            updates.append(EvidenceUpdate(
                "ui", "last_click_at", [int(x), int(y)], "code_check",
                "点击位置记录"
            ))
        return updates