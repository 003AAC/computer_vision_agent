"""
Experience Extractor - 经验提取器
==================================
任务结束后从执行轨迹生成经验卡：
- 成功工作流（可迁移步骤 + 验证方法）
- 失败恢复（problem/cause/solution）
不保存绝对坐标、固定布局。
"""
from typing import Dict, Any, List


# 工具行为归类（用于生成可迁移步骤描述）
_OBSERVE_TOOLS = {"visual_scan", "visual_scan_region", "visual_scan_grid",
                  "visual_read_text", "visual_read_region"}
_FIND_TOOLS = {"visual_locate", "visual_locate_region", "visual_find_text"}
_ACT_TOOLS = {"click_at", "drag_mouse", "type_text", "press_key", "hotkey"}
_POWER_TOOLS = {"run_powershell"}


def _tool_to_action(tool_name: str, args: Dict[str, Any]) -> str:
    """将工具调用转为可迁移的动作描述（丢弃坐标）"""
    if tool_name == "click_at":
        return "点击已找到的目标"
    if tool_name == "type_text":
        return f"输入文字: {str(args.get('text', ''))[:30]}"
    if tool_name == "press_key":
        return f"按下按键: {args.get('key', '')}"
    if tool_name == "hotkey":
        return f"组合键: {'+'.join(args.get('keys', []))}"
    if tool_name == "visual_find_text":
        return f"OCR查找文字: {args.get('target_text', '')}"
    if tool_name == "visual_locate":
        return f"视觉定位: {args.get('target_description', '')}"
    if tool_name == "run_powershell":
        cmd = str(args.get('command', ''))[:40]
        return f"PowerShell: {cmd}"
    if tool_name == "wait":
        return f"等待 {args.get('seconds', 1)} 秒"
    return f"执行 {tool_name}"


class ExperienceExtractor:
    """经验提取器：从执行轨迹生成经验卡"""

    @staticmethod
    def extract_steps(trace: List[Dict[str, Any]]) -> List[str]:
        """从成功轨迹提取可迁移步骤摘要"""
        steps = []
        for item in trace:
            if item.get("is_failure"):
                continue
            tool_name = item.get("tool", "")
            if tool_name in _ACT_TOOLS and not item.get("verified", False):
                continue
            action = _tool_to_action(tool_name, item.get("args", {}))
            if action not in steps:
                steps.append(action)
        return steps[:10]

    @staticmethod
    def extract_verification(trace: List[Dict[str, Any]]) -> List[str]:
        """提取验证方法（观察类工具 + PowerShell 查询）"""
        verifications = []
        for item in trace:
            tool = item.get("tool", "")
            if tool in _OBSERVE_TOOLS or tool == "run_powershell":
                desc = _tool_to_action(tool, item.get("args", {}))
                if desc not in verifications:
                    verifications.append(desc)
        return verifications[:5]

    @staticmethod
    def extract_failure_cases(error_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """从异常历史提取失败恢复案例"""
        cases = []
        for err in error_history:
            case = {
                "problem": f"{err.get('tool_name', '')} 执行失败",
                "cause": err.get("raw_error", "")[:80],
                "solution": err.get("fix_instruction", "")[:120],
            }
            if case not in cases:
                cases.append(case)
        return cases[:5]

    @staticmethod
    def build_experience_card(trace: List[Dict[str, Any]],
                              error_history: List[Dict[str, Any]],
                              task: str) -> Dict[str, Any]:
        """构建经验卡（不保存坐标）"""
        return {
            "task": task[:50],
            "strategy": ExperienceExtractor.extract_steps(trace),
            "verification": ExperienceExtractor.extract_verification(trace),
            "failure_cases": ExperienceExtractor.extract_failure_cases(error_history),
        }