"""
Agent 异常处理机制 - Exception Handler
====================================
统一处理工具执行过程中的错误：分类诊断 → 自愈修复 → 效果验证 → 经验沉淀 → 智能降级。

设计理念：
  - 复用已有的系统性错误修复模块（ErrorDiagnostician / SelfHealer / SuccessDetector）
  - 修复优先于降级：先自救，救不了再降级
  - 每次修复结果沉淀到知识库
  - 伪成功检测：AI 宣告完成前先验证

异常分类（增强）：
  1. 视觉失败（找不到目标）→ OCR / region scan / screenshot analysis
  2. 动作失败（click失败）→ retry / move cursor / foreground window
  3. 状态异常（弹窗/卡住）→ classify popup → dismiss
  4. 未知异常 → 保存快照（screenshot + OCR结果 + 当前memory）供LLM分析
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from agent.action_result import parse_action_result
from systematic_error_fix.error_diagnostician import ErrorDiagnostician
from systematic_error_fix.models import ErrorType
from self_healer import SelfHealer

logger = logging.getLogger(__name__)


class ToolException(Exception):
    """工具执行异常统一包装

    Attributes:
        tool_name: 出错的工具名
        raw_error: 原始错误信息
    """

    def __init__(self, tool_name: str, raw_error: str):
        super().__init__(f"[{tool_name}] {raw_error}")
        self.tool_name = tool_name
        self.raw_error = raw_error


class FailureDetector:
    """失败检测器（关键词匹配 + 成功信号反证）"""

    FAILURE_SIGNALS = [
        "失败", "错误", "异常", "超时", "未找到", "找不到",
        "not found", "error", "failed", "timeout",
        "拒绝访问", "权限", "不存在", "不是内部", "不是可识别",
        "安全策略拒绝", "工具不存在",
        "系统找不到", "无法访问", "定位失败", "点击失败", "输入失败",
        "按键失败", "组合键失败", "拖拽失败", "命令执行失败",
        "命令执行超时", "命令执行异常",
    ]

    SUCCESS_SIGNALS = [
        "已点击", "已输入", "已按下", "已拖拽", "命令执行成功",
        "✅", "成功", "已等待",
    ]

    def __init__(self):
        self._failure_signals = [s.lower() for s in self.FAILURE_SIGNALS]
        self._success_signals = [s.lower() for s in self.SUCCESS_SIGNALS]

    def detect(self, result_str: str) -> bool:
        """检测是否为失败

        Args:
            result_str: 工具返回结果字符串

        Returns:
            是否为失败
        """
        structured = parse_action_result(result_str)
        if structured:
            return (
                structured.get("execution_status") == "failed"
                or structured.get("verification_status") == "failed"
            )

        result_lower = result_str.lower()

        # Verification failure must take precedence over optimistic dispatch text.
        if any(s in result_lower for s in (
            "验证失败", "未验证", "verification failed", "not verified",
        )):
            return True

        # 成功信号优先（避免误判，如"打开失败处理成功"）
        if any(s in result_lower for s in self._success_signals):
            return False

        return any(s in result_lower for s in self._failure_signals)


class ExceptionCategory:
    """异常分类枚举（字符串常量）"""

    VISION_FAILURE = "vision_failure"       # 视觉失败：找不到目标
    ACTION_FAILURE = "action_failure"       # 动作失败：点击/拖拽/输入等失败
    STATE_ANOMALY = "state_anomaly"         # 状态异常：弹窗/卡住/界面不一致
    UNKNOWN = "unknown"                     # 未知异常


class ExceptionHandler:
    """Agent 异常处理器

    职责：
      1. 工具结果失败检测
      2. 错误类型诊断（ErrorDiagnostician）
      3. 异常分类（视觉/动作/状态/未知）
      4. 自愈修复动作生成（SelfHealer）
      5. 修复效果验证
      6. 经验沉淀
      7. 智能降级决策
      8. 伪成功检测（完成前验证）
      9. 未知异常快照保存
    """

    # 错误类型 → 降级建议映射
    DEGRADE_SUGGESTIONS = {
        ErrorType.PROGRAM_NOT_FOUND: "目标程序不存在，建议改用 PowerShell 搜索安装路径或提示用户安装",
        ErrorType.PATH_NOT_FOUND: "目标路径不存在，建议改用 PowerShell 验证路径或创建目录",
        ErrorType.ENCODING_ERROR: "出现编码问题，建议改用 PowerShell 命令设置 UTF-8 环境",
        ErrorType.PERMISSION_DENIED: "权限不足，建议改用 PowerShell 检查权限或更换用户目录",
        ErrorType.COMMAND_TIMEOUT: "命令超时，建议拆分命令或更换更简单的方式",
        ErrorType.NETWORK_ERROR: "网络异常，建议等待后重试或更换执行方式",
        ErrorType.UNKNOWN_ERROR: "未知错误，建议改用 PowerShell 方式尝试",
    }

    # 视觉工具（找不到目标 → 视觉失败）
    VISION_TOOLS = {
        "visual_scan", "visual_scan_region", "visual_scan_grid",
        "visual_locate", "visual_locate_region",
        "visual_read_text", "visual_read_region", "visual_find_text",
    }
    # 动作工具（失败 → 动作失败）
    ACTION_TOOLS = {
        "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
    }

    def __init__(self, knowledge_base=None, tools_registry=None):
        """初始化异常处理器

        Args:
            knowledge_base: 知识库实例（用于经验沉淀）
            tools_registry: 工具注册表 {tool_name: tool}
        """
        self.knowledge_base = knowledge_base
        self.tools_registry = tools_registry or {}

        # 错误诊断器（复用系统性错误修复模块）
        self.diagnostician = ErrorDiagnostician(knowledge_base)

        # 自愈闭环（复用 self_healer.py）
        self.healer = SelfHealer(knowledge_base) if knowledge_base else None

        # 失败检测器
        self.failure_detector = FailureDetector()

        # 内部状态
        self.error_history = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

        # 快照目录（未知异常时保存）
        self.snapshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "snapshots"
        )
        # 当前工作记忆（供快照引用）
        self.working_memory = None
        # 当前任务描述
        self.current_task = ""

    # ============================================================
    # 生命周期
    # ============================================================

    def reset(self, task_query: str = ""):
        """新任务开始时重置状态"""
        self.consecutive_failures = 0
        self.error_history = []
        self.current_task = task_query
        if self.healer:
            self.healer.reset()

    # ============================================================
    # 失败检测
    # ============================================================

    def is_success(self, result_str: str) -> bool:
        """检测工具结果是否为成功（非失败）"""
        return not self.failure_detector.detect(result_str)

    # ============================================================
    # 异常分类（增强）
    # ============================================================

    def classify_exception(self, result_str: str, tool_name: str) -> str:
        """对工具失败进行四大分类

        Args:
            result_str: 失败的工具结果
            tool_name: 工具名

        Returns:
            ExceptionCategory 之一
        """
        result_lower = result_str.lower()

        # 1. 视觉失败：视觉类工具未找到目标
        if tool_name in self.VISION_TOOLS:
            return ExceptionCategory.VISION_FAILURE

        # 2. 动作失败：动作类工具失败
        if tool_name in self.ACTION_TOOLS:
            return ExceptionCategory.ACTION_FAILURE

        # 3. PowerShell 失败：权限类问题 → 状态异常；其他 → 动作失败
        if tool_name == "run_powershell":
            if "拒绝" in result_str or "权限" in result_str:
                return ExceptionCategory.STATE_ANOMALY
            return ExceptionCategory.ACTION_FAILURE

        # 4. 默认
        return ExceptionCategory.UNKNOWN

    def process_failure(
        self,
        result_str: str,
        tool_name: str,
        task_query: str = "",
    ) -> Dict[str, Any]:
        """处理一次失败：分类 + 诊断 + 生成修复建议

        Args:
            result_str: 失败的工具结果
            tool_name: 出错的工具名
            task_query: 当前任务描述

        Returns:
            处理报告（分类、诊断、修复建议、是否降级等）
        """
        self.consecutive_failures += 1
        self.current_task = task_query or self.current_task

        # 步骤0：异常分类
        exception_category = self.classify_exception(result_str, tool_name)

        # 步骤1：错误诊断（复用 ErrorDiagnostician）
        diagnosis = None
        try:
            diagnosis = self.diagnostician.diagnose(
                result_str,
                context={"tool_name": tool_name, "task": task_query},
            )
        except Exception as e:
            logger.warning(f"错误诊断失败: {e}")

        # 步骤2：自愈诊断（复用 SelfHealer，若可用）
        heal_plan = None
        if self.healer:
            try:
                heal_plan = self.healer.diagnose(
                    result_str, tool_name, {"task": task_query}
                )
            except Exception as e:
                logger.warning(f"自愈诊断失败: {e}")

        # 汇总错误类型
        error_type = ErrorType.UNKNOWN_ERROR
        if diagnosis is not None:
            error_type = diagnosis.error_type
        elif heal_plan and heal_plan.get("error_type") != "unknown":
            error_type = self._map_heal_type_to_error_type(heal_plan.get("error_type", ""))

        # 生成分类修复指令
        fix_instruction = self._build_fix_instruction(
            diagnosis, heal_plan, tool_name, exception_category
        )

        # 智能降级判断
        should_degrade = self.consecutive_failures >= self.max_consecutive_failures
        degrade_suggestion = ""
        if should_degrade:
            degrade_suggestion = self.DEGRADE_SUGGESTIONS.get(
                error_type,
                self.DEGRADE_SUGGESTIONS[ErrorType.UNKNOWN_ERROR],
            )

        report = {
            "error_type": str(error_type),
            "error_type_enum": error_type,
            "exception_category": exception_category,
            "diagnosis": diagnosis.to_dict() if diagnosis else None,
            "heal_plan": heal_plan,
            "fix_instruction": fix_instruction,
            "consecutive_failures": self.consecutive_failures,
            "should_degrade": should_degrade,
            "degrade_suggestion": degrade_suggestion,
            "tool_name": tool_name,
            "raw_error": result_str[:300],
            "timestamp": datetime.now().isoformat(),
        }

        # 未知异常 → 保存快照供 LLM 分析
        if exception_category == ExceptionCategory.UNKNOWN:
            snapshot_path = self.save_snapshot(
                tool_name=tool_name,
                error=result_str[:300],
                extra={"fix_instruction": fix_instruction},
            )
            if snapshot_path:
                report["snapshot_path"] = snapshot_path

        self.error_history.append(report)
        return report

    def _map_heal_type_to_error_type(self, heal_type: str) -> ErrorType:
        """将 SelfHealer 的错误类型映射为 ErrorType"""
        mapping = {
            "program_not_found": ErrorType.PROGRAM_NOT_FOUND,
            "path_not_found": ErrorType.PATH_NOT_FOUND,
            "encoding_error": ErrorType.ENCODING_ERROR,
            "permission_denied": ErrorType.PERMISSION_DENIED,
            "command_timeout": ErrorType.COMMAND_TIMEOUT,
            "network_error": ErrorType.NETWORK_ERROR,
        }
        return mapping.get(heal_type, ErrorType.UNKNOWN_ERROR)

    def _build_fix_instruction(
        self,
        diagnosis,
        heal_plan: Optional[Dict[str, Any]],
        tool_name: str,
        exception_category: str = ExceptionCategory.UNKNOWN,
    ) -> str:
        """构建分类修复指令文本（插入对话让 LLM 执行）"""
        lines = []
        lines.append(f"【异常处理】工具 {tool_name} 执行失败，系统已诊断：")

        # 分类特异性建议
        category_hints = {
            ExceptionCategory.VISION_FAILURE: (
                "【视觉失败】找不到目标。建议："
                "1) 使用 visual_find_text() 或 visual_read_region() 尝试 OCR 定位；"
                "2) 使用 visual_scan_region() 在候选区域扫描；"
                "3) 截图分析当前界面状态。"
            ),
            ExceptionCategory.ACTION_FAILURE: (
                "【动作失败】操作执行失败。建议："
                "1) 重试操作（可能为瞬时失败）；"
                "2) 先移动鼠标到目标位置再点击；"
                "3) 检查目标窗口是否在前景（可用 Alt+Tab 或点击窗口标题栏）。"
            ),
            ExceptionCategory.STATE_ANOMALY: (
                "【状态异常】可能出现了弹窗或界面异常。建议："
                "1) 先用 visual_scan() 或 visual_read_text() 识别当前界面；"
                "2) 若检测到弹窗，识别弹窗类型并按需关闭（Enter/ESC/点击确定）；"
                "3) 确认是否卡在某个意外界面。"
            ),
            ExceptionCategory.UNKNOWN: (
                "【未知异常】系统已保存快照供分析。建议："
                "1) 先观察当前屏幕状态（visual_scan / visual_read_text）；"
                "2) 根据观察结果更换操作策略；"
                "3) 可尝试 PowerShell 命令完成。"
            ),
        }
        hint = category_hints.get(exception_category, "")
        if hint:
            lines.append(hint)

        # 诊断信息
        if diagnosis is not None:
            lines.append(f"- 错误类型: {diagnosis.error_type}（置信度 {diagnosis.confidence:.2f}）")
            lines.append(f"- 诊断推理: {diagnosis.reasoning}")
            if diagnosis.fix_actions:
                lines.append("- 建议修复动作:")
                for i, action in enumerate(diagnosis.fix_actions[:3], 1):
                    lines.append(f"  {i}. {action.description}")

        # 自愈信息
        if heal_plan and heal_plan.get("fix_actions"):
            lines.append("- 自愈方案:")
            for i, action in enumerate(heal_plan["fix_actions"], 1):
                desc = action.get("description", "")
                lines.append(f"  {i}. {desc}")

        lines.append("- 请根据以上诊断，立即执行修复动作或改用其他方式。")
        lines.append("- 若连续失败，请考虑使用 run_powershell 命令方式完成任务。")
        return "\n".join(lines)

    # ============================================================
    # 快照保存（未知异常）
    # ============================================================

    def save_snapshot(self, tool_name: str = "", error: str = "",
                      extra: Dict[str, Any] = None) -> str:
        """保存异常快照（截图 + OCR结果 + 当前memory）

        Args:
            tool_name: 出错的工具名
            error: 错误信息
            extra: 额外信息

        Returns:
            快照文件路径（成功）或 ""（失败）
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.snapshots_dir, exist_ok=True)
            base = os.path.join(self.snapshots_dir, f"snapshot_{timestamp}")

            # 截图
            screenshot_path = None
            try:
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot_path = f"{base}.png"
                screenshot.save(screenshot_path)
            except Exception as e:
                logger.warning(f"快照截图失败: {e}")

            # OCR 结果
            ocr_result = None
            try:
                from vision.ocr import ocr_screen
                ocr_result = ocr_screen()
            except Exception as e:
                logger.warning(f"快照OCR失败: {e}")

            # memory 快照
            memory_snapshot = None
            if self.working_memory is not None:
                try:
                    memory_snapshot = self.working_memory.snapshot()
                except Exception:
                    memory_snapshot = None

            # 汇总 JSON
            data = {
                "timestamp": timestamp,
                "tool_name": tool_name,
                "error": error,
                "task": self.current_task,
                "screenshot": os.path.basename(screenshot_path) if screenshot_path else None,
                "ocr_result": ocr_result[:2000] if ocr_result else None,
                "memory": memory_snapshot,
                "extra": extra or {},
            }
            json_path = f"{base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"异常快照已保存: {json_path}")
            return json_path
        except Exception as e:
            logger.warning(f"保存快照失败: {e}")
            return ""

    # ============================================================
    # 完成验证
    # ============================================================

    def validate_completion(
        self,
        content: str,
        recent_tool_results: List[str],
    ) -> Dict[str, Any]:
        """验证 AI 宣告的完成是否可信（伪成功检测）

        Args:
            content: AI 回复内容
            recent_tool_results: 最近的工具执行结果

        Returns:
            验证报告
        """
        # 检测是否有失败残留
        has_failure = any(
            self.failure_detector.detect(str(r)) for r in recent_tool_results
        )

        confidence = 0.7
        reasons = []

        if has_failure:
            confidence -= 0.3
            reasons.append("最近存在失败的工具调用，完成声明可疑")
        if not recent_tool_results:
            reasons.append("没有最近的工具执行记录")

        confidence = max(0.0, min(1.0, confidence))

        # Tool dispatch is not completion evidence. Objective verifiers decide
        # whether a completion claim is supported.
        is_pseudo_success = has_failure

        return {
            "is_pseudo_success": is_pseudo_success,
            "confidence": confidence,
            "reasons": reasons,
            "has_failure": has_failure,
            "has_success_operation": False,
        }

    def record_heal_outcome(
        self,
        report: Dict[str, Any],
        retry_success: bool,
    ):
        """记录修复结果（沉淀到知识库）"""
        if not self.healer or not report.get("heal_plan"):
            return

        try:
            self.healer.record_result(report["heal_plan"], {}, retry_success)
        except Exception as e:
            logger.warning(f"记录修复结果失败: {e}")

    def apply_fix(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """执行自愈修复动作（直接调用工具）

        Args:
            report: process_failure 返回的报告

        Returns:
            修复结果
        """
        if not self.healer or not self.tools_registry:
            return {"all_success": False, "message": "自愈不可用（缺少知识库或工具注册表）"}

        heal_plan = report.get("heal_plan")
        if not heal_plan or not heal_plan.get("fix_actions"):
            return {"all_success": False, "message": "无可用修复动作"}

        return self.healer.apply_fix(heal_plan, self.tools_registry)

    def get_statistics(self) -> Dict[str, Any]:
        """获取异常处理统计"""
        total = len(self.error_history)
        error_types = {}
        categories = {}
        for r in self.error_history:
            et = r["error_type"]
            error_types[et] = error_types.get(et, 0) + 1
            cat = r.get("exception_category", ExceptionCategory.UNKNOWN)
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_errors": total,
            "consecutive_failures": self.consecutive_failures,
            "error_types": error_types,
            "categories": categories,
        }