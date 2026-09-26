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

from systematic_error_fix.error_diagnostician import ErrorDiagnostician
from systematic_error_fix.models import ErrorType
from self_healer import SelfHealer

logger = logging.getLogger(__name__)


class ErrorCategory:
    """异常四大分类常量"""

    VISION_FAILURE = "vision_failure"      # 视觉失败：找不到目标
    ACTION_FAILURE = "action_failure"      # 动作失败：click 失败
    STATE_ANOMALY = "state_anomaly"        # 状态异常：弹窗/卡住
    UNKNOWN = "unknown"                    # 未知异常


# 视觉类工具
VISION_TOOLS = {
    "visual_scan", "visual_scan_region", "visual_scan_grid",
    "visual_locate", "visual_locate_region",
    "visual_read_text", "visual_read_region", "visual_find_text",
}
# 动作类工具
ACTION_TOOLS = {
    "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
}
# 状态异常关键词（弹窗/无响应）
STATE_ANOMALY_KEYWORDS = [
    "弹窗", "对话框", "popup", "未响应", "not responding",
    "卡住", "无响应", "阻止", "拦截",
]
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
        "系统找不到", "无法访问", "定位失败", "点击失败", "输入失败",
        "按键失败", "组合键失败", "拖拽失败", "命令执行失败",
        "命令执行超时", "命令执行异常",
    ]

    SUCCESS_SIGNALS = [
        "已点击", "已输入", "已按下", "已拖拽", "命令执行成功",
        "✅", "成功", "已等待",
    ]

    # 硬失败信号：即使文本含 "✅"/"已点击" 也必须优先判为失败
    # （修复"输入/点击无效却因 ✅ 被判成功"的假成功链）
    HARD_FAILURE_SIGNALS = [
        "点击失败", "输入失败", "按键失败", "组合键失败", "拖拽失败",
        "失败:", "失败：", "失败 ", "事件未送达", "未送达",
        "需以管理员身份", "uipi", "光标未到位", "鼠标可能被拦截",
        "被拦截", "窗口拒绝", "前台窗口拒绝",
        # 动作门卫（core._action_guard）的拒绝文案：
        #   若不含失败关键词，会被误判为"成功"，导致被拦截也算进展
        "动作被拒绝", "被拒绝", "不在允许", "禁止盲点",
        "命令执行失败", "命令执行超时", "命令执行异常",
        "定位失败", "operate failed", "unable to",
    ]

    def __init__(self):
        self._failure_signals = [s.lower() for s in self.FAILURE_SIGNALS]
        self._success_signals = [s.lower() for s in self.SUCCESS_SIGNALS]
        self._hard_failure_signals = [
            s.lower() for s in self.HARD_FAILURE_SIGNALS
        ]

    def detect(self, result_str: str) -> bool:
        """检测是否为失败

        判定优先级（关键）：
          1. 硬失败信号（"点击失败:" / "事件未送达" / "需以管理员身份" …）
             —— 优先于 "✅"，避免"✅ 已点击 … [鼠标可能被拦截]" 被误判成功
          2. 成功信号（"✅" / "已点击" …）
          3. 一般失败信号

        Args:
            result_str: 工具返回结果字符串

        Returns:
            是否为失败
        """
        result_lower = str(result_str).lower()

        # 1) 硬失败优先
        if any(s in result_lower for s in self._hard_failure_signals):
            return True

        # 2) 成功信号（避免误判，如"打开失败处理成功"）
        if any(s in result_lower for s in self._success_signals):
            return False

        # 3) 一般失败信号
        return any(s in result_lower for s in self._failure_signals)


class ExceptionHandler:
    """Agent 异常处理器

    职责：
      1. 工具结果失败检测
      2. 错误四大分类（视觉/动作/状态/未知）
      3. 错误类型诊断（ErrorDiagnostician）
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

    # 每类异常的修复建议
    CATEGORY_FIX_HINTS = {
        ErrorCategory.VISION_FAILURE: (
            "【视觉失败】请按顺序尝试：\n"
            "  1. visual_find_text(\"目标文字\") 用 OCR 直接找文字坐标\n"
            "  2. visual_read_region(x,y,w,h) 缩小范围 OCR\n"
            "  3. visual_scan_region() 局部放大视觉检测\n"
            "  4. 目标可能不在当前页面 → 先完成前置步骤"
        ),
        ErrorCategory.ACTION_FAILURE: (
            "【动作失败】请按顺序尝试：\n"
            "  1. 重新定位目标坐标后重试 click_at\n"
            "  2. 先点击窗口空白处激活窗口（foreground）\n"
            "  3. 改用 press_key / hotkey 走键盘路径\n"
            "  4. 若目标被遮挡，先关闭遮挡窗口"
        ),
        ErrorCategory.STATE_ANOMALY: (
            "【状态异常】检测到弹窗/无响应：\n"
            "  1. 先 visual_read_text() 读弹窗文字判断类型\n"
            "  2. 优先点\"确定/关闭/取消\"消除弹窗\n"
            "  3. 若无响应，等待几秒或发送 escape 键\n"
            "  4. 处理完弹窗后再继续原任务"
        ),
        ErrorCategory.UNKNOWN: (
            "【未知异常】已保存现场快照供分析。\n"
            "  1. 重新观察屏幕确认当前状态\n"
            "  2. 考虑换用 run_powershell 方式完成任务"
        ),
    }

    def __init__(self, knowledge_base=None,
                 tools_registry: Optional[Dict[str, Any]] = None,
                 snapshots_dir: str = None):
        """初始化异常处理器

        Args:
            knowledge_base: 知识库实例（用于经验沉淀）
            tools_registry: 工具注册表 {tool_name: tool}
            snapshots_dir: 快照保存目录（默认项目根/snapshots）
        """
        self.knowledge_base = knowledge_base
        self.tools_registry = tools_registry or {}

        # 错误诊断器（复用系统性错误修复模块）
        self.diagnostician = ErrorDiagnostician(knowledge_base)

        # 自愈闭环（复用 self_healer.py）
        self.healer = SelfHealer(knowledge_base) if knowledge_base else None

        # 失败检测器
        self.failure_detector = FailureDetector()

        # 快照目录
        if snapshots_dir is None:
            snapshots_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "snapshots"
            )
        self.snapshots_dir = snapshots_dir

        # 内部状态
        self.error_history: List[Dict[str, Any]] = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.category_counts: Dict[str, int] = {}

    def reset(self, task_query: str = ""):
        """新任务开始时重置状态"""
        self.consecutive_failures = 0
        self.error_history = []
        self.category_counts = {}
        if self.healer:
            self.healer.reset()

    def is_success(self, result_str: str) -> bool:
        """检测工具结果是否为成功（非失败）"""
        return not self.failure_detector.detect(result_str)

    # ============================================================
    # 异常分类（四大类）
    # ============================================================

    def classify_category(self, result_str: str, tool_name: str) -> str:
        """将失败分类为四大类之一

        Args:
            result_str: 失败结果
            tool_name: 工具名

        Returns:
            ErrorCategory 常量之一
        """
        result_lower = str(result_str).lower()

        # 1. 状态异常：含弹窗/无响应关键词
        if any(kw in result_lower for kw in STATE_ANOMALY_KEYWORDS):
            return ErrorCategory.STATE_ANOMALY

        # 2. 视觉失败：视觉类工具失败 或 含"未找到/找不到"
        not_found_kw = ["未找到", "找不到", "not found", "定位失败", "未检测到"]
        # 动作工具明确失败优先归为动作失败
        if tool_name in ACTION_TOOLS:
            return ErrorCategory.ACTION_FAILURE
        if tool_name in VISION_TOOLS or any(kw in result_lower for kw in not_found_kw):
            return ErrorCategory.VISION_FAILURE

        # 4. 未知
        return ErrorCategory.UNKNOWN


    # ============================================================
    # 失败处理
    # ============================================================

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

        # 步骤0：四大分类
        category = self.classify_category(result_str, tool_name)
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

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
            error_type = self._map_heal_type_to_error_type(
                heal_plan.get("error_type", "")
            )

        # 生成修复指令文本（注入提示词）
        fix_instruction = self._build_fix_instruction(
            diagnosis, heal_plan, tool_name, category
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
            "category": category,
            "diagnosis": diagnosis.to_dict() if diagnosis else None,
            "heal_plan": heal_plan,
            "fix_instruction": fix_instruction,
            "consecutive_failures": self.consecutive_failures,
            "should_degrade": should_degrade,
            "degrade_suggestion": degrade_suggestion,
            "tool_name": tool_name,
            "raw_error": result_str[:300],
        }
        self.error_history.append(report)

        # 未知异常 → 保存快照（供 LLM 分析）
        if category == ErrorCategory.UNKNOWN:
            try:
                report["snapshot_path"] = self.capture_snapshot(
                    result_str, tool_name, context={"task": task_query}
                )
            except Exception as e:
                logger.warning(f"快照保存失败: {e}")

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
        category: str = ErrorCategory.UNKNOWN,
    ) -> str:
        """构建修复指令文本（插入对话让 LLM 执行）"""
        lines = []
        lines.append(
            f"【异常处理】工具 {tool_name} 执行失败（分类: {category}）"
        )

        # 分类专用建议
        category_hint = self.CATEGORY_FIX_HINTS.get(category)
        if category_hint:
            lines.append(category_hint)

        # 诊断信息
        if diagnosis is not None:
            lines.append(
                f"- 错误类型: {diagnosis.error_type}"
                f"（置信度 {diagnosis.confidence:.2f}）"
            )
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
    # 快照（未知异常现场保存）
    # ============================================================

    def capture_snapshot(self, result_str: str, tool_name: str,
                         context: Dict[str, Any] = None) -> str:
        """保存异常现场快照（screenshot + OCR结果 + 上下文）

        Args:
            result_str: 失败结果
            tool_name: 工具名
            context: 附加上下文（memory 等）

        Returns:
            快照目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        snap_dir = os.path.join(self.snapshots_dir, f"snap_{timestamp}")
        os.makedirs(snap_dir, exist_ok=True)

        # 1. 截图
        screenshot_path = ""
        try:
            import pyautogui
            img = pyautogui.screenshot()
            screenshot_path = os.path.join(snap_dir, "screenshot.png")
            img.save(screenshot_path)
        except Exception as e:
            logger.warning(f"快照截图失败: {e}")

        # 2. OCR 结果
        ocr_texts = []
        try:
            from vision.ocr import ocr_screen
            ocr_json = ocr_screen()
            with open(os.path.join(snap_dir, "ocr.json"), "w",
                      encoding="utf-8") as f:
                f.write(ocr_json)
            ocr_texts = [
                t.get("text", "")
                for t in json.loads(ocr_json).get("texts", [])
            ]
        except Exception as e:
            logger.warning(f"快照 OCR 失败: {e}")

        # 3. 上下文（memory 等）
        snapshot_meta = {
            "timestamp": timestamp,
            "tool_name": tool_name,
            "error": result_str[:1000],
            "context": context or {},
            "ocr_texts": ocr_texts[:100],
            "screenshot": screenshot_path,
        }
        with open(os.path.join(snap_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(snapshot_meta, f, ensure_ascii=False, indent=2)

        logger.info(f"已保存异常快照: {snap_dir}")
        return snap_dir

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

        # 检测是否有有效操作
        has_success_operation = any(
            self.is_success(str(r)) for r in recent_tool_results
        )

        confidence = 0.5
        reasons = []

        if has_failure:
            confidence -= 0.3
            reasons.append("最近存在失败的工具调用，完成声明可疑")
        if not recent_tool_results:
            confidence -= 0.2
            reasons.append("没有任何工具执行记录，无法确认完成")
        elif has_success_operation:
            confidence += 0.3
            reasons.append("有成功的工具操作记录")

        confidence = max(0.0, min(1.0, confidence))

        # 伪成功判定
        is_pseudo_success = confidence < 0.7 and (
            has_failure or not recent_tool_results
        )

        return {
            "is_pseudo_success": is_pseudo_success,
            "confidence": confidence,
            "reasons": reasons,
            "has_failure": has_failure,
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
            return {
                "all_success": False,
                "message": "自愈不可用（缺少知识库或工具注册表）",
            }

        heal_plan = report.get("heal_plan")
        if not heal_plan or not heal_plan.get("fix_actions"):
            return {"all_success": False, "message": "无可用修复动作"}

        return self.healer.apply_fix(heal_plan, self.tools_registry)

    def get_statistics(self) -> Dict[str, Any]:
        """获取异常处理统计"""
        total = len(self.error_history)
        error_types = {}
        for r in self.error_history:
            et = r["error_type"]
            error_types[et] = error_types.get(et, 0) + 1
        return {
            "total_errors": total,
            "consecutive_failures": self.consecutive_failures,
            "error_types": error_types,
            "category_counts": dict(self.category_counts),
        }
