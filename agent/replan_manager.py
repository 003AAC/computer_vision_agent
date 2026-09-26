"""
Replan Manager - 过期启动器检测与强制重新规划
============================================
解决"反复启动过期启动器却不调整计划"的问题。

问题链路：
  启动器过期（快捷方式失效 / 程序已卸载 / 版本路径变更）
    → 点击或 Start-Process 失败
    → Agent 换个坐标继续点同一目标（死磕）
    → 计划从未调整 → 反复失败

本模块提供：
  1. LaunchFailureDetector : 识别"启动类失败"，并提取失效路径/目标
  2. ReplanManager         : 记录失效路径（复用 PathPlanner.NegativeExperienceManager）
                             → 同一目标连续失败达阈值 → 触发**强制重新规划**
                             → 生成"禁止路径清单 + 备选启动方式"约束注入规划提示词
"""
import logging
import re
from typing import Dict, Any, List, Optional

from systematic_error_fix.path_planner import PathPlanner
from systematic_error_fix.models import ErrorType

logger = logging.getLogger(__name__)


class LaunchFailureDetector:
    """启动类失败识别器"""

    # 启动相关关键词（命令/目标中出现）
    LAUNCH_HINTS = (
        "start-process", "start ", ".lnk", ".exe", "explorer ",
        "打开", "启动", "运行",
    )
    # 失败信号
    FAILURE_HINTS = (
        "不是内部", "不是可识别", "系统找不到", "找不到", "无法访问",
        "拒绝访问", "不存在", "无效", "过期", "failed", "not found",
        "cannot find", "no such",
    )
    # Windows 路径 / 快捷方式 / 可执行文件
    _PATH_RE = re.compile(
        r'([A-Za-z]:\\[^"\'\r\n]+?\.(?:lnk|exe|bat|cmd))', re.I
    )

    @classmethod
    def is_launch_failure(cls, tool_name: str, args: Dict[str, Any],
                          result_str: str) -> bool:
        """判断是否为"启动类失败"

        Args:
            tool_name: 工具名
            args: 工具参数
            result_str: 工具结果

        Returns:
            是否属于启动失败
        """
        result_lower = str(result_str).lower()
        # 必须有失败信号
        if not any(h in result_lower for h in cls.FAILURE_HINTS):
            return False
        # 命令类：run_powershell 启动失败
        if tool_name == "run_powershell":
            cmd = str(args.get("command", "")).lower()
            if any(h in cmd for h in cls.LAUNCH_HINTS):
                return True
            # 结果里出现路径也能判定
            if cls._PATH_RE.search(str(result_str)):
                return True
            return False
        # 点击类：点击后报错里带失效路径
        if tool_name in ("click_at", "drag_mouse"):
            return bool(cls._PATH_RE.search(str(result_str)))
        return False

    @classmethod
    def extract_target(cls, tool_name: str, args: Dict[str, Any],
                       result_str: str) -> str:
        """提取失效目标（路径优先，其次应用名）

        Returns:
            失效路径/目标标识；无法提取返回空串
        """
        # 1. 从结果中找路径
        m = cls._PATH_RE.search(str(result_str))
        if m:
            return m.group(1).strip()
        # 2. 从参数命令中找路径
        cmd = str(args.get("command", ""))
        m = cls._PATH_RE.search(cmd)
        if m:
            return m.group(1).strip()
        # 3. 从 Start-Process / start 后取应用名
        m = re.search(r'(?:start-process|start)\s+["\']?([^"\'\s;]+)',
                      cmd, re.I)
        if m:
            return m.group(1).strip()
        # 4. 从参数中取目标描述
        if tool_name in ("visual_locate", "visual_locate_region"):
            return str(args.get("target_description", "")).strip()
        if tool_name == "visual_find_text":
            return str(args.get("target_text", "")).strip()
        return ""


class ReplanManager:
    """重新规划管理器（失效路径记忆 + 触发策略）"""

    def __init__(self, threshold: int = 2, max_replans: int = 2,
                 planner: PathPlanner = None):
        """
        Args:
            threshold: 同一目标连续失败多少次触发重新规划
            max_replans: 单任务最多重新规划次数（防止无限重规划）
            planner: 路径规划器（提供失效路径记忆），None 则新建
        """
        self.planner = planner or PathPlanner()
        self.threshold = threshold
        self.max_replans = max_replans
        self.replan_count = 0
        self._fail_count: Dict[str, int] = {}
        self._last_failed_target = ""
        self.last_failure_detail: List[Dict[str, Any]] = []

    # ============================================================
    # 记录
    # ============================================================

    def record_launch_failure(self, target: str, reason: str = "",
                              tool_name: str = "") -> int:
        """记录一次启动失败

        Args:
            target: 失效路径/目标
            reason: 失败原因
            tool_name: 工具名

        Returns:
            该目标累计连续失败次数
        """
        target = (target or "").strip()
        if not target:
            return 0

        self._last_failed_target = target
        self._fail_count[target] = self._fail_count.get(target, 0) + 1

        # 沉淀到失效路径记忆（跨任务复用）
        try:
            self.planner.record_failure(
                target, ErrorType.PROGRAM_NOT_FOUND,
                reason or "启动失败/快捷方式可能已过期",
            )
        except Exception as e:
            logger.warning(f"记录失效路径失败: {e}")

        self.last_failure_detail.append({
            "target": target,
            "reason": (reason or "")[:120],
            "tool": tool_name,
            "count": self._fail_count[target],
        })
        logger.info(
            f"启动失败记录: {target}（连续 {self._fail_count[target]} 次）"
        )
        return self._fail_count[target]

    def record_success(self):
        """记录成功 → 重置连续失败计数（避免误触发重规划）"""
        self._fail_count.clear()
        self._last_failed_target = ""

    # ============================================================
    # 触发判断
    # ============================================================

    def should_replan(self) -> Optional[str]:
        """是否需要重新规划

        Returns:
            触发原因；不需要返回 None
        """
        if self.replan_count >= self.max_replans:
            return None

        for target, count in self._fail_count.items():
            if count >= self.threshold:
                return (
                    f"目标 [{target}] 连续启动失败 {count} 次"
                    f"（≥{self.threshold}），启动方式可能已过期，需重新规划"
                )
        return None

    def mark_replanned(self):
        """标记已重新规划（重置计数，保留失效路径记忆）"""
        self.replan_count += 1
        self._fail_count.clear()
        self._last_failed_target = ""

    # ============================================================
    # 约束构建
    # ============================================================

    def failed_paths(self) -> List[str]:
        """获取全部失效路径"""
        try:
            return self.planner.negative_manager.get_failed_paths()
        except Exception:
            return []

    def build_constraint(self, reason: str = "") -> str:
        """构建重新规划约束文本（注入规划提示词）"""
        lines = ["## ⚠️ 重新规划约束（重要）"]
        if reason:
            lines.append(f"触发原因: {reason}")

        failed = self.failed_paths()
        if failed:
            lines.append("以下启动方式**已确认失效，禁止再使用**：")
            for p in failed[:10]:
                lines.append(f"  - {p}")

        lines.append("")
        lines.append("请据此调整方案，必须满足：")
        lines.append("  1. 为启动步骤提供**至少 2 种备选方式**，且都不在禁用清单内")
        lines.append("     （如：where 命令定位 → 直接执行 exe；"
                     "开始菜单搜索；PowerShell Get-ChildItem 扫描目录）")
        lines.append("  2. 若目标程序可能未安装，先用 PowerShell 验证存在性，"
                     "再决定后续阶段")
        lines.append("  3. 禁止原样重复上一版计划")
        lines.append("  4. 每阶段标注 est_cost（预估成本，越小越优）")
        return "\n".join(lines)

    # ============================================================
    # 统计
    # ============================================================

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "replan_count": self.replan_count,
            "max_replans": self.max_replans,
            "failed_paths": self.failed_paths(),
            "fail_counts": dict(self._fail_count),
            "last_target": self._last_failed_target,
        }

