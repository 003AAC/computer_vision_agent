"""
Cost Planner - 成本感知规划
==========================
让规划与执行都"算成本"：

1. CostModel   : 为每个工具/动作估算成本（单位：成本点，≈ 延迟×风险）
2. PlanCandidate / select_best_plan : 多方案择优（成本最低且可行）
3. BudgetTracker : 执行中累计实际成本，超预算时给出"降级到低成本路径"提示

设计理念：
  - 规划阶段：优先选择成本最低的可行方案（如用 run_powershell 启动程序，
    而不是"截图→定位图标→双击"这类高成本多步路径）
  - 执行阶段：累计成本；超预算 → 注入降级指令，强制走低成本工具
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CostModel:
    """工具/动作成本模型（成本单位 ≈ 延迟 × 风险）"""

    # 单次调用成本（成本点）
    TOOL_COST: Dict[str, float] = {
        # —— 高成本：全屏 OCR / 全屏视觉 / 分块扫描（慢、易误检）——
        "visual_read_text": 6.0,
        "visual_scan": 5.0,
        "visual_scan_grid": 8.0,
        # —— 中成本：定位（可能触发 DINO 二次推理）——
        "visual_locate": 4.0,
        # —— 低成本：区域操作（快且准）——
        "visual_locate_region": 2.0,
        "visual_find_text": 2.0,
        "visual_read_region": 1.5,
        "run_powershell": 1.5,
        # —— 极低成本：直接动作 ——
        "click_at": 0.5,
        "drag_mouse": 1.0,
        "type_text": 1.0,
        "press_key": 0.3,
        "hotkey": 0.3,
        # wait 按秒计
        "wait": 0.0,
    }

    # 缓存命中的定位折扣（tracker 已记住对象，只需邻域重扫）
    CACHED_LOCATE_DISCOUNT = 0.4

    # 低成本工具（超预算时优先建议）
    CHEAP_TOOLS = [
        "run_powershell", "visual_find_text",
        "visual_read_region", "press_key", "hotkey",
    ]

    # 高成本工具（超预算时应避免）
    EXPENSIVE_TOOLS = [
        "visual_read_text", "visual_scan", "visual_scan_grid",
        "visual_locate",
    ]

    @classmethod
    def tool_cost(cls, tool_name: str, args: Dict[str, Any] = None,
                  cached: bool = False) -> float:
        """估算单次工具调用成本

        Args:
            tool_name: 工具名
            args: 工具参数（wait 需要秒数）
            cached: 是否命中视觉对象缓存（定位类可打折）

        Returns:
            成本点
        """
        args = args or {}
        if tool_name == "wait":
            try:
                return float(args.get("seconds", 1.0))
            except (TypeError, ValueError):
                return 1.0

        base = cls.TOOL_COST.get(tool_name, 1.0)
        # 定位类命中缓存 → 折扣（无需全屏重扫）
        if cached and tool_name in ("visual_locate", "visual_locate_region",
                                    "visual_find_text"):
            base *= cls.CACHED_LOCATE_DISCOUNT
        return base

    @classmethod
    def estimate_actions_cost(cls, allowed_actions: List[str]) -> float:
        """从阶段的 allowed_actions 粗估该阶段成本（取一次典型调用）"""
        if not allowed_actions:
            return 5.0
        costs = sorted(cls.TOOL_COST.get(a, 1.0) for a in allowed_actions)
        cheapest = costs[0]
        median = costs[len(costs) // 2]
        return round(median + cheapest, 2)


class PlanCandidate:
    """一个候选方案"""

    def __init__(self, name: str, approach: str = "",
                 est_total_cost: float = 0.0, risk: str = "medium",
                 phases: List[Dict[str, Any]] = None,
                 rationale: str = ""):
        self.name = name
        self.approach = approach
        self.est_total_cost = est_total_cost
        self.risk = risk
        self.phases = phases or []
        self.rationale = rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "approach": self.approach,
            "est_total_cost": self.est_total_cost,
            "risk": self.risk,
            "phase_count": len(self.phases),
            "rationale": self.rationale,
        }


def select_best_plan(candidates: List[PlanCandidate],
                     failed_paths: List[str] = None) -> Optional[PlanCandidate]:
    """成本感知择优：在可行方案中选成本最低者

    规则：
      1. 过滤掉"明显依赖失效路径"的方案
      2. 低风险优先，其次成本最低
      3. 若全被过滤，则退回成本最低者（避免无方案可用）

    Args:
        candidates: 候选方案列表
        failed_paths: 已失效路径/启动方式清单

    Returns:
        选中的方案；无候选返回 None
    """
    if not candidates:
        return None
    failed_paths = [p.lower() for p in (failed_paths or [])]

    def depends_on_failed(c: PlanCandidate) -> bool:
        if not failed_paths:
            return False
        text = (c.approach + " " + c.rationale + " " + " ".join(
            str(p) for p in c.phases)).lower()
        return any(fp in text for fp in failed_paths if fp)

    viable = [c for c in candidates if not depends_on_failed(c)]
    pool = viable or candidates   # 全失效则退回全部，至少能跑

    risk_rank = {"low": 0, "medium": 1, "high": 2}
    pool.sort(key=lambda c: (risk_rank.get(c.risk, 1), c.est_total_cost))
    return pool[0]


class BudgetTracker:
    """执行期成本预算追踪"""

    # 超预算后降级指令
    DOWNGRADE_HINT = (
        "【成本超预算】当前累计成本已超出预算。请立即改走**低成本路径**：\n"
        "  1. 优先 run_powershell 一条命令直达结果\n"
        "  2. 用 visual_find_text / visual_read_region（区域）代替全屏扫描/全屏OCR\n"
        "  3. 用 press_key / hotkey 代替多步鼠标点击\n"
        "  4. 不要重复 visual_read_text / visual_scan / visual_scan_grid"
    )

    def __init__(self, budget: float = None, safety_factor: float = 1.5,
                 default_budget: float = 60.0):
        """
        Args:
            budget: 显式预算（成本点）；None 则由计划成本推导
            safety_factor: 计划成本 → 预算的放大系数
            default_budget: 无计划时的默认预算
        """
        self.safety_factor = safety_factor
        self.default_budget = default_budget
        self.planned_cost = 0.0
        self.budget = float(budget) if budget is not None else default_budget
        self.spent = 0.0
        self._hint_given = False

    # ============================================================
    # 预算设置
    # ============================================================

    def set_from_plan(self, planned_cost: float):
        """用规划阶段成本设置预算"""
        self.planned_cost = float(planned_cost or 0.0)
        if self.planned_cost > 0:
            self.budget = round(self.planned_cost * self.safety_factor, 2)
        else:
            self.budget = self.default_budget

    # ============================================================
    # 记账
    # ============================================================

    def add(self, cost: float, tool_name: str = ""):
        """记录一次成本"""
        self.spent += float(cost or 0.0)

    def add_tool(self, tool_name: str, args: Dict[str, Any] = None,
                 cached: bool = False):
        """按工具名记账"""
        c = CostModel.tool_cost(tool_name, args, cached=cached)
        self.add(c, tool_name)
        return c

    # ============================================================
    # 查询
    # ============================================================

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    def over_budget(self) -> bool:
        return self.spent > self.budget

    def usage_ratio(self) -> float:
        if self.budget <= 0:
            return 0.0
        return min(1.0, self.spent / self.budget)

    def should_warn(self, threshold: float = 0.8) -> bool:
        """是否接近预算上限"""
        return self.usage_ratio() >= threshold

    def take_downgrade_hint(self) -> Optional[str]:
        """取降级指令（只提示一次，避免刷屏）"""
        if self.over_budget() and not self._hint_given:
            self._hint_given = True
            return self.DOWNGRADE_HINT
        return None

    def reset_hint(self):
        self._hint_given = False

    # ============================================================
    # 指令构建
    # ============================================================

    def build_instruction(self) -> str:
        """构建成本状态指令（注入提示词）"""
        lines = [
            "## [成本预算] 执行成本",
            f"- 已用: {self.spent:.1f} / 预算 {self.budget:.1f} 成本点 "
            f"({self.usage_ratio() * 100:.0f}%)",
        ]
        if self.over_budget():
            lines.append("- ⚠️ 已超预算，请改走低成本路径（见降级指令）")
        elif self.should_warn():
            lines.append("- ⚠️ 接近预算上限，请优先使用低成本工具")
        else:
            lines.append(
                f"- 低成本工具: {', '.join(CostModel.CHEAP_TOOLS)}"
            )
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "planned_cost": round(self.planned_cost, 2),
            "budget": round(self.budget, 2),
            "spent": round(self.spent, 2),
            "remaining": round(self.remaining, 2),
            "usage_ratio": round(self.usage_ratio(), 3),
            "over_budget": self.over_budget(),
        }


def estimate_phases_cost(phases: List[Any]) -> float:
    """估算整套阶段计划的成本

    Args:
        phases: 阶段列表（PhaseDefinition 或 dict）

    Returns:
        成本点
    """
    total = 0.0
    for p in phases or []:
        if hasattr(p, "allowed_actions"):
            actions = p.allowed_actions or []
        elif isinstance(p, dict):
            actions = p.get("allowed_actions", [])
        else:
            actions = []
        # 单阶段典型成本 = 一次定位 + 一次动作 + 一次确认
        total += CostModel.estimate_actions_cost(actions) + 1.0
    return round(total, 2)

