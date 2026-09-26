"""
Skill 系统基类 - Base Skill
===========================
为不同应用场景定义专属技能（goal / available_tools / strategy / verification）。

设计原则：
  - Skill 为工具选择提供推荐，不作为执行权限白名单
  - Skill 匹配失败 → 回退到通用探索模式（不崩溃）
  - 每个 Skill 定义领域专属的：
    - goal: 目标描述
    - available_tools: 允许的工具集（约束）
    - strategy: 执行策略（引导）
    - verification: 验收方式
"""
from typing import Dict, Any, List, Optional


class BaseSkill:
    """技能基类"""

    # 技能标识
    name: str = "base"
    description: str = "通用技能"

    # 匹配关键词（用于任务匹配）
    match_keywords: List[str] = []

    # 目标
    goal: str = ""

    # 允许的工具集（None = 不限制）
    available_tools: Optional[List[str]] = None

    # 策略提示（注入 LLM）
    strategy: str = ""

    # 验收方式
    verification: str = ""

    # 需要避免的操作
    avoid: List[str] = []

    # ============================================================
    # 匹配
    # ============================================================

    @classmethod
    def matches(cls, task: str) -> bool:
        """判断任务是否匹配此技能（子类可覆盖）"""
        if not cls.match_keywords:
            return False
        t = (task or "").lower()
        return any(kw.lower() in t for kw in cls.match_keywords)

    @classmethod
    def match_score(cls, task: str) -> float:
        """匹配分数（0~1），用于多技能竞争"""
        if not cls.match_keywords:
            return 0.0
        t = (task or "").lower()
        hits = sum(1 for kw in cls.match_keywords if kw.lower() in t)
        return hits / len(cls.match_keywords)

    # ============================================================
    # 构建指令
    # ============================================================

    @classmethod
    def build_instruction(cls) -> str:
        """构建技能指令（注入 LLM）"""
        lines = [f"## [技能] {cls.name}"]
        if cls.description:
            lines.append(f"- 说明: {cls.description}")
        if cls.goal:
            lines.append(f"- 目标: {cls.goal}")
        if cls.available_tools:
            lines.append(f"- 推荐工具: {', '.join(cls.available_tools)}")
        if cls.avoid:
            lines.append(f"- 避免: {', '.join(cls.avoid)}")
        if cls.strategy:
            lines.append(f"- 策略: {cls.strategy}")
        if cls.verification:
            lines.append(f"- 验收: {cls.verification}")
        return "\n".join(lines)

    @classmethod
    def is_tool_allowed(cls, tool_name: str) -> bool:
        """Skills guide tool choice but do not block task-required actions."""
        return True


def get_base_skills() -> List[type]:
    """获取所有可用技能类（自动发现 skills 包中的 Skill 子类）"""
    import importlib
    import pkgutil

    skill_classes: List[type] = []
    # 导入本包（注意：避免覆盖局部变量名）
    import skills as skills_pkg
    for _, modname, _ in pkgutil.iter_modules(skills_pkg.__path__):
        if modname in ("base",):
            continue
        try:
            module = importlib.import_module(f"skills.{modname}")
            for attr in dir(module):
                obj = getattr(module, attr)
                if (isinstance(obj, type)
                        and issubclass(obj, BaseSkill)
                        and obj is not BaseSkill
                        and obj.name != "base"):
                    skill_classes.append(obj)
        except Exception as e:
            print(f"  [Skill] 加载 {modname} 失败: {e}")
    return skill_classes


def select_skill(task: str, skill_classes: List[type] = None) -> Optional[type]:
    """为任务选择最匹配的 Skill

    Args:
        task: 用户任务描述
        skill_classes: 候选技能类列表（默认自动发现）

    Returns:
        匹配的技能类；无匹配返回 None
    """
    if skill_classes is None:
        skill_classes = get_base_skills()

    best_skill = None
    best_score = 0.0

    for skill in skill_classes:
        score = skill.match_score(task)
        if score > best_score:
            best_score = score
            best_skill = skill

    # 需要至少 1 个关键词命中（阈值 0）
    if best_skill is not None and best_score > 0:
        return best_skill
    return None


def build_skill_instruction(task: str, skill_classes: List[type] = None) -> str:
    """构建匹配技能的指令文本

    Returns:
        技能指令（无匹配返回空串）
    """
    skill = select_skill(task, skill_classes)
    if skill is None:
        return ""
    return skill.build_instruction()