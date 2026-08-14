"""
Skill 系统 - 领域技能包
=======================
- base.py: BaseSkill 基类 + 技能匹配器
- file_skill.py: 文件整理技能
- game_skill.py: 游戏操作技能
- browser_skill.py: 浏览器操作技能
"""
from skills.base import (
    BaseSkill,
    get_base_skills,
    select_skill,
    build_skill_instruction,
)

__all__ = [
    "BaseSkill",
    "get_base_skills",
    "select_skill",
    "build_skill_instruction",
]