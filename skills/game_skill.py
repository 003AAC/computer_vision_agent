"""
Game Skill - 游戏操作技能
==========================
处理游戏相关的桌面任务：启动游戏、操作菜单、游戏内交互等。
"""
from skills.base import BaseSkill


class GameSkill(BaseSkill):
    """游戏操作技能"""

    name = "game"
    description = "游戏操作技能：启动游戏、操作菜单、处理游戏内界面"
    match_keywords = [
        "游戏", "游玩", "打开游戏", "开始游戏", "进入游戏",
        "game", "play", "launch", "start",
    ]
    goal = "完成游戏的启动和游玩操作"
    available_tools = [
        "visual_scan", "visual_scan_region", "visual_scan_grid",
        "visual_locate", "visual_locate_region",
        "visual_find_text", "visual_read_text", "visual_read_region",
        "click_at", "press_key", "hotkey", "wait", "run_powershell",
    ]
    avoid = ["type_text"]  # 游戏中通常不需要输入长文本
    strategy = (
        "1. 启动游戏：用 run_powershell Start-Process 或桌面图标点击。\n"
        "2. 菜单操作：OCR 找按钮文字（开始游戏/继续/设置），再点击。\n"
        "3. 复杂语义（NPC等）：先 visual_scan 理解场景，再区域定位。\n"
        "4. 游戏加载需耐心：操作后等待，再验证界面变化。"
    )
    verification = (
        "通过 OCR 确认游戏界面元素（主菜单/加载画面/HUD）出现。"
    )