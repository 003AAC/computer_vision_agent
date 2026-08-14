"""
File Skill - 文件整理技能
==========================
处理文件相关的桌面任务：整理文件、重命名、移动、创建文件夹等。
"""
from skills.base import BaseSkill


class FileSkill(BaseSkill):
    """文件整理技能"""

    name = "file"
    description = "文件整理技能：处理文件/文件夹的创建、移动、删除、重命名"
    match_keywords = [
        "文件", "文件夹", "整理", "重命名", "移动文件", "创建文件夹",
        "删除文件", "复制文件", "保存到", "下载到",
        "file", "folder", "desktop", "documents", "rename", "organize",
    ]
    goal = "完成文件/文件夹的管理操作"
    available_tools = [
        "run_powershell", "visual_scan", "visual_scan_region",
        "visual_find_text", "click_at", "drag_mouse", "hotkey", "wait",
        "visual_locate", "visual_locate_region", "visual_read_text", "visual_read_region",
    ]
    avoid = ["type_text"]  # 文件操作通常用 PowerShell 更可靠，避免手动输入长路径
    strategy = (
        "1. 优先使用 run_powershell 执行文件操作（更可靠）。\n"
        "2. 需要可视化操作时，先 OCR 定位目标文件/文件夹。\n"
        "3. 拖拽移动用 drag_mouse，注意起点终点坐标准确。"
    )
    verification = (
        "用 run_powershell 的 Test-Path / Get-ChildItem 验证文件是否存在及状态。"
    )