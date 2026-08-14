"""
Browser Skill - 浏览器操作技能
==============================
处理浏览器相关的桌面任务：打开网页、搜索、下载、网页交互等。
"""
from skills.base import BaseSkill


class BrowserSkill(BaseSkill):
    """浏览器操作技能"""

    name = "browser"
    description = "浏览器操作技能：打开网页、搜索、下载、表单填写"
    match_keywords = [
        "网页", "浏览器", "网站", "搜索", "打开百度", "打开谷歌",
        "下载", "网页上", "在线",
        "browser", "web", "search", "website", "download", "chrome", "edge",
    ]
    goal = "完成浏览器相关的操作"
    available_tools = [
        "run_powershell", "visual_scan", "visual_scan_region",
        "visual_find_text", "visual_read_text", "visual_read_region",
        "click_at", "type_text", "press_key", "hotkey", "wait",
        "visual_locate", "visual_locate_region",
    ]
    avoid = []
    strategy = (
        "1. 打开浏览器：run_powershell Start-Process 或点击图标。\n"
        "2. 输入网址/搜索词：先点击地址栏/搜索框，再 type_text。\n"
        "3. 表单填写：OCR 找输入框 label，点击后用 type_text。\n"
        "4. 下载操作：点击下载按钮，等待完成后验证文件。"
    )
    verification = (
        "用 OCR 验证网页元素出现（搜索结果/页面标题/下载完成提示）。"
    )