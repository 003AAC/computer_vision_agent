"""
Desktop Icons - 桌面图标定位（UI Automation）
=============================================
通过 pywinauto (UIA) 直接读取 Windows 桌面图标名称与屏幕坐标。
这是桌面图标定位的最可靠方案：
  - 不受花哨壁纸影响（不依赖 OCR/视觉识别）
  - 精确到像素（系统级坐标）
  - 图标名 100% 准确（来自 Windows 资源管理器）

设计背景：
  EasyOCR 对花哨壁纸上的中文小图标识别率极低（如 "原神" → "IHaalts@f"），
  Florence-2 对桌面小图标检测也不可靠。而 UI Automation 直接从系统
  获取图标名称 + 矩形坐标，零识别误差。

用法：
  from system.desktop_icons import find_desktop_icon, list_desktop_icons

  find_desktop_icon("原神") -> {"found": True, "name": "原神", "x": 342, "y": 341, ...}
  list_desktop_icons() -> [{"name": ..., "x": ..., "y": ...}, ...]
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_icon_items():
    """获取桌面图标元素列表（pywinauto UIA）

    Returns:
        list of pywinauto ListItem；失败返回 []
    """
    try:
        from pywinauto import Desktop
        desktop = Desktop(backend="uia")
        progman = desktop.window(title_re=".*Program Manager.*")
        if not progman.exists():
            return []
        items = progman.descendants(control_type="ListItem")
        return items or []
    except Exception as e:
        logger.warning(f"pywinauto 获取桌面图标失败: {e}")
        return []


def list_desktop_icons() -> List[Dict[str, Any]]:
    """列出所有桌面图标及其屏幕坐标

    Returns:
        [{"name": str, "x": int, "y": int, "bbox": [l,t,r,b]}, ...]
        按屏幕位置排序（从上到下、从左到右）
    """
    items = _get_icon_items()
    icons = []
    for it in items:
        try:
            name = it.window_text()
            if not name or name in ("desktop.ini",):
                continue
            r = it.rectangle()
            cx = r.left + (r.right - r.left) // 2
            cy = r.top + (r.bottom - r.top) // 2
            icons.append({
                "name": name,
                "x": cx, "y": cy,
                "bbox": [r.left, r.top, r.right, r.bottom],
            })
        except Exception:
            continue
    # 排序：y 优先（行），x 其次（列）
    icons.sort(key=lambda i: (i["y"] // 50, i["x"]))
    return icons


def find_desktop_icon(name: str) -> Dict[str, Any]:
    """按名称查找桌面图标

    匹配策略：
      1. 精确匹配（完全同名）
      2. 包含匹配（目标名是图标名的子串，或反之）
      3. token 匹配（"原神" 或 "Genshin" 命中即返回）

    Args:
        name: 图标名（如 "原神"、"Hearts of Iron IV"、"米哈游启动器"）

    Returns:
        {"found": bool, "name": str, "x": int, "y": int,
         "bbox": [l,t,r,b], "match": str}
    """
    if not name or not name.strip():
        return {"found": False, "name": name, "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0], "match": ""}

    icons = list_desktop_icons()
    target = name.strip().lower()

    # 1. 精确匹配
    for i in icons:
        if i["name"].lower() == target:
            return {"found": True, **i, "match": "exact"}

    # 2. 包含匹配（目标名出现在图标名中，或图标名出现在目标名中）
    for i in icons:
        iname = i["name"].lower()
        if target in iname or iname in target:
            return {"found": True, **i, "match": "partial"}

    # 3. token 匹配（目标含多个词，任一完整词命中）
    import re as _re
    words = [w for w in _re.split(r"[\s.\-_]+", target) if len(w) >= 2]
    best = None
    for i in icons:
        iname = i["name"].lower()
        hits = sum(1 for w in words if w in iname)
        if hits >= 1 and (best is None or hits > best["hits"]):
            best = {**i, "hits": hits}
    if best:
        best.pop("hits")
        return {"found": True, **best, "match": "token"}

    return {"found": False, "name": name, "x": 0, "y": 0,
            "bbox": [0, 0, 0, 0], "match": ""}
