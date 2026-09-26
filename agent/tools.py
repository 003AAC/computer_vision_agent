"""
Agent 工具定义
==============
核心工具：视觉分析（结构化JSON）+ 键鼠操作 + PowerShell
"""
import json
import subprocess
import time
import ctypes
import os

# Make Win32 and screenshot coordinates use the same DPI space.
if os.name == "nt":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

import pyautogui
import pyperclip
from langchain_core.tools import tool

from agent.action_result import action_result
from vision.engine import (
    scan_screen, scan_region, scan_grid,
    locate_element, locate_in_region,
)
from vision.ocr import ocr_screen, ocr_region, find_text_on_screen
from vision.engine import take_screenshot, take_region_screenshot
from vision.router import suggest_detection_strategy
from agent.tracker import ObjectTracker

# ============================================================
# 视觉对象记忆（全局单例：加速重复定位，不改变工具接口）
# ============================================================
_tracker = ObjectTracker(max_objects=30)


def _tracked_locate(target_description: str) -> Dict:
    """对象记忆加速的 locate_element

    首次：全屏定位 → 缓存结果
    再次：优先在缓存邻域小范围重定位 → 命中直接返回；未命中回退全屏
    """
    try:
        # 屏幕签名（检测页面变化）
        shot = take_screenshot()
        from agent.tracker import _image_signature
        sig = _image_signature(shot)

        # 缓存可用 → 邻域重扫
        if _tracker.should_use_cached(target_description, sig):
            cached = _tracker.get(target_description)
            if cached is not None:
                region = _tracker.get_neighborhood(target_description)
                if region:
                    x, y, w, h = region
                    # 邻域内用 locate_in_region（比全屏快）
                    result = locate_in_region(target_description, x, y, w, h)
                    if result.get("found"):
                        # 平滑更新缓存
                        bbox = result.get("bbox", [])
                        if len(bbox) == 4:
                            if not _tracker.offset_exceeded(target_description, bbox):
                                _tracker.track(
                                    target_description, bbox,
                                    result.get("confidence", 0.5), sig,
                                    source="tracked",
                                )
                        return result
                    else:
                        # 邻域未找到 → 缓存失效，回退全屏
                        _tracker.invalidate(target_description)

        # 全屏定位
        result = locate_element(target_description)
        if result.get("found"):
            bbox = result.get("bbox", [])
            if len(bbox) == 4:
                _tracker.track(
                    target_description, bbox,
                    result.get("confidence", 0.5), sig,
                    source="locate",
                )
        return result
    except Exception:
        # 任何异常回退原始定位
        return locate_element(target_description)


def _tracked_locate_region(target_description: str, x: int, y: int,
                           width: int, height: int) -> Dict:
    """对象记忆加速的 locate_in_region（区域定位同样缓存）"""
    try:
        region_shot = take_region_screenshot(x, y, width, height)
        from agent.tracker import _image_signature
        sig = _image_signature(region_shot)
        key = f"region:{target_description}:{x},{y}"

        if _tracker.should_use_cached(key, sig):
            cached = _tracker.get(key)
            if cached is not None:
                # 直接复用缓存的相对位置（区域未变）
                result = {
                    "found": True,
                    "x": cached.center[0],
                    "y": cached.center[1],
                    "bbox": cached.bbox,
                    "confidence": cached.confidence,
                    "cached": True,
                }
                return result
            _tracker.invalidate(key)

        result = locate_in_region(target_description, x, y, width, height)
        if result.get("found"):
            bbox = result.get("bbox", [])
            if len(bbox) == 4:
                _tracker.track(key, bbox, result.get("confidence", 0.5), sig,
                               source="locate_region")
        return result
    except Exception:
        return locate_in_region(target_description, x, y, width, height)


# ============================================================
# Win32 API 低级鼠标/键盘操作（比 pyautogui 更可靠）
# ============================================================

try:
    import win32api
    import win32con
    import win32gui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


def _win32_set_cursor_pos(x: int, y: int):
    """使用 Win32 API 移动鼠标，失败时回退 pyautogui"""
    if WIN32_AVAILABLE:
        try:
            win32api.SetCursorPos((x, y))
            return
        except Exception:
            pass  # Win32 失败，回退 pyautogui
    pyautogui.moveTo(x, y, duration=0.1)


def _win32_mouse_click(button: str = "left"):
    """使用 Win32 API 发送真实鼠标点击事件"""
    if not WIN32_AVAILABLE:
        pyautogui.click(
            button="right" if button == "right" else "left",
            clicks=2 if button == "double" else 1,
            interval=0.05,
        )
        return

    if button == "double":
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif button == "right":
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    else:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def _win32_send_key(key: str):
    """使用 Win32 API 发送按键事件"""
    if not WIN32_AVAILABLE:
        pyautogui.press(key)
        return

    # 映射 pyautogui 键名到虚拟键码
    KEY_MAP = {
        'enter': 0x0D, 'return': 0x0D,
        'tab': 0x09,
        'escape': 0x1B, 'esc': 0x1B,
        'backspace': 0x08,
        'delete': 0x2E, 'del': 0x2E,
        'space': 0x20,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
        'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
        'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
        'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
        'home': 0x24, 'end': 0x23, 'pageup': 0x21, 'pagedown': 0x22,
        'insert': 0x2D,
        'win': 0x5B, 'lwin': 0x5B, 'rwin': 0x5C,
        'alt': 0x12, 'lalt': 0xA4, 'ralt': 0xA5,
        'ctrl': 0x11, 'lctrl': 0xA2, 'rctrl': 0xA3,
        'shift': 0x10, 'lshift': 0xA0, 'rshift': 0xA1,
        'capslock': 0x14,
        '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
        '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
        'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
        'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
        'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
        'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
        'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
        ';': 0xBA, "'": 0xDE, ',': 0xBC, '.': 0xBE, '/': 0xBF,
        '\\': 0xDC, '[': 0xDB, ']': 0xDD, '-': 0xBD, '=': 0xBB,
        '`': 0xC0,
    }

    key_lower = key.lower().strip()
    vk = KEY_MAP.get(key_lower)
    if vk is None:
        # 尝试用 ord 处理单个字符
        if len(key) == 1:
            vk = ord(key.upper())
        else:
            vk = 0

    if vk:
        win32api.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
        win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
    else:
        pyautogui.press(key)


def _win32_send_text(text: str):
    """使用 Win32 API 发送文本（通过剪贴板 + Ctrl+V）"""
    if not WIN32_AVAILABLE:
        pyperclip.copy(text)
        pyautogui.hotkey('ctrl', 'v')
        return

    # 保存当前剪贴板内容
    old_clip = None
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            old_clip = root.clipboard_get()
        except tk.TclError:
            pass
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
    except Exception:
        pyperclip.copy(text)

    _win32_multi_key(['ctrl', 'v'])
    time.sleep(0.1)

    # 恢复旧剪贴板
    if old_clip is not None:
        try:
            pyperclip.copy(old_clip)
        except Exception:
            pass


def _win32_multi_key(keys: list):
    """按下组合键，如 ['ctrl', 'v']"""
    if not WIN32_AVAILABLE:
        pyautogui.hotkey(*keys)
        return

    MOD_KEYS = ['ctrl', 'lctrl', 'rctrl', 'alt', 'lalt', 'ralt',
                'shift', 'lshift', 'rshift', 'win', 'lwin', 'rwin']

    # 按下修饰键
    mod_vks = []
    for k in keys:
        k_lower = k.lower()
        if k_lower in MOD_KEYS:
            vk = _key_name_to_vk(k_lower)
            if vk:
                win32api.keybd_event(vk, 0, 0, 0)
                mod_vks.append(vk)

    # 按普通键
    for k in keys:
        k_lower = k.lower()
        if k_lower not in MOD_KEYS:
            vk = _key_name_to_vk(k_lower)
            if vk:
                win32api.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

    # 释放修饰键（反向顺序）
    for vk in reversed(mod_vks):
        if vk:
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)


def _key_name_to_vk(key: str) -> int:
    """将键名转为虚拟键码"""
    KEY_MAP = {
        'enter': 0x0D, 'return': 0x0D,
        'tab': 0x09,
        'escape': 0x1B, 'esc': 0x1B,
        'backspace': 0x08,
        'delete': 0x2E, 'del': 0x2E,
        'space': 0x20,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
        'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
        'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
        'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
        'home': 0x24, 'end': 0x23, 'pageup': 0x21, 'pagedown': 0x22,
        'insert': 0x2D,
        'win': 0x5B, 'lwin': 0x5B, 'rwin': 0x5C,
        'alt': 0x12, 'lalt': 0xA4, 'ralt': 0xA5,
        'ctrl': 0x11, 'lctrl': 0xA2, 'rctrl': 0xA3,
        'shift': 0x10, 'lshift': 0xA0, 'rshift': 0xA1,
        'capslock': 0x14,
        '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
        '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
        'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
        'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
        'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
        'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
        'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
        ';': 0xBA, "'": 0xDE, ',': 0xBC, '.': 0xBE, '/': 0xBF,
        '\\': 0xDC, '[': 0xDB, ']': 0xDD, '-': 0xBD, '=': 0xBB,
        '`': 0xC0,
    }
    return KEY_MAP.get(key, 0)


# ============================================================
# 视觉分析工具（返回结构化 JSON，杜绝自然语言描述）
# ============================================================

@tool
def visual_scan() -> str:
    """【扫描全屏】扫描整个屏幕，返回结构化 JSON。
    JSON 格式：
    {
      "screen": "1920x1080",
      "scene": "整体场景描述（一句话）",
      "elements": [
        {"label": "icon", "x": 100, "y": 200, "w": 48, "h": 48, "cx": 124, "cy": 224, "confidence": 0.85},
        ...
      ],
      "count": 元素总数
    }
    每个 element 包含：label(类型), x/y(左上角), w/h(宽高), cx/cy(中心点), confidence(置信度)
    你可以根据 cx/cy 直接点击，或根据 x/y/w/h 定位区域。
    如果全屏扫描不够详细，可以用 visual_scan_region() 放大指定区域。"""
    try:
        return scan_screen()
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def visual_scan_region(x: int, y: int, width: int, height: int) -> str:
    """【扫描区域】扫描屏幕指定区域（放大后分析），返回结构化 JSON。
    先用 visual_scan() 全屏扫描找到感兴趣的区域，再用此工具放大查看。
    x: 区域左上角横坐标
    y: 区域左上角纵坐标
    width: 区域宽度
    height: 区域高度
    返回 JSON 格式同 visual_scan()，坐标已转换为全屏绝对坐标。
    如果区域太小（<200px），会自动放大以便识别。"""
    try:
        return scan_region(x, y, width, height)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def visual_scan_grid(rows: int = 2, cols: int = 3) -> str:
    """【分块扫描】将屏幕分成网格，逐块扫描，返回结构化 JSON。
    适合屏幕内容复杂、全屏扫描不够详细时使用。
    例如 2x3 会把屏幕分成6块，每块单独分析。
    rows: 行数（默认2）
    cols: 列数（默认3）
    返回 JSON 包含每块的 scene 描述和该块内的 elements 列表。"""
    try:
        return scan_grid(rows, cols)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def visual_locate(target_description: str) -> str:
    """【定位元素】在屏幕上查找指定元素，返回精确坐标。
    target_description: 要查找的元素描述，如"开始菜单"、"搜索框"、"确定按钮"、"最小化按钮"、"原神"等。
    返回格式：找到则返回坐标(x, y)，未找到则提示。
    找到坐标后，用 click_at() 点击该位置。
    如果全屏找不到，可以用 visual_locate_region() 在指定区域内找。

    ⚠️ 复杂语义描述（如"游戏里的NPC上司"）建议改用 visual_scan() 先理解场景，再用 visual_locate_region() 在候选区域定位。"""
    try:
        # 优先：UI Automation 桌面图标定位（对图标名最可靠，不受壁纸/OCR影响）
        # 桌面图标是常见目标（"原神"、"Hearts of Iron"、"微信"），直接系统级精确查找
        try:
            from system.desktop_icons import find_desktop_icon
            uia = find_desktop_icon(target_description)
            if uia.get("found"):
                return (
                    f"✅ 桌面图标[{uia['name']}] -> 坐标({uia['x']}, {uia['y']})，"
                    f"方法: UI自动化，置信度: 1.00"
                )
        except Exception as e:
            print(f"  [UIA] 桌面图标定位异常: {e}")

        # Vision Router：复杂语义 → 先给 LLM 策略提示
        strategy = suggest_detection_strategy(target_description)
        if strategy.get("strategy") == "vlm_guided":
            return (
                f"⚠️ '{target_description}' 属于复杂语义描述（VLM任务）。"
                f"{strategy['hint']} "
                f"图形检测模型（DINO）不适合直接理解此描述。"
            )

        result = _tracked_locate(target_description)
        # 置信度阈值：过低视为视觉误检（复杂壁纸常产生低置信假目标）
        if result.get("found") and result.get("confidence", 0) >= 0.35:
            return (f"✅ 找到[{target_description}] -> 坐标({result['x']}, {result['y']})，"
                    f"置信度: {result['confidence']:.2f}")
        else:
            # OCR 引导兜底：先找文字锚点，再在附近检测图标
            try:
                from vision.engine import locate_icon_near_text
                guided = locate_icon_near_text(target_description)
                if guided.get("found"):
                    return (
                        f"✅ OCR引导找到[{target_description}] -> 坐标({guided['x']}, {guided['y']})，"
                        f"方法: {guided.get('method', '')}，"
                        f"锚点文字: {guided.get('anchor_text', '')}({guided.get('anchor_match', '')})，"
                        f"置信度: {guided.get('confidence', 0):.2f}"
                    )
            except Exception as e:
                print(f"  [OCR引导] 兜底定位异常: {e}")
            return f"❌ 未找到[{target_description}]"
    except Exception as e:
        return f"定位失败: {str(e)}"


@tool
def visual_locate_region(target_description: str, x: int, y: int,
                         width: int, height: int) -> str:
    """【区域定位】在屏幕指定区域内查找元素，适合找小图标。
    先用 visual_scan() 或 visual_scan_grid() 找到目标的大致区域，
    再用此工具在该区域内精确定位。
    target_description: 要查找的元素描述
    x, y: 区域左上角坐标
    width, height: 区域宽高
    返回的坐标是相对于全屏的绝对坐标，可直接用于 click_at()。"""
    try:
        result = _tracked_locate_region(target_description, x, y, width, height)
        if result.get("found"):
            return (f"✅ 在区域({x},{y},{width}x{height})中找到[{target_description}] "
                    f"-> 坐标({result['x']}, {result['y']})，"
                    f"置信度: {result['confidence']:.2f}")
        else:
            return f"❌ 在区域({x},{y},{width}x{height})中未找到[{target_description}]"
    except Exception as e:
        return f"区域定位失败: {str(e)}"


# ============================================================
# 鼠标操作工具（Win32 API 实现，更可靠）
# ============================================================

@tool
def click_at(x: int, y: int, button: str = "left") -> str:
    """【点击】在屏幕指定坐标执行鼠标点击。
    x: 横坐标（像素），y: 纵坐标（像素）。
    button: 鼠标按键，可选 'left'（左键）、'right'（右键）、'double'（双击），默认左键。
    先用 visual_locate() 或 visual_locate_region() 获取坐标，再用此工具点击。"""
    try:
        target_x, target_y = int(x), int(y)
        if button not in ("left", "right", "double"):
            return action_result("click_at", "failed", f"不支持的鼠标按键: {button}")

        screen_width, screen_height = pyautogui.size()
        if not (0 <= target_x < screen_width and 0 <= target_y < screen_height):
            return action_result(
                "click_at", "failed",
                f"目标坐标超出屏幕范围: ({target_x}, {target_y}), "
                f"屏幕尺寸: {screen_width}x{screen_height}",
            )

        # 移动鼠标（Win32 API 优先，失败自动回退 pyautogui）
        _win32_set_cursor_pos(target_x, target_y)
        time.sleep(0.15)

        # 执行点击（Win32 API 优先）
        _win32_mouse_click(button)

        # 验证位置
        if WIN32_AVAILABLE:
            after = win32api.GetCursorPos()
        else:
            after = pyautogui.position()

        detail = (
            f"点击事件已发送到 ({target_x}, {target_y}) "
            f"({'双击' if button == 'double' else '右键' if button == 'right' else '左键'}); "
            "界面效果尚未验证"
        )

        # 防误报检测：如果鼠标实际位置偏离目标太多，标记警告
        if abs(after[0] - target_x) > 50 or abs(after[1] - target_y) > 50:
            detail += f"；鼠标可能被拦截，实际位置({after[0]},{after[1]})"

        return action_result("click_at", "dispatched", detail)

    except Exception as e:
        return action_result("click_at", "failed", f"点击事件发送失败: {e}")


@tool
def drag_mouse(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> str:
    """【拖拽】从起点到终点拖拽鼠标。适用于拖拽文件、滑动滑块、选择文本等操作。"""
    try:
        if WIN32_AVAILABLE:
            # Win32 API 拖拽
            win32api.SetCursorPos((int(start_x), int(start_y)))
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.1)

            # 渐变移动
            steps = max(int(duration * 20), 5)
            for i in range(1, steps + 1):
                cx = int(start_x + (end_x - start_x) * i / steps)
                cy = int(start_y + (end_y - start_y) * i / steps)
                win32api.SetCursorPos((cx, cy))
                time.sleep(duration / steps)

            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            # pyautogui 降级
            pyautogui.moveTo(int(start_x), int(start_y), duration=0.2)
            time.sleep(0.1)
            pyautogui.drag(int(end_x) - int(start_x), int(end_y) - int(start_y), duration=float(duration))

        return action_result(
            "drag_mouse", "dispatched",
            f"拖拽事件已发送: ({start_x}, {start_y}) -> ({end_x}, {end_y}); "
            "界面效果尚未验证",
        )
    except Exception as e:
        return action_result("drag_mouse", "failed", f"拖拽事件发送失败: {e}")


# ============================================================
# 键盘操作工具（Win32 API 实现，更可靠）
# ============================================================

@tool
def type_text(text: str) -> str:
    """【输入】在当前光标位置输入文字。支持中文，通过剪贴板粘贴实现。"""
    try:
        _win32_send_text(text)
        display = text[:50] + ('...' if len(text) > 50 else '')
        return action_result(
            "type_text", "dispatched",
            f"输入事件已发送: [{display}]; 输入效果尚未验证",
        )
    except Exception as e:
        return action_result("type_text", "failed", f"输入事件发送失败: {e}")


@tool
def press_key(key: str) -> str:
    """【按键】按下并释放一个键盘按键。
    key: 键名，如 'enter', 'tab', 'escape', 'backspace', 'delete', 'f5', 'space',
          'win', 'alt', 'ctrl', 'shift', 'up', 'down', 'left', 'right' 等。"""
    try:
        _win32_send_key(key)
        return action_result(
            "press_key", "dispatched",
            f"按键事件已发送: {key}; 操作效果尚未验证",
        )
    except Exception as e:
        return action_result("press_key", "failed", f"按键事件发送失败: {e}")


@tool
def hotkey(keys: list) -> str:
    """【组合键】按下键盘组合键。如 ['ctrl', 'c'], ['win', 'd'], ['alt', 'tab'], ['win', 'r']。"""
    try:
        _win32_multi_key(keys)
        return action_result(
            "hotkey", "dispatched",
            f"组合键事件已发送: {'+'.join(keys)}; 操作效果尚未验证",
        )
    except Exception as e:
        return action_result("hotkey", "failed", f"组合键事件发送失败: {e}")


# ============================================================
# OCR 文字识别工具（解决"眼瞎"问题的核心）
# ============================================================

@tool
def visual_read_text() -> str:
    """【读文字】全屏 OCR——最后手段！优先用 visual_read_region() 或 visual_find_text()。
    
    全屏 OCR 非常慢（CPU 跑需要 2-5 秒），只在两种情况下使用：
    1. 完全不知道目标在哪里
    2. 其他方式都失败了
    
    正常情况下优先：
      - visual_find_text("确定") → 直接找到文字并返回点击坐标
      - visual_read_region(x,y,w,h) → 只扫目标区域，快 10 倍
    
    返回 JSON: {screen, texts: [{text, x, y, w, h, cx, cy}], count}"""
    try:
        return ocr_screen()
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def visual_read_region(x: int, y: int, width: int, height: int) -> str:
    """【区域读文字】OCR 读取屏幕指定区域内的文字。
    先用 visual_read_text() 或 visual_scan() 找到感兴趣区域，再用此工具放大读。
    返回 JSON 格式同 visual_read_text()，坐标已转换为全屏绝对坐标。
    x, y: 区域左上角坐标
    width, height: 区域宽高"""
    try:
        return ocr_region(x, y, width, height)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def visual_find_text(target_text: str) -> str:
    """【找文字】在屏幕上查找指定文字，返回精确点击坐标。
    三级匹配：精确→包含→模糊→token，找到后直接返回可点击的坐标。
    target_text: 要查找的文字，如 "确定"、"保存"、"文件"、"开始" 等。
    返回: 找到则返回坐标(x,y)，可直接用 click_at() 点击。
    这是点击"确定/取消/保存"等文字按钮的最可靠方式！
    ⚠️ 若返回的匹配文本是长句子（含命令/任务描述），说明匹配到的是窗口/终端文字而非图标名，需谨慎。"""
    try:
        result = find_text_on_screen(target_text)
        if result.get("found"):
            matched_text = result['text']
            match_type = result['match']
            x, y = result['x'], result['y']
            # 判断是否为"长文本命令回显"（很可能不是图标）
            is_long = len(matched_text) > 10
            # 判断是否为终端/窗口上下文（含任务描述常见词）
            echo_like = any(w in matched_text for w in
                            ["帮我", "打开桌面", "开始执行", "执行任务", "桌面的",
                             "请帮我", "任务", "python", "import", "执行"])
            note = ""
            if is_long or echo_like:
                note = (f" ⚠️注意：匹配到的是长文本[{matched_text[:20]}]，"
                        f"可能是窗口/终端文字而非目标图标名，请确认该坐标是否为目标位置")
            return (f"✅ OCR 找到文字[{matched_text}]（{match_type}匹配）"
                    f" -> 坐标({x}, {y})，"
                    f"可直接 click_at({x}, {y}) 点击{note}")
        else:
            hint = result.get("hint", "")
            return f"❌ OCR 未找到文字[{target_text}]。{hint}"
    except Exception as e:
        return f"OCR 查找失败: {str(e)}"


# ============================================================
# PowerShell 命令执行工具
# ============================================================

def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode('gbk')
    except (UnicodeDecodeError, AttributeError):
        try:
            return data.decode('utf-8', errors='replace')
        except Exception:
            return data.decode('gbk', errors='replace')


@tool
def run_powershell(command: str) -> str:
    """【PowerShell】执行 PowerShell 命令。
    适用于文件操作、启动程序、查询系统信息、环境变量等。
    命令示例：
      - 启动程序：Start-Process notepad
      - 创建文件：New-Item -Path "C:\\Users\\华硕\\Desktop\\test.txt" -ItemType File
      - 查看目录：Get-ChildItem C:\\Users\\华硕\\Desktop
      - 检查路径：Test-Path "C:\\Users\\华硕\\Desktop"
    command: 要执行的 PowerShell 命令"""
    try:
        import base64
        encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
        full_cmd = [
            "powershell", "-NoLogo", "-NoProfile", "-NonInteractive",
            "-EncodedCommand", encoded,
        ]

        if command.strip().lower().startswith('start-process ') or command.strip().lower().startswith('start '):
            proc = subprocess.Popen(
                full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(0.5)
            if proc.poll() is None:
                return "命令执行成功（GUI程序已启动）"
            return f"命令执行成功（退出码: {proc.returncode}）" if proc.returncode == 0 else f"命令执行失败，退出码: {proc.returncode}"

        result = subprocess.run(full_cmd, capture_output=True, timeout=15)
        output = _decode_bytes(result.stdout).strip()
        error = _decode_bytes(result.stderr).strip()

        if result.returncode == 0:
            return f"命令执行成功\n{output}" if output else "命令执行成功（无输出）"
        return f"命令执行失败\n错误: {error}\n退出码: {result.returncode}"
    except subprocess.TimeoutExpired:
        return "命令执行超时（15秒）"
    except Exception as e:
        return f"命令执行异常: {str(e)}"


# ============================================================
# 辅助工具
# ============================================================

@tool
def wait(seconds: float = 1.0) -> str:
    """【等待】等待指定秒数。用于等待界面加载、动画完成等。"""
    time.sleep(float(seconds))
    return f"已等待 {seconds} 秒"


# ============================================================
# 工具注册表
# ============================================================

ALL_TOOLS = [
    visual_scan, visual_scan_region, visual_scan_grid,
    visual_locate, visual_locate_region,
    visual_read_text, visual_read_region, visual_find_text,
    click_at, drag_mouse,
    type_text, press_key, hotkey,
    run_powershell,
    wait,
]

TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}