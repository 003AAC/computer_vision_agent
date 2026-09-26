"""
Agent 工具定义
==============
核心工具：视觉分析（结构化JSON）+ 键鼠操作 + PowerShell
"""
import json
import subprocess
import time
import ctypes

import pyautogui
import pyperclip
from langchain_core.tools import tool

from vision.engine import (
    scan_screen, scan_region, scan_grid,
    locate_element, locate_in_region,
)
from vision.ocr import ocr_screen, ocr_region, find_text_on_screen
from vision.engine import take_screenshot, take_region_screenshot
from vision.router import suggest_detection_strategy
from agent.tracker import ObjectTracker
from system import input_controller as input_ctl

# ============================================================
# 视觉对象记忆（全局单例：加速重复定位，不改变工具接口）
# ============================================================
_tracker = ObjectTracker(max_objects=30)


def _current_foreground() -> str:
    """当前前台窗口标题（用于对象缓存的"跨窗口失效"判断）"""
    try:
        return input_ctl.get_window_title(input_ctl.get_foreground_window()) or ""
    except Exception:
        return ""


def _tracked_locate(target_description: str) -> Dict:
    """对象记忆加速的 locate_element

    首次：全屏定位 → 缓存结果（含前台窗口标题）
    再次：仅当"页面未变 且 前台窗口一致"才在缓存邻域小范围重定位
    """
    try:
        # 屏幕签名（检测页面变化）
        shot = take_screenshot()
        from agent.tracker import _image_signature
        sig = _image_signature(shot)
        fg = _current_foreground()

        # 缓存可用（页面未变 + 前台窗口一致）→ 邻域重扫
        if _tracker.should_use_cached(target_description, sig, fg):
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
                                    source="tracked", foreground=fg,
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
                    source="locate", foreground=fg,
                )
        return result
    except Exception:
        # 任何异常回退原始定位
        return locate_element(target_description)


def _tracked_locate_region(target_description: str, x: int, y: int,
                           width: int, height: int) -> Dict:
    """对象记忆加速的 locate_in_region（区域定位同样缓存）

    安全约束（P4）：仅当"区域截图未变 且 前台窗口一致 且 缓存新鲜(≤2s)"
    才直接复用缓存坐标；否则重新做区域定位，避免拿过期坐标去点。
    """
    try:
        region_shot = take_region_screenshot(x, y, width, height)
        from agent.tracker import _image_signature
        sig = _image_signature(region_shot)
        fg = _current_foreground()
        key = f"region:{target_description}:{x},{y}"

        if _tracker.should_use_cached(key, sig, fg):
            cached = _tracker.get(key)
            if cached is not None and (time.time() - cached.last_seen) <= 2.0:
                # 复用缓存的相对位置（区域未变 + 窗口一致 + 新鲜）
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
                               source="locate_region", foreground=fg)
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


# ============================================================
# 输入薄包装（已重构：全部委托 system.input_controller）
# ------------------------------------------------------------
# 旧实现直接用已废弃的 mouse_event / keybd_event，且
#   - 不激活前台窗口 → 首次点击被吞
#   - 不做权限(UIPI)预检 → 提权窗口静默丢弃输入
#   - 不做到位校验 → 输入无效仍返回成功
# 现统一委托 SendInput 实现，以下包装仅为向后兼容保留。
# ============================================================

def _win32_set_cursor_pos(x: int, y: int):
    """移动鼠标（SendInput 绝对定位 + 到位校验）"""
    r = input_ctl.move_to(x, y)
    if not r.get("ok") and not WIN32_AVAILABLE:
        pyautogui.moveTo(x, y, duration=0.1)
    return r


def _win32_mouse_click(button: str = "left"):
    """点击（SendInput；double 走双击序列）"""
    b = "right" if button == "right" else "left"
    return input_ctl.click(b, double=(button == "double"))


def _win32_send_key(key: str):
    """按键（委托 input_controller：自动为方向键/编辑键加扩展标志）"""
    return input_ctl.press_key(key)


def _win32_send_text(text: str):
    """输入文本（委托 input_controller：pyperclip 持久化 + 校验 + 粘贴等待）

    旧实现的问题（已修）：
      - 依赖 Tk 实例存活，实例销毁后剪贴板内容可能丢失
      - 粘贴后仅 sleep(0.1) 就还原旧剪贴板 → 慢应用会贴成旧内容
    """
    return input_ctl.type_text(text)


def _win32_multi_key(keys: list):
    """组合键（委托 input_controller：修饰键先按后放，扩展键带标志）"""
    return input_ctl.hotkey(keys)


def _key_name_to_vk(key: str) -> int:
    """键名 → 虚拟键码（委托 input_controller，唯一映射源）"""
    return input_ctl.key_name_to_vk(key)


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
# 鼠标操作工具（SendInput + 前台激活 + 权限预检 + 到位校验）
# ============================================================

# ============================================================
# 假点击检测：点击后界面是否真的发生变化
# ------------------------------------------------------------
# 现象：SendInput 报"事件已送达"，但目标程序并未响应
#   （窗口未真正聚焦 / 反作弊 / 目标已消失），于是返回 "✅ 已点击"
#   却毫无效果 —— 即"假点击"。更糟的是 Agent 会反复点同一坐标。
# 这里用"点击前后画面签名"做客观判定，并把"同一坐标重复点击且始终
# 无变化"升级为**失败**，确保 Agent 不会陷在假点击循环里。
# ============================================================
_LAST_CLICK = {"coord": None, "sig_before": "", "sig_after": ""}
_CLICK_EFFECT_WAIT = 0.35


def _screen_sig() -> str:
    """当前屏幕的轻量签名（用于判断界面是否变化）"""
    try:
        from agent.tracker import _image_signature
        return _image_signature(take_screenshot())
    except Exception:
        return ""


def reset_fake_click_state():
    """重置假点击检测状态（新任务开始时调用）"""
    _LAST_CLICK["coord"] = None
    _LAST_CLICK["sig_before"] = ""
    _LAST_CLICK["sig_after"] = ""


@tool
def click_at(x: int, y: int, button: str = "left") -> str:
    """【点击】在屏幕指定坐标执行鼠标点击（含真实性校验）。
    x: 横坐标（像素），y: 纵坐标（像素）。
    button: 鼠标按键，可选 'left'（左键）、'right'（右键）、'double'（双击），默认左键。
    系统会自动：激活目标窗口 → SendInput 绝对定位 → 校验光标到位 → 点击
    → 对比点击前后画面判断是否真的生效。
    若目标窗口以管理员权限运行而本程序未提权（UIPI 会丢弃输入），
    或同一坐标重复点击且界面始终无变化，将返回明确失败而非假成功。"""
    try:
        target_x, target_y = int(x), int(y)

        # 点击前画面签名（用于判断点击是否真的产生效果）
        sig_before = _screen_sig()

        r = input_ctl.click_at_point(
            target_x, target_y, button=("right" if button == "right" else "left"),
            double=(button == "double"),
        )

        label = ("双击" if button == "double"
                 else ("右键" if button == "right" else "左键"))

        if not r.get("ok"):
            reason = r.get("error", "未知原因")
            extra = " [需以管理员身份运行本程序]" if r.get("uipi_conflict") else ""
            return f"点击失败: {reason}{extra}"

        # 等待界面响应后取第二次签名
        time.sleep(_CLICK_EFFECT_WAIT)
        sig_after = _screen_sig()

        # —— 假点击判定：上一次同坐标点击也没引起任何变化 ——
        prev = _LAST_CLICK
        repeat_ineffective = False
        try:
            if (prev["coord"] and prev["sig_before"] and prev["sig_after"]
                    and prev["sig_before"] == prev["sig_after"]):
                px, py = prev["coord"]
                if abs(px - target_x) <= 8 and abs(py - target_y) <= 8:
                    repeat_ineffective = True
        except Exception:
            pass

        _LAST_CLICK["coord"] = (target_x, target_y)
        _LAST_CLICK["sig_before"] = sig_before
        _LAST_CLICK["sig_after"] = sig_after

        if repeat_ineffective:
            return (
                f"点击失败: 同一坐标 ({target_x}, {target_y}) 重复点击，"
                f"且界面始终没有任何变化 —— 点击未生效（假点击）。\n"
                f"请**立即停止重复点击**：重新观察屏幕确认目标是否仍然存在"
                f"（visual_find_text / visual_read_region），"
                f"或改用 press_key / hotkey / run_powershell 换一条路径。"
            )

        result = f"✅ 已点击 ({target_x}, {target_y}) {label}"
        fg = r.get("foreground", "")
        if fg:
            result += f" [前台窗口: {fg[:40]}]"
        if r.get("foreground_warning"):
            result += f" [警告：{r['foreground_warning']}]"
        # 单次点击无变化 → 给出显式警告（可能是慢加载，也可能是目标无效）
        if sig_before and sig_after and sig_before == sig_after:
            result += " [警告: 点击后界面无变化，若下一步仍无进展请换方式]"
        return result

    except Exception as e:
        return f"点击失败: {str(e)}"


@tool
def drag_mouse(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> str:
    """【拖拽】从起点到终点拖拽鼠标（SendInput 实现 + 起点到位校验）。
    适用于拖拽文件、滑动滑块、选择文本等操作。"""
    try:
        r = input_ctl.drag(int(start_x), int(start_y),
                           int(end_x), int(end_y), duration=float(duration))
        if not r.get("ok"):
            return f"拖拽失败: {r.get('error', '未知原因')}"
        return f"✅ 已从 ({start_x}, {start_y}) 拖拽到 ({end_x}, {end_y})"
    except Exception as e:
        return f"拖拽失败: {str(e)}"


# ============================================================
# 键盘操作工具（SendInput + 扩展键 + 发送校验）
# ============================================================

@tool
def type_text(text: str) -> str:
    """【输入】在当前光标位置输入文字（支持中文，经剪贴板粘贴）。
    内部会校验剪贴板写入并在粘贴完成后才还原旧剪贴板。
    若目标窗口为管理员权限而本程序未提权，将返回明确失败。"""
    try:
        r = input_ctl.type_text(text)
        if not r.get("ok"):
            return f"输入失败: {r.get('error', '未知原因')}"
        display = text[:50] + ('...' if len(text) > 50 else '')
        return f"✅ 已输入: [{display}]"
    except Exception as e:
        return f"输入失败: {str(e)}"


@tool
def press_key(key: str) -> str:
    """【按键】按下并释放一个键盘按键（方向键/编辑键自动带扩展标志）。
    key: 键名，如 'enter', 'tab', 'escape', 'backspace', 'delete', 'f5', 'space',
          'win', 'alt', 'ctrl', 'shift', 'up', 'down', 'left', 'right' 等。"""
    try:
        r = input_ctl.press_key(key)
        if not r.get("ok"):
            return f"按键失败: {r.get('error', '事件未送达')} (key={key})"
        ext = " [扩展键]" if r.get("extended") else ""
        return f"✅ 已按下 {key}{ext}"
    except Exception as e:
        return f"按键失败: {str(e)}"


@tool
def hotkey(keys: list) -> str:
    """【组合键】按下键盘组合键。如 ['ctrl', 'c'], ['win', 'd'], ['alt', 'tab'], ['win', 'r']。"""
    try:
        r = input_ctl.hotkey(keys)
        joined = '+'.join(str(k) for k in keys)
        if not r.get("ok"):
            return f"组合键失败: {r.get('error', '事件未送达')} ({joined})"
        return f"✅ 已按下 {joined}"
    except Exception as e:
        return f"组合键失败: {str(e)}"


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
        full_cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}'

        if command.strip().lower().startswith('start-process ') or command.strip().lower().startswith('start '):
            proc = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            if proc.poll() is None:
                return "命令执行成功（GUI程序已启动）"
            return f"命令执行成功（退出码: {proc.returncode}）" if proc.returncode == 0 else f"命令执行失败，退出码: {proc.returncode}"

        result = subprocess.run(full_cmd, shell=True, capture_output=True, timeout=15)
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