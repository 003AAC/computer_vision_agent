"""
Input Controller - Win32 SendInput 键鼠控制器
============================================
替代已废弃的 mouse_event / keybd_event，解决"键鼠输入无效却返回假成功"的问题。

关键能力：
  1. SendInput 绝对定位：MOUSEEVENTF_MOVE|MOUSEEVENTF_ABSOLUTE
     （0..65535 归一化，支持多显示器/虚拟屏原点）
  2. 扩展键自动加 KEYEVENTF_EXTENDEDKEY（方向键/编辑键/右 Ctrl/Alt）
  3. 前台窗口激活：ShowWindow(SW_RESTORE) + SetForegroundWindow
     （AttachThreadInput 兜底）——窗口不在前台时首次点击会被吞掉
  4. 权限自检：目标窗口提权而本进程未提权 → UIPI 会静默丢弃输入，
     此时**明确报错**而不是返回成功
  5. DPI 感知声明：避免非 100% 缩放下坐标错位
  6. 真实性校验：移动后比对 GetCursorPos，未到达目标位置即判失败
"""
import ctypes
import time
from ctypes import wintypes
from typing import Any, Dict, List, Optional, Tuple

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
try:
    _advapi32 = ctypes.windll.advapi32
except Exception:      # pragma: no cover
    _advapi32 = None

# ============================================================
# SendInput 结构体定义
# ============================================================
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 \
    else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


# ============================================================
# 常量
# ============================================================
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

SW_RESTORE = 9
SW_SHOW = 5

# 需要扩展标志的虚拟键（方向键/编辑键/右修饰键等）
VK_EXTENDED = {
    0x21,  # pageup
    0x22,  # pagedown
    0x23,  # end
    0x24,  # home
    0x25,  # left
    0x26,  # up
    0x27,  # right
    0x28,  # down
    0x2C,  # printscreen
    0x2D,  # insert
    0x2E,  # delete
    0x6F,  # numpad divide
    0x90,  # numlock
    0xA3,  # right ctrl
    0xA5,  # right alt
}

# 键名 → 虚拟键码（唯一映射源）
KEY_MAP = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09,
    "escape": 0x1B, "esc": 0x1B, "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "space": 0x20,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "insert": 0x2D,
    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
    "alt": 0x12, "lalt": 0xA4, "ralt": 0xA5,
    "ctrl": 0x11, "lctrl": 0xA2, "rctrl": 0xA3,
    "shift": 0x10, "lshift": 0xA0, "rshift": 0xA1,
    "capslock": 0x14,
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
    "f": 0x46, "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A,
    "k": 0x4B, "l": 0x4C, "m": 0x4D, "n": 0x4E, "o": 0x4F,
    "p": 0x50, "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
    "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58, "y": 0x59, "z": 0x5A,
    ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
    "\\": 0xDC, "[": 0xDB, "]": 0xDD, "-": 0xBD, "=": 0xBB,
    "`": 0xC0,
}

MOD_KEYS = {"ctrl", "lctrl", "rctrl", "alt", "lalt", "ralt",
            "shift", "lshift", "rshift", "win", "lwin", "rwin"}


def key_name_to_vk(key: str) -> int:
    """键名 → 虚拟键码（0 表示无法识别）"""
    k = str(key).lower().strip()
    if k in KEY_MAP:
        return KEY_MAP[k]
    if len(str(key)) == 1:
        return ord(str(key).upper())
    return 0


def is_extended_vk(vk: int) -> bool:
    """该虚拟键是否需要 KEYEVENTF_EXTENDEDKEY 标志"""
    return vk in VK_EXTENDED


# ============================================================
# DPI 感知
# ============================================================
_DPI_DONE = False


def set_dpi_awareness() -> bool:
    """声明进程 DPI 感知，避免非 100% 缩放下坐标错位

    依次尝试：PerMonitorV2 → PerMonitor → System DPI aware
    """
    global _DPI_DONE
    if _DPI_DONE:
        return True
    ok = False
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            ok = True
    except Exception:
        pass
    if not ok:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            ok = True
        except Exception:
            pass
    if not ok:
        try:
            ok = bool(_user32.SetProcessDPIAware())
        except Exception:
            ok = False
    _DPI_DONE = True
    return ok


# ============================================================
# 坐标归一化（绝对定位）
# ============================================================

def get_virtual_screen() -> Tuple[int, int, int, int]:
    """虚拟屏 (left, top, width, height)（多显示器合并范围）"""
    try:
        x = _user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        y = _user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = _user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = _user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    except Exception:
        x, y, w, h = 0, 0, 0, 0
    if w <= 1 or h <= 1:
        w = _user32.GetSystemMetrics(0) or 1920
        h = _user32.GetSystemMetrics(1) or 1080
        x, y = 0, 0
    return x, y, w, h


def to_absolute(x: int, y: int) -> Tuple[int, int]:
    """屏幕像素坐标 → SendInput 绝对坐标 (0..65535)"""
    vx, vy, vw, vh = get_virtual_screen()
    nx = int(round((x - vx) * 65535 / max(vw - 1, 1)))
    ny = int(round((y - vy) * 65535 / max(vh - 1, 1)))
    nx = max(0, min(65535, nx))
    ny = max(0, min(65535, ny))
    return nx, ny


# ============================================================
# 底层 SendInput
# ============================================================

def _send(inputs: List[INPUT]) -> int:
    """发送输入事件，返回成功发送的条数"""
    if not inputs:
        return 0
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    try:
        _user32.SendInput.argtypes = [
            wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int
        ]
        _user32.SendInput.restype = wintypes.UINT
    except Exception:
        pass
    return int(_user32.SendInput(n, arr, ctypes.sizeof(INPUT)))


def _mouse_input(flags: int, dx: int = 0, dy: int = 0,
                 data: int = 0) -> INPUT:
    mi = MOUSEINPUT(dx, dy, data, flags, 0, 0)
    return INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=mi))


def _key_input(vk: int, flags: int = 0) -> INPUT:
    ki = KEYBDINPUT(vk, 0, flags, 0, 0)
    return INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=ki))


# ============================================================
# 鼠标操作
# ============================================================

def get_cursor_pos() -> Tuple[int, int]:
    """当前光标屏幕坐标"""
    pt = wintypes.POINT()
    try:
        _user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    except Exception:
        return -1, -1


def move_to(x: int, y: int, tolerance: int = 2) -> Dict[str, Any]:
    """移动鼠标到屏幕像素坐标（SendInput 绝对定位 + 到位校验）

    Returns:
        {"ok": bool, "x": int, "y": int, "actual": [x, y], "error": str}
    """
    set_dpi_awareness()
    x, y = int(x), int(y)
    nx, ny = to_absolute(x, y)
    sent = _send([_mouse_input(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny
    )])
    if sent < 1:
        return {
            "ok": False, "x": x, "y": y, "actual": list(get_cursor_pos()),
            "error": ("SendInput 移动失败（可能被 UIPI 拦截或桌面被锁定）"),
        }
    ax, ay = get_cursor_pos()
    if abs(ax - x) > tolerance or abs(ay - y) > tolerance:
        return {
            "ok": False, "x": x, "y": y, "actual": [ax, ay],
            "error": (f"光标未到位（目标 ({x},{y}) 实际 ({ax},{ay})，"
                      f"可能有程序用 ClipCursor 限制光标）"),
        }
    return {"ok": True, "x": x, "y": y, "actual": [ax, ay], "error": ""}


_BUTTON_FLAGS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


def mouse_down(button: str = "left") -> bool:
    down, _ = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    return _send([_mouse_input(down)]) >= 1


def mouse_up(button: str = "left") -> bool:
    _, up = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    return _send([_mouse_input(up)]) >= 1


def click(button: str = "left", double: bool = False,
          interval: float = 0.06) -> Dict[str, Any]:
    """在当前光标处点击（按下 + 抬起）

    Args:
        button: left / right / middle
        double: 是否双击
        interval: 双击两次之间的间隔秒数
    """
    down, up = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    if double:
        first = _send([_mouse_input(down), _mouse_input(up)])
        time.sleep(interval)
        second = _send([_mouse_input(down), _mouse_input(up)])
        sent = first + second
        expect = 4
    else:
        sent = _send([_mouse_input(down), _mouse_input(up)])
        expect = 2
    return {
        "ok": sent == expect, "sent": int(sent), "expect": expect,
        "error": "" if sent == expect else
        f"SendInput 点击事件未全部送达（{sent}/{expect}）",
    }


def drag(x1: int, y1: int, x2: int, y2: int,
         duration: float = 0.5) -> Dict[str, Any]:
    """拖拽：绝对定位到起点 → 按下 → 分步移动 → 抬起"""
    set_dpi_awareness()
    start = move_to(x1, y1)
    if not start.get("ok"):
        return {"ok": False, "error": f"拖拽起点移动失败: {start.get('error')}"}
    time.sleep(0.05)
    if not mouse_down("left"):
        return {"ok": False, "error": "按下左键失败"}
    time.sleep(0.05)

    steps = max(int(duration * 20), 5)
    for i in range(1, steps + 1):
        cx = int(x1 + (x2 - x1) * i / steps)
        cy = int(y1 + (y2 - y1) * i / steps)
        nx, ny = to_absolute(cx, cy)
        _send([_mouse_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny)])
        time.sleep(max(duration / steps, 0.005))

    time.sleep(0.05)
    if not mouse_up("left"):
        return {"ok": False, "error": "抬起左键失败"}
    ax, ay = get_cursor_pos()
    return {"ok": True, "actual": [ax, ay], "error": ""}


def scroll(clicks: int) -> Dict[str, Any]:
    """滚轮滚动（正数向上，负数向下，单位 120）"""
    data = int(clicks) * 120
    sent = _send([_mouse_input(MOUSEEVENTF_WHEEL, 0, 0, data)])
    return {"ok": sent >= 1, "error": "" if sent >= 1 else "滚轮事件发送失败"}


# ============================================================
# 键盘操作
# ============================================================

def _ext_flag(vk: int) -> int:
    """虚拟键对应的扩展标志（方向键/编辑键/右修饰键必须带）"""
    return KEYEVENTF_EXTENDEDKEY if is_extended_vk(vk) else 0


def key_down(key: str) -> bool:
    """按下按键（不释放）"""
    vk = key_name_to_vk(key)
    if not vk:
        return False
    return _send([_key_input(vk, _ext_flag(vk))]) >= 1


def key_up(key: str) -> bool:
    """释放按键"""
    vk = key_name_to_vk(key)
    if not vk:
        return False
    return _send([_key_input(vk, _ext_flag(vk) | KEYEVENTF_KEYUP)]) >= 1


def press_key(key: str, hold: float = 0.03) -> Dict[str, Any]:
    """按下并释放一个键

    Returns:
        {"ok": bool, "vk": int, "extended": bool, "error": str}
    """
    vk = key_name_to_vk(key)
    if not vk:
        return {"ok": False, "vk": 0, "extended": False,
                "error": f"未知键名: {key}"}
    ext = _ext_flag(vk)
    down = _send([_key_input(vk, ext)])
    time.sleep(max(hold, 0.01))
    up = _send([_key_input(vk, ext | KEYEVENTF_KEYUP)])
    ok = down >= 1 and up >= 1
    return {
        "ok": ok, "vk": vk, "extended": bool(ext),
        "error": "" if ok else "SendInput 按键事件未送达（可能被 UIPI 拦截）",
    }


def hotkey(keys: List[str]) -> Dict[str, Any]:
    """按下组合键，如 ['ctrl','c']、['win','r']、['alt','tab']

    修饰键先按下 → 普通键按下并释放 → 修饰键反序释放
    """
    if not keys:
        return {"ok": False, "error": "组合键为空"}

    vks = []
    unknown = []
    for k in keys:
        vk = key_name_to_vk(k)
        if not vk:
            unknown.append(str(k))
        else:
            vks.append((str(k).lower().strip(), vk))
    if unknown:
        return {"ok": False, "error": f"未知键名: {', '.join(unknown)}"}

    mods = [vk for name, vk in vks if name in MOD_KEYS]
    normals = [vk for name, vk in vks if name not in MOD_KEYS]

    total = 0
    expect = 0

    # 1) 修饰键按下
    if mods:
        seq = [_key_input(vk, _ext_flag(vk)) for vk in mods]
        total += _send(seq)
        expect += len(seq)
        time.sleep(0.02)

    # 2) 普通键按下+释放
    if normals:
        seq = []
        for vk in normals:
            f = _ext_flag(vk)
            seq.append(_key_input(vk, f))
            seq.append(_key_input(vk, f | KEYEVENTF_KEYUP))
        total += _send(seq)
        expect += len(seq)
        time.sleep(0.02)

    # 3) 修饰键反序释放
    if mods:
        seq = [_key_input(vk, _ext_flag(vk) | KEYEVENTF_KEYUP)
               for vk in reversed(mods)]
        total += _send(seq)
        expect += len(seq)

    ok = total == expect
    return {
        "ok": ok, "sent": int(total), "expect": expect,
        "error": "" if ok else
        f"组合键事件未全部送达（{total}/{expect}，可能被 UIPI 拦截）",
    }


def type_text(text: str, settle: float = 0.2,
              restore_clipboard: bool = True) -> Dict[str, Any]:
    """通过剪贴板粘贴输入文本（支持中文）

    相比旧实现的改进：
      - 使用 pyperclip（内容持久化到 Win32 剪贴板，不依赖 Tk 实例存活）
      - 写入后**校验剪贴板内容**，避免被其他程序抢占导致贴错
      - 粘贴后等待 settle 秒（默认 0.2s）再恢复旧剪贴板，
        避免"目标应用还没粘贴完就被还原"→ 贴成旧内容
    """
    if text is None:
        text = ""
    try:
        import pyperclip
    except Exception as e:
        return {"ok": False, "error": f"pyperclip 不可用: {e}"}

    old = None
    if restore_clipboard:
        try:
            old = pyperclip.paste()
        except Exception:
            old = None

    try:
        pyperclip.copy(text)
    except Exception as e:
        return {"ok": False, "error": f"剪贴板写入失败: {e}"}

    time.sleep(0.06)
    # 校验剪贴板确实写入成功
    try:
        got = pyperclip.paste()
        if got != text:
            return {"ok": False,
                    "error": "剪贴板内容校验失败（可能被其他程序抢占）"}
    except Exception:
        pass

    r = hotkey(["ctrl", "v"])
    if not r.get("ok"):
        return {"ok": False,
                "error": f"粘贴组合键发送失败: {r.get('error', '')}"}

    time.sleep(max(float(settle), 0.05))

    if restore_clipboard and old is not None:
        try:
            pyperclip.copy(old)
        except Exception:
            pass

    return {"ok": True, "error": ""}


# ============================================================
# 窗口 / 前台激活
# ============================================================

def get_foreground_window() -> int:
    """当前前台窗口句柄"""
    try:
        return int(_user32.GetForegroundWindow())
    except Exception:
        return 0


def get_window_title(hwnd: int) -> str:
    """窗口标题"""
    try:
        buf = ctypes.create_unicode_buffer(512)
        _user32.GetWindowTextW(int(hwnd), buf, 512)
        return buf.value
    except Exception:
        return ""


def get_window_pid(hwnd: int) -> int:
    """窗口所属进程 ID"""
    try:
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        return int(pid.value)
    except Exception:
        return 0


def is_window_iconic(hwnd: int) -> bool:
    """窗口是否最小化"""
    try:
        return bool(_user32.IsIconic(int(hwnd)))
    except Exception:
        return False


def find_window(title_substr: str) -> int:
    """按标题子串查找可见窗口（不区分大小写），返回句柄（0 表示未找到）"""
    if not title_substr:
        return 0
    key = str(title_substr).lower()
    found = []

    def _cb(hwnd, _lparam):
        try:
            if not _user32.IsWindowVisible(hwnd):
                return True
            title = get_window_title(hwnd)
            if title and key in title.lower():
                found.append(hwnd)
                return False
        except Exception:
            pass
        return True

    try:
        CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND,
                                wintypes.LPARAM)
        _user32.EnumWindows(CB(_cb), 0)
    except Exception:
        return 0
    return int(found[0]) if found else 0


def window_from_point(x: int, y: int) -> int:
    """屏幕坐标处的最顶层根窗口句柄（用于判断点击目标窗口）"""
    try:
        pt = wintypes.POINT(int(x), int(y))
        hwnd = _user32.WindowFromPoint(pt)
        if not hwnd:
            return 0
        # GA_ROOT = 2：取顶层窗口，便于 SetForegroundWindow
        root = _user32.GetAncestor(int(hwnd), 2)
        return int(root or hwnd)
    except Exception:
        return 0


def ensure_foreground(hwnd: int = 0, title: str = "") -> Dict[str, Any]:
    """把目标窗口激活到前台（避免首次点击被吞掉）

    Args:
        hwnd: 窗口句柄（优先）
        title: 窗口标题子串（hwnd 为 0 时用它查找）

    Returns:
        {"ok": bool, "hwnd": int, "title": str, "error": str}
    """
    target = int(hwnd or 0) or find_window(title)
    if not target:
        return {"ok": False, "hwnd": 0, "title": "",
                "error": f"未找到窗口（hwnd={hwnd}, title='{title}'）"}

    if get_foreground_window() == target:
        return {"ok": True, "hwnd": target,
                "title": get_window_title(target), "error": ""}

    try:
        if is_window_iconic(target):
            _user32.ShowWindow(target, SW_RESTORE)
        else:
            _user32.ShowWindow(target, SW_SHOW)
        time.sleep(0.05)
    except Exception:
        pass

    ok = False
    try:
        ok = bool(_user32.SetForegroundWindow(target))
    except Exception:
        ok = False

    if not ok:
        # AttachThreadInput 兜底（跨线程前台限制常见于 Windows 前台锁定）
        try:
            fg = get_foreground_window()
            tid_cur = _kernel32.GetCurrentThreadId()
            tid_fg = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
            tid_tgt = _user32.GetWindowThreadProcessId(target, None)
            attached = []
            for tid in (tid_fg, tid_tgt):
                if tid and tid != tid_cur:
                    if _user32.AttachThreadInput(tid_cur, tid, True):
                        attached.append(tid)
            try:
                _user32.BringWindowToTop(target)
                ok = bool(_user32.SetForegroundWindow(target))
                if not ok:
                    _user32.SetFocus(target)
                    ok = (get_foreground_window() == target)
            finally:
                for tid in attached:
                    _user32.AttachThreadInput(tid_cur, tid, False)
        except Exception:
            ok = ok or (get_foreground_window() == target)

    time.sleep(0.08)
    if get_foreground_window() == target:
        ok = True

    return {
        "ok": bool(ok), "hwnd": target,
        "title": get_window_title(target),
        "error": "" if ok else "SetForegroundWindow 被系统拒绝（前台窗口锁定）",
    }


# ============================================================
# 权限自检（UIPI）
# ============================================================
_TOKEN_ELEVATION = 20
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008

try:
    _kernel32.OpenProcess.argtypes = [
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
    ]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
except Exception:
    pass
if _advapi32 is not None:
    try:
        _advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        _advapi32.OpenProcessToken.restype = wintypes.BOOL
    except Exception:
        pass


def is_self_elevated() -> bool:
    """本进程是否以管理员权限运行"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def is_process_elevated(pid: int) -> Optional[bool]:
    """指定进程是否提权运行

    Returns:
        True / False / None（无法判定）
    """
    if _advapi32 is None or not pid:
        return None
    handle = None
    token = None
    try:
        handle = _kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return None
        token = wintypes.HANDLE()
        if not _advapi32.OpenProcessToken(
                handle, _TOKEN_QUERY, ctypes.byref(token)):
            return None

        class _TOKEN_ELEVATION_STRUCT(ctypes.Structure):
            _fields_ = [("TokenIsElevated", wintypes.DWORD)]

        te = _TOKEN_ELEVATION_STRUCT()
        ret = wintypes.DWORD()
        ok = _advapi32.GetTokenInformation(
            token, _TOKEN_ELEVATION, ctypes.byref(te),
            ctypes.sizeof(te), ctypes.byref(ret),
        )
        if not ok:
            return None
        return bool(te.TokenIsElevated)
    except Exception:
        return None
    finally:
        try:
            if token:
                _kernel32.CloseHandle(token)
        except Exception:
            pass
        try:
            if handle:
                _kernel32.CloseHandle(handle)
        except Exception:
            pass


def check_uipi_conflict(hwnd: int = 0) -> Dict[str, Any]:
    """检测 UIPI 冲突：目标窗口提权而本进程未提权 → 模拟输入会被丢弃

    Returns:
        {"conflict": bool, "self_elevated": bool,
         "target_elevated": bool|None, "target_title": str}
    """
    hwnd = int(hwnd or 0) or get_foreground_window()
    self_elev = is_self_elevated()
    target_title = get_window_title(hwnd) if hwnd else ""
    pid = get_window_pid(hwnd) if hwnd else 0
    target_elev = is_process_elevated(pid) if pid else None

    conflict = False
    if target_elev is True and not self_elev:
        conflict = True

    return {
        "conflict": conflict,
        "self_elevated": self_elev,
        "target_elevated": target_elev,
        "target_title": target_title,
    }


# ============================================================
# 统一入口：带校验的点击
# ============================================================

def click_at_point(x: int, y: int, button: str = "left",
                   double: bool = False, activate: bool = True,
                   settle: float = 0.15) -> Dict[str, Any]:
    """在指定坐标点击（含权限预检 + 前台激活 + 到位校验）

    流程：
      1. UIPI 预检：目标窗口提权而本进程未提权 → 直接判失败并说明原因
      2. 激活坐标处的根窗口（避免首次点击只用于激活、被吞掉）
      3. SendInput 绝对定位移动 + GetCursorPos 到位校验
      4. 发送点击事件并确认事件全部送达

    Returns:
        {"ok": bool, "x": int, "y": int, "error": str,
         "uipi_conflict": bool, "foreground": str}
    """
    x, y = int(x), int(y)
    set_dpi_awareness()

    result: Dict[str, Any] = {
        "ok": False, "x": x, "y": y, "error": "",
        "uipi_conflict": False, "foreground": "",
    }

    # 1) UIPI 预检
    target_hwnd = window_from_point(x, y) if activate else 0
    conflict = check_uipi_conflict(target_hwnd)
    if conflict["conflict"]:
        result["uipi_conflict"] = True
        result["error"] = (
            f"目标窗口[{conflict['target_title']}]以管理员权限运行，"
            f"而本程序未提权 → Windows(UIPI) 会丢弃模拟输入。"
            f"请以**管理员身份**运行本程序后重试。"
        )
        return result

    # 2) 激活前台窗口
    if activate and target_hwnd:
        fg = ensure_foreground(hwnd=target_hwnd)
        result["foreground"] = fg.get("title", "") or get_window_title(
            get_foreground_window()
        )
        if not fg.get("ok"):
            # 激活失败不直接判死，继续尝试点击（可能窗口已在前台）
            result["foreground_warning"] = fg.get("error", "")
    else:
        result["foreground"] = get_window_title(get_foreground_window())

    # 3) 移动 + 到位校验
    mv = move_to(x, y)
    if not mv.get("ok"):
        result["error"] = f"移动失败: {mv.get('error')}"
        result["actual"] = mv.get("actual")
        return result
    time.sleep(max(float(settle), 0.0))

    # 4) 点击
    ck = click(button=button, double=double)
    if not ck.get("ok"):
        result["error"] = f"点击事件未送达: {ck.get('error')}"
        return result

    result["ok"] = True
    result["actual"] = mv.get("actual")
    return result


def describe_environment() -> Dict[str, Any]:
    """输入环境诊断（供启动自检 / 故障排查）"""
    vx, vy, vw, vh = get_virtual_screen()
    fg = get_foreground_window()
    return {
        "dpi_aware": set_dpi_awareness(),
        "virtual_screen": {"left": vx, "top": vy, "width": vw, "height": vh},
        "cursor": list(get_cursor_pos()),
        "self_elevated": is_self_elevated(),
        "foreground_title": get_window_title(fg),
        "foreground_elevated": is_process_elevated(get_window_pid(fg)),
        "sendinput_available": True,
    }






