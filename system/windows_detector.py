"""
Windows Detector - 系统/窗口检测器
==================================
使用 psutil + pywin32 获取客观的 system/window 证据，
供给 WorldState 验收使用（高置信度，非 OCR/视觉推测）。

接口：
  get_process(name)          → [{name, pid, running}]
  get_windows()              → [{title, pid, visible}]
  get_window_by_pid(pid)     → {title, pid, visible} | None
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# try import psutil（可选依赖）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil 未安装 → 进程检测不可用")

# try import pywin32（可选依赖）
try:
    import win32gui
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.warning("pywin32 未安装 → 窗口检测不可用")


def get_process(name: str) -> List[Dict[str, Any]]:
    """按名称获取进程

    Args:
        name: 进程名（如 tetris / tetris.exe / notepad）

    Returns:
        [{name, pid, running}]
    """
    if not PSUTIL_AVAILABLE:
        return []

    name_lower = name.lower().strip()
    if not name_lower.endswith(".exe"):
        name_lower += ".exe"

    results = []
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pname = (proc.info.get('name') or '').lower()
                if name_lower in pname or pname in name_lower:
                    results.append({
                        "name": proc.info['name'],
                        "pid": proc.info['pid'],
                        "running": True,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.warning(f"get_process 失败: {e}")
    return results


def get_windows() -> List[Dict[str, Any]]:
    """获取所有可见窗口

    Returns:
        [{title, pid, visible}]
    """
    if not WIN32_AVAILABLE:
        return []

    windows = []

    def _enum_callback(hwnd, results):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title.strip():
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            results.append({
                "title": title,
                "pid": pid,
                "visible": True,
            })
        except Exception:
            pass

    try:
        win32gui.EnumWindows(_enum_callback, windows)
    except Exception as e:
        logger.warning(f"get_windows 失败: {e}")
    return windows


def get_window_by_pid(pid: int) -> Optional[Dict[str, Any]]:
    """按 PID 获取窗口（标题 + 是否可见）

    Args:
        pid: 进程 ID

    Returns:
        {title, pid, visible} | None
    """
    if not WIN32_AVAILABLE:
        return None

    try:
        windows = get_windows()
        for w in windows:
            if w["pid"] == pid:
                return w
    except Exception as e:
        logger.warning(f"get_window_by_pid 失败: {e}")
    return None


def get_process_status(pid: int) -> bool:
    """进程是否仍在运行（按 PID）

    Returns:
        True if running
    """
    if not PSUTIL_AVAILABLE:
        # fallback: 尝试 Unix / 简单检查
        try:
            import os
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False