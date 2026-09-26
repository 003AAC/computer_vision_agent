"""
Computer Use Agent - 配置文件
API Key 从本地文件读取，不要硬编码
"""
import os
import sys


def setup_console_encoding() -> bool:
    """统一控制台编码兜底

    背景：中文 Windows 控制台代码页为 936(GBK)，非 GBK 字符
    （✅ ❌ ✓ ⚠️ ⚡ 等）在 **重定向/管道** 场景下 `print` 会抛
    UnicodeEncodeError，足以中断整个任务（例如打印工具结果 "✅ 已点击…"）。

    策略（逐流尝试）：
      1. 重配置为 UTF-8 + errors='replace'
      2. 失败则仅把 errors 设为 'replace'（保证 print 永不抛异常）

    Returns:
        是否至少完成一次兜底
    """
    ok = False
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            ok = True
            continue
        except Exception:
            pass
        try:
            stream.reconfigure(errors="replace")
            ok = True
        except Exception:
            pass
    return ok


def safe_print(*args, **kwargs):
    """永不抛 UnicodeEncodeError 的 print（日志兜底）"""
    try:
        print(*args, **kwargs)
        return
    except UnicodeEncodeError:
        pass
    except Exception:
        return
    # 降级：编码不可表示字符 → 用 replace 输出
    try:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        text = " ".join(str(a) for a in args)
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
        sys.stdout.write("\n")
    except Exception:
        pass


# 模块导入即生效（agent.core / agent.tools / computer_use_agent 均会先导入本模块）
setup_console_encoding()


def _load_key(filename, fallback=""):
    """从本地文件加载 API Key"""
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"[警告] 未找到 {filename}，请在 {project_dir} 下创建该文件并填入 API Key")
        return fallback

# === DeepSeek API（决策层 LLM）===
DEEPSEEK_API_KEY = _load_key("deepseek_key.txt")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# === Agent 参数 ===
MAX_STEPS = 200
ACTION_DELAY = 1.5
