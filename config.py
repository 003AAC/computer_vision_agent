"""
Computer Use Agent - 配置文件
API Key 从本地文件读取，不要硬编码
"""
import os

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
