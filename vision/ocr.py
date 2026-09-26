"""
OCR 感知模块 — EasyOCR 引擎
=============================
让 Agent 能"读"屏幕上的文字，而不只是"看"形状。
这是解决"眼瞎"问题的核心补丁。

EasyOCR 返回格式：每个元素 (bbox, text, confidence)
  bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 四个角坐标
"""
import json
import logging
import re
import sys
import time
import warnings
from typing import Dict, Any, List, Optional

import pyautogui
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# 日志 / 告警兜底
# ============================================================
# ⚡ 1) 抑制 EasyOCR 的 pin_memory 刷屏
#   EasyOCR 每次 readtext() 都创建 DataLoader(..., pin_memory=True)，
#   CPU-only 环境下每条识别都会告警一次，观察轮询时刷屏几十条。
# ⚡ 2) 兜底 print：GBK 控制台下打印 ✓/✅ 会抛 UnicodeEncodeError
#   （曾导致本模块首次 OCR 调用直接失败）

_QUIET_WARNING_PATTERNS = (
    "pin_memory",
    "no accelerator is found",
)


def _safe_print(*args, **kwargs):
    """永不抛 UnicodeEncodeError 的 print"""
    try:
        print(*args, **kwargs)
        return
    except UnicodeEncodeError:
        pass
    except Exception:
        return
    try:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        text = " ".join(str(a) for a in args)
        sys.stdout.write(
            text.encode(enc, "replace").decode(enc, "replace") + "\n"
        )
    except Exception:
        pass


_ORIGINAL_SHOWWARNING = warnings.showwarning


def _quiet_showwarning(message, category, filename, lineno,
                       file=None, line=None):
    """丢弃 pin_memory 类刷屏告警，其余交回原处理器

    为何不用 filterwarnings 唯一兜底：只要有库调用
    `warnings.simplefilter()` / `resetwarnings()`，已注册规则就会被清空。
    替换 showwarning 不会被重置，最稳。
    """
    text = str(message)
    if any(p in text for p in _QUIET_WARNING_PATTERNS):
        return
    try:
        _ORIGINAL_SHOWWARNING(message, category, filename, lineno,
                              file=file, line=line)
    except Exception:
        pass


def _install_quiet_warnings():
    """安装告警静音（幂等，可重复调用）"""
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*argument is set as true.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*no accelerator is found.*",
    )
    try:
        warnings.showwarning = _quiet_showwarning
    except Exception:
        pass


_install_quiet_warnings()

# ============================================================
# EasyOCR 单例（懒加载，常驻内存）
# ============================================================
_reader = None


def _get_reader():
    """获取 EasyOCR Reader 实例（懒加载）"""
    global _reader
    if _reader is None:
        _safe_print("  [OCR] 加载 EasyOCR（中英文）...")
        import easyocr
        # import easyocr 可能带来其它库对 warnings 过滤器的重置，这里再装一次
        _install_quiet_warnings()
        _reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        _safe_print("  [OCR] [OK] EasyOCR 就绪")
    return _reader


def preload_ocr():
    """预加载 OCR 模型（Agent 启动时调用）"""
    _get_reader()


# ============================================================
# 核心 OCR 函数
# ============================================================

def _ocr_image(image: Image.Image) -> List[Dict]:
    """对图片执行 OCR，返回结构化文字列表
    
    EasyOCR 返回格式：
      [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text", confidence), ...]
    """
    reader = _get_reader()
    # PIL Image → numpy array
    img_array = np.array(image)
    results = reader.readtext(img_array)

    items = []
    for bbox, text, confidence in results:
        if not text.strip():
            continue
        # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        items.append({
            "text": text.strip(),
            "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
            "confidence": round(float(confidence), 3),
        })
    return items


# ============================================================
# 对外接口
# ============================================================

# ⚡ 全屏 OCR 短 TTL 缓存
#   同一动作内的多处调用（预测观察 / 预测对比 / 阶段推进 / 验收）
#   常常在同一瞬间重复识别同一帧（每次 CPU 2-5s）。
#   这里做 0.8s 复用；**动作类工具执行后必须 invalidate**，保证不乱用旧帧。
_OCR_CACHE = {"ts": 0.0, "value": ""}
_OCR_CACHE_TTL = 0.8


def invalidate_ocr_cache():
    """使全屏 OCR 缓存失效（动作改变屏幕后调用）"""
    _OCR_CACHE["ts"] = 0.0
    _OCR_CACHE["value"] = ""


def _ocr_screen_uncached() -> str:
    """真正执行全屏 OCR（内部使用）"""
    try:
        screenshot = pyautogui.screenshot()
        screen_w, screen_h = pyautogui.size()

        texts = _ocr_image(screenshot)

        # 去重：文字相同且距离 < 20px 的合并
        seen = []
        for t in texts:
            is_dup = False
            for s in seen:
                if t["text"] == s["text"] and abs(t["cx"] - s["cx"]) < 20 and abs(t["cy"] - s["cy"]) < 20:
                    is_dup = True
                    break
            if not is_dup:
                seen.append(t)

        result = {
            "screen": f"{screen_w}x{screen_h}",
            "texts": seen,
            "count": len(seen),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def ocr_screen(use_cache: bool = True, ttl: float = None) -> str:
    """全屏 OCR — 识别屏幕上所有可见文字及其坐标

    使用 EasyOCR（中英文），返回所有可见文字块及坐标。
    比 visual_scan() 更适合"找按钮/菜单/对话框上的文字"场景。

    Args:
        use_cache: 是否使用短 TTL 缓存（默认 True）
        ttl: 缓存有效期秒数（默认 0.8s）

    Returns:
        JSON: {screen, texts: [{text, x, y, w, h, cx, cy, confidence}], count}
    """
    ttl = _OCR_CACHE_TTL if ttl is None else float(ttl)
    if use_cache and ttl > 0:
        age = time.time() - _OCR_CACHE["ts"]
        if _OCR_CACHE["value"] and age < ttl:
            return _OCR_CACHE["value"]

    result_json = _ocr_screen_uncached()

    if use_cache and ttl > 0:
        _OCR_CACHE["ts"] = time.time()
        _OCR_CACHE["value"] = result_json
    return result_json


def _screen_texts(use_cache: bool = True) -> List[Dict]:
    """获取全屏文字块列表（复用 ocr_screen 缓存）"""
    try:
        data = json.loads(ocr_screen(use_cache=use_cache))
        if isinstance(data, dict):
            return data.get("texts", []) or []
    except Exception:
        pass
    return []


def ocr_region(x: int, y: int, width: int, height: int) -> str:
    """区域 OCR — 识别屏幕指定区域内的文字
    
    先截取区域，放大后 OCR，坐标转换回全屏绝对坐标。
    适合小区域文字密集场景（如对话框、菜单栏）。
    
    Returns:
        JSON: {screen, region, texts, count}
    """
    try:
        full = pyautogui.screenshot()
        screen_w, screen_h = pyautogui.size()
        x = max(0, min(x, screen_w - 1))
        y = max(0, min(y, screen_h - 1))
        width = min(width, screen_w - x)
        height = min(height, screen_h - y)
        region = full.crop((x, y, x + width, y + height))
        
        # 小区域放大以提高 OCR 准确率
        scale_factor = 1
        if width < 300 or height < 300:
            scale = max(2, min(4, 600 // min(width, height)))
            region = region.resize((width * scale, height * scale), Image.LANCZOS)
            scale_factor = scale

        texts = _ocr_image(region)

        # 转换坐标回全屏
        for t in texts:
            t["x"] = t["x"] // scale_factor + x
            t["y"] = t["y"] // scale_factor + y
            t["cx"] = t["cx"] // scale_factor + x
            t["cy"] = t["cy"] // scale_factor + y
            t["w"] = t["w"] // scale_factor
            t["h"] = t["h"] // scale_factor

        result = {
            "screen": f"{screen_w}x{screen_h}",
            "region": f"({x},{y}) {width}x{height}",
            "texts": texts,
            "count": len(texts),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def find_text_on_screen(target_text: str) -> Dict[str, Any]:
    """在屏幕上查找指定文字，返回精确坐标

    先用 OCR 读所有文字，再在结果中搜索目标文字。
    匹配策略（逐级降级）：
      1. 精确匹配（完全相同）
      2. 包含匹配（互为子串）
      3. 模糊匹配（去空格标点后比较）
      4. token 级匹配（目标拆词，OCR 碎片含多数关键词即匹配——容忍 EasyOCR 识别碎片）

    若多级匹配有多个候选，取坐标最接近屏幕中心的（减少误点边缘）。

    Args:
        target_text: 要查找的文字（如 "确定"、"保存"、"Hearts of Iron"）

    Returns:
        {"found": bool, "text": str, "x": int, "y": int, "match": str,
         "candidates": [...], ...}
    """
    try:
        # 复用全屏 OCR 缓存（同一 step 内已有观察结果时不再重复识别，省 2-5s）
        texts = _screen_texts(use_cache=True)
        if not texts:
            screenshot = pyautogui.screenshot()
            texts = _ocr_image(screenshot)

        def _make(t, match, score_extra=0.0):
            return {"found": True, "text": t["text"], "x": t["cx"], "y": t["cy"],
                    "bbox": [t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]],
                    "confidence": t["confidence"], "match": match,
                    "score": round(min(1.0, t["confidence"] + score_extra), 3)}

        # Level 1: 精确匹配（最可靠：图标名/按钮文字通常是短文本）
        exact_matches = [t for t in texts if t["text"] == target_text]
        if exact_matches:
            return _make(exact_matches[0], "exact", 0.3)

        # Level 2: 包含匹配
        #   ⚠️ 关键：目标文字出现在长文本中（如 "帮我打开桌面的原神"）很可能是
        #   终端命令/窗口标题/界面文字，不是可点击的图标名。真正的图标名通常短（2-6 字）。
        #   因此按"文本块长度"排序：越短越可能是图标名，优先返回。
        contains_matches = []
        for t in texts:
            if target_text in t["text"] or t["text"] in target_text:
                # 短文本块（<= 8 字符）= 极可能是图标名 → 高优先级
                # 长文本块 = 命令回显/界面句子 → 低优先级
                if len(t["text"]) <= 8:
                    return _make(t, "partial", 0.25)
                contains_matches.append(t)
        if contains_matches:
            # 长文本候选：全部返回，让调用方/LLM 判断是否可信
            return _make(contains_matches[0], "partial", 0.0)

        # Level 3: 模糊匹配（去空格标点后比较）
        target_clean = re.sub(r'[\s.…,，。！？、]', '', target_text)
        for t in texts:
            text_clean = re.sub(r'[\s.…,，。！？、]', '', t["text"])
            if len(target_clean) >= 2 and len(text_clean) >= 2:
                if target_clean in text_clean or text_clean in target_clean:
                    return _make(t, "fuzzy")

        # Level 4: token 级匹配（容忍 OCR 碎片）
        #   目标拆成词（如 "Hearts of Iron" → ["hearts", "of", "iron"]）
        #   对每个 OCR 文字块，统计命中关键词数；>= 最小命中数即视为候选。
        target_words = [
            w for w in re.split(r'[\s.…,，。！？、]+', target_text.lower())
            if len(w) >= 2
        ]
        if target_words:
            candidates = []
            min_hits = 1 if len(target_words) >= 2 else 1
            for t in texts:
                t_low = t["text"].lower()
                hits = sum(1 for w in target_words if w in t_low)
                if hits >= min_hits:
                    # 评分：命中数/词数 + 置信度
                    score = (hits / len(target_words)) * 0.7 + t["confidence"] * 0.3
                    candidates.append((score, t))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_t = candidates[0]
                return _make(best_t, "token")

        return {"found": False, "text": target_text, "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0], "match": "none",
                "hint": f"OCR 未找到匹配文字（共识别 {len(texts)} 个文字块），可尝试 visual_locate()"}
    except Exception as e:
        logger.error(f"OCR 查找失败: {e}")
        return {"found": False, "text": target_text, "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0], "match": "error", "error": str(e)}


def find_text_candidates(target_text: str, limit: int = 5) -> Dict[str, Any]:
    """在屏幕上查找目标文字，返回全部候选坐标（按匹配分排序）

    与 find_text_on_screen 的区别：
      - 不匹配失败就返回，而是返回所有命中候选（含 token 级碎片命中）
      - 供 OCR 引导定位：即使内容识别有误（如 "Hearts ot"），位置仍是可靠锚点

    Args:
        target_text: 要查找的文字
        limit: 最多返回候选数

    Returns:
        {"found": bool, "candidates": [{"text", "x", "y", "cx", "cy", "bbox", "confidence", "score", "match"}]}
    """
    try:
        screenshot = pyautogui.screenshot()
        texts = _ocr_image(screenshot)
        target_low = target_text.lower()
        target_clean = re.sub(r'[\s.…,，。！？、]', '', target_low)
        target_words = [
            w for w in re.split(r'[\s.…,，。！？、]+', target_low)
            if len(w) >= 2
        ]

        scored = []
        for t in texts:
            t_low = t["text"].lower()
            t_clean = re.sub(r'[\s.…,，。！？、]', '', t_low)
            score = 0.0
            match = ""

            if t_low == target_low:
                score, match = 1.0, "exact"
            elif target_low in t_low or t_low in target_low:
                score, match = 0.9, "partial"
            elif target_clean and t_clean and (
                    target_clean in t_clean or t_clean in target_clean):
                score, match = 0.8, "fuzzy"
            elif target_words:
                hits = sum(1 for w in target_words if w in t_low)
                if hits >= 1:
                    score = (hits / len(target_words)) * 0.7
                    match = "token"

            if score > 0:
                # 置信度加成（位置锚点不要求内容高置信）
                conf = t.get("confidence", 0.0)
                final = score * 0.8 + conf * 0.2
                # ⚠️ 短文本（图标名）优先：长句子很可能是终端命令/窗口标题
                #    图标名通常 2-8 字符，命令回显通常 > 12 字符
                text_len = len(t["text"])
                if text_len <= 8:
                    final += 0.15          # 短文本强加分（图标名特征）
                elif text_len >= 15:
                    final -= 0.15          # 长文本降权（命令回显特征）
                # 无空格长文本（连续字符）更可能是图标名；含中文句子的长文本更像命令
                if text_len <= 8:
                    final += 0.10
                scored.append({
                    "text": t["text"], "x": t["x"], "y": t["y"],
                    "cx": t["cx"], "cy": t["cy"],
                    "bbox": [t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]],
                    "confidence": conf, "score": round(final, 3), "match": match,
                })

        scored.sort(key=lambda c: c["score"], reverse=True)
        return {
            "found": bool(scored),
            "candidates": scored[:limit],
            "total_candidates": len(scored),
        }
    except Exception as e:
        logger.error(f"OCR 候选查找失败: {e}")
        return {"found": False, "candidates": [], "error": str(e)}