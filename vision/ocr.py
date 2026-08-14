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
from typing import Dict, Any, List, Optional

import pyautogui
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# EasyOCR 单例（懒加载，常驻内存）
# ============================================================
_reader = None


def _get_reader():
    """获取 EasyOCR Reader 实例（懒加载）"""
    global _reader
    if _reader is None:
        print("  [OCR] 加载 EasyOCR（中英文）...")
        import easyocr
        _reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        print("  [OCR] ✓ EasyOCR 就绪")
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

def ocr_screen() -> str:
    """全屏 OCR — 识别屏幕上所有可见文字及其坐标
    
    使用 EasyOCR（中英文），返回所有可见文字块及坐标。
    比 visual_scan() 更适合"找按钮/菜单/对话框上的文字"场景。
    
    Returns:
        JSON: {screen, texts: [{text, x, y, w, h, cx, cy, confidence}], count}
    """
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
    
    先用 OCR 读所有文字，再在结果中搜索目标文字（三级匹配：精确→包含→模糊）。
    这是定位"确定按钮"、"保存"、"文件菜单"等文字型元素的最可靠方式。
    
    Args:
        target_text: 要查找的文字（如 "确定"、"保存"、"开始"）
    
    Returns:
        {"found": bool, "text": str, "x": int, "y": int, "match": str, ...}
    """
    try:
        screenshot = pyautogui.screenshot()
        texts = _ocr_image(screenshot)

        # Level 1: 精确匹配
        for t in texts:
            if t["text"] == target_text:
                return {"found": True, "text": t["text"], "x": t["cx"], "y": t["cy"],
                        "bbox": [t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]],
                        "confidence": t["confidence"],
                        "match": "exact"}

        # Level 2: 包含匹配
        for t in texts:
            if target_text in t["text"] or t["text"] in target_text:
                return {"found": True, "text": t["text"], "x": t["cx"], "y": t["cy"],
                        "bbox": [t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]],
                        "confidence": t["confidence"],
                        "match": "partial"}

        # Level 3: 模糊匹配（去空格标点后比较）
        target_clean = re.sub(r'[\s.…,，。！？、]', '', target_text)
        for t in texts:
            text_clean = re.sub(r'[\s.…,，。！？、]', '', t["text"])
            if len(target_clean) >= 2 and len(text_clean) >= 2:
                if target_clean in text_clean or text_clean in target_clean:
                    return {"found": True, "text": t["text"], "x": t["cx"], "y": t["cy"],
                            "bbox": [t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]],
                            "confidence": t["confidence"],
                            "match": "fuzzy"}

        return {"found": False, "text": target_text, "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0], "match": "none",
                "hint": f"OCR 未找到匹配文字（共识别 {len(texts)} 个文字块），可尝试 visual_locate()"}
    except Exception as e:
        logger.error(f"OCR 查找失败: {e}")
        return {"found": False, "text": target_text, "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0], "match": "error", "error": str(e)}