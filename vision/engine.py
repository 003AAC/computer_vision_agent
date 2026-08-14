"""
视觉感知引擎 - 本地视觉模型
============================
本地视觉模型只负责看，返回结构化 JSON。
所有决策由 DeepSeek（云端主脑）负责。

检测策略：
  - scan_screen() → F2-base 开放词汇检测，返回所有可见元素坐标
  - scan_region() → 局部放大检测
  - scan_grid() → 分块检测
  - locate_element() → 精确查找指定元素

核心改进：
  - 置信度阈值过滤（>= 0.35），减少误检
  - 元素数量上限（30个），防止大量低质检测淹没主脑
  - 场景类型判断：纯文本场景提示使用 OCR
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional

import pyautogui
import torch
from PIL import Image

from vision.models import F2_BASE_PATH, GD_PATH, F2_LARGE_PATH

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
MIN_CONFIDENCE = 0.35          # 最低置信度阈值（低于此值的检测丢弃）
MAX_ELEMENTS = 30              # 单次扫描最大元素数
TEXT_SCENE_PIXEL_RATIO = 0.85  # 纯色像素占比超过此值 => 纯文本场景

# ============================================================
# 模型实例（懒加载，启动后常驻）
# ============================================================
_f2_base_processor = None
_f2_base_model = None
_gd_model = None
_gd_processor = None


def preload_models():
    """预加载所有视觉模型（Agent 启动时调用，常驻内存）"""
    print("  [视觉] 预加载视觉模型...")
    _load_f2_base()
    print("  [视觉] ✓ Florence-2-base 已常驻（开放词汇检测）")
    _load_grounding_dino()
    print("  [视觉] ✓ Grounding DINO 已常驻（精确查找用）")


def _load_f2_base():
    global _f2_base_processor, _f2_base_model
    if _f2_base_model is not None:
        return
    print("  [视觉] 加载 Florence-2-base...")
    from transformers import AutoProcessor, AutoModelForCausalLM
    _f2_base_processor = AutoProcessor.from_pretrained(F2_BASE_PATH, trust_remote_code=True)
    _f2_base_model = AutoModelForCausalLM.from_pretrained(F2_BASE_PATH, trust_remote_code=True)


def _load_grounding_dino():
    global _gd_model, _gd_processor
    if _gd_model is not None:
        return
    print("  [视觉] 加载 Grounding DINO...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _gd_processor = AutoProcessor.from_pretrained(GD_PATH)
    _gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_PATH)


# ============================================================
# 截图工具
# ============================================================

def take_screenshot() -> Image.Image:
    return pyautogui.screenshot()


def take_region_screenshot(x: int, y: int, width: int, height: int) -> Image.Image:
    """截取屏幕指定区域并放大"""
    full = pyautogui.screenshot()
    screen_w, screen_h = pyautogui.size()
    x = max(0, min(x, screen_w - 1))
    y = max(0, min(y, screen_h - 1))
    width = min(width, screen_w - x)
    height = min(height, screen_h - y)
    region = full.crop((x, y, x + width, y + height))
    if width < 200 or height < 200:
        scale = max(2, min(4, 400 // min(width, height)))
        region = region.resize((width * scale, height * scale), Image.LANCZOS)
    return region


# ============================================================
# 场景判断：检测当前屏幕是否为"纯文本"模式（如终端、日志）
# ============================================================

def _is_text_dominated_scene(image: Image.Image, uniform_ratio_threshold: float = 0.85) -> bool:
    """判断屏幕是否为纯文本场景（终端/日志/代码编辑器等）
    
    策略：统计大面积纯色区域占比。如果画面绝大部分是统一背景色
    （如黑色终端 + 白色字符），说明是文本界面，不适合做视觉检测。
    
    Returns:
        True 表示是文本为主的场景，建议用 OCR
    """
    try:
        # 缩小采样以提高速度
        small = image.resize((80, 60), Image.NEAREST)
        pixels = list(small.getdata())
        
        # 统计出现频率最高的颜色（背景色）
        from collections import Counter
        color_counts = Counter(pixels)
        most_common_color = color_counts.most_common(1)[0][0]
        
        # 统计"接近"背景色的像素占比（允许 ±15 偏差）
        r0, g0, b0 = most_common_color[:3]
        uniform_count = 0
        total = len(pixels)
        for pixel in pixels:
            r, g, b = pixel[:3]
            if abs(r - r0) < 30 and abs(g - g0) < 30 and abs(b - b0) < 30:
                uniform_count += 1
        
        ratio = uniform_count / total
        return ratio >= uniform_ratio_threshold
    except Exception:
        return False


# ============================================================
# F2-large 推理（开放词汇检测）
# ============================================================


def _run_f2_base(image: Image.Image, task_prompt: str, text_input: str = None,
                  max_tokens: int = 512) -> str:
    """Florence-2-base 推理（支持 OVD）"""
    _load_f2_base()
    prompt = task_prompt + (text_input or "")
    inputs = _f2_base_processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = _f2_base_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_tokens, num_beams=2,
    )
    result = _f2_base_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if task_prompt in result:
        result = result.replace(task_prompt, "").strip()
    return result


# ============================================================
# 解析 F2-large 的 OD/OVD 输出为结构化元素列表
# ============================================================

def _parse_f2_detection(od_text: str, min_confidence: float = MIN_CONFIDENCE) -> List[Dict]:
    """解析 Florence-2 的目标检测输出
    
    F2 的 OD 输出格式：
      icon: <x1>,<y1>,<x2>,<y2>; button: <x1>,<y1>,<x2>,<y2>
    或
      icon(0.85): <x1>,<y1>,<x2>,<y2>; button(0.92): <x1>,<y1>,<x2>,<y2>
    
    Args:
        od_text: F2 模型输出
        min_confidence: 最低置信度阈值，低于此值的丢弃
    
    Returns:
        过滤后的元素列表
    """
    elements = []
    if not od_text:
        return elements

    # 按分号分割
    parts = od_text.split(";")
    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        try:
            label_part, coords = part.split(":", 1)
            label_part = label_part.strip()
            coords = coords.strip()

            # 提取标签和置信度
            confidence = 1.0
            label = label_part
            if "(" in label_part and ")" in label_part:
                import re as _re
                m = _re.match(r'(.+?)\(([\d.]+)\)', label_part)
                if m:
                    label = m.group(1).strip()
                    confidence = float(m.group(2))

            # 置信度过滤
            if confidence < min_confidence:
                continue

            # 解析坐标: <x1>,<y1>,<x2>,<y2>
            nums = coords.replace("<", "").replace(">", "").split(",")
            if len(nums) == 4:
                x1, y1, x2, y2 = map(int, nums)
                # 过滤：框面积太小（< 16px）或太大（> 屏幕 90%）都是误检
                w, h = x2 - x1, y2 - y1
                if w < 4 or h < 4:
                    continue
                elements.append({
                    "label": label,
                    "x": x1, "y": y1,
                    "w": w, "h": h,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "confidence": round(confidence, 3),
                })
        except (ValueError, IndexError):
            continue

    return elements


# ============================================================
# 中文→英文映射
# ============================================================

_QUERY_TRANSLATIONS = {
    "开始菜单": "start menu", "开始": "start button",
    "任务栏": "taskbar", "桌面": "desktop",
    "此电脑": "this PC", "我的电脑": "my computer",
    "回收站": "recycle bin", "控制面板": "control panel",
    "搜索框": "search box", "搜索": "search",
    "输入框": "input box", "文本框": "text box",
    "确定": "OK button", "取消": "cancel button",
    "关闭": "close button", "保存": "save button",
    "打开": "open button", "下一步": "next button",
    "上一步": "previous button", "返回": "back button",
    "提交": "submit button", "登录": "login button",
    "注册": "register button", "搜索按钮": "search button",
    "菜单": "menu", "设置": "settings",
    "帮助": "help", "关于": "about", "退出": "exit",
    "最小化": "minimize", "最大化": "maximize", "还原": "restore",
    "记事本": "notepad", "浏览器": "browser",
    "Chrome": "Chrome", "Edge": "Edge",
    "微信": "WeChat", "QQ": "QQ", "WPS": "WPS",
    "VS Code": "Visual Studio Code", "代码": "Visual Studio Code",
    "终端": "terminal", "命令提示符": "command prompt",
    "PowerShell": "PowerShell", "文件资源管理器": "file explorer",
    "资源管理器": "file explorer", "计算器": "calculator",
    "日历": "calendar", "时钟": "clock", "相机": "camera",
    "截图": "screenshot", "画图": "paint",
    "原神": "Genshin Impact", "游戏": "game",
    "图标": "icon", "按钮": "button", "链接": "link",
    "图片": "image", "文字": "text", "标题": "title",
    "标签": "tab", "滚动条": "scroll bar",
    "下拉菜单": "dropdown menu", "复选框": "checkbox",
    "单选按钮": "radio button", "开关": "toggle switch",
    "滑块": "slider", "进度条": "progress bar",
    "通知": "notification", "弹窗": "popup window",
    "对话框": "dialog", "窗口": "window",
}


def _translate_query(text: str) -> str:
    if text in _QUERY_TRANSLATIONS:
        return _QUERY_TRANSLATIONS[text]
    for cn, en in _QUERY_TRANSLATIONS.items():
        if cn in text:
            return en
    return text


# ============================================================
# 对外接口 1：scan_screen() — 全屏扫描（F2-base 开放词汇检测）
# ============================================================

def scan_screen() -> str:
    """扫描全屏，返回结构化 JSON（所有可见元素及其坐标）
    
    使用 F2-large 的开放词汇检测能力，检测屏幕上所有常见元素。
    返回每个元素的 label、坐标、置信度。
    
    Returns:
        JSON: {screen, elements: [{label, x, y, w, h, cx, cy, confidence}]}
    """
    try:
        screenshot = take_screenshot()
        screen_w, screen_h = pyautogui.size()

        # 场景判断：如果是纯文本场景，提醒用 OCR
        if _is_text_dominated_scene(screenshot):
            return json.dumps({
                "screen": f"{screen_w}x{screen_h}",
                "scene": "文本界面",
                "note": "当前屏幕以文本为主（终端/日志/代码），视觉检测可能不准确，"
                        "建议使用 visual_find_text() 或 visual_read_region() 等 OCR 工具",
                "elements": [],
                "count": 0,
            }, ensure_ascii=False)

        # 用 F2-base 做开放词汇检测
        # 注意：不包含 image/logo 等易误检词
        query = "button. text. icon. window. taskbar. menu. input. search bar. checkbox."
        od_result = _run_f2_base(screenshot, "<OPEN_VOCABULARY_DETECTION>", query, max_tokens=1024)

        elements = _parse_f2_detection(od_result, min_confidence=MIN_CONFIDENCE)

        # 按置信度排序，取置信度最高的前 MAX_ELEMENTS 个
        elements.sort(key=lambda e: e["confidence"], reverse=True)
        elements = elements[:MAX_ELEMENTS]

        # 去重：IoU > 70% 的保留置信度高的
        deduped = []
        for e in elements:
            is_dup = False
            for d in deduped:
                ix = max(e["x"], d["x"])
                iy = max(e["y"], d["y"])
                ix2 = min(e["x"] + e["w"], d["x"] + d["w"])
                iy2 = min(e["y"] + e["h"], d["y"] + d["h"])
                if ix < ix2 and iy < iy2:
                    inter = (ix2 - ix) * (iy2 - iy)
                    union = e["w"] * e["h"] + d["w"] * d["h"] - inter
                    if union > 0 and inter / union > 0.7:
                        is_dup = True
                        break
            if not is_dup:
                deduped.append(e)

        result = {
            "screen": f"{screen_w}x{screen_h}",
            "elements": deduped,
            "count": len(deduped),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# 对外接口 2：scan_region() — 局部扫描
# ============================================================

def scan_region(x: int, y: int, width: int, height: int) -> str:
    """扫描屏幕指定区域（放大后分析）
    
    Returns:
        JSON: {screen, region, elements}
    """
    try:
        region = take_region_screenshot(x, y, width, height)
        screen_w, screen_h = pyautogui.size()
        actual_w = min(width, screen_w - x)
        actual_h = min(height, screen_h - y)

        # 场景判断
        if _is_text_dominated_scene(region):
            return json.dumps({
                "screen": f"{screen_w}x{screen_h}",
                "region": f"({x},{y}) {actual_w}x{actual_h}",
                "scene": "文本区域",
                "note": "该区域以文本为主，建议用 visual_read_region() OCR 读取",
                "elements": [],
                "count": 0,
            }, ensure_ascii=False)

        query = "button. text. icon. window. menu. input. search bar. checkbox."
        od_result = _run_f2_base(region, "<OPEN_VOCABULARY_DETECTION>", query, max_tokens=1024)

        elements = _parse_f2_detection(od_result, min_confidence=MIN_CONFIDENCE)
        # 转换坐标为全屏绝对坐标
        for e in elements:
            e["x"] += x
            e["y"] += y
            e["cx"] += x
            e["cy"] += y

        result = {
            "screen": f"{screen_w}x{screen_h}",
            "region": f"({x},{y}) {actual_w}x{actual_h}",
            "elements": elements,
            "count": len(elements),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# 对外接口 3：scan_grid() — 分块扫描
# ============================================================

def scan_grid(rows: int = 2, cols: int = 3) -> str:
    """将屏幕分成网格，逐块扫描
    
    Returns:
        JSON: 每块包含该块内的元素列表
    """
    try:
        screen_w, screen_h = pyautogui.size()
        cell_w = screen_w // cols
        cell_h = screen_h // rows

        all_elements = []
        grid_info = []

        for r in range(rows):
            for c in range(cols):
                cx = c * cell_w
                cy = r * cell_h
                region = take_region_screenshot(cx, cy, cell_w, cell_h)

                # 跳过纯文本块
                if _is_text_dominated_scene(region):
                    grid_info.append({
                        "grid": f"({r+1},{c+1})",
                        "position": f"({cx},{cy})-{cx+cell_w},{cy+cell_h}",
                        "scene": "文本区域",
                        "count": 0,
                    })
                    continue

                query = "button. text. icon. window. menu. input. search bar."
                od_result = _run_f2_base(region, "<OPEN_VOCABULARY_DETECTION>", query, max_tokens=512)

                elements = _parse_f2_detection(od_result, min_confidence=MIN_CONFIDENCE)
                for e in elements:
                    e["x"] += cx
                    e["y"] += cy
                    e["cx"] += cx
                    e["cy"] += cy
                    e["grid"] = f"({r+1},{c+1})"

                all_elements.extend(elements)
                grid_info.append({
                    "grid": f"({r+1},{c+1})",
                    "position": f"({cx},{cy})-{cx+cell_w},{cy+cell_h}",
                    "count": len(elements),
                })

        result = {
            "screen": f"{screen_w}x{screen_h}",
            "grid": f"{rows}x{cols}",
            "grids": grid_info,
            "elements": all_elements[:MAX_ELEMENTS],  # 全局也加限制
            "count": min(len(all_elements), MAX_ELEMENTS),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# 对外接口 4：locate_element() — 精确查找指定元素
# ============================================================

def locate_element(target_description: str) -> Dict[str, Any]:
    """在屏幕上精确查找指定元素
    
    先用 F2-base 的开放词汇检测找，找不到再用 Grounding DINO。
    
    Args:
        target_description: 要查找的元素描述（中文或英文）
    
    Returns:
        {"found": bool, "x": int, "y": int, "confidence": float}
    """
    en_query = _translate_query(target_description)

    # 先用 F2-base 检测
    try:
        screenshot = take_screenshot()
        od_result = _run_f2_base(screenshot, "<OPEN_VOCABULARY_DETECTION>", en_query, max_tokens=512)
        elements = _parse_f2_detection(od_result, min_confidence=MIN_CONFIDENCE)
        if elements:
            best = elements[0]
            return {
                "found": True,
                "x": best["cx"],
                "y": best["cy"],
                "bbox": [best["x"], best["y"], best["x"] + best["w"], best["y"] + best["h"]],
                "confidence": best["confidence"],
            }
    except Exception as e:
        logger.warning(f"F2-base 检测失败: {e}")

    # 再用 Grounding DINO
    try:
        _load_grounding_dino()
        screenshot = take_screenshot()

        inputs = _gd_processor(images=screenshot, text=en_query, return_tensors="pt")
        with torch.no_grad():
            outputs = _gd_model(**inputs)

        results = _gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.25, text_threshold=0.2,
            target_sizes=[screenshot.size[::-1]]
        )
        result = results[0]

        if len(result["boxes"]) > 0:
            best_idx = result["scores"].argmax().item()
            box = result["boxes"][best_idx].tolist()
            score = result["scores"][best_idx].item()
            x1, y1, x2, y2 = box
            return {
                "found": True,
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": score,
            }
    except Exception as e:
        logger.error(f"DINO 定位失败: {e}")

    return {"found": False, "x": 0, "y": 0, "bbox": [0, 0, 0, 0], "confidence": 0.0}


def locate_in_region(target_description: str, x: int, y: int,
                     width: int, height: int) -> Dict[str, Any]:
    """在屏幕指定区域内精确查找元素"""
    en_query = _translate_query(target_description)

    # 先用 F2-base
    try:
        region = take_region_screenshot(x, y, width, height)

        # 如果是文本区域，直接提示用 OCR
        if _is_text_dominated_scene(region):
            return {
                "found": False,
                "x": 0, "y": 0,
                "bbox": [0, 0, 0, 0],
                "confidence": 0.0,
                "hint": "该区域以文本为主，建议使用 visual_find_text()",
            }

        od_result = _run_f2_base(region, "<OPEN_VOCABULARY_DETECTION>", en_query, max_tokens=512)
        elements = _parse_f2_detection(od_result, min_confidence=MIN_CONFIDENCE)
        if elements:
            best = elements[0]
            return {
                "found": True,
                "x": best["cx"] + x,
                "y": best["cy"] + y,
                "bbox": [best["x"] + x, best["y"] + y,
                         best["x"] + best["w"] + x, best["y"] + best["h"] + y],
                "confidence": best["confidence"],
            }
    except Exception as e:
        logger.warning(f"F2-base 区域检测失败: {e}")

    # 再用 DINO
    try:
        _load_grounding_dino()
        region = take_region_screenshot(x, y, width, height)

        inputs = _gd_processor(images=region, text=en_query, return_tensors="pt")
        with torch.no_grad():
            outputs = _gd_model(**inputs)

        results = _gd_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.25, text_threshold=0.2,
            target_sizes=[region.size[::-1]]
        )
        result = results[0]

        if len(result["boxes"]) > 0:
            best_idx = result["scores"].argmax().item()
            box = result["boxes"][best_idx].tolist()
            score = result["scores"][best_idx].item()
            return {
                "found": True,
                "x": int((box[0] + box[2]) / 2) + x,
                "y": int((box[1] + box[3]) / 2) + y,
                "bbox": [int(box[0]) + x, int(box[1]) + y,
                         int(box[2]) + x, int(box[3]) + y],
                "confidence": score,
            }
    except Exception as e:
        logger.error(f"DINO 区域定位失败: {e}")

    return {"found": False, "x": 0, "y": 0, "bbox": [0, 0, 0, 0], "confidence": 0.0}


# ============================================================
# 兼容旧接口
# ============================================================

def describe_screen(mode: str = "quick") -> str:
    return scan_screen()


def describe_region(x: int, y: int, width: int, height: int, mode: str = "quick") -> str:
    return scan_region(x, y, width, height)


def describe_grid(rows: int = 2, cols: int = 3) -> str:
    return scan_grid(rows, cols)


def analyze_screen(mode: str = "auto") -> str:
    return scan_screen()


def find_element(target_description: str, **kwargs) -> Dict[str, Any]:
    return locate_element(target_description)