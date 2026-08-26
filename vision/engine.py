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
    """Florence-2-base 推理（支持 OVD）

    注意：使用 skip_special_tokens=False 解码，保留 <loc_N>/<poly> 坐标 token，
    供 _parse_f2_ovd_tokens() 解析。文本任务（如 <CAPTION>）的输出不受影响，
    调用方自行 strip <s>/</s> 即可。
    """
    _load_f2_base()
    prompt = task_prompt + (text_input or "")
    inputs = _f2_base_processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = _f2_base_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_tokens, num_beams=2,
    )
    result = _f2_base_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # 去掉上下文 token，保留 <loc_N> 坐标 token
    result = result.replace("<s>", "").replace("</s>", "").strip()
    if task_prompt in result:
        result = result.replace(task_prompt, "").strip()
    return result


def _parse_f2_ovd_tokens(od_text: str, image_size: Tuple[int, int],
                         min_confidence: float = MIN_CONFIDENCE) -> List[Dict]:
    """解析 Florence-2 OVD 输出的 <loc_N> token 序列为元素列表

    Florence-2 开放词汇检测的实际输出格式：
      label<loc_x1><loc_y1><loc_x2><loc_y2>                 # 单框
      label<poly><loc_x1><loc_y1><loc_x2><loc_y2>...</poly> # 多边形
      label. another_label<loc_x1>...                       # 多标签

    <loc_N> 的 N 为 0~999 的归一化坐标，映射到图像尺寸：
      px = N / 999 * (width | height)

    Args:
        od_text: _run_f2_base 返回的（保留 loc token 的）文本
        image_size: (width, height) 图像像素尺寸
        min_confidence: 保留的最低置信度（F2 OVD 无显式置信度，忽略过滤）

    Returns:
        元素列表 [{label, x, y, w, h, cx, cy, confidence}]
    """
    if not od_text or not od_text.strip():
        return []

    w, h = image_size
    elements: List[Dict] = []

    def _px(n: int, dim: int) -> int:
        return int(round(n / 999 * dim))

    # 屏幕尺寸（用于过滤全屏误检框）
    screen_w, screen_h = w, h

    # 1) 提取所有坐标 token 序列（可能包在 <poly></poly> 内）
    #    同时保留标签文本（非 loc token 部分）
    #    按 <poly>...</poly> 或连续 <loc_N> 块切分
    import re as _re

    # 移除 poly 标签，保留内部 loc token
    cleaned = _re.sub(r"</?poly>", "", od_text)

    # 2) 将"标签文本 + loc序列"拆成段
    #    模式：非 loc 文本作为 label，随后紧跟的 loc 序列作为其坐标
    parts = _re.split(r"(<loc_\d+>)", cleaned)
    # parts 形如 ['icon', '<loc_161>', '<loc_0>', '<loc_195>', '<loc_61>', ...]

    cur_label = ""
    cur_locs: List[int] = []

    def _flush():
        nonlocal cur_label, cur_locs
        if cur_label and len(cur_locs) >= 4:
            # loc token 两两一组 (x,y)，取多边形/框的包围盒
            xs = [_px(cur_locs[i], w) for i in range(0, len(cur_locs), 2)]
            ys = [_px(cur_locs[i + 1], h) for i in range(0, len(cur_locs), 2)]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            fw, fh = x2 - x1, y2 - y1
            # 过滤误检：
            #   - 框太小（< 4px）
            #   - 覆盖超过屏幕 90%（多标签 OVD 常退化为整屏默认框 <loc_0><loc_998>）
            #   - 面积占比超过 85%
            if fw < 4 or fh < 4:
                cur_label = ""
                cur_locs = []
                return
            if fw > screen_w * 0.9 or fh > screen_h * 0.9:
                cur_label = ""
                cur_locs = []
                return
            if (fw * fh) > (screen_w * screen_h) * 0.85:
                cur_label = ""
                cur_locs = []
                return
            if fw >= 4 and fh >= 4:
                elements.append({
                    "label": cur_label.strip(),
                    "x": x1, "y": y1,
                    "w": fw, "h": fh,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "confidence": 1.0,  # F2 OVD 无置信度，默认高置信
                })
        cur_label = ""
        cur_locs = []

    for part in parts:
        if not part:
            continue
        if part.startswith("<loc_"):
            n = _re.sub(r"\D", "", part)
            if n:
                cur_locs.append(int(n))
        else:
            _flush()
            cur_label = part.strip()

    _flush()

    # 3) 标签可能含多个（空格/句点分隔），拆分到独立元素
    expanded: List[Dict] = []
    for e in elements:
        labels = [lab.strip() for lab in _re.split(r"[.\s]+", e["label"])
                  if lab.strip()]
        if not labels:
            continue
        for lab in labels:
            if lab.lower() in ("s", "/s", ""):
                continue
            item = dict(e)
            item["label"] = lab
            expanded.append(item)
    return expanded


def _run_f2_ovd(image: Image.Image, query: str,
                max_tokens: int = 1024) -> List[Dict]:
    """Florence-2 开放词汇检测（正确解析坐标 token）

    替代旧的 _run_f2_base + _parse_f2_detection 组合——
    旧组合因 skip_special_tokens=True 丢弃 <loc_N> 坐标，永远返回空。

    Args:
        image: 输入图像（全屏或区域）
        query: 检测目标描述（如 "icon. button. text."）

    Returns:
        元素列表 [{label, x, y, w, h, cx, cy, confidence}]
    """
    od_text = _run_f2_base(image, "<OPEN_VOCABULARY_DETECTION>", query,
                           max_tokens=max_tokens)
    return _parse_f2_ovd_tokens(od_text, image.size, min_confidence=0.0)


# 用于 scan_* 兜底的通用检测标签（每个单独查询，避免多标签退化全屏框）
_SCAN_FALLBACK_LABELS = [
    "icon", "button", "window", "menu", "text",
    "checkbox", "search bar", "taskbar", "input",
]


def _run_f2_ovd_scan(image: Image.Image, query: str,
                     max_tokens: int = 1024) -> List[Dict]:
    """F2 开放词汇检测（带逐标签兜底 + 跨标签去重）

    多标签 OVD 在复杂背景下常退化为整屏默认框 <loc_0><loc_0><loc_998><loc_998>
    或多个标签指向同一框（如 button/text/icon 都在同一位置）。
    先对多标签结果做 IoU 去重，若仍为空则逐个单标签重新检测。

    Args:
        image: 输入图像
        query: 多标签查询（如 "button. text. icon. window. ..."）

    Returns:
        元素列表 [{label, x, y, w, h, cx, cy, confidence}]
    """
    def _dedup(elements: List[Dict]) -> List[Dict]:
        """IoU > 70% 的框只保留第一个（跨标签去重）"""
        deduped: List[Dict] = []
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
        return deduped

    elements = _dedup(_run_f2_ovd(image, query, max_tokens=max_tokens))
    if elements:
        # 按框面积从小到大排序（优先返回小目标，如图标/按钮）
        elements.sort(key=lambda e: e["w"] * e["h"])
        return elements[:MAX_ELEMENTS]

    # 兜底：逐标签检测
    collected: List[Dict] = []
    used_labels = set()
    for lab in _SCAN_FALLBACK_LABELS:
        if lab in used_labels:
            continue
        used_labels.add(lab)
        try:
            els = _run_f2_ovd(image, lab, max_tokens=512)
            collected.extend(els)
        except Exception:
            continue

    deduped = _dedup(collected)
    # 按框面积从小到大排序（优先返回小目标，如图标/按钮）
    deduped.sort(key=lambda e: e["w"] * e["h"])
    return deduped[:MAX_ELEMENTS]


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


def _region_to_screen(x_rel: int, y_rel: int, region_x: int, region_y: int,
                      scale_x: float, scale_y: float) -> Tuple[int, int]:
    """将放大区域内的相对坐标转换回全屏绝对坐标

    区域截图若被放大（take_region_screenshot 对小区放大 2-4 倍），
    F2/DINO 返回的是放大图的坐标，必须除以缩放因子再偏移区域原点。

    Args:
        x_rel, y_rel: 模型返回的区域图内坐标
        region_x, region_y: 区域在全屏的左上角
        scale_x, scale_y: 缩放因子（放大了多少倍）

    Returns:
        (全屏绝对 x, 全屏绝对 y)
    """
    return int(x_rel / scale_x) + region_x, int(y_rel / scale_y) + region_y


def locate_icon_near_text(target_text: str,
                          max_anchor_distance: int = 120) -> Dict[str, Any]:
    """OCR 引导的图标定位：先找文字锚点，再在附近检测图标

    设计背景（解决"桌面图标找不到"根因）：
      - EasyOCR 内容识别可能出错（如 "Hearts ot"），但文字块位置可靠
      - Florence-2 在局部放大小区域检测 icon 比全屏可靠
      - 因此：EasyOCR 定位文字 → 以文字为中心扩展区域 → F2 检测图标

    Windows 桌面图标布局：图形在文字上方（图标名在图形下方）。
    因此扩展区域取"文字上方为主"（y 向上偏移）。

    Args:
        target_text: 目标文字（如 "Hearts of Iron" / "钢铁雄心"）
        max_anchor_distance: 文字锚点距屏幕边缘的最远距离（过滤越界）

    Returns:
        {
          "found": bool,
          "x": int, "y": int,             # 图标中心（全屏绝对坐标）
          "bbox": [x1,y1,x2,y2],
          "anchor_text": str,             # OCR 匹配到的文字块
          "anchor_match": str,            # exact/partial/fuzzy/token
          "anchor_x": int, "anchor_y": int,
          "method": str,                  # ocr_guided / direct_f2 / dino
          "confidence": float,
        }
    """
    # 步骤0: 优先用 UI Automation 精确查找桌面图标（最可靠，不依赖 OCR/视觉）
    try:
        from system.desktop_icons import find_desktop_icon
        icon = find_desktop_icon(target_text)
        if icon.get("found"):
            return {
                "found": True,
                "x": icon["x"], "y": icon["y"],
                "bbox": icon.get("bbox", [0, 0, 0, 0]),
                "anchor_text": icon.get("name", ""),
                "anchor_match": f"uia_{icon.get('match', '')}",
                "anchor_x": icon["x"], "anchor_y": icon["y"],
                "method": "uia_desktop_icon",
                "confidence": 1.0,
            }
    except Exception as e:
        logger.warning(f"UI Automation 桌面图标定位失败: {e}")

    # 步骤1: OCR 定位文字锚点（多候选）
    from vision.ocr import find_text_candidates
    cand = find_text_candidates(target_text, limit=5)
    if not cand.get("found"):
        return {"found": False, "x": 0, "y": 0, "bbox": [0, 0, 0, 0],
                "anchor_text": "", "anchor_match": "", "anchor_x": 0,
                "anchor_y": 0, "method": "no_anchor", "confidence": 0.0}

    # 步骤2: 对每个锚点候选，扩展区域 → F2 检测 icon
    #   候选已按分数排序（短文本图标名优先、长命令回显降权）
    for anchor in cand["candidates"]:
        ax, ay = anchor["cx"], anchor["cy"]
        # 过滤边缘文字（离边缘太近可能区域越界）
        if ax < 20 or ay < 20:
            continue

        # 过滤过短碎片锚点（单字符/纯符号，如 "O"、"E"、"I"）
        #   token 匹配可能命中单字母碎片，这些不是有效图标名
        anchor_text = str(anchor.get("text", "")).strip()
        anchor_clean = re.sub(r'[\s.…,，。！？、"\']', '', anchor_text)
        if len(anchor_clean) < 2:
            continue
        # 过滤明显是命令/代码回显的锚点（含括号、引号、方括号、等号、python 关键词等）
        if any(sym in anchor_text for sym in
               ['(', ')', '[', ']', '"', '=', 'import', 'print', 'python',
                'text', 'candidates', 'locate', 'find', ':', ';']):
            continue
        # 过滤超长锚点（> 14 字符）：很可能是终端命令/窗口标题整句，而非图标名
        if len(anchor_text) > 14:
            continue
        # 过滤含任务动词/介词的长句特征（"打开"、"帮我"、"桌面的"等命令回显词）
        if any(w in anchor_text for w in
               ["帮我", "打开桌面", "开始执行", "执行任务", "桌面的", "请帮我"]):
            continue

        # 图标在文字上方：区域取文字上方为主，x 以文字为中心
        rw, rh = 160, 110
        rx = max(0, ax - rw // 2)
        ry = max(0, ay - rh + 10)   # 图标区域：文字上方约 100px
        # 区域截图（可能被放大）
        region = take_region_screenshot(rx, ry, rw, rh)
        scale_x = region.size[0] / min(rw, region.size[0])
        scale_y = region.size[1] / min(rh, region.size[1])

        # F2 检测 icon（本地图中找图形）
        elements = _run_f2_ovd(region, "icon", max_tokens=512)
        if elements:
            best = elements[0]
            abs_x, abs_y = _region_to_screen(
                best["cx"], best["cy"], rx, ry, scale_x, scale_y
            )
            return {
                "found": True,
                "x": abs_x, "y": abs_y,
                "bbox": [abs_x, abs_y, abs_x, abs_y],
                "anchor_text": anchor["text"],
                "anchor_match": anchor["match"],
                "anchor_x": ax, "anchor_y": ay,
                "method": "ocr_guided_f2",
                "confidence": best["confidence"],
            }

        # F2 失败 → DINO 兜底（本地图）
        try:
            _load_grounding_dino()
            inputs = _gd_processor(images=region, text="icon",
                                   return_tensors="pt")
            with torch.no_grad():
                outputs = _gd_model(**inputs)
            results = _gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=0.25, text_threshold=0.2,
                target_sizes=[region.size[::-1]]
            )
            r = results[0]
            if len(r["boxes"]) > 0:
                box = r["boxes"][0].tolist()
                abs_x, abs_y = _region_to_screen(
                    int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2),
                    rx, ry, scale_x, scale_y
                )
                return {
                    "found": True,
                    "x": abs_x, "y": abs_y,
                    "bbox": [abs_x, abs_y, abs_x, abs_y],
                    "anchor_text": anchor["text"],
                    "anchor_match": anchor["match"],
                    "anchor_x": ax, "anchor_y": ay,
                    "method": "ocr_guided_dino",
                    "confidence": float(r["scores"][0]),
                }
        except Exception as e:
            logger.warning(f"OCR 引导 DINO 定位失败: {e}")

    # 步骤3: 全部锚点失败
    first = cand["candidates"][0]
    return {"found": False, "x": 0, "y": 0, "bbox": [0, 0, 0, 0],
            "anchor_text": first.get("text", ""),
            "anchor_match": first.get("match", ""),
            "anchor_x": first["cx"], "anchor_y": first["cy"],
            "method": "anchor_found_no_icon", "confidence": 0.0}


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
        elements = _run_f2_ovd_scan(screenshot, query, max_tokens=1024)

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
        # 区域缩放因子（take_region_screenshot 对小区会放大）
        scale_x = region.size[0] / max(1, actual_w)
        scale_y = region.size[1] / max(1, actual_h)

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
        elements = _run_f2_ovd_scan(region, query, max_tokens=1024)
        # 转换坐标为全屏绝对坐标（除以缩放因子 + 偏移区域原点）
        for e in elements:
            e["x"], e["y"] = _region_to_screen(e["x"], e["y"], x, y, scale_x, scale_y)
            e["cx"], e["cy"] = _region_to_screen(e["cx"], e["cy"], x, y, scale_x, scale_y)
            e["w"] = int(e["w"] / scale_x)
            e["h"] = int(e["h"] / scale_y)

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
                elements = _run_f2_ovd_scan(region, query, max_tokens=512)
                # 每个格子的缩放因子
                g_scale_x = region.size[0] / max(1, cell_w)
                g_scale_y = region.size[1] / max(1, cell_h)
                for e in elements:
                    e["x"], e["y"] = _region_to_screen(
                        e["x"], e["y"], cx, cy, g_scale_x, g_scale_y)
                    e["cx"], e["cy"] = _region_to_screen(
                        e["cx"], e["cy"], cx, cy, g_scale_x, g_scale_y)
                    e["w"] = int(e["w"] / g_scale_x)
                    e["h"] = int(e["h"] / g_scale_y)
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
        elements = _run_f2_ovd(screenshot, en_query, max_tokens=512)
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
        scale_x = region.size[0] / max(1, width)
        scale_y = region.size[1] / max(1, height)

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
        elements = _parse_f2_ovd_tokens(od_result, region.size, min_confidence=0.0)
        if elements:
            best = elements[0]
            abs_x, abs_y = _region_to_screen(
                best["cx"], best["cy"], x, y, scale_x, scale_y)
            abs_x1, abs_y1 = _region_to_screen(
                best["x"], best["y"], x, y, scale_x, scale_y)
            abs_w = int(best["w"] / scale_x)
            abs_h = int(best["h"] / scale_y)
            return {
                "found": True,
                "x": abs_x, "y": abs_y,
                "bbox": [abs_x1, abs_y1, abs_x1 + abs_w, abs_y1 + abs_h],
                "confidence": best["confidence"],
            }
    except Exception as e:
        logger.warning(f"F2-base 区域检测失败: {e}")

    # 再用 DINO
    try:
        _load_grounding_dino()
        region = take_region_screenshot(x, y, width, height)
        scale_x = region.size[0] / max(1, width)
        scale_y = region.size[1] / max(1, height)

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
            abs_x, abs_y = _region_to_screen(
                int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2),
                x, y, scale_x, scale_y)
            abs_x1, abs_y1 = _region_to_screen(
                int(box[0]), int(box[1]), x, y, scale_x, scale_y)
            abs_w = int((box[2] - box[0]) / scale_x)
            abs_h = int((box[3] - box[1]) / scale_y)
            return {
                "found": True,
                "x": abs_x, "y": abs_y,
                "bbox": [abs_x1, abs_y1, abs_x1 + abs_w, abs_y1 + abs_h],
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