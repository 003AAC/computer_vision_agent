"""
Vision Router - 视觉任务路由器
==============================
解决"视觉模块职责混乱"的问题。

按任务类型选择正确的视觉模块：
  - 纯文字提取  → OCR（EasyOCR）
  - 已知类目标   → Grounding DINO（person / cat / window / icon）
  - 类别判断     → CLIP（或回退 Florence-2）
  - 复杂场景理解 → Florence-2 VLM

设计原则：
  - 保持 vision/engine.py 的旧接口（scan_screen / locate_element 等）签名不变
  - Router 是**新增的推荐层**，不强制替换现有调用
  - 模型加载失败 → 自动降级，不崩溃
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class VisionTaskType:
    """视觉任务类型"""

    OCR = "ocr"                 # 文字读取
    DETECT = "detect"           # 已知目标定位（DINO）
    CLASSIFY = "classify"       # 类别判断（CLIP）
    SCENE = "scene"             # 复杂场景理解（VLM）
    SCAN = "scan"               # 全屏扫描


# ============================================================
# 任务类型分类
# ============================================================

# 明确文字任务关键词
_OCR_KEYWORDS = [
    "文字", "文本", "字", "标题", "按钮文字", "标签", "内容",
    "read text", "ocr", "find text", "text", "word", "label",
    # 常见 UI 文字按钮（用 OCR 定位最可靠）
    "确定", "取消", "保存", "继续", "下一步", "上一步", "完成",
    "确认", "重试", "关闭", "打开", "删除", "移动",
]

# 明确已知类目标关键词（DINO 擅长）
# 注意：不包含 "NPC" 等复杂语义词（应为 SCENE）
_DETECT_KEYWORDS = [
    "人", "人物", "窗口", "按钮", "图标", "猫", "狗", "车", "桌子",
    "person", "people", "window", "button", "icon", "cat", "dog",
    "car", "chair", "坐",
]

# 明确类别判断关键词（CLIP 擅长）
_CLASSIFY_KEYWORDS = [
    "是不是", "是否是", "判断", "有没有", "属于", "类别", "种类", "还是",
    "这是", "这是否", "is this", "belongs to", "category", "type", "looks like", "or not",
]

# 复杂场景理解关键词（VLM 擅长）
_SCENE_KEYWORDS = [
    "场景", "界面", "布局", "发生了什么", "画面", "情境",
    "scene", "screen", "ui", "interface", "screenshot",
    "关系", "位置", "在哪里", "哪个", "find the", "where is",
]


def classify_task(query: str) -> str:
    """根据查询文字判断视觉任务类型

    Args:
        query: 查询描述（如"游戏里的NPC上司"）

    Returns:
        VisionTaskType 之一
    """
    q = (query or "").lower().strip()
    if not q:
        return VisionTaskType.SCAN

    # 优先级：明确关键词
    if any(kw in q for kw in _OCR_KEYWORDS):
        return VisionTaskType.OCR
    if any(kw in q for kw in _CLASSIFY_KEYWORDS):
        return VisionTaskType.CLASSIFY
    if any(kw in q for kw in _DETECT_KEYWORDS):
        return VisionTaskType.DETECT
    if any(kw in q for kw in _SCENE_KEYWORDS):
        return VisionTaskType.SCENE

    # 启发式：包含"哪里/位置/区域" → 定位类，用 DETECT
    if any(kw in q for kw in ["哪里", "位置", "坐标", "区域", "where", "position", "location"]):
        return VisionTaskType.DETECT

    # 复杂语义（如"游戏里的NPC上司"）→ 默认 VLM
    return VisionTaskType.SCENE


def get_router_description() -> str:
    """返回路由说明（供诊断/提示词）"""
    return (
        "Vision Router：\n"
        "  - OCR: 文字读取\n"
        "  - Detect: 已知目标定位（DINO）\n"
        "  - Classify: 类别判断（CLIP）\n"
        "  - Scene: 复杂场景理解（Florence-2 VLM）\n"
        "  - Scan: 全屏扫描"
    )


# ============================================================
# 类别判断（CLIP / 回退）
# ============================================================

_clip_model = None
_clip_processor = None


def _load_clip():
    """加载 CLIP 模型（懒加载）"""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return True
    except Exception as e:
        logger.warning(f"CLIP 加载失败（回退 Florence-2）: {e}")
        return False


def classify_image(image, candidates: List[str]) -> Dict[str, Any]:
    """对图像进行类别判断

    Args:
        image: PIL Image
        candidates: 候选类别列表（如 ["NPC", "背景", "箱子"]）

    Returns:
        {
          "label": str,           # 最可能的类别
          "scores": {label: score},
          "model": "clip" | "florence2"
        }
    """
    if not candidates:
        return {"label": "", "scores": {}, "model": "none"}

    # 优先 CLIP
    if _load_clip():
        try:
            import torch
            inputs = _clip_processor(
                text=candidates, images=image, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            scores = {
                candidates[i]: round(float(probs[i]), 4)
                for i in range(len(candidates))
            }
            best = max(scores, key=scores.get)
            return {"label": best, "scores": scores, "model": "clip"}
        except Exception as e:
            logger.warning(f"CLIP 推理失败（回退 Florence-2）: {e}")

    # 回退：用 Florence-2 描述 + 关键词匹配
    try:
        from vision.engine import _run_f2_base
        desc = _run_f2_base(image, "<CAPTION>", max_tokens=100)
        desc_lower = desc.lower()
        scores = {}
        for cand in candidates:
            scores[cand] = 1.0 if cand.lower() in desc_lower else 0.0
        best = max(scores, key=scores.get)
        return {
            "label": best if scores[best] > 0 else candidates[0],
            "scores": scores,
            "model": "florence2",
            "description": desc,
        }
    except Exception as e:
        logger.error(f"Florence-2 回退分类失败: {e}")
        return {"label": candidates[0] if candidates else "", "scores": {}, "model": "fallback"}


# ============================================================
# 路由到精确模块（供 tools.py 使用）
# ============================================================

def route_locate(target_description: str) -> str:
    """路由 locate 任务到正确的视觉模块描述

    返回一个"建议使用的模块"标记，供 tools.py 决定如何定位：
      - "ocr"    → 用 OCR 找文字
      - "detect" → 用 DINO/F2 定位已知目标
      - "scene"  → 用 F2 复杂语义理解

    Args:
        target_description: 目标描述

    Returns:
        VisionTaskType 之一（用于路由决策）
    """
    return classify_task(target_description)


def suggest_detection_strategy(target_description: str) -> Dict[str, Any]:
    """为 locate 返回建议的检测策略（供 tools/_tracked_locate 参考）

    Returns:
        {
          "task_type": str,
          "strategy": str,      # 建议的检测路径
          "hint": str
        }
    """
    task_type = classify_task(target_description)

    # 文字型 → OCR 优先
    if task_type == VisionTaskType.OCR:
        return {
            "task_type": task_type,
            "strategy": "ocr_first",
            "hint": f"建议用视觉找文字：visual_find_text('{target_description}') 或区域 OCR",
        }

    # 已知目标 → DINO / F2 开放词汇检测
    if task_type == VisionTaskType.DETECT:
        return {
            "task_type": task_type,
            "strategy": "detector",
            "hint": f"使用目标检测（F2/DINO）定位已知目标: {target_description}",
        }

    # 复杂语义 → VLM 场景理解 + 区域定位
    if task_type == VisionTaskType.SCENE:
        return {
            "task_type": task_type,
            "strategy": "vlm_guided",
            "hint": (
                f"'{target_description}' 是复杂语义描述。"
                f"建议：先 visual_scan() 理解场景，再在候选区域 visual_locate_region() 定位。"
                f"避免直接给 DINO 复杂句子。"
            ),
        }

    # 默认
    return {
        "task_type": task_type,
        "strategy": "default",
        "hint": "",
    }