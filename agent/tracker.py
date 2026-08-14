"""
Object Tracker - 视觉对象记忆
==============================
解决"每次 visual_locate 都全屏重扫"的重复扫描问题。

作为本地加速层，集成进 tools.py 的 visual_locate / visual_locate_region：
  1. 首次找到对象 → 缓存 bbox + 置信度 + 截图 hash
  2. 再次查找 → 优先在缓存 bbox 邻域小范围重扫（快 10 倍）
  3. 仅以下情况触发全屏重检：
     - confidence 下降
     - 页面变化（截图 hash 差异大）
     - 点击后目标消失（invalidate）
     - 邻域重扫未找到

设计原则：
  - Agent / LLM 无感知（不新增工具）
  - 不存储固定坐标用于跨任务（任务结束清空；无任务只需短期缓存）
  - Kalman 式平滑：位置用加权平均微调，避免抖动
"""
import hashlib
import time
from typing import Dict, Any, List, Optional

from PIL import Image


def _image_signature(image: Image.Image, size: int = 32) -> str:
    """计算截图签名（缩略图 hash），用于检测页面变化"""
    try:
        small = image.resize((size, size), Image.NEAREST)
        # 量化为 16 级灰度，生成特征串
        gray = small.convert("L")
        pixels = list(gray.getdata())
        # 分块平均，每块 8 个像素 → 128 长度特征
        blocks = []
        for i in range(0, len(pixels), 8):
            chunk = pixels[i:i+8]
            blocks.append(str(sum(chunk) // len(chunk)))
        return hashlib.md5(",".join(blocks).encode()).hexdigest()[:16]
    except Exception:
        return ""


class TrackedObject:
    """被跟踪对象"""

    def __init__(
        self,
        key: str,
        bbox: List[int],
        confidence: float,
        screen_signature: str = "",
        source: str = "locate",
    ):
        """
        Args:
            key: 对象键（如 "NPC"、"start_button"）
            bbox: [x1, y1, x2, y2]
            confidence: 检测置信度
            screen_signature: 发现时的屏幕签名
            source: 来源（'locate' / 'locate_region' / 'scan'）
        """
        self.key = key
        self.bbox = bbox
        self.confidence = confidence
        self.screen_signature = screen_signature
        self.source = source
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.detect_count = 1
        self.hit_count = 0      # 邻域确认命中次数
        self.miss_count = 0     # 邻域未命中次数
        self.invalidated = False

    @property
    def center(self) -> List[int]:
        return [
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        ]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def update(self, new_bbox: List[int], new_confidence: float,
               screen_signature: str = ""):
        """更新位置（平滑：50% 新位置 + 50% 旧位置）"""
        # 平滑
        smooth_bbox = [
            int(0.5 * self.bbox[0] + 0.5 * new_bbox[0]),
            int(0.5 * self.bbox[1] + 0.5 * new_bbox[1]),
            int(0.5 * self.bbox[2] + 0.5 * new_bbox[2]),
            int(0.5 * self.bbox[3] + 0.5 * new_bbox[3]),
        ]
        self.bbox = smooth_bbox
        self.confidence = new_confidence
        if screen_signature:
            self.screen_signature = screen_signature
        self.last_seen = time.time()
        self.detect_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "last_seen": self.last_seen,
            "detect_count": self.detect_count,
        }


class ObjectTracker:
    """视觉对象跟踪器（单任务生命周期缓存）"""

    # 邻域放大比例（在缓存 bbox 基础上外扩，用于小范围重扫）
    NEIGHBORHOOD_EXPAND = 1.5
    # 邻域重扫命中后允许的最大偏移（超过则重新全屏检测）
    MAX_TRACK_OFFSET = 100
    # 页面变化敏感度（签名不同块比例）
    SCREEN_CHANGE_THRESHOLD = 0.4

    def __init__(self, max_objects: int = 20):
        self._objects: Dict[str, TrackedObject] = {}
        self._max_objects = max_objects
        self._last_screen_signature: Dict[str, str] = {}

    # ============================================================
    # 记录 / 查找
    # ============================================================

    def track(self, key: str, bbox: List[int], confidence: float,
              screen_signature: str = "", source: str = "locate") -> TrackedObject:
        """记录/更新对象位置"""
        key = self._normalize_key(key)
        now = time.time()

        if key in self._objects:
            obj = self._objects[key]
            # 若之前的对象已失效，重新跟踪
            if obj.invalidated:
                self._objects[key] = TrackedObject(
                    key, bbox, confidence, screen_signature, source
                )
            else:
                obj.update(bbox, confidence, screen_signature)
            # 更新屏幕签名记录
            self._last_screen_signature[key] = screen_signature
            return self._objects[key]

        # 新增
        if len(self._objects) >= self._max_objects:
            # 淘汰最久未使用的
            oldest_key = min(
                self._objects,
                key=lambda k: self._objects[k].last_seen
            )
            del self._objects[oldest_key]

        obj = TrackedObject(key, bbox, confidence, screen_signature, source)
        self._objects[key] = obj
        self._last_screen_signature[key] = screen_signature
        return obj

    def get(self, key: str) -> Optional[TrackedObject]:
        """获取缓存对象"""
        key = self._normalize_key(key)
        obj = self._objects.get(key)
        if obj is not None and obj.invalidated:
            return None
        return obj

    def _normalize_key(self, key: str) -> str:
        """规范化对象键（去空格小写）"""
        return key.strip().lower()

    # ============================================================
    # 状态判断
    # ============================================================

    def should_use_cached(self, key: str, screen_signature: str = "") -> bool:
        """判断是否应优先使用缓存（在邻域进行小范围重扫）

        条件：
          - 对象已缓存且未失效
          - 无屏幕签名 或 签名未发生显著变化
        """
        obj = self.get(key)
        if obj is None:
            return False
        if screen_signature:
            last = self._last_screen_signature.get(self._normalize_key(key), "")
            if last and last != screen_signature:
                # 页面变化 → 需要全屏重检
                return False
        return True

    def screen_changed(self, key: str, screen_signature: str) -> bool:
        """检测页面是否相对上次检测发生变化"""
        obj = self.get(key)
        if obj is None:
            return True
        return bool(obj.screen_signature) and obj.screen_signature != screen_signature

    def get_neighborhood(self, key: str, expand: float = None) -> List[int]:
        """获取对象邻域范围（用于小范围重扫）

        Returns:
            [x, y, width, height] 相对全屏坐标
        """
        obj = self.get(key)
        if obj is None:
            return []
        expand = expand or self.NEIGHBORHOOD_EXPAND

        w = obj.width
        h = obj.height
        if w <= 0:
            w = 50
        if h <= 0:
            h = 50

        pad_x = int(w * (expand - 1))
        pad_y = int(h * (expand - 1))

        x1 = max(0, obj.bbox[0] - pad_x)
        y1 = max(0, obj.bbox[1] - pad_y)
        x2 = obj.bbox[2] + pad_x
        y2 = obj.bbox[3] + pad_y
        return [x1, y1, x2 - x1, y2 - y1]

    def offset_exceeded(self, key: str, new_bbox: List[int]) -> bool:
        """新检测结果是否偏离缓存过远"""
        obj = self.get(key)
        if obj is None:
            return True
        if len(new_bbox) != 4:
            return True
        old_cx, old_cy = obj.center
        new_cx = (new_bbox[0] + new_bbox[2]) // 2
        new_cy = (new_bbox[1] + new_bbox[3]) // 2
        dist = ((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5
        return dist > self.MAX_TRACK_OFFSET

    # ============================================================
    # 失效
    # ============================================================

    def invalidate(self, key: str):
        """失效指定对象（如点击后目标消失）"""
        key = self._normalize_key(key)
        if key in self._objects:
            self._objects[key].invalidated = True

    def invalidate_by_click(self, click_bbox: List[int], margin: int = 50):
        """点击后失效点击位置附近的被跟踪对象

        点击坐标落在某个跟踪对象 bbox（外扩 margin）内 → 该对象可能被触发消失
        """
        if len(click_bbox) != 2:
            return
        cx, cy = click_bbox
        for key, obj in list(self._objects.items()):
            if obj.invalidated:
                continue
            x1, y1, x2, y2 = obj.bbox
            if (x1 - margin <= cx <= x2 + margin and
                    y1 - margin <= cy <= y2 + margin):
                obj.invalidated = True

    def invalidate_all(self):
        """失效全部（页面大变 / 新任务）"""
        for obj in self._objects.values():
            obj.invalidated = True
        self._last_screen_signature.clear()

    def clear(self):
        """清空全部"""
        self._objects.clear()
        self._last_screen_signature.clear()

    # ============================================================
    # 统计
    # ============================================================

    def get_statistics(self) -> Dict[str, Any]:
        active = [o for o in self._objects.values() if not o.invalidated]
        return {
            "total_tracked": len(self._objects),
            "active": len(active),
            "objects": [o.to_dict() for o in active],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracked": [o.to_dict() for o in self._objects.values()],
            "invalidate_all": False,
        }