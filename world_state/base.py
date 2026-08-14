"""
World State Base - 核心基础模型
==============================
Evidence（证据） + Belief（信念）模型。

不再使用"单一值+覆盖置信度"，而是证据列表 + 按可靠性与新鲜度计算 belief。

来源可靠性优先：
  system_api > code_check > powershell > ocr > vision
"""
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# 来源固有可靠性
SOURCE_CONFIDENCE = {
    "system_api": 0.99,    # 系统 API / 注册表 / vswhere
    "code_check": 0.95,    # 代码逻辑检测（Test-Path 等确定性检查）
    "filesystem": 0.95,    # 文件系统检测
    "window": 0.95,        # 窗口 API（win32/Get-Process MainWindowTitle）
    "powershell": 0.90,    # PowerShell 输出解析
    "ocr": 0.70,           # OCR 文字识别
    "vision": 0.50,        # 视觉模型推测
    "user": 1.0,           # 用户明确确认
    "task": 0.90,          # 任务/规划器内部状态
    "default": 0.50,
}

# 新鲜度衰减因子（半衰期分钟）
FRESHNESS_HALF_LIFE_SECONDS = 1800.0  # 30 分钟


def now_iso() -> str:
    """当前 ISO 时间"""
    return datetime.now(timezone.utc).isoformat()


def source_reliability(source: str) -> float:
    """获取来源固有可靠性"""
    return SOURCE_CONFIDENCE.get(source, SOURCE_CONFIDENCE["default"])


class Evidence:
    """一条证据"""

    def __init__(self, source: str, value: Any,
                 timestamp: Optional[float] = None,
                 note: str = "", description: str = None):
        self.source = source
        self.value = value
        self.source_confidence = source_reliability(source)
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.note = note
        # description: 人类可读描述（验收条件引用）
        self.description = description or note or ""

    @property
    def confidence(self) -> float:
        """证据置信度（当前有效置信度 = 来源可靠性 × 新鲜度）"""
        return self.effective_confidence()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_confidence": self.source_confidence,
            "value": self.value,
            "timestamp": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "note": self.note,
            "description": self.description,
        }

    def freshness(self, now: float = None) -> float:
        """证据新鲜度（0~1，随时间指数衰减）"""
        now = now or time.time()
        age = max(0.0, now - self.timestamp)
        return math.exp(-age / FRESHNESS_HALF_LIFE_SECONDS)

    def effective_confidence(self, now: float = None) -> float:
        """有效置信度 = 来源可靠性 × 新鲜度"""
        return self.source_confidence * self.freshness(now)


class Belief:
    """信念：由多个证据计算得出的当前值 + 置信度"""

    def __init__(self):
        self.value: Any = None
        self.supporting: List[Evidence] = []
        self.contradicting: List[Evidence] = []
        self.status: str = "unknown"   # unknown/confirmed/uncertain/contradicted

    @property
    def confidence(self) -> float:
        """置信度：基于支持证据的有效置信度融合（Dempster-Shafer 简化）"""
        if not self.supporting:
            return 0.0

        # 支持证据：并集概率 → 1 - Π(1-c_i)
        support_conf = 1.0
        for ev in self.supporting:
            c = ev.effective_confidence()
            if c > 0:
                support_conf *= (1 - c)
        support_conf = 1 - support_conf

        # 矛盾证据削弱
        contradiction = 0.0
        if self.contradicting:
            for ev in self.contradicting:
                contradiction = max(
                    contradiction, ev.effective_confidence()
                )

        # 最终置信度 = 支持 × (1 - 最强矛盾)
        return round(support_conf * (1 - contradiction), 3)

    def add_evidence(self, source: str, value: Any,
                     note: str = "", timestamp: Optional[float] = None) -> Evidence:
        """添加证据并重新计算 belief"""
        ev = Evidence(source, value, timestamp, note)

        if self.value is None:
            # 首次
            self.value = value
            self.supporting.append(ev)
            self._recompute_status()
            return ev

        # 支持当前值
        if value == self.value:
            self.supporting.append(ev)
        else:
            # 矛盾
            self.contradicting.append(ev)
            # 若矛盾证据有效置信度显著高于所有支持，切换值
            strongest_support = max(
                (e.effective_confidence() for e in self.supporting), default=0.0
            )
            if ev.effective_confidence() > strongest_support:
                self.value = value
                # 原支持降级为矛盾
                self.contradicting.extend(self.supporting)
                self.supporting = [ev]

        self._recompute_status()
        return ev

    def _recompute_status(self):
        """根据证据情况更新 status"""
        conf = self.confidence
        if not self.supporting and not self.contradicting:
            self.status = "unknown"
        elif self.contradicting and self.value is not None and \
                any(e.value != self.value for e in self.contradicting):
            if conf >= 0.8:
                self.status = "confirmed"
            elif conf >= 0.4:
                self.status = "uncertain"
            else:
                self.status = "contradicted"
        else:
            self.status = "confirmed" if conf >= 0.8 else "uncertain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "status": self.status,
            "supporting": [e.to_dict() for e in self.supporting],
            "contradicting": [e.to_dict() for e in self.contradicting],
        }

    # ============================================================
    # 领域 State 类的辅助工具
    # ============================================================

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Belief":
        """从字典恢复（持久化用）"""
        belief = cls()
        if not d:
            return belief
        # 恢复证据
        for ed in d.get("supporting", []):
            belief.add_evidence(
                ed.get("source", "default"), ed.get("value"),
                note=ed.get("note", ""),
            )
        for ed in d.get("contradicting", []):
            belief.add_evidence(
                ed.get("source", "default"), ed.get("value"),
                note=ed.get("note", ""),
            )
        return belief


class StateField:
    """状态字段容器（简化访问 Belief 的包装）"""

    def __init__(self, name: str = ""):
        self.name = name
        self.belief = Belief()

    @property
    def value(self):
        return self.belief.value

    @property
    def confidence(self) -> float:
        return self.belief.confidence

    @property
    def status(self) -> str:
        return self.belief.status

    def add_evidence(self, source: str, value: Any, note: str = ""):
        self.belief.add_evidence(source, value, note)
        return self


class StateBase:
    """所有领域 State 的基类"""

    domain: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        """序列化所有字段"""
        result = {}
        for attr, val in vars(self).items():
            if isinstance(val, Belief):
                result[attr] = val.to_dict()
            elif isinstance(val, StateField):
                result[attr] = val.belief.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateBase":
        """从字典恢复"""
        state = cls()
        for attr, val in vars(state).items():
            if isinstance(val, Belief) and attr in d:
                state.__dict__[attr] = Belief.from_dict(d[attr])
        return state