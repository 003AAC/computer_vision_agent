"""
Abstract Experience - 抽象经验存储
=================================
增强记忆层：只存**行为模式**，不存固定坐标。

与 Episodic/Semantic 记忆的区别：
  - Episodic: 任务流程 + 失败恢复（侧重"怎么做"）
  - Semantic: 稳定知识（如何找到/如何验证）
  - Abstract: 跨任务的行为模式 + 检测方法 + 状态转换（侧重"什么方式有效"）

示例（不存坐标）：
{
  "application": "Beholder",
  "action_pattern": "NPC交互通常通过点击人物模型触发",
  "detection_method": ["OCR寻找提示文本", "人物区域检测"],
  "state_transitions": ["MAIN_MENU → GAME_SCENE"],
  "failure_cases": ["不要依赖固定坐标"]
}

存储位置：agent/memories/abstract_experience.json
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AbstractExperience:
    """一条抽象经验"""

    def __init__(
        self,
        application: str,
        action_pattern: str = "",
        detection_method: List[str] = None,
        state_transitions: List[str] = None,
        failure_cases: List[str] = None,
        confidence: float = 0.5,
    ):
        self.application = application
        self.action_pattern = action_pattern
        self.detection_method = detection_method or []
        self.state_transitions = state_transitions or []
        self.failure_cases = failure_cases or []
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()
        self.usage_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application": self.application,
            "action_pattern": self.action_pattern,
            "detection_method": self.detection_method,
            "state_transitions": self.state_transitions,
            "failure_cases": self.failure_cases,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AbstractExperience":
        exp = cls(
            application=d.get("application", ""),
            action_pattern=d.get("action_pattern", ""),
            detection_method=d.get("detection_method", []),
            state_transitions=d.get("state_transitions", []),
            failure_cases=d.get("failure_cases", []),
            confidence=d.get("confidence", 0.5),
        )
        exp.timestamp = d.get("timestamp", exp.timestamp)
        exp.usage_count = d.get("usage_count", 0)
        return exp


class AbstractExperienceStore:
    """抽象经验存储（JSON 持久化）"""

    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memories", "abstract_experience.json"
            )
        self.path = path
        self._experiences: Dict[str, AbstractExperience] = {}
        self._load()

    # ============================================================
    # 持久化
    # ============================================================

    def _load(self):
        """从磁盘加载"""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key, item in data.items():
                    self._experiences[key] = AbstractExperience.from_dict(item)
        except Exception as e:
            logger.warning(f"抽象经验加载失败: {e}")

    def _save(self):
        """保存到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._experiences.items()},
                    f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            logger.warning(f"抽象经验保存失败: {e}")

    # ============================================================
    # CRUD
    # ============================================================

    def get_key(self, application: str, action_pattern: str = "") -> str:
        """生成存储键"""
        pattern = action_pattern[:30] if action_pattern else "*"
        return f"{application.lower()}::{pattern}"

    def add(self, application: str, action_pattern: str = "",
            detection_method: List[str] = None,
            state_transitions: List[str] = None,
            failure_cases: List[str] = None,
            confidence: float = 0.5) -> AbstractExperience:
        """新增或更新抽象经验"""
        key = self.get_key(application, action_pattern)
        if key in self._experiences:
            exp = self._experiences[key]
            # 合并更新
            if detection_method:
                exp.detection_method = _merge_unique(exp.detection_method, detection_method)
            if state_transitions:
                exp.state_transitions = _merge_unique(exp.state_transitions, state_transitions)
            if failure_cases:
                exp.failure_cases = _merge_unique(exp.failure_cases, failure_cases)
            if action_pattern and not exp.action_pattern:
                exp.action_pattern = action_pattern
            exp.confidence = min(1.0, exp.confidence + 0.1)
            exp.timestamp = datetime.now().isoformat()
        else:
            exp = AbstractExperience(
                application=application,
                action_pattern=action_pattern,
                detection_method=detection_method,
                state_transitions=state_transitions,
                failure_cases=failure_cases,
                confidence=confidence,
            )
            self._experiences[key] = exp
        self._save()
        return exp

    def get(self, application: str) -> List[AbstractExperience]:
        """获取某应用的所有抽象经验"""
        app_key = application.lower()
        return [
            exp for key, exp in self._experiences.items()
            if key.startswith(app_key)
        ]

    def search(self, query: str, top_n: int = 3) -> List[AbstractExperience]:
        """按关键词搜索抽象经验"""
        query_l = query.lower()
        results = []
        for exp in self._experiences.values():
            field_text = " ".join([
                exp.application, exp.action_pattern,
                " ".join(exp.detection_method),
                " ".join(exp.state_transitions),
                " ".join(exp.failure_cases),
            ]).lower()
            score = 0
            for kw in query_l.split():
                if kw in field_text:
                    score += 1
            # 应用名精确匹配加分
            if exp.application.lower() in query_l:
                score += 2
            if score > 0:
                results.append((score, exp))
        results.sort(key=lambda x: x[0] * x[1].confidence, reverse=True)
        return [exp for _, exp in results[:top_n]]

    def record_usage(self, key: str):
        """记录经验被成功使用（提升置信度）"""
        if key in self._experiences:
            exp = self._experiences[key]
            exp.usage_count += 1
            exp.confidence = min(1.0, exp.confidence + 0.05)
            self._save()

    def build_instruction(self, query: str) -> str:
        """构建经验指令（注入提示词）"""
        exps = self.search(query)
        if not exps:
            return ""
        lines = ["## [抽象经验] 相关行为模式"]
        for exp in exps:
            lines.append(f"- 应用 [{exp.application}]（置信度 {exp.confidence:.2f}）：")
            if exp.action_pattern:
                lines.append(f"  - 行为模式: {exp.action_pattern}")
            if exp.detection_method:
                lines.append(f"  - 检测方法: {' / '.join(exp.detection_method[:3])}")
            if exp.state_transitions:
                lines.append(f"  - 状态转换: {' → '.join(exp.state_transitions[:3])}")
            if exp.failure_cases:
                lines.append(f"  - 经验教训: {' / '.join(exp.failure_cases[:3])}")
        lines.append("⚠️ 以上为抽象经验，不包含固定坐标。使用前先验证当前屏幕。")
        return "\n".join(lines)

    def all(self) -> List[AbstractExperience]:
        return list(self._experiences.values())

    def clear(self):
        self._experiences.clear()
        self._save()


def _merge_unique(base: List[str], additions: List[str]) -> List[str]:
    """合并去重"""
    merged = list(base)
    for item in additions:
        if item not in merged:
            merged.append(item)
    return merged[:10]