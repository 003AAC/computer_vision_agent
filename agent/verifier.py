"""
Goal Verifier - 任务验收模块
============================
独立的任务验收器，不依赖执行过程。

任务开始时由 LLM 生成 Goal Specification：
{
  "goal": "country_selected",
  "evidence": ["出现苏联国旗", "国家名称包含 USSR/苏联", "进入国家界面"],
  "failure_condition": ["仍处于国家选择界面"]
}

Agent 宣告完成时，由 verifier 独立用 OCR 全屏扫描验证：
  - evidence 中的关键词出现在屏幕上 → 验收通过
  - failure_condition 中的关键词出现 → 验收失败（伪完成）
  - evidence 和 failure_condition 都不匹配 → 低置信度，要求额外验证

这解决了 HOI4 案例的根因：验收由独立模块判定，而非执行 agent 自述。
"""
import json
import logging
from typing import Dict, Any, List, Optional

from vision.ocr import ocr_screen

logger = logging.getLogger(__name__)


class GoalSpec:
    """目标验收规格"""

    def __init__(
        self,
        goal: str,
        evidence: List[str] = None,
        failure_condition: List[str] = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            goal: 目标描述（如 "country_selected"）
            evidence: 验收证据（屏幕上应出现的关键词/元素描述）
            failure_condition: 失败条件（出现则视为未完成）
            confidence_threshold: 验收通过所需的最低置信度
        """
        self.goal = goal
        self.evidence = evidence or []
        self.failure_condition = failure_condition or []
        self.confidence_threshold = confidence_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "evidence": self.evidence,
            "failure_condition": self.failure_condition,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoalSpec":
        return cls(
            goal=d.get("goal", ""),
            evidence=d.get("evidence", []),
            failure_condition=d.get("failure_condition", []),
            confidence_threshold=d.get("confidence_threshold", 0.6),
        )


class GoalVerifier:
    """独立目标验收器（基于 OCR 证据 + 可注入工具结果文本）"""

    def __init__(self, use_ocr: bool = True):
        """
        Args:
            use_ocr: 是否使用 OCR 验证（可注入 mock 供测试）
        """
        self._use_ocr = use_ocr
        self._current_spec: Optional[GoalSpec] = None
        self._verify_history: List[Dict[str, Any]] = []

    # ============================================================
    # 规格管理
    # ============================================================

    def set_spec(self, spec: GoalSpec):
        """设置当前任务的验收规格"""
        self._current_spec = spec

    def set_spec_from_dict(self, d: Dict[str, Any]):
        """从字典设置验收规格（Planning 结果）"""
        self._current_spec = GoalSpec.from_dict(d)

    @property
    def has_spec(self) -> bool:
        return self._current_spec is not None

    @property
    def current_spec(self) -> Optional[GoalSpec]:
        return self._current_spec

    def clear(self):
        """任务结束清除"""
        self._current_spec = None
        self._verify_history = []

    # ============================================================
    # 验收
    # ============================================================

    def verify(self, ocr_texts: List[str] = None) -> Dict[str, Any]:
        """独立验收当前页面状态

        Args:
            ocr_texts: 屏幕文字列表（不传则自动全屏 OCR）

        Returns:
            {
              "verified": bool,
              "confidence": float,
              "matched_evidence": [...],
              "matched_failures": [...],
              "reasons": [...]
            }
        """
        if self._current_spec is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "matched_evidence": [],
                "matched_failures": [],
                "reasons": ["未设置验收规格"],
            }

        if ocr_texts is None:
            ocr_texts = self._get_screen_texts()

        spec = self._current_spec
        matched_evidence = self._match_keywords(spec.evidence, ocr_texts)
        matched_failures = self._match_keywords(spec.failure_condition, ocr_texts)

        # 置信度计算
        confidence = 0.0
        if spec.evidence:
            confidence = len(matched_evidence) / len(spec.evidence)
        # 失败条件出现 → 强负信号
        if matched_failures:
            confidence = min(confidence, 0.2)

        # 判定
        verified = (
            confidence >= spec.confidence_threshold
            and not matched_failures
        )

        reasons = []
        if matched_evidence:
            reasons.append(f"匹配到证据: {', '.join(matched_evidence)}")
        if matched_failures:
            reasons.append(f"检测到失败条件: {', '.join(matched_failures)}")
        if not verified:
            reasons.append(
                f"置信度 {confidence:.2f} 低于阈值 {spec.confidence_threshold}"
            )

        result = {
            "verified": verified,
            "confidence": round(confidence, 3),
            "matched_evidence": matched_evidence,
            "matched_failures": matched_failures,
            "reasons": reasons,
            "goal": spec.goal,
        }
        self._verify_history.append(result)
        return result

    def verify_wrapped(self) -> str:
        """验收并返回可注入提示词的文本（供 core.py 使用）"""
        result = self.verify()
        if result["verified"]:
            return (
                f"【独立验收通过】目标 [{result['goal']}] 已验证达成。"
                f"置信度 {result['confidence']:.2f}。"
                f"证据: {', '.join(result['matched_evidence'])}"
            )
        else:
            return (
                f"【独立验收未通过】目标 [{result['goal']}] 未确认达成。"
                f"置信度 {result['confidence']:.2f}。"
                f"原因: {'; '.join(result['reasons'])}"
            )

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_screen_texts(self) -> List[str]:
        """获取屏幕所有文字（OCR）"""
        try:
            result_json = ocr_screen()
            data = json.loads(result_json)
            return [t.get("text", "") for t in data.get("texts", [])]
        except Exception as e:
            logger.warning(f"OCR 获取屏幕文字失败: {e}")
            return []

    def _match_keywords(self, keywords: List[str],
                        ocr_texts: List[str]) -> List[str]:
        """匹配关键词（支持模糊匹配：去除空格标点后包含关系）"""
        if not keywords:
            return []
        import re as _re
        obs = [t.lower().strip() for t in ocr_texts]
        matched = []
        for kw in keywords:
            kw_l = kw.lower().strip()
            # 关键词可能含空格（如 "USSR / 苏联"），拆成多个词，全部出现才匹配
            parts = [
                p.strip() for p in _re.split(r'[\s/|，,、]+', kw_l)
                if p.strip() and len(p.strip()) >= 1
            ]
            if not parts:
                continue
            # 关键词本身作为整体先试
            if any(kw_l in o for o in obs):
                matched.append(kw)
                continue
            # 多词条件：全部出现才匹配
            if len(parts) > 1 and all(any(p in o for o in obs) for p in parts):
                matched.append(kw)
                continue
            # 单次包含匹配
            if len(parts) == 1 and any(parts[0] in o for o in obs):
                matched.append(kw)
        return matched

    def get_verify_history(self) -> List[Dict[str, Any]]:
        """获取验收历史"""
        return self._verify_history