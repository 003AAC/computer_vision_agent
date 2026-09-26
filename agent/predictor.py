"""
Action Predictor - 动作后预测闭环
=================================
解决"agent 点击按钮以后不知道应该发生什么"的问题。

执行流程（由 core.py 协调）：
  click()
  ↓ predict_next_state()       # 设置预测：预期状态 + 应出现的元素
  ↓ wait()                     # 等待界面变化
  ↓ observe()                  # 观察（OCR 获取屏幕文字）
  ↓ compare()                  # 对比预测 vs 观察
  ↓ classify()                 # 分类：点击失败 / 加载中 / 异常弹窗 / 状态异常

预测来源优先级：
  1. LLM 回复中的 <predict> 块（显式声明）
  2. TaskPhaseMachine 当前阶段的 expected_next + entry_conditions
  3. WorkingMemory 设定的 expected_transition
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from vision.ocr import ocr_screen

logger = logging.getLogger(__name__)


class Prediction:
    """一次动作预测"""

    def __init__(
        self,
        action: str,
        args: Dict[str, Any],
        predicted_state: str = "",
        should_appear: List[str] = None,
        timeout: float = 3.0,
        source: str = "llm",
    ):
        self.action = action
        self.args = dict(args)
        self.predicted_state = predicted_state
        self.should_appear = should_appear or []
        self.timeout = timeout
        self.source = source
        self.timestamp = time.time()
        self.result: Optional[Dict[str, Any]] = None

    def is_expired(self, now: float = None) -> bool:
        now = now or time.time()
        return now - self.timestamp > self.timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "predicted_state": self.predicted_state,
            "should_appear": self.should_appear,
            "timeout": self.timeout,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class ActionPredictor:
    """动作预测协调器"""

    # 等待观察轮询间隔（秒）
    POLL_INTERVAL = 0.5
    # 最大观察次数（CPU 全屏 OCR 每次 2-5s，严格限制重扫次数）
    # 首次观察 + 退避重扫：0.5s→1s→2s，约 3.5s 窗口（与默认 timeout=3s 匹配）
    MAX_OBSERVATIONS = 3
    # 最大预测数量（短期内只保留最近的活动预测）
    MAX_PREDICTIONS = 3

    def __init__(self):
        self._active: List[Prediction] = []

    # ============================================================
    # 预测管理
    # ============================================================

    def predict(
        self,
        action: str,
        args: Dict[str, Any],
        predicted_state: str = "",
        should_appear: List[str] = None,
        timeout: float = 3.0,
        source: str = "llm",
    ) -> Prediction:
        """建立一次动作预测

        Args:
            action: 已执行的动作工具名
            args: 动作参数
            predicted_state: 预期的下一阶段/状态
            should_appear: 动作后应出现的元素/文字列表
            timeout: 等待超时秒数
            source: 预测来源（llm / phase / memory）

        Returns:
            创建的 Prediction
        """
        pred = Prediction(
            action, args, predicted_state, should_appear, timeout, source
        )
        self._active.append(pred)
        if len(self._active) > self.MAX_PREDICTIONS:
            self._active = self._active[-self.MAX_PREDICTIONS:]
        logger.info(
            f"建立预测: {action} → {predicted_state} "
            f"(应出现: {', '.join(should_appear or [])}, 超时 {timeout}s)"
        )
        return pred

    def clear(self):
        """清除所有预测"""
        self._active.clear()

    def get_latest(self) -> Optional[Prediction]:
        """获取最新的未过期预测"""
        for pred in reversed(self._active):
            if not pred.is_expired():
                return pred
        return None

    # ============================================================
    # 观察 / 对比 / 分类
    # ============================================================

    def observe(self, ocr_texts: List[str] = None) -> List[str]:
        """观察当前屏幕（获取文字列表）

        Args:
            ocr_texts: 可直接传入观察结果（供调用方复用 / mock 测试）
        """
        if ocr_texts is not None:
            return ocr_texts
        try:
            result_json = ocr_screen()
            data = json.loads(result_json)
            return [t.get("text", "") for t in data.get("texts", [])]
        except Exception as e:
            logger.warning(f"OCR 观察失败: {e}")
            return []

    def wait_and_observe(
        self,
        pred: Prediction,
        poll_interval: float = None,
        ocr_texts: List[str] = None,
    ) -> Dict[str, Any]:
        """等待并观察，直到预测条件满足或超时

        流程：
          1. 首次观察（OCR 获取屏幕文字）
          2. 若未命中，退避重扫（0.5s→1s→2s→4s，最多 MAX_OBSERVATIONS 次）
          3. 出现 → 预测通过；超时 → 预测失败

        ⚠️ 性能优化：不再每 poll_interval 全屏 OCR 轮询（CPU 每次 2-5s，
        会刷屏 + 拖慢节奏）。改为指数退避 + 观察次数上限。

        Args:
            pred: 预测对象
            poll_interval: 轮询间隔（已废弃，改用退避）
            ocr_texts: 可注入首次观察结果

        Returns:
            {
              "matched": bool,
              "found": [...],
              "missed": [...],
              "elapsed": float,
              "observation_count": int
            }
        """
        deadline = pred.timestamp + pred.timeout
        should_appear = pred.should_appear or []

        found: List[str] = []
        missed: List[str] = list(should_appear)
        elapsed = 0.0
        observation_count = 0
        last_observation: List[str] = list(ocr_texts) if ocr_texts else []

        # 首次观察
        if should_appear:
            first_obs = self.observe(ocr_texts)
            observation_count += 1
            if first_obs:
                last_observation = first_obs
            found, missed = self._split_matches(should_appear, first_obs)

        # 退避重扫（最多 MAX_OBSERVATIONS 次全屏 OCR，避免 CPU 无限刷屏）
        backoff = 0.5
        while missed and time.time() < deadline \
                and observation_count < self.MAX_OBSERVATIONS:
            time.sleep(backoff)
            backoff = min(backoff * 2, 2.0)
            obs = self.observe()
            observation_count += 1
            if obs:
                last_observation = obs
                found, missed = self._split_matches(should_appear, obs)
                if not missed:
                    break

        elapsed = time.time() - pred.timestamp
        result = {
            "matched": not missed,
            "found": found,
            "missed": missed,
            "elapsed": round(elapsed, 2),
            "observation_count": observation_count,
            # ⚡ 复用：调用方无需再全屏 OCR 一次（省 2-5s）
            "last_observation": last_observation,
        }
        pred.result = result
        return result

    def compare(
        self,
        pred: Prediction,
        observed_phase: str = "",
        ocr_texts: List[str] = None,
    ) -> Dict[str, Any]:
        """对比预测与观察结果

        Args:
            pred: 预测对象
            observed_phase: 观察到的任务阶段（可选）
            ocr_texts: 观察到的屏幕文字（可选）

        Returns:
            {
              "is_prediction_correct": bool,
              "matched_elements": [...],
              "missed_elements": [...],
              "phase_match": bool,
              "predicted_phase": str,
              "observed_phase": str
            }
        """
        should_appear = pred.should_appear or []
        obs = ocr_texts if ocr_texts is not None else self.observe()

        if should_appear:
            found, missed = self._split_matches(should_appear, obs)
        else:
            found, missed = [], []

        phase_match = True
        if pred.predicted_state and observed_phase:
            phase_match = pred.predicted_state.lower() == observed_phase.lower()

        is_correct = (not missed) and phase_match

        result = {
            "is_prediction_correct": is_correct,
            "matched_elements": found,
            "missed_elements": missed,
            "phase_match": phase_match,
            "predicted_phase": pred.predicted_state,
            "observed_phase": observed_phase,
        }
        pred.result = result
        return result

    def classify_failure(self, compare_result: Dict[str, Any], action: str) -> str:
        """对预测失败进行分类

        Returns:
            分类结果：'click_failed' / 'loading' / 'popup' / 'state_anomaly' / 'unknown'
        """
        if compare_result.get("is_prediction_correct"):
            return "ok"

        # 分类策略（启发式）
        missed = compare_result.get("missed_elements", [])

        # 动作是点击/拖拽但目标未出现 → 可能是点击失败
        if action in ("click_at", "drag_mouse") and missed:
            return "click_failed"

        # 动作是输入后界面未变 → 可能是需要加载
        if action in ("type_text", "press_key", "hotkey") and missed:
            return "loading"

        # 预测阶段与观察阶段不一致 → 状态异常（可能弹窗/卡住）
        if not compare_result.get("phase_match"):
            return "state_anomaly"

        return "unknown"

    def extended_wait(self, seconds: float = 2.0):
        """延长等待（加载中场景）"""
        time.sleep(seconds)

    # ============================================================
    # 内部工具
    # ============================================================

    def _split_matches(self, should_appear: List[str],
                       ocr_texts: List[str]) -> Tuple[List[str], List[str]]:
        """将应出现的元素分为已找到/未找到"""
        if not should_appear:
            return [], []
        obs = [t.lower().strip() for t in ocr_texts]
        found = []
        missed = []
        for target in should_appear:
            t_l = target.lower().strip()
            # 多词条件（空格分隔）全部出现才算
            parts = [p.strip() for p in t_l.replace("，", " ").replace(",", " ").split() if p.strip()]
            if not parts:
                continue
            if all(any(p in o for o in obs) for p in parts):
                found.append(target)
            else:
                missed.append(target)
        return found, missed