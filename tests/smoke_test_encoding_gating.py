"""
编码兜底 + 动作门卫 + OCR 缓存 + tracker 前台校验 - 冒烟测试
==========================================================
覆盖（不产生真实点击/按键；OCR 全程打桩）：
  1. config.setup_console_encoding / safe_print（GBK 流不崩）
  2. core._action_guard：阶段门控 + 禁止盲点（click_at 需坐标来源）
  3. vision.ocr 全屏 OCR 短 TTL 缓存 + invalidate
  4. predictor.wait_and_observe 返回 last_observation；MAX_OBSERVATIONS=3
  5. tracker 前台窗口一致性校验

运行：  python tests/smoke_test_encoding_gating.py
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _GbkStrictStream(io.TextIOBase):
    """模拟 GBK 严格编码的控制台（打印 ✅/✓ 会抛 UnicodeEncodeError），
    且没有 reconfigure（模拟被包装过的流）"""

    encoding = "gbk"
    errors = "strict"

    def __init__(self):
        super().__init__()
        self.data = ""

    def write(self, s):
        s.encode("gbk")          # 非 GBK 字符 → 抛异常
        self.data += s
        return len(s)


class TestEncodingFallback(unittest.TestCase):
    def test_safe_print_survives_gbk_stream(self):
        from config import safe_print
        old = sys.stdout
        stream = _GbkStrictStream()
        sys.stdout = stream
        try:
            # 不应抛异常（旧代码这里就是崩溃点）
            safe_print("  <- ✅ 已点击 (100, 200) 左键")
            safe_print("✓ ⚠️ ⚡ ❌")
        finally:
            sys.stdout = old

    def test_setup_console_encoding_no_raise(self):
        from config import setup_console_encoding
        old = sys.stdout
        sys.stdout = _GbkStrictStream()
        try:
            # 无 reconfigure 方法的流 → 返回 False 但不抛异常
            r = setup_console_encoding()
            self.assertIsInstance(r, bool)
        finally:
            sys.stdout = old

    def test_setup_works_on_real_stream(self):
        import subprocess
        # 真实进程内：调用后 print ✅ 不应崩
        code = (
            "import sys; sys.path.insert(0, r'%s'); "
            "from config import setup_console_encoding; "
            "setup_console_encoding(); "
            "print('  <- ✅ 已点击 (1, 2) 左键')"
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # 注意：子进程已把 stdout 重配置为 UTF-8，父进程显式按 UTF-8 解码
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, timeout=60)
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        self.assertEqual(p.returncode, 0, err[-500:])
        self.assertIn("已点击", out)


class TestActionGuard(unittest.TestCase):
    def _phase(self, allowed=None, name="DESKTOP"):
        from agent.task_phase import TaskPhaseMachine, PhaseDefinition
        return TaskPhaseMachine(
            [PhaseDefinition(name, allowed_actions=allowed or [])],
            start_phase=name,
        )

    def test_phase_gating_blocks_click(self):
        from agent.core import _action_guard
        phase = self._phase(allowed=["run_powershell", "visual_scan"])
        reason = _action_guard("click_at", {"x": 10, "y": 10}, phase,
                               {(10, 10)})
        self.assertIn("动作被拒绝", reason)
        self.assertIn("DESKTOP", reason)

    def test_phase_gating_allows_when_listed(self):
        from agent.core import _action_guard
        phase = self._phase(allowed=["click_at", "visual_find_text"])
        self.assertEqual(_action_guard("click_at", {"x": 10, "y": 10},
                                       phase, {(10, 10)}), "")

    def test_no_gating_when_phase_allows_all(self):
        """EXPLORING 兜底阶段不限制任何工具"""
        from agent.task_phase import TaskPhaseMachine
        from agent.core import _action_guard
        phase = TaskPhaseMachine()
        self.assertEqual(_action_guard("click_at", {"x": 1, "y": 2},
                                       phase, {(1, 2)}), "")

    def test_blind_click_blocked(self):
        from agent.core import _action_guard
        phase = self._phase(allowed=["click_at"])
        reason = _action_guard("click_at", {"x": 999, "y": 888}, phase, set())
        self.assertIn("禁止盲点", reason)
        self.assertIn("点击失败", reason)

    def test_known_coordinate_passes_with_tolerance(self):
        from agent.core import _action_guard
        phase = self._phase(allowed=["click_at"])
        # 容差 12px 内视为同一目标
        self.assertEqual(
            _action_guard("click_at", {"x": 353, "y": 421},
                          phase, {(350, 420)}), "")
        # 超出容差 → 拦截
        self.assertIn("禁止盲点",
                      _action_guard("click_at", {"x": 400, "y": 420},
                                    phase, {(350, 420)}))

    def test_non_guarded_tool_not_blocked(self):
        """run_powershell 是逃生通道，不受阶段门控限制"""
        from agent.core import _action_guard
        phase = self._phase(allowed=["visual_scan"])
        self.assertEqual(_action_guard("run_powershell",
                                       {"command": "echo 1"}, phase, set()), "")

    def test_guard_rejection_detected_as_failure(self):
        """闭环关键：被门卫拒绝 → 必须被判为失败（否则算"进展"会掩盖问题）"""
        from agent.core import _action_guard
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()

        phase = self._phase(allowed=["visual_scan"])
        gated = _action_guard("click_at", {"x": 1, "y": 1}, phase, {(1, 1)})
        self.assertTrue(gated)
        self.assertFalse(h.is_success(gated), gated)

        blind = _action_guard("click_at", {"x": 777, "y": 777},
                              self._phase(allowed=["click_at"]), set())
        self.assertFalse(h.is_success(blind), blind)

    def test_coordinate_extraction(self):
        from agent.core import _collect_coordinates
        sink = set()
        _collect_coordinates(
            '✅ OCR 找到文字[开始游戏]（exact匹配） -> 坐标(350, 420)，'
            '可直接 click_at(350, 420) 点击', sink)
        self.assertIn((350, 420), sink)
        _collect_coordinates(
            '{"texts":[{"text":"确定","cx": 900, "cy": 500}]}', sink)
        self.assertIn((900, 500), sink)

    def test_recent_window(self):
        from agent.core import _recent_known_coords
        history = [{(1, 1)}, {(2, 2)}, {(3, 3)}, {(4, 4)}]
        merged = _recent_known_coords(history, step=4, window=3)
        self.assertIn((2, 2), merged)
        self.assertNotIn((1, 1), merged)


class TestOcrCache(unittest.TestCase):
    def test_ttl_cache_and_invalidate(self):
        from vision import ocr
        calls = {"n": 0}
        orig = ocr._ocr_screen_uncached

        def fake():
            calls["n"] += 1
            return '{"screen":"1x1","texts":[],"count":0}'

        try:
            ocr._ocr_screen_uncached = fake
            ocr.invalidate_ocr_cache()

            a = ocr.ocr_screen()
            b = ocr.ocr_screen()
            self.assertEqual(calls["n"], 1)      # 命中缓存，只算一次
            self.assertEqual(a, b)

            ocr.invalidate_ocr_cache()
            ocr.ocr_screen()
            self.assertEqual(calls["n"], 2)      # 失效后重算

            ocr.ocr_screen(use_cache=False)
            self.assertEqual(calls["n"], 3)      # 强制重算
        finally:
            ocr._ocr_screen_uncached = orig
            ocr.invalidate_ocr_cache()

    def test_screen_texts_helper(self):
        from vision import ocr
        orig = ocr._ocr_screen_uncached
        try:
            ocr._ocr_screen_uncached = lambda: (
                '{"texts":[{"text":"确定","cx":10,"cy":20}],"count":1}'
            )
            ocr.invalidate_ocr_cache()
            texts = ocr._screen_texts()
            self.assertEqual(len(texts), 1)
            self.assertEqual(texts[0]["text"], "确定")
        finally:
            ocr._ocr_screen_uncached = orig
            ocr.invalidate_ocr_cache()


class TestPredictorObservation(unittest.TestCase):
    def test_max_observations_tightened(self):
        from agent.predictor import ActionPredictor
        self.assertEqual(ActionPredictor.MAX_OBSERVATIONS, 3)

    def test_last_observation_matched(self):
        from agent.predictor import ActionPredictor
        p = ActionPredictor()
        pred = p.predict("click_at", {}, predicted_state="MAIN_MENU",
                         should_appear=["开始游戏"], timeout=0.05)
        r = p.wait_and_observe(pred, ocr_texts=["开始游戏", "退出"])
        self.assertTrue(r["matched"])
        self.assertIn("last_observation", r)
        self.assertIn("开始游戏", r["last_observation"])

    def test_last_observation_on_miss(self):
        """未命中时也应返回末次观察（供调用方复用，避免再全屏 OCR）"""
        from agent.predictor import ActionPredictor
        p = ActionPredictor()
        # 打桩 observe，避免真实全屏 OCR
        p.observe = lambda ocr_texts=None: (
            ocr_texts if ocr_texts is not None else ["无关文字"]
        )
        pred = p.predict("click_at", {}, predicted_state="X",
                         should_appear=["不可能出现的文字"],
                         timeout=0.2)
        r = p.wait_and_observe(pred)
        self.assertFalse(r["matched"])
        self.assertIn("last_observation", r)
        self.assertEqual(r["last_observation"], ["无关文字"])
        # 退避次数受限（≤ MAX_OBSERVATIONS）
        self.assertLessEqual(r["observation_count"],
                             ActionPredictor.MAX_OBSERVATIONS)


class TestTrackerForeground(unittest.TestCase):
    def test_foreground_mismatch_invalidates_cache(self):
        from agent.tracker import ObjectTracker
        t = ObjectTracker()
        t.track("NPC", [100, 100, 200, 200], 0.9,
                screen_signature="s1", foreground="Game A")
        self.assertTrue(t.should_use_cached("NPC", "s1", "Game A"))
        self.assertFalse(t.should_use_cached("NPC", "s1", "Game B"))
        self.assertTrue(t.foreground_changed("NPC", "Game B"))
        obj = t.get("NPC")
        self.assertEqual(obj.foreground, "Game A")
        self.assertIn("foreground", obj.to_dict())

    def test_backward_compat_without_foreground(self):
        from agent.tracker import ObjectTracker
        t = ObjectTracker()
        t.track("X", [1, 1, 9, 9], 0.5, screen_signature="s")
        self.assertTrue(t.should_use_cached("X", "s"))
        self.assertFalse(t.foreground_changed("X", "Anything"))


class TestFakeClickDetection(unittest.TestCase):
    """假点击检测：SendInput 报成功但界面无变化 → 不能算成功"""

    def _setup(self, sig_sequence):
        from agent import tools
        from system import input_controller as ic
        tools.reset_fake_click_state()
        orig_click = ic.click_at_point
        orig_sig = tools._screen_sig
        ic.click_at_point = lambda *a, **k: {
            "ok": True, "actual": [10, 10], "foreground": "Game", "error": "",
        }
        seq = list(sig_sequence)

        def _sig():
            return seq.pop(0) if seq else "END"

        tools._screen_sig = _sig
        return tools, ic, orig_click, orig_sig

    def _teardown(self, tools, ic, orig_click, orig_sig):
        ic.click_at_point = orig_click
        tools._screen_sig = orig_sig
        tools.reset_fake_click_state()

    def test_single_click_no_change_warns(self):
        tools, ic, oc, os_ = self._setup(["SAME", "SAME"])
        try:
            out = tools.click_at.invoke({"x": 100, "y": 200})
            self.assertIn("✅ 已点击", out)
            self.assertIn("界面无变化", out)      # 单次只警告
        finally:
            self._teardown(tools, ic, oc, os_)

    def test_repeat_click_no_change_is_failure(self):
        from agent.exception_handler import ExceptionHandler
        tools, ic, oc, os_ = self._setup(["SAME", "SAME", "SAME", "SAME"])
        try:
            tools.click_at.invoke({"x": 100, "y": 200})
            second = tools.click_at.invoke({"x": 100, "y": 200})
            self.assertIn("点击失败", second)
            self.assertIn("假点击", second)
            self.assertFalse(ExceptionHandler().is_success(second),
                             "假点击必须被判为失败")
        finally:
            self._teardown(tools, ic, oc, os_)

    def test_different_target_not_flagged(self):
        tools, ic, oc, os_ = self._setup(["SAME", "SAME", "SAME", "SAME"])
        try:
            tools.click_at.invoke({"x": 100, "y": 200})
            # 换一个明显不同的坐标 → 不算重复点击
            out = tools.click_at.invoke({"x": 900, "y": 800})
            self.assertIn("✅ 已点击", out)
            self.assertNotIn("假点击", out)
        finally:
            self._teardown(tools, ic, oc, os_)

    def test_screen_changed_not_flagged(self):
        tools, ic, oc, os_ = self._setup(["A", "B", "A", "B"])
        try:
            tools.click_at.invoke({"x": 100, "y": 200})
            out = tools.click_at.invoke({"x": 100, "y": 200})
            self.assertIn("✅ 已点击", out)
            self.assertNotIn("假点击", out)
        finally:
            self._teardown(tools, ic, oc, os_)

    def test_repeat_within_tolerance_flagged(self):
        """容差 8px 内视为同一坐标（重复点击同一按钮）"""
        tools, ic, oc, os_ = self._setup(["S", "S", "S", "S"])
        try:
            tools.click_at.invoke({"x": 100, "y": 200})
            out = tools.click_at.invoke({"x": 104, "y": 203})
            self.assertIn("假点击", out)
        finally:
            self._teardown(tools, ic, oc, os_)


if __name__ == "__main__":
    unittest.main(verbosity=2)

