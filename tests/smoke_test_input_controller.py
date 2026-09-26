"""
输入控制器 + 假成功链修复 - 冒烟测试（标准库 unittest）
======================================================
覆盖：
  system.input_controller：坐标归一化 / 扩展键 / 按键映射 / UIPI 预检 / 失败路径
  agent.exception_handler.FailureDetector：硬失败信号优先于 "✅"
  agent.tools 输入工具：失败时返回失败信号（不再是假成功）

安全声明：本测试**不会**真实点击/按键/移动鼠标
（对 input_controller._send 与 click_at_point 做了打桩）。

运行：  python tests/smoke_test_input_controller.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoordinateNormalization(unittest.TestCase):
    def test_corners_and_center(self):
        from system import input_controller as ic
        # 左上角 → 0
        self.assertEqual(ic.to_absolute(0, 0), (0, 0))
        vx, vy, vw, vh = ic.get_virtual_screen()
        # 右下角 → 65535
        self.assertEqual(ic.to_absolute(vw - 1, vh - 1), (65535, 65535))
        # 中心 ≈ 32767/32768
        cx, cy = ic.to_absolute(vw // 2, vh // 2)
        self.assertTrue(32000 <= cx <= 33500, cx)
        self.assertTrue(32000 <= cy <= 33500, cy)

    def test_clamped(self):
        from system import input_controller as ic
        nx, ny = ic.to_absolute(999999, 999999)
        self.assertEqual((nx, ny), (65535, 65535))
        nx, ny = ic.to_absolute(-500, -500)
        self.assertEqual((nx, ny), (0, 0))

    def test_virtual_screen_valid(self):
        from system import input_controller as ic
        vx, vy, vw, vh = ic.get_virtual_screen()
        self.assertGreater(vw, 0)
        self.assertGreater(vh, 0)


class TestKeyMapping(unittest.TestCase):
    def test_known_keys(self):
        from system import input_controller as ic
        self.assertEqual(ic.key_name_to_vk("enter"), 0x0D)
        self.assertEqual(ic.key_name_to_vk("left"), 0x25)
        self.assertEqual(ic.key_name_to_vk("delete"), 0x2E)
        self.assertEqual(ic.key_name_to_vk("a"), 0x41)
        self.assertEqual(ic.key_name_to_vk("ctrl"), 0x11)

    def test_unknown_key(self):
        from system import input_controller as ic
        self.assertEqual(ic.key_name_to_vk("nonexistentkey"), 0)

    def test_extended_flags(self):
        """方向键/编辑键/右修饰键必须带扩展标志（本次修复的核心）"""
        from system import input_controller as ic
        for name in ("left", "right", "up", "down",
                     "insert", "delete", "home", "end",
                     "pageup", "pagedown", "rctrl", "ralt"):
            vk = ic.key_name_to_vk(name)
            self.assertTrue(ic.is_extended_vk(vk), f"{name} 应带扩展标志")
            self.assertEqual(ic._ext_flag(vk), ic.KEYEVENTF_EXTENDEDKEY)

        for name in ("a", "enter", "esc", "space", "f5", "ctrl", "shift"):
            vk = ic.key_name_to_vk(name)
            self.assertFalse(ic.is_extended_vk(vk), f"{name} 不应带扩展标志")
            self.assertEqual(ic._ext_flag(vk), 0)

    def test_extended_constant_value(self):
        from system import input_controller as ic
        self.assertEqual(ic.KEYEVENTF_EXTENDEDKEY, 0x0001)
        self.assertEqual(ic.MOUSEEVENTF_ABSOLUTE, 0x8000)


class TestFailurePathsNoRealInput(unittest.TestCase):
    """失败路径测试：全程打桩，不发送任何真实输入"""

    def test_move_to_reports_sendinput_failure(self):
        from system import input_controller as ic
        orig = ic._send
        try:
            ic._send = lambda inputs: 0          # 模拟输入被拦截
            r = ic.move_to(100, 100)
            self.assertFalse(r["ok"])
            self.assertIn("SendInput", r["error"])
        finally:
            ic._send = orig

    def test_press_key_unknown(self):
        from system import input_controller as ic
        r = ic.press_key("not_a_key_at_all")
        self.assertFalse(r["ok"])
        self.assertIn("未知键名", r["error"])

    def test_hotkey_unknown(self):
        from system import input_controller as ic
        r = ic.hotkey(["ctrl", "not_a_key"])
        self.assertFalse(r["ok"])
        self.assertIn("未知键名", r["error"])

    def test_uipi_conflict_blocks_before_input(self):
        """UIPI 冲突时必须在发送输入之前判失败，并提示以管理员运行"""
        from system import input_controller as ic
        orig_check = ic.check_uipi_conflict
        orig_send = ic._send
        sent = {"n": 0}

        def _count(inputs):
            sent["n"] += 1
            return len(inputs)

        try:
            ic.check_uipi_conflict = lambda hwnd=0: {
                "conflict": True, "self_elevated": False,
                "target_elevated": True, "target_title": "Setup.exe",
            }
            ic._send = _count
            r = ic.click_at_point(500, 500)
            self.assertFalse(r["ok"])
            self.assertTrue(r["uipi_conflict"])
            self.assertIn("管理员", r["error"])
            self.assertEqual(sent["n"], 0, "冲突时不应发送任何输入事件")
        finally:
            ic.check_uipi_conflict = orig_check
            ic._send = orig_send

    def test_describe_environment(self):
        from system import input_controller as ic
        env = ic.describe_environment()
        for key in ("dpi_aware", "virtual_screen", "cursor",
                    "self_elevated", "foreground_title"):
            self.assertIn(key, env)
        self.assertGreater(env["virtual_screen"]["width"], 0)


class TestFalseSuccessChainBroken(unittest.TestCase):
    """验证"输入无效却返回假成功"的链路已被打断"""

    def test_hard_failure_overrides_checkmark(self):
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()
        # 旧实现会因 "✅" 判成功 —— 现在必须判失败
        s = "✅ 已点击 (100, 200) 左键 [注意：鼠标可能被拦截，实际位置(0,0)]"
        self.assertFalse(h.is_success(s))

    def test_plain_success_still_ok(self):
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()
        self.assertTrue(h.is_success("✅ 已点击 (100, 200) 左键"))
        self.assertTrue(h.is_success("✅ 已输入: [hello]"))
        self.assertTrue(h.is_success("✅ 已按下 enter"))
        self.assertTrue(h.is_success("✅ 已按下 ctrl+c"))

    def test_failure_strings_detected(self):
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()
        for s in (
            "点击失败: 目标窗口[Setup.exe]以管理员权限运行，需以管理员身份运行",
            "输入失败: 剪贴板内容校验失败（可能被其他程序抢占）",
            "按键失败: SendInput 按键事件未送达（可能被 UIPI 拦截）(key=left)",
            "组合键失败: 组合键事件未全部送达（2/4，可能被 UIPI 拦截）",
            "拖拽失败: 拖拽起点移动失败: 光标未到位",
        ):
            self.assertFalse(h.is_success(s), s)

    def test_extended_key_success_not_flagged(self):
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()
        self.assertTrue(h.is_success("✅ 已按下 left [扩展键]"))


class TestToolsDelegateToController(unittest.TestCase):
    """工具层：失败时返回失败信号，成功时返回 ✅（全程打桩，不真实点击）"""

    def _patch(self, returned):
        from system import input_controller as ic
        orig = ic.click_at_point
        ic.click_at_point = lambda *a, **k: dict(returned)
        return ic, orig

    def test_click_at_uipi_failure(self):
        from agent import tools
        ic, orig = self._patch({
            "ok": False, "uipi_conflict": True,
            "error": "目标窗口[Setup.exe]以管理员权限运行 → UIPI 会丢弃模拟输入。",
        })
        try:
            out = tools.click_at.invoke({"x": 500, "y": 300})
            self.assertIn("点击失败", out)
            self.assertIn("管理员", out)
            # 该输出必须被判为失败（链路闭合）
            from agent.exception_handler import ExceptionHandler
            self.assertFalse(ExceptionHandler().is_success(out))
        finally:
            ic.click_at_point = orig

    def test_click_at_move_failure(self):
        from agent import tools
        ic, orig = self._patch({
            "ok": False, "error": "移动失败: SendInput 移动失败（可能被 UIPI 拦截）",
        })
        try:
            out = tools.click_at.invoke({"x": 1, "y": 2})
            self.assertIn("点击失败", out)
        finally:
            ic.click_at_point = orig

    def test_click_at_success(self):
        from agent import tools
        ic, orig = self._patch({
            "ok": True, "actual": [500, 300],
            "foreground": "记事本", "error": "",
        })
        try:
            out = tools.click_at.invoke({"x": 500, "y": 300})
            self.assertIn("✅ 已点击", out)
            self.assertIn("记事本", out)
            from agent.exception_handler import ExceptionHandler
            self.assertTrue(ExceptionHandler().is_success(out))
        finally:
            ic.click_at_point = orig

    def test_press_key_failure_and_success(self):
        from agent import tools
        from system import input_controller as ic
        orig = ic.press_key
        try:
            ic.press_key = lambda k, hold=0.03: {
                "ok": False, "vk": 0, "extended": False,
                "error": "SendInput 按键事件未送达（可能被 UIPI 拦截）",
            }
            out = tools.press_key.invoke({"key": "left"})
            self.assertIn("按键失败", out)

            ic.press_key = lambda k, hold=0.03: {
                "ok": True, "vk": 0x25, "extended": True, "error": "",
            }
            out = tools.press_key.invoke({"key": "left"})
            self.assertIn("✅ 已按下 left", out)
            self.assertIn("扩展键", out)
        finally:
            ic.press_key = orig

    def test_type_text_failure(self):
        from agent import tools
        from system import input_controller as ic
        orig = ic.type_text
        try:
            ic.type_text = lambda text, settle=0.2, restore_clipboard=True: {
                "ok": False, "error": "剪贴板内容校验失败（可能被其他程序抢占）",
            }
            out = tools.type_text.invoke({"text": "你好"})
            self.assertIn("输入失败", out)
        finally:
            ic.type_text = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)

