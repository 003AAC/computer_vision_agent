"""
回归测试（合并版）- 闭环智能体 + 成本规划 + 重规划 + 输入控制器
===============================================================
覆盖既有能力，确保键鼠模块重构后无回归：
  WorkingMemory / TaskPhaseMachine / GoalVerifier / ObjectTracker
  ActionPredictor / ExecutionLogger / AbstractExperience
  CostModel+BudgetTracker+select_best_plan
  LaunchFailureDetector+ReplanManager
  parse_planning_response(<plan>/<goal>/<plans>)
  ExceptionHandler 四大分类 + 硬失败信号
  input_controller 坐标/扩展键

运行：  python tests/smoke_test_regression.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWorkingMemory(unittest.TestCase):
    def test_lifecycle(self):
        from agent.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.init_task("打开Beholder并游玩", application="Beholder",
                     initial_state="desktop")
        wm.record_action("visual_locate", {"target_description": "start_button"})
        wm.set_expected_transition("main_menu", ["开始游戏"], timeout=3)
        wm.set_prediction("main_menu", timeout=3)
        wm.update_state_after_observation("main_menu", 0.85)
        d = wm.to_dict()
        self.assertEqual(d["current_application"], "Beholder")
        self.assertEqual(d["current_state"], "main_menu")
        self.assertIn("工作记忆", wm.build_instruction())

    def test_coordinate_extraction(self):
        from agent.memory.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.init_task("t")
        la = wm.record_action("click_at", {"x": 1292, "y": 386})
        self.assertEqual(la["coordinate"], [1292, 386])


class TestTaskPhaseMachine(unittest.TestCase):
    def test_evidence_transition(self):
        from agent.task_phase import TaskPhaseMachine, PhaseDefinition
        phases = [
            PhaseDefinition("DESKTOP", entry_conditions=["桌面"],
                            expected_next=["MAIN_MENU"]),
            PhaseDefinition("MAIN_MENU", entry_conditions=["开始游戏"],
                            expected_next=["GAME_RUNNING"], est_cost=4),
            PhaseDefinition("GAME_RUNNING", entry_conditions=["生命值"]),
        ]
        sm = TaskPhaseMachine(phases, start_phase="DESKTOP")
        self.assertEqual(sm.try_transition(["开始游戏", "退出"]), "MAIN_MENU")
        self.assertEqual(sm.try_transition(["生命值"]), "GAME_RUNNING")

    def test_est_cost_field(self):
        from agent.task_phase import PhaseDefinition
        p = PhaseDefinition("A", est_cost=7, alternatives=["alt1"])
        self.assertEqual(p.est_cost, 7.0)
        self.assertEqual(p.alternatives, ["alt1"])
        d = p.to_dict()
        self.assertIn("est_cost", d)
        self.assertEqual(PhaseDefinition.from_dict(d).est_cost, 7.0)

    def test_action_gating(self):
        from agent.task_phase import TaskPhaseMachine, PhaseDefinition
        sm = TaskPhaseMachine(
            [PhaseDefinition("P1", allowed_actions=["visual_find_text"])],
            start_phase="P1",
        )
        self.assertTrue(sm.is_action_allowed("visual_find_text"))
        self.assertFalse(sm.is_action_allowed("click_at"))


class TestGoalVerifier(unittest.TestCase):
    def test_hoi4_case(self):
        from agent.verifier import GoalVerifier
        v = GoalVerifier()
        v.set_spec_from_dict({
            "goal": "country_selected",
            "evidence": ["苏联", "USSR"],
            "failure_condition": ["选择国家"],
            "confidence_threshold": 0.5,
        })
        r = v.verify(ocr_texts=["USSR", "苏联"])
        self.assertTrue(r["verified"])

    def test_failure_condition_blocks(self):
        from agent.verifier import GoalVerifier
        v = GoalVerifier()
        v.set_spec_from_dict({"goal": "g", "evidence": ["苏联"],
                              "failure_condition": ["选择国家"],
                              "confidence_threshold": 0.5})
        r = v.verify(ocr_texts=["选择国家", "苏联"])
        self.assertFalse(r["verified"])


class TestTrackerPredictorLoggerExperience(unittest.TestCase):
    def test_tracker(self):
        from agent.tracker import ObjectTracker
        t = ObjectTracker()
        t.track("NPC", [700, 380, 800, 460], 0.91, screen_signature="s1")
        self.assertEqual(t.get("NPC").center, [750, 420])
        self.assertTrue(t.should_use_cached("NPC", "s1"))
        self.assertFalse(t.should_use_cached("NPC", "s2"))
        t.invalidate_by_click([750, 420])
        self.assertIsNone(t.get("NPC"))

    def test_predictor(self):
        from agent.predictor import ActionPredictor
        p = ActionPredictor()
        pred = p.predict("click_at", {"x": 1, "y": 2},
                         predicted_state="main_menu",
                         should_appear=["开始游戏"], timeout=0.1)
        ok = p.compare(pred, observed_phase="main_menu",
                       ocr_texts=["开始游戏"])
        self.assertTrue(ok["is_prediction_correct"])
        self.assertEqual(p.classify_failure(ok, "click_at"), "ok")
        bad = p.compare(pred, observed_phase="main_menu", ocr_texts=["桌面"])
        self.assertEqual(p.classify_failure(bad, "click_at"), "click_failed")

    def test_logger(self):
        from agent.logger import ExecutionLogger
        with tempfile.TemporaryDirectory() as tmp:
            lg = ExecutionLogger(logs_dir=tmp)
            lg.start_task("任务", goal_spec={"goal": "g"})
            lg.record_step({"step": 1, "action": {"tool": "click_at"}})
            lg.end_task(True, "完成", {"steps": 1})
            files = os.listdir(tmp)
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0]), encoding="utf-8") as f:
                content = f.read()
            self.assertIn("task_start", content)
            self.assertIn("task_end", content)

    def test_abstract_experience_no_coords(self):
        from agent.experience import AbstractExperienceStore
        with tempfile.TemporaryDirectory() as tmp:
            store = AbstractExperienceStore(path=os.path.join(tmp, "a.json"))
            store.add(application="Beholder",
                      action_pattern="NPC交互通过点击人物模型触发",
                      detection_method=["OCR寻找提示文本"],
                      failure_cases=["不要依赖固定坐标"])
            self.assertTrue(store.search("Beholder NPC"))
            instr = store.build_instruction("Beholder")
            self.assertIn("抽象经验", instr)
            self.assertNotIn("750", instr)


class TestCostPlanning(unittest.TestCase):
    def test_cost_ordering(self):
        from agent.cost_planner import CostModel
        self.assertGreater(CostModel.tool_cost("visual_read_text"),
                           CostModel.tool_cost("visual_find_text"))
        self.assertGreater(CostModel.tool_cost("visual_scan"),
                           CostModel.tool_cost("visual_locate_region"))

    def test_cached_discount_and_wait(self):
        from agent.cost_planner import CostModel
        self.assertLess(CostModel.tool_cost("visual_locate", cached=True),
                        CostModel.tool_cost("visual_locate", cached=False))
        self.assertEqual(CostModel.tool_cost("wait", {"seconds": 2.5}), 2.5)

    def test_budget(self):
        from agent.cost_planner import BudgetTracker
        b = BudgetTracker(safety_factor=1.5)
        b.set_from_plan(10.0)          # budget 15
        b.add_tool("visual_read_text")  # 6
        b.add_tool("visual_scan")       # 5 → 11
        self.assertFalse(b.over_budget())
        b.add_tool("visual_scan_grid")  # +8 → 19
        self.assertTrue(b.over_budget())
        self.assertIsNotNone(b.take_downgrade_hint())
        self.assertIsNone(b.take_downgrade_hint())   # 只提示一次
        self.assertIn("成本", b.build_instruction())

    def test_select_best_plan(self):
        from agent.cost_planner import (PlanCandidate, select_best_plan,
                                        estimate_phases_cost)
        cands = [
            PlanCandidate("A", approach="截图定位双击", est_total_cost=25,
                          risk="low"),
            PlanCandidate("B", approach="run_powershell 启动", est_total_cost=6,
                          risk="low"),
        ]
        self.assertEqual(select_best_plan(cands).name, "B")
        # 失效路径 → 排除依赖它的方案
        cands2 = [
            PlanCandidate("A", approach="开始菜单搜索", est_total_cost=12),
            PlanCandidate("B", approach="执行 D:\\原神.lnk", est_total_cost=3),
        ]
        self.assertEqual(
            select_best_plan(cands2, failed_paths=["D:\\原神.lnk"]).name, "A")
        self.assertIsNone(select_best_plan([]))
        self.assertGreater(estimate_phases_cost(
            [{"allowed_actions": ["visual_scan"]}]), 0)


class TestReplan(unittest.TestCase):
    def test_launch_failure_detection(self):
        from agent.replan_manager import LaunchFailureDetector
        args = {"command": 'Start-Process "D:\\原神.lnk"'}
        result = "命令执行失败\n错误: 系统找不到指定的文件。"
        self.assertTrue(LaunchFailureDetector.is_launch_failure(
            "run_powershell", args, result))
        self.assertEqual(LaunchFailureDetector.extract_target(
            "run_powershell", args, result), "D:\\原神.lnk")
        # 非启动失败
        self.assertFalse(LaunchFailureDetector.is_launch_failure(
            "visual_find_text", {"target_text": "开始"},
            "❌ OCR 未找到文字[开始]"))

    def test_replan_trigger_and_limit(self):
        from agent.replan_manager import ReplanManager
        rm = ReplanManager(threshold=2, max_replans=1)
        rm.record_launch_failure("D:\\原神.lnk", "快捷方式无效")
        self.assertIsNone(rm.should_replan())            # 1 次不够
        rm.record_launch_failure("D:\\原神.lnk", "快捷方式无效")
        reason = rm.should_replan()
        self.assertIsNotNone(reason)
        self.assertIn("D:\\原神.lnk", reason)
        self.assertIn("D:\\原神.lnk", rm.failed_paths())
        self.assertIn("禁止", rm.build_constraint(reason))
        rm.mark_replanned()
        rm.record_launch_failure("X.lnk")
        self.assertIsNone(rm.should_replan())            # 已达上限

    def test_success_resets(self):
        from agent.replan_manager import ReplanManager
        rm = ReplanManager(threshold=2)
        rm.record_launch_failure("A.lnk")
        rm.record_success()
        rm.record_launch_failure("A.lnk")
        self.assertIsNone(rm.should_replan())


class TestPlanningParse(unittest.TestCase):
    def test_parse_all_blocks(self):
        from agent.core import parse_planning_response
        content = """
<plan>
{"application": "Beholder", "initial_state": "DESKTOP",
 "phases": [{"name": "LAUNCH", "entry_conditions": ["开始"],
             "allowed_actions": ["run_powershell"], "est_cost": 2}],
 "est_total_cost": 9}
</plan>
<goal>
{"goal": "game_started", "evidence": ["开始游戏"], "failure_condition": []}
</goal>
<plans>
[{"name": "方案A", "approach": "PS启动", "est_total_cost": 9, "risk": "low"},
 {"name": "方案B", "approach": "图标双击", "est_total_cost": 20, "risk": "medium"}]
</plans>
"""
        p = parse_planning_response(content)
        self.assertEqual(p.get("application"), "Beholder")
        self.assertEqual(p.get("est_total_cost"), 9)
        self.assertEqual(len(p.get("phases", [])), 1)
        self.assertEqual(p.get("goal_spec", {}).get("goal"), "game_started")
        self.assertEqual(len(p.get("plan_candidates", [])), 2)

    def test_phase_from_dict_carries_cost(self):
        from agent.core import phase_from_dict
        pd = phase_from_dict({"name": "A", "est_cost": 5,
                              "alternatives": ["x"]})
        self.assertEqual(pd.est_cost, 5.0)
        self.assertEqual(pd.alternatives, ["x"])


class TestExceptionCategories(unittest.TestCase):
    def test_four_categories(self):
        from agent.exception_handler import ExceptionHandler, ErrorCategory
        h = ExceptionHandler()
        self.assertEqual(h.classify_category("❌ 未找到[开始]", "visual_locate"),
                         ErrorCategory.VISION_FAILURE)
        self.assertEqual(h.classify_category("点击失败: x", "click_at"),
                         ErrorCategory.ACTION_FAILURE)
        self.assertEqual(h.classify_category("检测到弹窗未响应", "visual_scan"),
                         ErrorCategory.STATE_ANOMALY)
        self.assertEqual(h.classify_category("weird", "run_powershell"),
                         ErrorCategory.UNKNOWN)

    def test_hard_failure_priority(self):
        from agent.exception_handler import ExceptionHandler
        h = ExceptionHandler()
        self.assertFalse(h.is_success(
            "✅ 已点击 (1,2) 左键 [注意：鼠标可能被拦截，实际位置(0,0)]"))
        self.assertTrue(h.is_success("✅ 已点击 (100, 200) 左键"))


class TestInputControllerCore(unittest.TestCase):
    """仅校验纯逻辑（不发真实输入）"""

    def test_normalization(self):
        from system import input_controller as ic
        self.assertEqual(ic.to_absolute(0, 0), (0, 0))
        vx, vy, vw, vh = ic.get_virtual_screen()
        self.assertEqual(ic.to_absolute(vw - 1, vh - 1), (65535, 65535))

    def test_extended_flags(self):
        from system import input_controller as ic
        for name in ("left", "delete", "home", "pageup", "rctrl"):
            self.assertTrue(ic.is_extended_vk(ic.key_name_to_vk(name)), name)
        for name in ("a", "enter", "esc", "ctrl"):
            self.assertFalse(ic.is_extended_vk(ic.key_name_to_vk(name)), name)

    def test_uipi_conflict_blocks_input(self):
        from system import input_controller as ic
        orig_check, orig_send = ic.check_uipi_conflict, ic._send
        counter = {"n": 0}
        try:
            ic.check_uipi_conflict = lambda hwnd=0: {
                "conflict": True, "self_elevated": False,
                "target_elevated": True, "target_title": "Setup.exe"}
            ic._send = lambda inputs: counter.__setitem__("n", counter["n"] + 1) or len(inputs)
            r = ic.click_at_point(400, 400)
            self.assertFalse(r["ok"])
            self.assertTrue(r["uipi_conflict"])
            self.assertIn("管理员", r["error"])
            self.assertEqual(counter["n"], 0)
        finally:
            ic.check_uipi_conflict = orig_check
            ic._send = orig_send


if __name__ == "__main__":
    unittest.main(verbosity=2)


