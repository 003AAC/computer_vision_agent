import unittest
from unittest.mock import patch

from agent.action_result import action_result, annotate_action_result, parse_action_result
from agent.powershell_policy import requires_powershell_confirmation
from agent.task_phase import PhaseDefinition, TaskPhaseMachine
from agent.tools import _win32_mouse_click
from agent.tool_policy import requires_sequential_execution
from world_state.manager import WorldStateManager
from agent.exception_handler import FailureDetector
from agent.state_machine import AgentStateMachine
from agent.predictor import ActionPredictor
from agent.memory.working_memory import WorkingMemory
from agent.core import _run_prediction_closure


class ActionResultTests(unittest.TestCase):
    def test_dispatched_action_is_not_reported_as_verified(self):
        result = parse_action_result(
            action_result("click_at", "dispatched", "click sent")
        )
        self.assertEqual(result["execution_status"], "dispatched")
        self.assertEqual(result["verification_status"], "unverified")

    def test_verification_status_and_observation_are_preserved(self):
        raw = action_result("click_at", "dispatched", "click sent")
        result = parse_action_result(annotate_action_result(raw, "verified", ["Next"]))
        self.assertEqual(result["verification_status"], "verified")
        self.assertEqual(result["observed_texts"], ["Next"])

    def test_failure_detector_distinguishes_dispatch_from_failed_verification(self):
        detector = FailureDetector()
        dispatched = action_result("click_at", "dispatched", "click sent")
        failed = annotate_action_result(dispatched, "failed", [])
        self.assertFalse(detector.detect(dispatched))
        self.assertTrue(detector.detect(failed))

    def test_pyautogui_fallback_honors_mouse_button_and_double_click(self):
        with patch("agent.tools.WIN32_AVAILABLE", False), patch(
            "agent.tools.pyautogui.click"
        ) as click:
            _win32_mouse_click("right")
            click.assert_called_once_with(button="right", clicks=1, interval=0.05)

        with patch("agent.tools.WIN32_AVAILABLE", False), patch(
            "agent.tools.pyautogui.click"
        ) as click:
            _win32_mouse_click("double")
            click.assert_called_once_with(button="left", clicks=2, interval=0.05)


class PhasePolicyTests(unittest.TestCase):
    def test_phase_action_list_is_recommendation_not_click_acl(self):
        phase = PhaseDefinition("TEST", allowed_actions=["visual_scan"])
        machine = TaskPhaseMachine([phase], start_phase="TEST")
        self.assertTrue(machine.is_action_allowed("click_at"))
        instruction = machine.build_instruction()
        self.assertIn("推荐操作", instruction)
        self.assertIn("不是硬性限制", instruction)

    def test_phase_transition_uses_observed_evidence(self):
        machine = TaskPhaseMachine([
            PhaseDefinition("START", expected_next=["READY"]),
            PhaseDefinition("READY", entry_conditions=["Continue"]),
        ], start_phase="START")
        self.assertEqual(machine.try_transition(["Continue"]), "READY")

    def test_phase_prediction_conditions_accept_any_configured_entry_condition(self):
        from agent.predictor import ActionPredictor

        predictor = ActionPredictor()
        self.assertEqual(
            predictor._split_matches(
                ["Continue", "Next"], ["Next"], match_any=True
            ),
            (["Next"], []),
        )
        self.assertEqual(
            predictor._split_matches(["Continue", "Next"], ["Next"]),
            (["Next"], ["Continue"]),
        )

    def test_action_batch_is_rejected_before_any_follow_up_action(self):
        actions = {"click_at", "type_text", "run_powershell"}
        self.assertTrue(requires_sequential_execution(
            ["visual_find_text", "click_at"], actions
        ))
        self.assertFalse(requires_sequential_execution(
            ["visual_scan", "visual_find_text"], actions
        ))
        self.assertFalse(requires_sequential_execution(["click_at"], actions))


class ProgressTests(unittest.TestCase):
    def test_only_verified_progress_resets_failure_streak(self):
        machine = AgentStateMachine(max_steps=20)
        machine.start()
        machine.record_step(["click_at"], has_progress=False, is_failure=True)
        self.assertEqual(machine.get_status_payload()["consecutive_failures"], 1)
        machine.record_step(["click_at"], has_progress=False, is_failure=False)
        self.assertEqual(machine.get_status_payload()["consecutive_failures"], 1)
        machine.record_step(["click_at"], has_progress=True, is_failure=False)
        self.assertEqual(machine.get_status_payload()["consecutive_failures"], 0)

    def test_preexisting_expected_text_cannot_verify_a_click(self):
        phase = TaskPhaseMachine([
            PhaseDefinition(
                "START",
                expected_next=["READY"],
            ),
            PhaseDefinition(
                "READY",
                entry_conditions=["Continue", "Next"],
            ),
        ], start_phase="START")
        predictor = ActionPredictor()
        predictor.wait_and_observe = lambda pred: {
            "matched": True,
            "found": ["Continue"],
            "missed": [],
            "observed_texts": ["Continue"],
            "elapsed": 0.1,
            "observation_count": 1,
        }
        result = _run_prediction_closure(
            predictor, phase, WorkingMemory(),
            "click_at", {"x": 1, "y": 1}, "",
            pre_action_texts=["Continue"],
            pre_observation_valid=True,
        )
        self.assertEqual(result["verification_status"], "unverified")
        self.assertEqual(result["classification"], "expected_evidence_preexisted")

    def test_new_expected_screen_text_verifies_the_action(self):
        phase = TaskPhaseMachine([
            PhaseDefinition("START", expected_next=["READY"]),
            PhaseDefinition("READY", entry_conditions=["Continue", "Next"]),
        ], start_phase="START")
        predictor = ActionPredictor()
        predictor.wait_and_observe = lambda pred: {
            "matched": True,
            "found": ["Next"],
            "missed": [],
            "observed_texts": ["Next"],
            "elapsed": 0.1,
            "observation_count": 1,
        }
        result = _run_prediction_closure(
            predictor, phase, WorkingMemory(),
            "click_at", {"x": 1, "y": 1}, "",
            pre_action_texts=[],
            pre_observation_valid=True,
        )
        self.assertEqual(result["verification_status"], "verified")

    def test_invalid_post_action_ocr_is_not_classified_as_action_failure(self):
        phase = TaskPhaseMachine([
            PhaseDefinition("START", expected_next=["READY"]),
            PhaseDefinition("READY", entry_conditions=["Continue"]),
        ], start_phase="START")
        predictor = ActionPredictor()
        predictor.wait_and_observe = lambda pred: {
            "matched": False,
            "found": [],
            "missed": ["Continue"],
            "observed_texts": [],
            "observation_valid": False,
            "elapsed": 0.1,
            "observation_count": 1,
        }
        result = _run_prediction_closure(
            predictor, phase, WorkingMemory(),
            "click_at", {"x": 1, "y": 1}, "",
            pre_action_texts=[],
            pre_observation_valid=True,
        )
        self.assertEqual(result["verification_status"], "unverified")
        self.assertEqual(result["classification"], "post_observation_unavailable")


class PowerShellPolicyTests(unittest.TestCase):
    def test_destructive_and_file_mutation_commands_require_confirmation(self):
        for command in (
            "Remove-Item C:\\temp\\file.txt",
            "rm -rf C:\\temp",
            "Set-Content C:\\temp\\file.txt hello",
            "Stop-Process -Name notepad",
            "Set-ExecutionPolicy Unrestricted",
            "powershell -EncodedCommand SQBFAFgA",
            "cmd.exe /c del C:\\temp\\file.txt",
        ):
            with self.subTest(command=command):
                self.assertTrue(requires_powershell_confirmation(command))

    def test_read_only_and_normal_launch_commands_do_not_require_confirmation(self):
        self.assertFalse(requires_powershell_confirmation("Get-Process notepad"))
        self.assertFalse(requires_powershell_confirmation("Start-Process notepad"))


class WorldStateEvidenceTests(unittest.TestCase):
    def test_visible_text_state_represents_latest_observation(self):
        manager = WorldStateManager()
        manager.observe_screen_texts(["Old screen"])
        manager.observe_screen_texts(["New screen"])
        self.assertEqual(manager.ui.visible_texts.value, ["New screen"])
        self.assertEqual(manager.ui.visible_texts.status, "uncertain")

    def test_test_path_false_is_not_written_as_existing(self):
        manager = WorldStateManager()
        manager._autowrite_powershell(
            "命令执行成功\nFalse",
            "Test-Path -LiteralPath 'C:\\temp\\missing file.txt'",
        )
        file_state = manager.files["C:\\temp\\missing file.txt"]
        self.assertIs(file_state.exists.value, False)
        self.assertIsNone(file_state.is_dir.value)

    def test_start_process_acknowledgement_does_not_claim_process_running(self):
        manager = WorldStateManager()
        manager._autowrite_powershell(
            "命令执行成功（GUI程序已启动）",
            "Start-Process notepad",
        )
        self.assertEqual(manager.applications, {})


if __name__ == "__main__":
    unittest.main()
