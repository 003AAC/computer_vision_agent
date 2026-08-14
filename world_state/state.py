"""
World State - 模块化领域状态类
===============================
不再使用单一 JSON，而是 5 个独立的模块化 State 类，各自自治。

每个 State 类：
  - 用 StateField（包装 Belief）表示字段
  - add_evidence() 更新证据 → 自动计算 belief
  - to_dict() / from_dict() 序列化
"""
from typing import Any, Dict, List, Optional

from world_state.base import StateBase, StateField


class EnvironmentState(StateBase):
    """环境状态：OS / 分辨率 / 活动窗口 / 时区"""

    domain = "environment"

    def __init__(self):
        self.os = StateField("os")
        self.screen_resolution = StateField("screen_resolution")
        self.active_window = StateField("active_window")
        self.locale = StateField("locale")


class UIState(StateBase):
    """UI 状态：活动窗口 / 弹窗 / 焦点 / 可见元素 / 视觉验收"""

    domain = "ui"

    def __init__(self):
        self.active_window = StateField("active_window")
        self.popup_present = StateField("popup_present")
        self.popup_title = StateField("popup_title")
        self.focus_lost = StateField("focus_lost")
        self.visible_texts = StateField("visible_texts")   # 最近 OCR 识别的关键文字
        # Florence 视觉验收（辅助证据）
        self.visual_verification = StateField("visual_verification")


class ApplicationState(StateBase):
    """应用状态：安装 / 版本 / 打开 / 进程 / 窗口"""

    domain = "applications"

    def __init__(self, name: str = ""):
        self.name = name
        self.installed = StateField("installed")
        self.version = StateField("version")
        self.opened = StateField("opened")
        self.status = StateField("status")   # ready / starting / running / crashed
        # 进程证据
        self.process_running = StateField("process_running")
        self.pid = StateField("pid")
        # 窗口证据
        self.window_title = StateField("window_title")
        self.visible = StateField("visible")


class FileState(StateBase):
    """文件状态：存在 / 路径 / 类型 / 编译"""

    domain = "files"

    def __init__(self, path: str = ""):
        self.path = path
        self.exists = StateField("exists")
        self.is_dir = StateField("is_dir")
        self.size = StateField("size")
        self.compiled = StateField("compiled")


class TaskState(StateBase):
    """任务状态：goal / stage / completed / pending"""

    domain = "task"

    def __init__(self, goal: str = ""):
        self.goal = StateField("goal")
        self.current_stage = StateField("current_stage")
        self.completed = StateField("completed")
        self.pending = StateField("pending")

    def init_goal(self, goal: str):
        self.goal.add_evidence("task", goal, "任务初始化")

    def set_stage(self, stage: str):
        self.current_stage.add_evidence("task", stage, "阶段推进")

    def mark_completed(self, item: str):
        completed = self.completed.value or []
        if item not in completed:
            new_list = list(completed) + [item]
            self.completed.add_evidence("task", new_list, f"完成: {item}")
        pending = self.pending.value or []
        if item in pending:
            new_pending = [p for p in pending if p != item]
            self.pending.add_evidence("task", new_pending, f"移除待办: {item}")

    def add_pending(self, item: str):
        pending = self.pending.value or []
        if item not in pending:
            new_list = list(pending) + [item]
            self.pending.add_evidence("task", new_list, f"新增待办: {item}")