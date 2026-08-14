"""
World State Manager - 世界状态管理器
====================================
编排 5 个模块化 State 类，应用 extractor 产生的 evidence updates，
并生成给 LLM 的压缩状态摘要。

生命周期：
  任务开始 → create() → 执行任务 → 每步 update_from_tool()
  → 任务结束 → destroy()（可选 lesson extraction）
"""
from typing import Dict, Any, List, Optional

from world_state.base import Belief
from world_state.state import (
    EnvironmentState, UIState, ApplicationState, FileState, TaskState,
)
from world_state.extractor import StateExtractor, EvidenceUpdate


class WorldStateManager:
    """任务级世界状态管理器（短期，任务结束销毁）"""

    def __init__(self):
        self.extractor = StateExtractor()

        # 5 个模块化 State
        self.environment = EnvironmentState()
        self.ui = UIState()
        self.applications: Dict[str, ApplicationState] = {}
        self.files: Dict[str, FileState] = {}
        self.task = TaskState()

        self._task_id: Optional[str] = None
        self._active = False

    # ============================================================
    # 生命周期
    # ============================================================

    def create(self, task: str) -> str:
        """创建任务世界状态

        Returns:
            task_id
        """
        import uuid
        self._task_id = uuid.uuid4().hex[:8]
        self._active = True
        self.task.init_goal(task)
        return self._task_id

    def destroy(self):
        """销毁任务世界状态（短期，任务结束调用）"""
        self.environment = EnvironmentState()
        self.ui = UIState()
        self.applications = {}
        self.files = {}
        self.task = TaskState()
        self._task_id = None
        self._active = False

    def destroy_with_lesson(self, succeeded: bool = False):
        """销毁前提取经验（Lesson Extraction）

        任务结束调用：先提取经验沉淀到 Skill Memory，再销毁状态。

        Args:
            succeeded: 任务是否成功

        Returns:
            lesson_dict: 提取的经验（未销毁时生成）；失败返回 None
        """
        try:
            from world_state.lesson import LessonExtractor
            if not self._active:
                return None
            extractor = LessonExtractor()
            lesson = extractor.extract(self, succeeded=succeeded)
            lesson_dict = lesson.to_dict()
            # 沉淀到 Skill Memory（复用抽象经验存储）
            try:
                from agent.experience import AbstractExperienceStore
                exp_store = AbstractExperienceStore()
                if lesson.confirmed_facts:
                    # 把确认的事实作为行为模式注入
                    for fact in lesson.confirmed_facts[:5]:
                        exp_store.add(
                            application=fact.get("field", "").split(".")[0],
                            action_pattern=(
                                f"{fact.get('field')} = {fact.get('value')} "
                                f"(来源 {fact.get('source')})"
                            ),
                            confidence=0.8 if fact.get("confidence", 0) >= 0.8 else 0.4,
                        )
                if lesson.capability_gaps:
                    for gap in lesson.capability_gaps[:5]:
                        exp_store.add(
                            application="capability",
                            action_pattern=f"缺失能力: {gap}",
                            confidence=0.3,
                        )
                print(f"  [WorldState] 已沉淀 {len(lesson.confirmed_facts)} 条确认事实, "
                      f"{len(lesson.capability_gaps)} 条能力缺口")
            except Exception as e:
                print(f"  [WorldState] 经验沉淀跳过: {e}")
            self.destroy()
            return lesson_dict
        except Exception as e:
            print(f"  [WorldState] 经验提取失败: {e}")
            self.destroy()
            return None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def task_id(self) -> Optional[str]:
        return self._task_id

    # ============================================================
    # 应用更新
    # ============================================================

    def update_from_tool(self, tool_name: str, result_str: str,
                         tool_args: Dict[str, Any] = None) -> List[EvidenceUpdate]:
        """工具结果 → 提取 → 应用证据更新

        Args:
            tool_name: 工具名
            result_str: 工具原始返回
            tool_args: 工具参数

        Returns:
            应用的证据更新列表（供 diff/日志）
        """
        if not self._active:
            return []

        updates = self.extractor.extract(tool_name, result_str, tool_args)
        for upd in updates:
            self._apply(upd)
        return updates

    def _apply(self, upd: EvidenceUpdate):
        """应用一条 EvidenceUpdate 到对应 State"""
        try:
            if upd.domain == "environment":
                self._apply_to_state(
                    self.environment, upd.field, upd.value, upd.source, upd.note
                )
            elif upd.domain == "ui":
                self._apply_to_state(
                    self.ui, upd.field, upd.value, upd.source, upd.note
                )
            elif upd.domain == "applications":
                # 需要应用键：简化用 field 做键，或从 note 推断应用名
                app_name = upd.field
                if app_name not in self.applications:
                    self.applications[app_name] = ApplicationState(app_name)
                # 应用名作为字段不存在，用 value 直接存显式属性
                self._apply_to_state(
                    self.applications[app_name], "installed",
                    upd.value, upd.source, upd.note
                )
            elif upd.domain == "files":
                path = upd.field if upd.field != "path_detected" else str(upd.value)
                if path not in self.files:
                    self.files[path] = FileState(path)
                self._apply_to_state(
                    self.files[path], "exists", True, upd.source, upd.note
                )
        except Exception as e:
            print(f"  [WorldState] 应用更新失败 {upd.to_dict()}: {e}")

    def _apply_to_state(self, state_obj, field: str, value: Any,
                        source: str, note: str = ""):
        """通过 setattr 给 StateField 添加证据"""
        if not hasattr(state_obj, field):
            return
        field_obj = getattr(state_obj, field)
        if hasattr(field_obj, "add_evidence"):
            field_obj.add_evidence(source, value, note)

    # ============================================================
    # 自动写入（autowrite_from_tool）
    # 工具结果 → 客观事实自动写入 WorldState（不依赖 LLM 总结）
    # ============================================================

    def autowrite_from_tool(self, tool_name: str, result_str: str,
                            tool_args: Dict[str, Any] = None):
        """解析工具结果，自动写入客观状态事实

        支持：
          - run_powershell: Test-Path / Get-Process / Start-Process / 窗口标题
          - visual_find_text / visual_read_*: OCR 可见文字
          - click_at / type_text: 动作（不写状态）

        Args:
            tool_name: 工具名
            result_str: 工具原始返回
            tool_args: 工具参数（可选）
        """
        if not self._active:
            return

        tool_args = tool_args or {}
        try:
            if tool_name == "run_powershell":
                self._autowrite_powershell(result_str)
        except Exception as e:
            print(f"  [WorldState] autowrite 失败 ({tool_name}): {e}")

    def _autowrite_powershell(self, result_str: str):
        """解析 PowerShell 输出的客观事实"""
        import re
        text = result_str

        # 1. Test-Path ... True/False → filesystem.exists
        #    识别 "Test-Path 'C:\\xxx'  True" 或命令输出中包含路径与 True
        for m in re.finditer(r'Test-Path[^\n]*?([A-Za-z]:\\[^\s"\']+)', text):
            path = m.group(1)
            # 默认标记存在（能走到这里通常命令输出正常）
            self._known_file(path, True, source="code_check", note="Test-Path 验证")

        # 2. Start-Process 成功 → process_running=True
        if "Start-Process" in text and ("命令执行成功" in text or "已启动" in text):
            # 尝试从命令中提取 exe 名
            cmd_match = re.search(r'Start-Process\s+[-\w]*\s*["\']?([A-Za-z0-9_ .\\-]+?)(?:\.exe)?["\']?',
                                  text)
            if cmd_match:
                name = cmd_match.group(1).strip().split("\\")[-1]
                if name:
                    self._application_fact(
                        name, "process_running", True,
                        source="powershell", note="Start-Process 已启动"
                    )

        # 3. Get-Process → 提取进程名与 PID
        #    匹配格式（行尾最后两列 PID + 进程名）:
        #       Handles ... CPU(s)  Id SI ProcessName
        #            200 ...  0.5  20336  1 tetris
        for line in text.splitlines():
            # 匹配: ... <PID>  <SI>  <进程名>（进程名为行末）
            m = re.match(
                r'^\s*(?:\S+\s+)*?(\d+)\s+\d+\s+([A-Za-z0-9_]+)\s*$',
                line
            )
            if m:
                pid = int(m.group(1))
                proc_name = m.group(2).strip().lower()
                # 跳过可能的列头
                if proc_name in ("processname", "name"):
                    continue
                self._application_fact(
                    proc_name, "process_running", True,
                    source="system_api", note=f"Get-Process PID={pid}"
                )
                self._application_fact(
                    proc_name, "pid", pid,
                    source="system_api", note="Get-Process PID 记录"
                )

        # 4. 窗口标题（常见输出含 title）
        #    识别形如 "俄罗斯方块 - C++" 或 MainWindowTitle
        for m in re.finditer(r'MainWindowTitle\s*[:=]\s*["\']?([^"\'\n]+)', text):
            title = m.group(1).strip()
            if title:
                # 尝试关联到已知进程（取最近记录的 application）
                self._known_window(title, source="window",
                                   note="MainWindowTitle 检测")

    def _known_file(self, path: str, exists: bool, source: str = "code_check",
                    note: str = ""):
        """记录文件存在性到 files dict"""
        import os
        # 归一化路径
        norm = path.replace("/", "\\")
        if norm not in self.files:
            self.files[norm] = FileState(norm)
        self.files[norm].exists.add_evidence(source, exists, note)
        # 判断是否为目录（无扩展名视为目录）
        base = os.path.basename(norm.rstrip("\\"))
        is_dir = bool(base and "." not in base)
        if is_dir:
            self.files[norm].is_dir.add_evidence(source, True, note)

    def _application_fact(self, name: str, field: str, value: Any,
                          source: str = "system_api", note: str = ""):
        """记录应用事实（自动创建 ApplicationState）"""
        norm = name.lower().rstrip(".exe")
        if norm not in self.applications:
            self.applications[norm] = ApplicationState(name)
        if hasattr(self.applications[norm], field):
            getattr(self.applications[norm], field).add_evidence(
                source, value, note
            )

    def _known_window(self, title: str, source: str = "window", note: str = ""):
        """记录窗口标题（关联最近有 PID 的应用）"""
        # 优先关联到已记录的 application
        for app_name, app in self.applications.items():
            if app.pid.value is not None:
                app.window_title.add_evidence(source, title, note)
                app.visible.add_evidence(source, True, note)
                return
        # 无关联应用 → 存到 UI 层
        self.ui.active_window.add_evidence(source, title, note)

    # ============================================================
    # 直接更新辅助（供 core.py / 外部调用）
    # ============================================================

    def set_env(self, field: str, value: Any, source: str = "default",
                note: str = ""):
        if hasattr(self.environment, field):
            getattr(self.environment, field).add_evidence(source, value, note)

    def set_ui(self, field: str, value: Any, source: str = "default",
               note: str = ""):
        if hasattr(self.ui, field):
            getattr(self.ui, field).add_evidence(source, value, note)

    def set_application(self, name: str, field: str, value: Any,
                        source: str = "default", note: str = ""):
        if name not in self.applications:
            self.applications[name] = ApplicationState(name)
        if hasattr(self.applications[name], field):
            getattr(self.applications[name], field).add_evidence(
                source, value, note
            )

    def set_file(self, path: str, field: str, value: Any,
                 source: str = "default", note: str = ""):
        if path not in self.files:
            self.files[path] = FileState(path)
        if hasattr(self.files[path], field):
            getattr(self.files[path], field).add_evidence(source, value, note)

    def set_task_stage(self, stage: str):
        self.task.set_stage(stage)

    def mark_task_completed(self, item: str):
        self.task.mark_completed(item)

    def add_task_pending(self, item: str):
        self.task.add_pending(item)

    # ============================================================
    # 序列化 / 摘要
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """完整世界状态（供调试/持久化）"""
        return {
            "environment": self.environment.to_dict(),
            "ui": self.ui.to_dict(),
            "applications": {k: v.to_dict() for k, v in self.applications.items()},
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "task": self.task.to_dict(),
        }

    def to_summary(self) -> str:
        """压缩摘要（注入 LLM）—— 只显示高置信/关键状态"""
        lines = ["## [世界状态] 结构化环境理解"]

        # 环境
        env_summary = self._summarize_state(self.environment, "环境")
        if env_summary:
            lines.append(env_summary)

        # UI
        ui_summary = self._summarize_state(self.ui, "界面")
        if ui_summary:
            lines.append(ui_summary)

        # 应用
        if self.applications:
            app_lines = ["- 应用:"]
            for name, app in self.applications.items():
                installed = app.installed.belief
                opened = app.opened.belief
                parts = []
                if installed.value is not None:
                    parts.append(f"安装={'是' if installed.value else '否'}"
                                 if installed.confidence >= 0.8
                                 else f"安装=?(置信{installed.confidence:.2f})")
                if opened.value is not None:
                    parts.append(f"打开={'是' if opened.value else '否'}"
                                 if opened.confidence >= 0.8
                                 else f"打开=?(置信{opened.confidence:.2f})")
                if app.version.value:
                    parts.append(f"版本={app.version.value}")
                if parts:
                    app_lines.append(f"  · {name}: {', '.join(parts)}")
                else:
                    app_lines.append(f"  · {name}: (未知)")
            lines.append("\n".join(app_lines))

        # 文件
        if self.files:
            file_lines = ["- 文件:"]
            for path, fstate in list(self.files.items())[:8]:
                exists = fstate.exists.value
                status = "存在" if exists else \
                    ("不存在" if exists is False else "未知")
                file_lines.append(f"  · {path}: {status}")
            lines.append("\n".join(file_lines))

        # 任务
        ts = self.task
        if ts.current_stage.value:
            lines.append(f"- 任务阶段: {ts.current_stage.value}")
        completed = ts.completed.value
        if completed:
            lines.append(f"- 已完成: {', '.join(list(completed)[:8])}")
        pending = ts.pending.value
        if pending:
            lines.append(f"- 待办: {', '.join(list(pending)[:8])}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _summarize_state(self, state_obj, label: str) -> str:
        """对 StateBase 对象生成摘要"""
        fields = []
        for attr, val in vars(state_obj).items():
            if isinstance(val, type) and hasattr(val, "belief"):
                belief = val.belief
            elif hasattr(val, "belief"):
                belief = val.belief
            else:
                continue
            if belief.value is not None:
                fields.append(
                    f"{attr}={belief.value}"
                    if belief.confidence >= 0.8
                    else f"{attr}=?(置信{belief.confidence:.2f})"
                )
        if not fields:
            return ""
        return f"- {label}: {', '.join(fields)}"

    # ============================================================
    # 快照
    # ============================================================

    def snapshot(self) -> Dict[str, Any]:
        """状态快照（供异常处理/日志）"""
        return {
            "task_id": self._task_id,
            "active": self._active,
            "state": self.to_dict(),
        }