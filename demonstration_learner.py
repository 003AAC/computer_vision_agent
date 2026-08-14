"""
Demonstration Learner - 演示学习模块
======================================
让 AI 通过观察用户操作来学习，提取操作模式存入知识库。

核心理念：
  不是"录制→回放"的机械模仿，而是"观察→理解→提取模式→存入知识库"
  
  当 AI 遇到不会操作的任务时：
  1. 请用户演示一遍操作
  2. 记录操作序列（鼠标点击、键盘输入等）
  3. 分析操作，提取出"操作模式"和"关键步骤"
  4. 将模式存入知识库，供以后参考
  5. 以后遇到类似任务，AI 参考知识库中的模式，灵活运用

用法：
  from demonstration_learner import DemonstrationLearner
  
  learner = DemonstrationLearner(knowledge_base)
  
  # 开始录制
  learner.start_recording("打开原神，清理每日委托")
  
  # ... 用户执行操作 ...
  
  # 停止录制，自动提取模式并存入知识库
  recording = learner.stop_recording()
  
  # 查询学习到的模式
  patterns = learner.query_learned_patterns("原神")
"""
import time
import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict

try:
    import pyautogui
except ImportError:
    # 测试时可能没有 pyautogui
    class _MockPyAutoGUI:
        @staticmethod
        def size():
            return (1920, 1080)
        @staticmethod
        def position():
            return _MockPoint(0, 0)
    class _MockPoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    pyautogui = _MockPyAutoGUI()



# ============================================================
# 数据结构
# ============================================================

@dataclass
class MouseAction:
    """鼠标操作记录"""
    action_type: str  # click, double_click, right_click, move, scroll
    x: int
    y: int
    button: str = "left"
    clicks: int = 1
    timestamp: float = 0.0
    screen_width: int = 0
    screen_height: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MouseAction':
        return cls(**data)


@dataclass
class KeyboardAction:
    """键盘操作记录"""
    action_type: str  # press, hotkey, type
    keys: list = field(default_factory=list)
    text: str = ""
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'KeyboardAction':
        return cls(**data)


@dataclass
class WaitAction:
    """等待操作记录"""
    duration: float
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WaitAction':
        return cls(**data)


@dataclass
class OperationPattern:
    """
    操作模式 - 从演示中提取的抽象操作模式
    
    不是记录具体的坐标和按键，而是记录"操作意图"和"操作逻辑"。
    例如：
      - "打开程序"模式: 按Win键 -> 输入程序名 -> 按回车
      - "保存文件"模式: 按Ctrl+S -> 等待 -> 输入文件名 -> 按回车
    """
    pattern_id: str
    name: str  # 模式名称，如"打开程序"、"保存文件"
    description: str  # 自然语言描述的操作步骤
    steps: List[dict]  # 抽象步骤序列
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    example_task: str = ""  # 从哪个任务中提取的
    success_count: int = 0  # 成功应用次数
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OperationPattern':
        return cls(**data)


# ============================================================
# 操作录制器
# ============================================================

class ActionRecorder:
    """
    操作录制器 - 在后台线程中监听并记录用户的键鼠操作
    
    使用轮询方式检测鼠标位置变化。
    注意：纯 Python 实现，不依赖全局钩子，通过 pyautogui 轮询。
    """
    
    def __init__(self, poll_interval: float = 0.05):
        self.poll_interval = poll_interval
        self._recording = False
        self._thread = None
        self._last_mouse_pos = None
        self._last_mouse_click_time = 0
        
        # 录制数据
        self.mouse_actions: List[MouseAction] = []
        self.keyboard_actions: List[KeyboardAction] = []
        self.wait_actions: List[WaitAction] = []
        self._start_time = 0
        self._last_action_time = 0
        
        # 屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.on_action: Optional[Callable] = None
    
    def start(self):
        """开始录制"""
        if self._recording:
            return
        
        self._recording = True
        self._last_mouse_pos = pyautogui.position()
        self._start_time = time.time()
        self._last_action_time = self._start_time
        
        self.mouse_actions.clear()
        self.keyboard_actions.clear()
        self.wait_actions.clear()
        
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        
        print(f"  [录制] 开始录制操作... (轮询间隔: {self.poll_interval*1000:.0f}ms)")
        print(f"  [录制] 屏幕分辨率: {self.screen_width}x{self.screen_height}")
    
    def stop(self) -> dict:
        """停止录制并返回录制数据"""
        if not self._recording:
            return {"error": "没有正在进行的录制"}
        
        self._recording = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        duration = time.time() - self._start_time
        self._merge_waits()
        
        result = {
            "duration": round(duration, 2),
            "mouse_actions": len(self.mouse_actions),
            "keyboard_actions": len(self.keyboard_actions),
            "wait_actions": len(self.wait_actions),
            "total_actions": len(self.mouse_actions) + len(self.keyboard_actions) + len(self.wait_actions),
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
        }
        
        print(f"  [录制] 录制完成! 共 {result['total_actions']} 个操作，耗时 {duration:.1f}秒")
        
        return result
    
    def is_recording(self) -> bool:
        return self._recording
    
    def get_recording_data(self) -> dict:
        return {
            "mouse_actions": [a.to_dict() for a in self.mouse_actions],
            "keyboard_actions": [a.to_dict() for a in self.keyboard_actions],
            "wait_actions": [a.to_dict() for a in self.wait_actions],
        }
    
    def _record_loop(self):
        """录制主循环（在后台线程中运行）"""
        last_pos = pyautogui.position()
        
        while self._recording:
            now = time.time()
            current_pos = pyautogui.position()
            
            # 检测鼠标移动
            if current_pos != last_pos:
                dx = abs(current_pos.x - last_pos.x)
                dy = abs(current_pos.y - last_pos.y)
                if dx > 3 or dy > 3:
                    self._record_wait_if_needed(now)
                    self.mouse_actions.append(MouseAction(
                        action_type="move",
                        x=current_pos.x, y=current_pos.y,
                        timestamp=now,
                        screen_width=self.screen_width,
                        screen_height=self.screen_height,
                    ))
                    self._last_action_time = now
                last_pos = current_pos
            
            time.sleep(self.poll_interval)
    
    def record_click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """记录鼠标点击（由外部监听调用）"""
        if not self._recording:
            return
        
        now = time.time()
        self._record_wait_if_needed(now)
        
        action_type = "double_click" if clicks == 2 else "click"
        self.mouse_actions.append(MouseAction(
            action_type=action_type, x=x, y=y,
            button=button, clicks=clicks,
            timestamp=now,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        ))
        self._last_action_time = now
    
    def record_key_press(self, key: str):
        """记录按键操作"""
        if not self._recording:
            return
        
        now = time.time()
        self._record_wait_if_needed(now)
        
        self.keyboard_actions.append(KeyboardAction(
            action_type="press", keys=[key], timestamp=now,
        ))
        self._last_action_time = now
    
    def record_hotkey(self, keys: list):
        """记录组合键操作"""
        if not self._recording:
            return
        
        now = time.time()
        self._record_wait_if_needed(now)
        
        self.keyboard_actions.append(KeyboardAction(
            action_type="hotkey", keys=keys, timestamp=now,
        ))
        self._last_action_time = now
    
    def record_text(self, text: str):
        """记录文字输入"""
        if not self._recording:
            return
        
        now = time.time()
        self._record_wait_if_needed(now)
        
        self.keyboard_actions.append(KeyboardAction(
            action_type="type", text=text, timestamp=now,
        ))
        self._last_action_time = now
    
    def _record_wait_if_needed(self, now: float):
        """如果距离上次操作超过阈值，记录等待操作"""
        elapsed = now - self._last_action_time
        if elapsed > 0.3:
            self.wait_actions.append(WaitAction(
                duration=round(elapsed, 2), timestamp=now,
            ))
    
    def _merge_waits(self):
        """合并连续的等待操作"""
        if len(self.wait_actions) <= 1:
            return
        
        merged = []
        for wait in self.wait_actions:
            if merged and wait.duration < 0.5:
                merged[-1].duration += wait.duration
            else:
                merged.append(wait)
        
        self.wait_actions = merged


# ============================================================
# 操作模式提取器（核心：从操作序列中提取抽象模式）
# ============================================================

class PatternExtractor:
    """
    操作模式提取器 - 从录制的操作序列中提取抽象的操作模式
    
    核心思想：
      不是保存"在(100,200)点击左键"这样的具体操作，
      而是提取"打开开始菜单 → 搜索程序名 → 回车启动"这样的操作逻辑。
    
    提取策略：
      1. 识别操作意图（打开、保存、关闭、切换等）
      2. 将具体坐标泛化为"目标位置"
      3. 将具体按键泛化为"快捷键组合"
      4. 提取操作间的依赖关系（先A后B）
    """
    
    # 已知的操作模式模板
    PATTERN_TEMPLATES = {
        "打开程序": {
            "keywords": ["打开", "启动", "运行", "start", "launch"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["win"], "meaning": "打开开始菜单"},
                {"action": "type", "meaning": "输入程序名称"},
                {"action": "press", "key": "enter", "meaning": "回车启动"},
            ]
        },
        "保存文件": {
            "keywords": ["保存", "save", "另存为"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["ctrl", "s"], "meaning": "Ctrl+S保存"},
                {"action": "wait", "meaning": "等待保存对话框"},
                {"action": "type", "meaning": "输入文件名"},
                {"action": "press", "key": "enter", "meaning": "确认保存"},
            ]
        },
        "关闭窗口": {
            "keywords": ["关闭", "退出", "关掉", "close", "exit"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["alt", "f4"], "meaning": "Alt+F4关闭窗口"},
            ]
        },
        "复制粘贴": {
            "keywords": ["复制", "粘贴", "copy", "paste"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["ctrl", "c"], "meaning": "Ctrl+C复制"},
                {"action": "click", "meaning": "点击目标位置"},
                {"action": "hotkey", "keys": ["ctrl", "v"], "meaning": "Ctrl+V粘贴"},
            ]
        },
        "切换窗口": {
            "keywords": ["切换", "切到", "switch", "alt+tab"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["alt", "tab"], "meaning": "Alt+Tab切换窗口"},
            ]
        },
        "搜索内容": {
            "keywords": ["搜索", "查找", "search", "find"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["ctrl", "f"], "meaning": "Ctrl+F打开搜索"},
                {"action": "type", "meaning": "输入搜索关键词"},
                {"action": "press", "key": "enter", "meaning": "确认搜索"},
            ]
        },
        "新建文件": {
            "keywords": ["新建", "创建", "new", "create"],
            "steps_pattern": [
                {"action": "hotkey", "keys": ["ctrl", "n"], "meaning": "Ctrl+N新建"},
            ]
        },
    }
    
    @classmethod
    def extract_patterns(cls, task_description: str,
                         mouse_actions: List[MouseAction],
                         keyboard_actions: List[KeyboardAction],
                         wait_actions: List[WaitAction]) -> List[OperationPattern]:
        """
        从操作序列中提取操作模式
        
        Args:
            task_description: 任务描述
            mouse_actions: 鼠标操作列表
            keyboard_actions: 键盘操作列表
            wait_actions: 等待操作列表
        
        Returns:
            提取到的操作模式列表
        """
        patterns = []
        task_lower = task_description.lower()
        
        # 1. 生成操作摘要（自然语言描述）
        summary = cls._generate_operation_summary(
            mouse_actions, keyboard_actions, wait_actions
        )
        
        # 2. 识别匹配的模式模板
        matched_templates = cls._match_templates(task_lower, summary)
        
        for template_name in matched_templates:
            template = cls.PATTERN_TEMPLATES[template_name]
            
            # 从实际操作中提取具体信息
            concrete_steps = cls._extract_concrete_steps(
                template, mouse_actions, keyboard_actions
            )
            
            pattern = OperationPattern(
                pattern_id=f"pattern_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=template_name,
                description=cls._generate_pattern_description(
                    template_name, concrete_steps, task_description
                ),
                steps=concrete_steps,
                tags=[template_name, "演示学习"],
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                example_task=task_description,
            )
            patterns.append(pattern)
        
        # 3. 如果没有匹配到已知模板，生成通用模式
        if not patterns:
            generic_pattern = cls._create_generic_pattern(
                task_description, mouse_actions, keyboard_actions, wait_actions
            )
            if generic_pattern:
                patterns.append(generic_pattern)
        
        return patterns
    
    @classmethod
    def _generate_operation_summary(cls, mouse_actions, keyboard_actions, wait_actions) -> str:
        """生成操作序列的自然语言摘要"""
        parts = []
        
        # 合并所有操作并按时间排序
        all_ops = []
        for a in mouse_actions:
            all_ops.append(("mouse", a.action_type, a.timestamp, a))
        for a in keyboard_actions:
            all_ops.append(("keyboard", a.action_type, a.timestamp, a))
        for a in wait_actions:
            all_ops.append(("wait", "wait", a.timestamp, a))
        
        all_ops.sort(key=lambda x: x[2])
        
        for op_type, action_type, ts, data in all_ops:
            if op_type == "mouse":
                if action_type in ("click", "double_click"):
                    parts.append(f"点击")
                elif action_type == "right_click":
                    parts.append(f"右键")
                elif action_type == "move":
                    parts.append(f"移动鼠标")
            elif op_type == "keyboard":
                if action_type == "hotkey":
                    parts.append(f"按快捷键{'+'.join(data.keys)}")
                elif action_type == "type":
                    parts.append(f"输入文字")
                elif action_type == "press":
                    parts.append(f"按{data.keys[0]}")
            elif op_type == "wait":
                if data.duration > 1:
                    parts.append(f"等待{data.duration:.0f}秒")
        
        return " → ".join(parts)
    
    @classmethod
    def _match_templates(cls, task_lower: str, summary: str) -> List[str]:
        """匹配已知的操作模式模板"""
        matched = []
        
        for name, template in cls.PATTERN_TEMPLATES.items():
            # 检查任务描述中是否包含关键词
            if any(kw in task_lower for kw in template["keywords"]):
                matched.append(name)
                continue
            
            # 检查操作摘要中是否包含模式特征
            pattern_actions = [s["action"] for s in template["steps_pattern"]]
            pattern_str = " ".join(pattern_actions)
            if any(a in summary for a in pattern_actions):
                matched.append(name)
        
        return matched
    
    @classmethod
    def _extract_concrete_steps(cls, template: dict,
                                mouse_actions, keyboard_actions) -> List[dict]:
        """从实际操作中提取具体步骤信息"""
        steps = []
        
        for step_template in template["steps_pattern"]:
            step = {
                "action": step_template["action"],
                "meaning": step_template["meaning"],
                "detail": ""
            }
            
            # 从实际操作中提取具体值
            if step_template["action"] == "hotkey":
                for ka in keyboard_actions:
                    if ka.action_type == "hotkey" and ka.keys:
                        step["detail"] = f"快捷键: {'+'.join(ka.keys)}"
                        break
            
            elif step_template["action"] == "type":
                for ka in keyboard_actions:
                    if ka.action_type == "type" and ka.text:
                        text = ka.text[:20]
                        step["detail"] = f"输入内容: '{text}'"
                        break
            
            elif step_template["action"] == "press":
                for ka in keyboard_actions:
                    if ka.action_type == "press" and ka.keys:
                        step["detail"] = f"按键: {ka.keys[0]}"
                        break
            
            elif step_template["action"] == "click":
                for ma in mouse_actions:
                    if ma.action_type in ("click", "double_click"):
                        step["detail"] = f"点击位置: ({ma.x}, {ma.y})"
                        break
            
            steps.append(step)
        
        return steps
    
    @classmethod
    def _generate_pattern_description(cls, pattern_name: str,
                                      steps: List[dict],
                                      task_description: str) -> str:
        """生成模式的自然语言描述"""
        step_descs = []
        for s in steps:
            desc = s["meaning"]
            if s["detail"]:
                desc += f" ({s['detail']})"
            step_descs.append(desc)
        
        return (f"【{pattern_name}】操作模式\n"
                f"  来源任务: {task_description}\n"
                f"  操作步骤:\n    " + "\n    ".join(
                    f"{i+1}. {d}" for i, d in enumerate(step_descs)
                ))
    
    @classmethod
    def _create_generic_pattern(cls, task_description: str,
                                 mouse_actions, keyboard_actions,
                                 wait_actions) -> Optional[OperationPattern]:
        """为未匹配到模板的操作创建通用模式"""
        if not mouse_actions and not keyboard_actions:
            return None
        
        # 统计操作类型
        click_count = sum(1 for a in mouse_actions if a.action_type in ("click", "double_click"))
        type_count = sum(1 for a in keyboard_actions if a.action_type == "type")
        hotkey_count = sum(1 for a in keyboard_actions if a.action_type == "hotkey")
        
        # 生成步骤描述
        steps = []
        all_ops = []
        for a in mouse_actions:
            all_ops.append(("mouse", a.timestamp, a))
        for a in keyboard_actions:
            all_ops.append(("keyboard", a.timestamp, a))
        for a in wait_actions:
            all_ops.append(("wait", a.timestamp, a))
        
        all_ops.sort(key=lambda x: x[1])
        
        for op_type, ts, data in all_ops:
            if op_type == "mouse" and data.action_type in ("click", "double_click"):
                steps.append({
                    "action": "click",
                    "meaning": "点击目标位置",
                    "detail": f"坐标: ({data.x}, {data.y})"
                })
            elif op_type == "keyboard" and data.action_type == "hotkey":
                steps.append({
                    "action": "hotkey",
                    "meaning": f"按下快捷键 {'+'.join(data.keys)}",
                    "detail": f"快捷键: {'+'.join(data.keys)}"
                })
            elif op_type == "keyboard" and data.action_type == "type":
                steps.append({
                    "action": "type",
                    "meaning": "输入文字",
                    "detail": f"内容: '{data.text[:30]}'"
                })
            elif op_type == "keyboard" and data.action_type == "press":
                steps.append({
                    "action": "press",
                    "meaning": f"按下 {data.keys[0]}",
                    "detail": f"按键: {data.keys[0]}"
                })
            elif op_type == "wait" and data.duration > 1:
                steps.append({
                    "action": "wait",
                    "meaning": f"等待 {data.duration:.0f} 秒",
                    "detail": ""
                })
        
        if not steps:
            return None
        
        # 生成模式名称
        if click_count > 0 and type_count > 0:
            pattern_name = "点击输入"
        elif hotkey_count > 0:
            pattern_name = "快捷键操作"
        elif click_count > 0:
            pattern_name = "点击操作"
        else:
            pattern_name = "键盘操作"
        
        step_descs = "\n    ".join(f"{i+1}. {s['meaning']} {s['detail']}" for i, s in enumerate(steps))
        
        return OperationPattern(
            pattern_id=f"pattern_generic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=pattern_name,
            description=f"【{pattern_name}】操作模式\n"
                        f"  来源任务: {task_description}\n"
                        f"  操作步骤:\n    {step_descs}",
            steps=steps,
            tags=[pattern_name, "演示学习", "通用模式"],
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            example_task=task_description,
        )


# ============================================================
# 演示学习器（主类）
# ============================================================

class DemonstrationLearner:
    """
    演示学习器 - 观察用户操作，提取模式存入知识库
    
    工作流程：
    1. AI 遇到不会操作的任务
    2. 调用 start_recording() 开始录制
    3. 告诉用户"请演示一遍操作"
    4. 用户执行操作，AI 在后台录制
    5. 用户操作完成，调用 stop_recording()
    6. 系统自动分析操作序列，提取操作模式
    7. 模式存入知识库，供以后参考
    8. 以后遇到类似任务，AI 查询知识库中的模式，灵活运用
    """
    
    def __init__(self, knowledge_base=None):
        """
        初始化演示学习器
        
        Args:
            knowledge_base: KnowledgeBase 实例，用于存储和检索学习到的模式
        """
        self.kb = knowledge_base
        self.recorder = ActionRecorder(poll_interval=0.05)
        self.pattern_extractor = PatternExtractor()
        
        # 当前录制信息
        self._current_recording_id = None
        self._current_task = None
        self._current_tags = []
        
        # 演示存储目录（保留原始数据供分析）
        self._recordings_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "knowledge_data",
            "demonstrations"
        )
        os.makedirs(self._recordings_dir, exist_ok=True)
    
    # ============================================================
    # 录制接口
    # ============================================================
    
    def start_recording(self, task_description: str, tags: list = None) -> str:
        """
        开始录制用户的操作演示
        
        Args:
            task_description: 任务描述
            tags: 标签列表
        
        Returns:
            录制ID
        """
        if self.recorder.is_recording():
            return "已有录制在进行中，请先停止当前录制"
        
        self._current_task = task_description
        self._current_tags = tags or []
        self._current_recording_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'=' * 60}")
        print(f"🎓 演示学习 - 开始录制")
        print(f"{'=' * 60}")
        print(f"  任务: {task_description}")
        print(f"  录制ID: {self._current_recording_id}")
        print(f"  标签: {', '.join(self._current_tags) if self._current_tags else '无'}")
        print(f"{'=' * 60}")
        print(f"  提示: 请用户开始执行操作...")
        print(f"  完成后 AI 会调用 stop_recording() 停止录制")
        print(f"{'=' * 60}")
        
        self.recorder.start()
        
        return self._current_recording_id
    
    def stop_recording(self) -> dict:
        """
        停止录制，提取操作模式并存入知识库
        
        Returns:
            学习结果，包含提取到的模式信息
        """
        if not self.recorder.is_recording():
            return {"error": "没有正在进行的录制"}
        
        # 1. 停止录制
        stats = self.recorder.stop()
        
        if stats.get("total_actions", 0) == 0:
            print(f"  [录制] ⚠ 没有录制到任何操作")
            self._current_recording_id = None
            self._current_task = None
            return {"error": "没有录制到任何操作"}
        
        # 2. 获取录制数据
        recording_data = self.recorder.get_recording_data()
        mouse_actions = [MouseAction.from_dict(a) for a in recording_data["mouse_actions"]]
        keyboard_actions = [KeyboardAction.from_dict(a) for a in recording_data["keyboard_actions"]]
        wait_actions = [WaitAction.from_dict(a) for a in recording_data["wait_actions"]]
        
        # 3. 保存原始录制数据（供后续分析）
        self._save_raw_recording(
            mouse_actions, keyboard_actions, wait_actions, stats
        )
        
        # 4. 提取操作模式（核心步骤）
        print(f"\n  [分析] 正在分析操作序列，提取操作模式...")
        patterns = self.pattern_extractor.extract_patterns(
            self._current_task,
            mouse_actions,
            keyboard_actions,
            wait_actions
        )
        
        # 5. 将模式存入知识库
        saved_patterns = []
        if self.kb:
            for pattern in patterns:
                self._save_pattern_to_knowledge_base(pattern)
                saved_patterns.append(pattern.name)
        
        # 6. 生成操作摘要
        summary = self._generate_summary(
            mouse_actions, keyboard_actions, wait_actions
        )
        
        print(f"\n{'=' * 60}")
        print(f"✅ 演示学习 - 完成")
        print(f"{'=' * 60}")
        print(f"  任务: {self._current_task}")
        print(f"  操作数: {stats['total_actions']}")
        print(f"  提取模式: {len(patterns)} 个")
        for p in patterns:
            print(f"    - {p.name}: {len(p.steps)} 步")
        print(f"{'=' * 60}")
        
        result = {
            "recording_id": self._current_recording_id,
            "task": self._current_task,
            "total_actions": stats["total_actions"],
            "duration": stats["duration"],
            "patterns_extracted": len(patterns),
            "patterns": [p.name for p in patterns],
            "summary": summary,
        }
        
        self._current_recording_id = None
        self._current_task = None
        
        return result
    
    def is_recording(self) -> bool:
        return self.recorder.is_recording()
    
    # ============================================================
    # 模式查询
    # ============================================================
    
    def query_learned_patterns(self, task_description: str) -> List[dict]:
        """
        查询与任务相关的已学习模式
        
        这是核心方法：当 AI 接到新任务时，调用此方法
        从知识库中查找相关的操作模式，参考模式来完成任务
        
        Args:
            task_description: 任务描述
        
        Returns:
            相关的操作模式列表
        """
        if not self.kb:
            return []
        
        try:
            # ====== 策略1: 用完整任务描述搜索 ======
            experiences = self.kb.search_experience(
                f"操作模式: {task_description}",
                n_results=5
            )
            
            patterns = []
            for exp in experiences:
                meta = exp.get("metadata", {})
                if "操作模式" in meta.get("task", ""):
                    patterns.append({
                        "name": meta.get("task", "").replace("操作模式: ", ""),
                        "description": meta.get("lesson", ""),
                        "steps": meta.get("steps", 0),
                        "tags": meta.get("tags", []),
                        "relevance": 1.0 - exp.get("distance", 0),
                    })
            
            # ====== 策略2: 如果没找到，用关键词拆解搜索 ======
            if not patterns:
                # 提取关键词（去掉"打开""启动"等动词，保留名词）
                keywords = []
                for kw in ["打开", "启动", "运行", "创建", "保存", "关闭", "删除"]:
                    if kw in task_description:
                        rest = task_description.replace(kw, "").strip()
                        if rest:
                            keywords.append(rest)
                
                for keyword in keywords:
                    experiences2 = self.kb.search_experience(
                        f"操作模式: {keyword}",
                        n_results=3
                    )
                    for exp in experiences2:
                        meta = exp.get("metadata", {})
                        if "操作模式" in meta.get("task", ""):
                            # 避免重复
                            name = meta.get("task", "").replace("操作模式: ", "")
                            if not any(p["name"] == name for p in patterns):
                                patterns.append({
                                    "name": name,
                                    "description": meta.get("lesson", ""),
                                    "steps": meta.get("steps", 0),
                                    "tags": meta.get("tags", []),
                                    "relevance": 0.7,  # 关键词匹配，置信度稍低
                                })
            
            # ====== 策略3: 如果还没找到，获取所有操作模式 ======
            if not patterns:
                try:
                    all_data = self.kb.experience_collection.get(
                        include=['metadatas', 'documents']
                    )
                    for i, meta in enumerate(all_data['metadatas']):
                        task_field = meta.get("task", "")
                        if "操作模式" in task_field:
                            patterns.append({
                                "name": task_field.replace("操作模式: ", ""),
                                "description": meta.get("lesson", ""),
                                "steps": meta.get("steps", 0),
                                "tags": meta.get("tags", []),
                                "relevance": 0.5,  # 全量匹配，置信度最低
                            })
                except Exception:
                    pass
            
            return patterns
            
        except Exception as e:
            print(f"  [查询] 查询模式失败: {e}")
            return []

    
    def get_pattern_recommendation(self, task_description: str) -> str:
        """
        获取针对任务的操作模式推荐（自然语言描述）
        
        Args:
            task_description: 任务描述
        
        Returns:
            推荐的操作模式描述
        """
        patterns = self.query_learned_patterns(task_description)
        
        if not patterns:
            return ""
        
        lines = ["📚 从之前的学习中找到了相关操作模式:"]
        for i, p in enumerate(patterns, 1):
            lines.append(f"\n  {i}. 【{p['name']}】(匹配度: {p.get('relevance', 0):.0%})")
            lines.append(f"     {p['description'][:200]}")
        
        lines.append(f"\n💡 提示: 参考以上模式来完成任务，但要根据实际情况灵活调整")
        
        return "\n".join(lines)
    
    # ============================================================
    # 存储
    # ============================================================
    
    def _save_raw_recording(self, mouse_actions, keyboard_actions,
                            wait_actions, stats):
        """保存原始录制数据到文件"""
        recording_data = {
            "recording_id": self._current_recording_id,
            "task_description": self._current_task,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": stats["duration"],
            "total_actions": stats["total_actions"],
            "mouse_actions": [a.to_dict() for a in mouse_actions],
            "keyboard_actions": [a.to_dict() for a in keyboard_actions],
            "wait_actions": [a.to_dict() for a in wait_actions],
            "screen_width": stats["screen_width"],
            "screen_height": stats["screen_height"],
            "tags": self._current_tags,
        }
        
        filepath = os.path.join(
            self._recordings_dir,
            f"{self._current_recording_id}.json"
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recording_data, f, ensure_ascii=False, indent=2)
        print(f"  [存储] 原始录制数据已保存")
    
    def _save_pattern_to_knowledge_base(self, pattern: OperationPattern):
        """将操作模式存入知识库"""
        if not self.kb:
            return
        
        experience = {
            "task": f"操作模式: {pattern.name}",
            "result": "模式已提取",
            "lesson": pattern.description,
            "date": pattern.created_at[:10],
            "steps": len(pattern.steps),
            "tools_used": ["demonstration_learner"],
            "tool_counts": json.dumps({
                "pattern_steps": len(pattern.steps),
            }, ensure_ascii=False),
            "tags": pattern.tags + [f"示例: {pattern.example_task[:20]}"],
        }
        
        try:
            self.kb.add_experience(experience)
            print(f"  [知识库] ✓ 操作模式已保存: {pattern.name}")
        except Exception as e:
            print(f"  [知识库] ✗ 保存失败: {e}")
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    @staticmethod
    def _generate_summary(mouse_actions, keyboard_actions, wait_actions) -> str:
        """生成操作摘要"""
        click_count = sum(1 for a in mouse_actions if a.action_type in ("click", "double_click"))
        move_count = sum(1 for a in mouse_actions if a.action_type == "move")
        right_click = sum(1 for a in mouse_actions if a.action_type == "right_click")
        press_count = sum(1 for a in keyboard_actions if a.action_type == "press")
        hotkey_count = sum(1 for a in keyboard_actions if a.action_type == "hotkey")
        type_count = sum(1 for a in keyboard_actions if a.action_type == "type")
        
        parts = []
        if click_count:
            parts.append(f"点击{click_count}次")
        if move_count:
            parts.append(f"移动{move_count}次")
        if right_click:
            parts.append(f"右键{right_click}次")
        if press_count:
            parts.append(f"按键{press_count}次")
        if hotkey_count:
            parts.append(f"组合键{hotkey_count}次")
        if type_count:
            parts.append(f"输入{type_count}次")
        
        return ", ".join(parts) if parts else "无操作"


# ============================================================
# 测试入口
# ============================================================

def test_demonstration_learner():
    """测试演示学习模块"""
    print("=" * 60)
    print("Demonstration Learner - 单元测试")
    print("=" * 60)
    
    # 模拟知识库
    class MockKB:
        def __init__(self):
            self.experiences = []
        def add_experience(self, exp):
            self.experiences.append(exp)
            print(f"  [MockKB] 保存经验: {exp['task']}")
        def search_experience(self, query, n_results=3):
            return [{
                "metadata": {
                    "task": "操作模式: 打开程序",
                    "lesson": "【打开程序】操作模式\n  来源任务: 打开记事本\n  操作步骤:\n    1. 打开开始菜单\n    2. 输入程序名称\n    3. 回车启动",
                    "steps": 3,
                    "tags": ["打开程序", "演示学习"],
                },
                "distance": 0.1,
            }]
    
    learner = DemonstrationLearner(MockKB())
    
    # 测试1: 录制和保存
    print("\n[测试1] 录制演示（模拟）")
    rid = learner.start_recording("打开记事本并输入hello", tags=["测试", "记事本"])
    print(f"  录制ID: {rid}")
    
    # 模拟一些操作
    import time
    learner.recorder.mouse_actions.append(MouseAction(
        action_type="click", x=100, y=200, timestamp=time.time(),
        screen_width=1920, screen_height=1080
    ))
    time.sleep(0.1)
    learner.recorder.keyboard_actions.append(KeyboardAction(
        action_type="type", text="hello world", timestamp=time.time()
    ))
    time.sleep(0.1)
    learner.recorder.wait_actions.append(WaitAction(duration=0.5, timestamp=time.time()))
    
    result = learner.stop_recording()
    print(f"  结果: {result.get('total_actions', '?')} 个操作")
    print(f"  提取模式: {result.get('patterns_extracted', 0)} 个")
    for p in result.get('patterns', []):
        print(f"    - {p}")
    
    # 测试2: 查询模式推荐
    print("\n[测试2] 查询操作模式推荐")
    recommendation = learner.get_pattern_recommendation("打开记事本")
    if recommendation:
        print(recommendation)
    else:
        print("  无推荐（知识库为空时正常）")
    
    # 测试3: 模式提取器测试
    print("\n[测试3] 模式提取器测试")
    from demonstration_learner import PatternExtractor
    patterns = PatternExtractor.extract_patterns(
        "保存文件到桌面",
        [MouseAction(action_type="click", x=500, y=300, timestamp=time.time(),
                     screen_width=1920, screen_height=1080)],
        [KeyboardAction(action_type="hotkey", keys=["ctrl", "s"], timestamp=time.time()),
         KeyboardAction(action_type="type", text="test.txt", timestamp=time.time()),
         KeyboardAction(action_type="press", keys=["enter"], timestamp=time.time())],
        [WaitAction(duration=1.0, timestamp=time.time())]
    )
    print(f"  提取到 {len(patterns)} 个模式:")
    for p in patterns:
        print(f"    - {p.name}: {len(p.steps)} 步")
        print(f"      描述: {p.description[:100]}...")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


