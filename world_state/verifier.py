"""
WorldState Verifier - 基于世界状态的验收
======================================
LLM 只能提出 completion claim，不能决定完成。
Verifier 根据 WorldState 中的真实证据判断任务是否完成。

核心接口：
  class VerificationCondition:
      id, description, weight, evidence_sources
      evaluate(world_state) -> {found, confidence, evidence}

  class WorldStateVerifier:
      verify(task_goal, world_state) -> {
          success, confidence, evidence, missing
      }

设计：
  - confidence = Σ(weight_i × evidence_confidence_i)
  - 不依赖 LLM 自述，只查 WorldState 事实
  - OCR/Florence 只作辅助证据，不作为主要判断
"""
import re
from typing import Any, Dict, List, Optional


class EvidenceItem:
    """一条验收证据（提取自 WorldState）"""

    def __init__(self, condition_id: str, description: str,
                 source: str, value: Any, confidence: float):
        self.condition_id = condition_id
        self.description = description
        self.source = source
        self.value = value
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition_id,
            "description": self.description,
            "source": self.source,
            "value": self.value,
            "confidence": self.confidence,
        }


class MissingEvidence:
    """一条缺失的验收证据"""

    def __init__(self, condition_id: str, description: str):
        self.condition_id = condition_id
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition_id,
            "description": self.description,
        }


class VerificationCondition:
    """一条验收条件：判断某类事实是否满足"""

    def __init__(self, condition_id: str, description: str,
                 weight: float, evidence_sources: List[str],
                 check):
        """
        Args:
            condition_id: 条件 ID（如 process_running）
            description: 人类可读描述（如 程序正在运行）
            weight: 权重（所有条件之和 = 1.0）
            evidence_sources: 允许的证据来源（system_api/filesystem/window/ocr/vision）
            check: 函数 (world_state, evidence_sources) -> (found, confidence, evidence_list)
        """
        self.condition_id = condition_id
        self.description = description
        self.weight = weight
        self.evidence_sources = evidence_sources
        self.check = check

    def evaluate(self, world_state) -> Dict[str, Any]:
        """评估当前状态是否满足本条件

        Returns:
            {found, confidence, evidence: [EvidenceItem]}
        """
        try:
            found, confidence, evidence_list = self.check(
                world_state, self.evidence_sources
            )
        except Exception:
            found, confidence, evidence_list = False, 0.0, []
        return {
            "found": found,
            "confidence": confidence,
            "evidence": evidence_list,
        }


class WorldStateVerifier:
    """世界状态验收器"""

    # 验收通过所需的置信度下限
    SUCCESS_THRESHOLD = 0.8

    def __init__(self):
        self._conditions: Dict[str, List[VerificationCondition]] = {}
        self._register_templates()

    # ============================================================
    # 通用任务模板（不写死俄罗斯方块）
    # ============================================================

    def _register_templates(self):
        """注册任务模板 → 验收条件列表"""
        self._conditions["compile_project"] = [
            VerificationCondition(
                "executable_exists", "可执行文件已生成", 1.0,
                ["filesystem"],
                check=check_executable_exists,
            ),
        ]

        self._conditions["create_project"] = [
            VerificationCondition(
                "file_exists", "项目文件存在", 0.6,
                ["filesystem"],
                check=check_project_file_exists,
            ),
            VerificationCondition(
                "directory_exists", "项目目录存在", 0.4,
                ["filesystem"],
                check=check_directory_exists,
            ),
        ]

        self._conditions["run_program"] = [
            VerificationCondition(
                "process_running", "程序正在运行", 0.4,
                ["system_api", "powershell"],
                check=check_process_running,
            ),
            VerificationCondition(
                "pid_exists", "进程 PID 已获取", 0.3,
                ["system_api"],
                check=check_pid_exists,
            ),
            VerificationCondition(
                "window_visible", "窗口可见", 0.3,
                ["window", "ocr", "vision"],
                check=check_window_visible,
            ),
        ]

    def get_conditions(self, task_type: str) -> List[VerificationCondition]:
        """获取任务模板的验收条件"""
        return self._conditions.get(task_type, [])

    def list_templates(self) -> List[str]:
        return list(self._conditions.keys())

    # ============================================================
    # 主验收
    # ============================================================

    def verify(self, task_goal: str, world_state) -> Dict[str, Any]:
        """验收：根据任务目标选择模板，查询 WorldState 证据

        Args:
            task_goal: 用户任务描述（如 "创建C++俄罗斯方块项目并运行"）
            world_state: WorldStateManager 实例

        Returns:
            {
              "success": bool,
              "confidence": float,
              "evidence": [EvidenceItem...],
              "missing": [MissingEvidence...],
              "task_type": str,
            }
        """
        task_type = self._infer_task_type(task_goal)
        conditions = self._conditions.get(task_type, [])

        if not conditions:
            # 无匹配模板 → 无法客观验收，返回 low confidence
            return {
                "success": False,
                "confidence": 0.0,
                "evidence": [],
                "missing": [MissingEvidence("template", "无匹配任务验收模板").to_dict()],
                "task_type": "unknown",
            }

        total_confidence = 0.0
        evidence_list: List[Dict[str, Any]] = []
        missing_list: List[Dict[str, Any]] = []
        total_weight = sum(c.weight for c in conditions)

        for cond in conditions:
            result = cond.evaluate(world_state)
            if result["found"]:
                # 多证据提权：取最高置信度（或可用 Belief 融合，这里用 max 简单化）
                ev_conf = max((e.confidence for e in result["evidence"]),
                              default=0.0)
                # 归一化权重
                weight = cond.weight / total_weight if total_weight > 0 else 0
                contribution = weight * ev_conf
                total_confidence += contribution

                for ev in result["evidence"]:
                    evidence_list.append(ev.to_dict())
            else:
                missing_list.append(MissingEvidence(
                    cond.condition_id, cond.description
                ).to_dict())

        success = total_confidence >= self.SUCCESS_THRESHOLD and not missing_list

        return {
            "success": success,
            "confidence": round(total_confidence, 3),
            "evidence": evidence_list,
            "missing": missing_list,
            "task_type": task_type,
        }

    # ============================================================
    # 任务类型推断（基于关键词）
    # ============================================================

    @staticmethod
    def _infer_task_type(task_goal: str) -> str:
        """从任务描述推断任务类型

        优先级：compile/run/create/open/install
        """
        t = (task_goal or "").lower()

        if any(kw in t for kw in ["编译", "compile", "build"]):
            return "compile_project"
        if any(kw in t for kw in ["运行", "启动", "执行", "run", "launch", "start"]):
            return "run_program"
        if any(kw in t for kw in ["创建", "新建", "生成", "create", "new", "generate"]):
            return "create_project"
        return "unknown"


# ============================================================
# Condition check 实现
# ============================================================

def check_executable_exists(world_state, sources) -> tuple:
    """可执行文件存在性（检查 WorldState.files 中后缀 .exe 的文件）"""
    return _check_file_pattern(world_state, sources, r"\.exe$",
                               "可执行文件", "executable_exists")


def check_project_file_exists(world_state, sources) -> tuple:
    """项目文件存在性（检查 .cpp/.c/.py/.h 等源码文件）"""
    return _check_file_pattern(world_state, sources,
                               r"\.(cpp|c|py|h|cs|java)$",
                               "项目源码文件", "file_exists")


def check_directory_exists(world_state, sources) -> tuple:
    """目录存在性（WorldState.files 中存在 is_dir=True 或路径最后不是文件扩展名）"""
    evidence_list = []
    found = False
    best_conf = 0.0

    for path, fstate in (world_state.files or {}).items():
        exists = fstate.exists.value
        if exists is not True:
            continue
        # 目录: 通过 is_dir 判断（如果字段有值）；否则通过路径无扩展名粗判
        is_dir = fstate.is_dir.value
        if is_dir is True or (is_dir is None and
                              not re.search(r'\.\w+$', os.path.basename(str(path) if path else ""))):
            conf = fstate.is_dir.confidence if fstate.is_dir.confidence > 0 else 0.95
            evidence_list.append(EvidenceItem(
                "directory_exists", f"目录 {path} 存在",
                "filesystem", True, conf,
            ))
            found = True
            best_conf = max(best_conf, conf)

    return found, best_conf, evidence_list


def check_process_running(world_state, sources) -> tuple:
    """程序是否在运行（ApplicationState.process_running）"""
    evidence_list = []
    found = False
    best_conf = 0.0

    for app_name, app in (world_state.applications or {}).items():
        running = app.process_running.value
        if running is True:
            conf = app.process_running.confidence
            evidence_list.append(EvidenceItem(
                "process_running", f"程序 {app_name} 正在运行",
                "system_api", True, conf,
            ))
            found = True
            best_conf = max(best_conf, conf)

    return found, best_conf, evidence_list


def check_pid_exists(world_state, sources) -> tuple:
    """PID 是否已获取（ApplicationState.pid）"""
    evidence_list = []
    found = False
    best_conf = 0.0

    for app_name, app in (world_state.applications or {}).items():
        pid = app.pid.value
        if pid is not None and pid != 0:
            conf = app.pid.confidence if app.pid.confidence > 0 else 1.0
            evidence_list.append(EvidenceItem(
                "pid_exists", f"进程 {app_name} PID={pid}",
                "system_api", pid, conf,
            ))
            found = True
            best_conf = max(best_conf, conf)

    return found, best_conf, evidence_list


def check_window_visible(world_state, sources) -> tuple:
    """窗口是否可见（ApplicationState.visible / window_title 或视觉证据）"""
    evidence_list = []
    found = False
    best_conf = 0.0

    # 来源1: ApplicationState.visible / window_title（window API）
    for app_name, app in (world_state.applications or {}).items():
        visible = app.visible.value
        title = app.window_title.value
        if visible is True or (title and title.strip()):
            conf = max(app.visible.confidence if app.visible.confidence > 0 else 0.95,
                       app.window_title.confidence if app.window_title.confidence > 0 else 0.95)
            evidence_list.append(EvidenceItem(
                "window_visible",
                f"应用 {app_name} 窗口可见 (标题: {title or '未知'})",
                "window", True, conf,
            ))
            found = True
            best_conf = max(best_conf, conf)

    # 来源2: 视觉辅助证据（Florence）—— 只作辅助
    if "vision" in sources:
        ui = world_state.ui
        vis = getattr(ui, "visual_verification", None)
        if vis and vis.value is True and vis.confidence > 0:
            evidence_list.append(EvidenceItem(
                "window_visible", "Florence 视觉确认窗口可见",
                "vision", True, vis.confidence,
            ))
            found = True
            best_conf = max(best_conf, vis.confidence)

    # 来源3: OCR 文字（辅助，置信度较低）
    if "ocr" in sources:
        ui = world_state.ui
        texts = getattr(ui, "visible_texts", None)
        if texts and texts.value:
            evidence_list.append(EvidenceItem(
                "window_visible", "OCR 检测到界面文字",
                "ocr", True, texts.confidence if texts.confidence > 0 else 0.7,
            ))
            found = True
            best_conf = max(best_conf, 0.7)

    return found, best_conf, evidence_list


# ============================================================
# 辅助工具
# ============================================================

import os


def _check_file_pattern(world_state, sources, pattern, desc,
                        condition_id) -> tuple:
    """通用文件匹配检查"""
    import re as _re
    evidence_list = []
    found = False
    best_conf = 0.0

    for path, fstate in (world_state.files or {}).items():
        if not path:
            continue
        p = str(path)
        if _re.search(pattern, p, _re.I):
            exists = fstate.exists.value
            if exists is True:
                conf = fstate.exists.confidence if fstate.exists.confidence > 0 else 0.95
                evidence_list.append(EvidenceItem(
                    condition_id, f"{desc} {p} 存在",
                    "filesystem", True, conf,
                ))
                found = True
                best_conf = max(best_conf, conf)

    return found, best_conf, evidence_list