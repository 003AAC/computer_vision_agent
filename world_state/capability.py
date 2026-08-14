"""
Capability System - 能力系统
=============================
替代"action → precondition"的绑定方式，改为"capability → requires"。

流程：
  LLM 想执行某动作 → 先检查对应 capability 是否可用
  → 若缺失，不调用工具，返回 missing 列表给 LLM 重新规划。

capability 值路径使用 "state.domain.field" 形式，从 manager 状态树中读取。
"""
from typing import Any, Dict, List, Optional

from world_state.manager import WorldStateManager


class Capability:
    """能力定义：执行某类动作所需的世界状态条件"""

    def __init__(self, name: str, requires: Dict[str, Any],
                 description: str = "", on_missing: str = ""):
        """
        Args:
            name: 能力名（如 "compile_cpp"）
            requires: 所需状态条件 { "applications.visual_studio.installed": True, ... }
            description: 能力描述
            on_missing: 缺失时的提示（注入 LLM）
        """
        self.name = name
        self.requires = requires
        self.description = description
        self.on_missing = on_missing


class CapabilityRegistry:
    """能力注册表"""

    # 内置能力（可扩展）
    CAPABILITIES: Dict[str, Capability] = {
        "compile_cpp": Capability(
            name="compile_cpp",
            requires={
                "applications.visual_studio.installed": True,
            },
            description="使用 Visual Studio 编译 C++ 代码",
            on_missing="缺少编译能力：需要 Visual Studio 已安装。请先通过 PowerShell 检测安装路径。",
        ),
        "run_powershell": Capability(
            name="run_powershell",
            requires={},
            description="执行任意 PowerShell 命令（始终可用）",
            on_missing="",
        ),
        "use_visual_interaction": Capability(
            name="use_visual_interaction",
            requires={
                "environment.screen_resolution": None,  # 仅需有值
            },
            description="视觉定位/点击（需屏幕可读）",
            on_missing="缺少视觉交互能力：无法读取屏幕。",
        ),
    }

    def __init__(self, extra_capabilities: Dict[str, Capability] = None):
        self._capabilities = dict(self.CAPABILITIES)
        if extra_capabilities:
            self._capabilities.update(extra_capabilities)

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def register(self, capability: Capability):
        self._capabilities[capability.name] = capability

    def check(self, name: str, manager: WorldStateManager) -> Dict[str, Any]:
        """检查能力是否可用

        Args:
            name: 能力名
            manager: WorldStateManager（读状态）

        Returns:
            {
              "capability": str,
              "available": bool,
              "missing": [{path, expected, actual}],
              "hint": str
            }
        """
        capability = self._capabilities.get(name)
        if capability is None:
            return {
                "capability": name,
                "available": False,
                "missing": [{"path": "<capability未注册>", "expected": "已注册",
                             "actual": name}],
                "hint": f"能力 {name} 未注册。",
            }

        missing = self._find_missing(capability.requires, manager)
        available = not missing

        return {
            "capability": name,
            "available": available,
            "missing": missing,
            "hint": "" if available else capability.on_missing,
        }

    def _find_missing(self, requires: Dict[str, Any],
                      manager: WorldStateManager) -> List[Dict[str, Any]]:
        """找出缺失的条件"""
        missing = []
        for path, expected in requires.items():
            actual = self._resolve_path(path, manager)
            # 判定是否满足
            satisfied = False
            if expected is None:
                satisfied = actual is not None
            elif isinstance(expected, str):
                satisfied = actual == expected
            else:
                satisfied = actual is expected

            if not satisfied:
                missing.append({
                    "path": path,
                    "expected": expected,
                    "actual": actual,
                })
        return missing

    @staticmethod
    def _resolve_path(path: str, manager: WorldStateManager) -> Any:
        """从状态树解析 "domain.field" 路径的值

        支持：
          - "applications.visual_studio.installed"
          - "environment.screen_resolution"
          - "ui.popup_present"
          - "files.<path>.exists"
        """
        parts = path.split(".")
        if len(parts) < 2:
            return None

        domain = parts[0]
        try:
            if domain == "environment":
                state = manager.environment
                return getattr(state, parts[1]).value \
                    if hasattr(state, parts[1]) else None
            elif domain == "ui":
                state = manager.ui
                return getattr(state, parts[1]).value \
                    if hasattr(state, parts[1]) else None
            elif domain == "applications":
                app_name = parts[1]
                app = manager.applications.get(app_name)
                if app is None:
                    return None
                field = parts[2] if len(parts) > 2 else "installed"
                return getattr(app, field).value \
                    if hasattr(app, field) else None
            elif domain == "files":
                # "files.<path>.exists" —— path 可能含点，取最后段为 field
                field = parts[-1]
                # 中间为路径（可能含"_"）
                path_key = ".".join(parts[1:-1])
                fstate = manager.files.get(path_key)
                if fstate is None:
                    return None
                return getattr(fstate, field).value \
                    if hasattr(fstate, field) else None
            elif domain == "task":
                state = manager.task
                return getattr(state, parts[1]).value \
                    if hasattr(state, parts[1]) else None
        except Exception:
            return None
        return None

    def all_capabilities(self) -> List[str]:
        return list(self._capabilities.keys())