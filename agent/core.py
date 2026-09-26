"""
Agent 主循环
============
持续运行模式：启动后常驻，预加载模型，交互式接收任务
核心功能：视觉分析（结构化JSON）+ OCR 文字识别 + 键鼠/PowerShell 操作
两层降级：Tier 1 多模态 → Tier 2 PowerShell

补充机制：
  - 状态感知机（agent.state_machine）：卡住检测、死循环检测、状态注入
  - 异常处理（agent.exception_handler）：错误诊断、自愈修复、伪成功检测
"""
import json
import re
import time
from typing import Dict, Any, List, Optional

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_STEPS,
    safe_print,
)
from agent.tools import ALL_TOOLS, TOOL_REGISTRY
from agent.state_machine import AgentStateMachine, AgentState
from agent.exception_handler import ExceptionHandler
from agent.memory.memory_manager import MemoryManager
from agent.memory.working_memory import WorkingMemory
from agent.state_manager import StateManager
from agent.task_phase import TaskPhaseMachine, PhaseDefinition
from agent.verifier import GoalVerifier, GoalSpec
from agent.tracker import ObjectTracker
from agent.predictor import ActionPredictor
from agent.logger import ExecutionLogger
from agent.experience import AbstractExperienceStore
from agent.tool_manager import ToolManager
from agent.cost_planner import (
    CostModel,
    BudgetTracker,
    PlanCandidate,
    select_best_plan,
    estimate_phases_cost,
)
from agent.replan_manager import LaunchFailureDetector, ReplanManager
from skills.base import build_skill_instruction, get_base_skills
from world_state.manager import WorldStateManager
from world_state.diff import StateDiff
from world_state.verifier import WorldStateVerifier
from vision.engine import preload_models

# 任务级失败计数（跨工具）
TIER_MAX_FAILURES = 3


def build_tools_schema():
    """构建工具schema给DeepSeek的function calling用"""
    tools_schema = []
    for t in ALL_TOOLS:
        if hasattr(t, 'args_schema') and t.args_schema:
            try:
                params = t.args_schema.model_json_schema()
            except Exception:
                params = {"type": "object", "properties": {}}
        else:
            params = {"type": "object", "properties": {}}
        tools_schema.append({
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": params}
        })
    return tools_schema


def parse_planning_response(content: str) -> Dict[str, Any]:
    """解析 LLM 规划响应

    期望 LLM 回复包含两类块：
      1. <plan> ... </plan>  JSON：{application, initial_state, phases: [...]}
      2. <goal> ... </goal>  JSON：{goal, evidence[], failure_condition[]}

    解析失败返回空字典（回退到无状态探索模式）。

    Args:
        content: LLM 回复内容

    Returns:
        {"application": str, "initial_state": str, "phases": [...], "goal_spec": {...}}
    """
    import re

    result: Dict[str, Any] = {}

    # 解析 <plan> 块
    plan_m = re.search(r'<plan>(.*?)</plan>', content, re.S | re.I)
    if plan_m:
        try:
            plan_data = json.loads(plan_m.group(1).strip())
            result["application"] = plan_data.get("application", "")
            result["initial_state"] = plan_data.get("initial_state", "")
            result["phases"] = plan_data.get("phases", [])
            result["est_total_cost"] = plan_data.get("est_total_cost", 0)
            result["approach"] = plan_data.get("approach", "")
        except json.JSONDecodeError:
            print("  [规划] <plan> 块解析失败，使用默认探索模式")

    # 解析 <goal> 块
    goal_m = re.search(r'<goal>(.*?)</goal>', content, re.S | re.I)
    if goal_m:
        try:
            result["goal_spec"] = json.loads(goal_m.group(1).strip())
        except json.JSONDecodeError:
            print("  [规划] <goal> 块解析失败，使用默认验收")

    # 解析 <plans> 块（成本感知多方案择优）
    plans_m = re.search(r'<plans>(.*?)</plans>', content, re.S | re.I)
    if plans_m:
        try:
            candidates = json.loads(plans_m.group(1).strip())
            if isinstance(candidates, list) and candidates:
                result["plan_candidates"] = candidates
        except json.JSONDecodeError:
            print("  [规划] <plans> 块解析失败，使用单一方案")

    return result


def phase_from_dict(d: Dict[str, Any]) -> PhaseDefinition:
    """从规划字典构建阶段定义"""
    return PhaseDefinition(
        name=d.get("name", ""),
        entry_conditions=d.get("entry_conditions", []),
        allowed_actions=d.get("allowed_actions", []),
        expected_next=d.get("expected_next", []),
        verification_method=d.get("verification_method", []),
        description=d.get("description", ""),
        est_cost=d.get("est_cost", 0.0),
        alternatives=d.get("alternatives", []),
    )


def plan_task(client, tools_schema, user_query: str,
              failed_paths: List[str] = None,
              replan_reason: str = "",
              cost_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
    """任务开始前的 Planning 调用：生成任务阶段 + 验收规格（成本感知）

    这是架构升级的第一步：让 LLM 在开跑前理解任务并建立状态。

    Args:
        client: OpenAI 客户端
        tools_schema: 工具 schema（不用，纯文本规划）
        user_query: 任务描述
        failed_paths: 已失效的启动路径/方式（重新规划时禁用，防止死磕过期启动器）
        replan_reason: 重新规划原因（非空表示本次是 replan）
        cost_feedback: 上一轮成本反馈 {spent, budget, over_budget}

    Returns:
        {
          "application": str,
          "initial_state": str,
          "phases": [PhaseDefinition],
          "goal_spec": Dict or None,
          "raw": str
        }
    """
    # ---- 重规划 / 成本反馈约束（仅在需要时非空）----
    extra_constraint = ""
    if replan_reason:
        _lines = [f"## ⚠️ 本次为**重新规划**（触发原因: {replan_reason}）"]
        if failed_paths:
            _lines.append("以下启动方式已确认失效，**禁止再使用**：")
            for _p in failed_paths[:10]:
                _lines.append(f"  - {_p}")
        _lines.append("必须为启动步骤提供至少 2 种**不在禁用清单内**的备选方式。")
        _lines.append("禁止原样重复上一版计划。")
        extra_constraint += "\n".join(_lines) + "\n\n"
    if cost_feedback and cost_feedback.get("over_budget"):
        extra_constraint += (
            "## 💰 成本反馈：上一轮已超预算"
            f"（已用 {float(cost_feedback.get('spent', 0)):.1f} / "
            f"预算 {float(cost_feedback.get('budget', 0)):.1f} 成本点）。\n"
            "本轮必须显著降低预计成本：优先单条 run_powershell 直达，"
            "减少全屏扫描/全屏OCR阶段，减少阶段数。\n\n"
        )

    planning_prompt = f"""你是一个Windows桌面自动化的**任务规划器**。在 Agent 开始执行前，你负责生成任务的结构化规划。

## 任务
{user_query}

{extra_constraint}## 输出格式（严格遵循）

请用以下块输出规划：

<plan>
{{
  "application": "目标应用名（如 Beholder / 记事本 / Visual Studio Code）",
  "initial_state": "起始界面状态（如 DESKTOP / MAIN_MENU / INSTALLER_STARTED）",
  "phases": [
    {{
      "name": "阶段名（大写英文，如 LAUNCH_APP）",
      "entry_conditions": ["该阶段屏幕应出现的文字/元素关键词"],
      "allowed_actions": ["该阶段允许的工具名"],
      "expected_next": ["预期下一阶段名"],
      "verification_method": ["如何验证当前处于此阶段"],
      "est_cost": 3,
      "alternatives": ["该阶段若有多种做法，列出备选方法"]
    }}
  ],
  "est_total_cost": 12
}}
</plan>

<goal>
{{
  "goal": "目标描述（简短）",
  "evidence": ["验收证据：屏幕上应出现的关键词/元素"],
  "failure_condition": ["失败条件：出现则视为未完成"]
}}
</goal>

<plans>
[
  {{"name": "方案A（名称）", "approach": "一句话做法", "est_total_cost": 10, "risk": "low", "rationale": "为什么选它（步骤越少越好）"}},
  {{"name": "方案B", "approach": "...", "est_total_cost": 18, "risk": "medium", "rationale": "..."}}
]
</plans>

## 要求
1. phases 覆盖任务的主要阶段（3-6 个），从初始状态到完成。
2. entry_conditions 是**屏幕真实会出现的文字**（如 "开始游戏"、"Play"、"单人游戏"、"确定"、应用标题栏文字），
   必须能被 OCR 直接识别到的实际屏幕文字。**禁止使用抽象概念描述**（如 "桌面图标"、"游戏主菜单"、"界面已打开"），
   因为 OCR 无法匹配这些抽象词。若某阶段确实无文字证据，请写该阶段窗口/按钮上的具体文字。
3. allowed_actions 从这些工具中选择：visual_scan, visual_scan_region, visual_scan_grid, visual_locate, visual_locate_region, visual_read_text, visual_read_region, visual_find_text, click_at, drag_mouse, type_text, press_key, hotkey, run_powershell, wait
4. goal-spec 的 evidence 是可独立验证的**屏幕特征**（用于验收，禁止依赖执行过程）。
   evidence 同样必须是**屏幕真实文字**（如 "开始游戏"、"Play"、游戏标题文字），禁止抽象描述。
5. **成本感知**：为每阶段给出 est_cost（预估调用次数，越小越省），并给出 est_total_cost。
6. **多方案**：<plans> 给出 2-3 个候选方案并各自估成本；系统会选择"成本最低且可行"的方案执行。
   优先低成本路径（一条 run_powershell 直达 > 多步截图定位点击）。
   **禁止**把"反复点同一目标"当方案；若某启动方式已在其失败清单中，必须换方式。
7. 只输出 <plan>、<goal>、<plans> 块，不要其他解释文字。
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是Windows桌面自动化的任务规划器，输出结构化 JSON 规划。"},
                {"role": "user", "content": planning_prompt},
            ],
            temperature=0.2,
            timeout=30,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_planning_response(content)

        # 构建 PhaseDefinition 列表
        phases = []
        for pd in parsed.get("phases", []):
            try:
                phases.append(phase_from_dict(pd))
            except Exception as e:
                print(f"  [规划] 阶段解析跳过: {e}")
        parsed["phases"] = phases
        parsed["raw"] = content

        # ---- 成本感知择优：从 <plans> 候选中选"成本最低且可行"的方案 ----
        try:
            raw_candidates = parsed.get("plan_candidates") or []
            if raw_candidates:
                candidates = []
                for c in raw_candidates:
                    if not isinstance(c, dict):
                        continue
                    c_phases = []
                    for pd in (c.get("phases") or []):
                        try:
                            c_phases.append(phase_from_dict(pd))
                        except Exception:
                            pass
                    est = c.get("est_total_cost", 0) or 0
                    try:
                        est = float(est)
                    except (TypeError, ValueError):
                        est = 0.0
                    if est <= 0:
                        est = estimate_phases_cost(c_phases)
                    candidates.append(PlanCandidate(
                        name=c.get("name", "候选方案"),
                        approach=c.get("approach", ""),
                        est_total_cost=est,
                        risk=str(c.get("risk", "medium")).lower(),
                        phases=c_phases,
                        rationale=c.get("rationale", ""),
                    ))
                best = select_best_plan(candidates, failed_paths=failed_paths)
                if best is not None:
                    print(f"  [规划|成本] 候选 {len(candidates)} 个，"
                          f"选中 [{best.name}] est_cost={best.est_total_cost}")
                    for c in candidates:
                        print(f"    - {c.name}: cost={c.est_total_cost} "
                              f"risk={c.risk}")
                    if best.phases:
                        phases = best.phases
                        parsed["phases"] = phases
                    parsed["selected_plan"] = best.to_dict()
                    if best.est_total_cost > 0:
                        parsed["est_total_cost"] = best.est_total_cost
        except Exception as e:
            print(f"  [规划|成本] 择优跳过: {e}")

        # 补齐成本：规划未给 est_total_cost 时按阶段估算
        if not parsed.get("est_total_cost"):
            parsed["est_total_cost"] = estimate_phases_cost(
                parsed.get("phases", [])
            )

        # 校验：至少应有阶段或 goal
        if phases or parsed.get("goal_spec"):
            print("  [规划] ✅ 任务规划生成成功")
            print(f"    - 应用: {parsed.get('application', '未知')}")
            print(f"    - 阶段数: {len(phases)}")
            print(f"    - 预估总成本: {parsed.get('est_total_cost')} 成本点")
            if parsed.get("goal_spec"):
                gs = parsed["goal_spec"]
                print(f"    - 目标: {gs.get('goal', '')} | 证据: {len(gs.get('evidence', []))}条")
        else:
            print("  [规划] ⚠️ 规划结果无效，回退到默认探索模式")
        return parsed
    except Exception as e:
        print(f"  [规划] 规划调用失败，回退到默认探索模式: {e}")
        return {}


def build_system_prompt(
    current_tier: int = 1,
    escalation_msg: str = "",
    state_machine: AgentStateMachine = None,
    working_memory: WorkingMemory = None,
    task_phase: TaskPhaseMachine = None,
) -> str:
    """构建系统提示词"""
    base_prompt = f"""你是一个Windows桌面自动化Agent。你是整个系统的**唯一主脑**，负责所有决策。

## 架构说明

- **你（DeepSeek，云端API）** = 主脑：负责所有决策
  - 拆解任务、决定下一步做什么
  - 判断操作结果、决定是否重试或换方案
  - 决定是否降级到 PowerShell
- **本地视觉模型** = 你的眼睛：只负责看，返回结构化 JSON
  - visual_scan() → 全屏扫描，场景描述
  - visual_scan_region() → 局部放大扫描
  - visual_scan_grid() → 分块扫描
  - visual_locate() → 图形元素定位
  - visual_locate_region() → 区域内图形定位
- **OCR 文字识别** = 你的眼睛（读文字版）
  - visual_read_text() → 全屏 OCR，读所有文字及坐标（最后手段，速度慢）
  - visual_read_region() → 区域 OCR，只扫目标区域（优先用！快10倍）
  - visual_find_text("按钮名") → 直接查找文字返回点击坐标（最常用！）

## OCR 返回格式

visual_read_text() 返回 JSON：
```json
{{"screen": "1920x1080", "texts": [{{"text": "确定", "cx": 350, "cy": 420}}, ...], "count": 15}}
```

visual_find_text("确定") 返回：
```
✅ OCR 找到文字[确定]（exact匹配） -> 坐标(350, 420)，可直接 click_at(350, 420) 点击
```

## OCR 优先原则

找按钮、菜单、链接等**文字型元素**时，优先级：
1. **visual_find_text("文字")** → 最可靠，直接返回坐标
2. **visual_read_region(x,y,w,h)** → 区域 OCR，快且准
3. **visual_scan() / visual_locate()** → 图形匹配（对桌面小图标不可靠）
4. **visual_read_text()** → 全屏 OCR，最后手段（CPU 慢）

## 工作流程

1. 拆解任务 → 决定下一步
2. 需要看屏幕 → visual_scan() 看场景
3. 需要找按钮文字 → visual_find_text("文字") 直接拿坐标
4. 需要操作 → click_at() / type_text() / press_key()
5. 操作后 → visual_scan() 确认效果
6. 重复直到完成

## 两层降级执行策略

### Tier 1 - 多模态操作（优先使用）
可用工具：
- visual_scan(): 全屏扫描
- visual_scan_region(x,y,w,h): 局部放大
- visual_scan_grid(rows,cols): 分块扫描
- visual_locate("元素"): 图形定位
- visual_locate_region("元素",x,y,w,h): 区域图形定位
- visual_read_text(): 全屏 OCR
- visual_read_region(x,y,w,h): 区域 OCR
- visual_find_text("文字"): 查找文字坐标
- click_at(x,y,button): 点击
- drag_mouse(x1,y1,x2,y2): 拖拽
- type_text("内容"): 输入文字
- press_key("enter"): 按键
- hotkey(["ctrl","s"]): 组合键

### Tier 2 - PowerShell 命令（Tier 1 失败后降级）
- run_powershell("命令"): 执行 PowerShell 命令

## 降级规则
1. 默认 Tier 1（多模态）
2. Tier 1 连续失败 3 次 → 降级 Tier 2（PowerShell）
3. Tier 2 也失败 → 任务无法完成，报告原因
4. 当前层级: Tier {current_tier}

## 经验记忆准则（Experience Memory）
你是**有操作经验**的 Computer Use Agent，不是每次都从零开始的新手。

执行前：
1. 查看系统注入的 [历史经验] 段落（语义经验 + 相似任务经验）。
2. 可复用已验证的成功工作流，跳过无意义的重复探索。
3. 已知失败案例（failure_recoveries）应主动规避。

⚠️ 记忆是指导，不是真理：
4. 应用记忆中的操作前，必须先验证当前屏幕状态（OCR/视觉）。
5. 若当前界面与记忆不符（分辨率变化/图标整理/软件更新），
   以当前观测为准，回退到正常视觉定位流程。

经验记忆不包含绝对坐标 —— 它告诉你怎么找目标，而不是目标在哪。
"""
    if escalation_msg:
        base_prompt += f"\n## [降级提示] {escalation_msg}\n"

    # 注入状态感知信息
    if state_machine is not None:
        base_prompt += "\n" + state_machine.build_state_instruction() + "\n"

    # 注入工作记忆（短期状态）
    if working_memory is not None:
        wm_ins = working_memory.build_instruction()
        if wm_ins:
            base_prompt += "\n" + wm_ins + "\n"

    # 注入任务阶段信息
    if task_phase is not None:
        base_prompt += "\n" + task_phase.build_instruction() + "\n"

    # 动作预测指令：让 LLM 在重要动作前声明预期
    base_prompt += """
## 动作预测规范（Action Prediction）
当你执行**会改变界面状态**的动作（click_at / type_text / press_key / hotkey / run_powershell）时，
在调用工具之前的文本回复中可以附带 <predict> 块，声明你的预期：

<predict>
{"next_state": "预期进入的阶段", "should_appear": ["应出现的元素/文字"], "timeout": 3}
</predict>

系统会：
1. 执行你的动作
2. 根据你的 <predict> 等待并观察
3. 对比预期 vs 实际，判断"点击失败/加载中/弹窗/成功"

如果未提供 <predict>，系统将使用任务阶段机定义的预期作为兜底。
"""

    # ⛔ 禁止盲点：硬约束（配合 core.py 的动作门卫，双保险）
    base_prompt += """
## ⛔ 禁止盲点（强制约束）
1. **没有坐标来源就绝不点击**：可用于 `click_at` 的坐标只允许来自
   `visual_find_text` / `visual_locate` / `visual_locate_region` /
   `visual_read_text` / `visual_read_region`（或视觉对象缓存命中）。
   凭猜测、凭记忆、"大概在那个位置"的坐标会被系统**直接拦截**，
   并返回"点击失败: 禁止盲点"。
2. **遵守当前任务阶段**：系统按 [任务阶段] 的 allowed_actions 拦截越权动作。
   若动作被拒绝，请阅读拒绝原因并改用本阶段允许的操作，
   **不要换个坐标继续点同一个东西**。
3. **先确认、再动作**：无法通过 OCR/视觉确认目标存在时，禁止"试探性点击"；
   应继续观察（换区域/换工具）或改用 `run_powershell` 直达。
4. **假点击会判失败**：若点击后界面毫无变化，或**同一坐标重复点击仍无变化**，
   系统会返回"点击失败: …假点击…"。此时**禁止原样重试**，
   必须先重新观察（确认目标是否还存在/界面是否已变），再换坐标或换方式。
"""

    return base_prompt


def _extract_ocr_texts(result_str: str) -> List[str]:
    """从工具结果字符串中提取 OCR 文字列表

    OCR 工具（visual_read_text / visual_read_region）返回 JSON：
      {"texts": [{"text": "确定", ...}, ...]}
    也兼容 visual_scan 的 elements label。

    Args:
        result_str: 工具返回字符串

    Returns:
        文字列表
    """
    texts = []
    try:
        data = json.loads(result_str)
        if isinstance(data, dict):
            # OCR 工具结果
            for t in data.get("texts", []):
                txt = t.get("text", "")
                if txt:
                    texts.append(txt)
            # 视觉扫描的元素 label
            for e in data.get("elements", []):
                label = e.get("label", "")
                if label:
                    texts.append(label)
            # 场景描述
            scene = data.get("scene", "")
            if scene:
                texts.append(scene)
    except (json.JSONDecodeError, TypeError):
        # 非 JSON（如 visual_find_text 的文本返回）
        pass
    return texts


# ============================================================
# ⚡ 动作门卫（P3）：阶段门控 + 禁止盲点
# ============================================================
# 仅对"会改变界面状态"的 UI 动作生效；run_powershell 作为逃生通道不拦截
GUARDED_ACTION_TOOLS = {
    "click_at", "drag_mouse", "type_text", "press_key", "hotkey",
}
# 坐标来源识别（从工具结果中提取真实坐标）
_COORD_PATTERNS = (
    re.compile(r"坐标\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
    re.compile(r'"cx"\s*:\s*(\d+)\s*,\s*"cy"\s*:\s*(\d+)'),
)
_COORD_TOLERANCE = 12
_COORD_WINDOW_STEPS = 3


def _collect_coordinates(result_str: str, sink: set) -> None:
    """从工具结果中提取真实坐标（供"禁止盲点"门卫校验）"""
    if not result_str or sink is None:
        return
    text = str(result_str)
    for pat in _COORD_PATTERNS:
        for m in pat.finditer(text):
            try:
                sink.add((int(m.group(1)), int(m.group(2))))
            except (TypeError, ValueError):
                continue
    while len(sink) > 400:          # 防止无限增长
        sink.pop()


def _recent_known_coords(coord_history: List[set], step: int,
                         window: int = _COORD_WINDOW_STEPS) -> set:
    """最近 window 步内获得的坐标集合"""
    merged: set = set()
    if not coord_history:
        return merged
    start = max(0, step - window)
    for i in range(start, min(step, len(coord_history))):
        merged |= coord_history[i]
    return merged


def _is_known_coordinate(x: int, y: int, known: set) -> bool:
    """坐标是否来自真实定位结果（容差内匹配）"""
    for (kx, ky) in known:
        if abs(kx - x) <= _COORD_TOLERANCE and abs(ky - y) <= _COORD_TOLERANCE:
            return True
    return False


def _action_guard(tool_name: str, tool_args: Dict[str, Any],
                  task_phase: TaskPhaseMachine, known_coords: set) -> str:
    """动作执行前门卫：返回非空字符串 = 拒绝执行（内容为拒绝原因）

    规则：
      1. 阶段门控：阶段机定义了 allowed_actions 且本工具不在其中 → 拒绝
      2. 禁止盲点：click_at 坐标必须来自最近几步的真实定位/OCR 结果

    失败开放（fail-open）：任何内部异常都不拦截，避免误伤正常流程。
    """
    try:
        # 规则 1：阶段门控（让"计划"真正约束"动作"）
        if tool_name in GUARDED_ACTION_TOOLS and task_phase is not None:
            if not task_phase.is_action_allowed(tool_name):
                return (
                    f"动作被拒绝: {task_phase.get_disallowed_hint(tool_name)}\n"
                    f"当前阶段为 [{task_phase.current_phase}]，"
                    f"请先完成该阶段目标，或改用本阶段允许的操作。"
                )

        # 规则 2：禁止盲点（click_at 必须有坐标来源）
        if tool_name == "click_at":
            raw_x = tool_args.get("x")
            raw_y = tool_args.get("y")
            try:
                x, y = int(raw_x), int(raw_y)
            except (TypeError, ValueError):
                return (f"点击失败: 坐标缺失或非法 (x={raw_x!r}, y={raw_y!r})，"
                        f"禁止盲点。")
            if not _is_known_coordinate(x, y, known_coords):
                return (
                    f"点击失败: 坐标 ({x}, {y}) 没有任何定位来源 —— 禁止盲点。\n"
                    f"请先调用 visual_find_text(\"目标文字\") / visual_locate() / "
                    f"visual_read_region() 取得真实坐标，再点击该坐标；"
                    f"若无法确认目标存在，请继续观察或更换方案。"
                )
    except Exception:
        return ""
    return ""


def _parse_predict_block(content: str) -> Dict[str, Any]:
    """解析 LLM 回复中的 <predict> 块

    Args:
        content: LLM 回复文本

    Returns:
        {"next_state": str, "should_appear": [...], "timeout": float}
        解析失败返回空字典
    """
    import re
    if not content:
        return {}
    m = re.search(r'<predict>(.*?)</predict>', content, re.S | re.I)
    if not m:
        return {}
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return {}


def _auto_predict_from_phase(task_phase: TaskPhaseMachine,
                             working_memory: WorkingMemory) -> Dict[str, Any]:
    """从任务阶段机自动构建预测（LLM 未提供 <predict> 时兜底）"""
    expected_next = task_phase.get_expected_next()
    if not expected_next:
        return {}

    # 取下一阶段作为预测
    next_phase_name = expected_next[0]
    next_phase = task_phase.get_phase(next_phase_name)
    if next_phase is None:
        return {}

    return {
        "next_state": next_phase_name,
        "should_appear": next_phase.entry_conditions,
        "timeout": 3.0,
    }


def _drive_phase_transition(task_phase: TaskPhaseMachine,
                            working_memory: WorkingMemory,
                            observed_texts: List[str]) -> Optional[str]:
    """用观察证据驱动任务阶段转换，并同步工作记忆"""
    if not observed_texts:
        return None
    new_phase = task_phase.try_transition(observed_texts)
    if new_phase:
        print(f"  [阶段推进] {task_phase.current_phase} ← 证据: {', '.join(observed_texts[:5])}")
        # 同步 WorkingMemory 旧字段
        working_memory.update(current_state=new_phase)
        # ⚡ StateManager: 同步分层世界状态（current_stage）
        if hasattr(working_memory, 'set_stage'):
            working_memory.set_stage(new_phase)
    return new_phase


def _run_prediction_closure(
    predictor: ActionPredictor,
    task_phase: TaskPhaseMachine,
    working_memory: WorkingMemory,
    tool_name: str,
    tool_args: Dict[str, Any],
    llm_content: str,
) -> Dict[str, Any]:
    """动作预测闭环：predict → wait → observe → compare → classify

    仅在动作类工具（改变界面状态）上执行。

    Returns:
        {
          "prediction_made": bool,
          "result": compare_result or None,
          "classification": str
        }
    """
    # 仅动作类工具才做预测
    if tool_name not in WorkingMemory.ACTION_TOOLS:
        return {"prediction_made": False, "result": None, "classification": "ok"}

    # 预测来源 1：LLM <predict> 块
    predict_data = _parse_predict_block(llm_content or "")
    source = "llm"
    # 预测来源 2：任务阶段兜底
    if not predict_data:
        predict_data = _auto_predict_from_phase(task_phase, working_memory)
        source = "phase"
    if not predict_data:
        # 无预测 → 仍记录动作，但不等待
        return {"prediction_made": False, "result": None, "classification": "ok"}

    predicted_state = predict_data.get("next_state", "")
    should_appear = predict_data.get("should_appear", [])
    timeout = float(predict_data.get("timeout", 3.0))

    # 过滤抽象描述（OCR 无法匹配的"概念词"），避免无意义空等 25s
    # 例: "桌面图标"、"游戏主菜单"、"界面已打开" —— 屏幕不会显示这些字
    # ⚠️ 注意：不能粗暴过滤短词（如 "游戏"），因为 "开始游戏"/"单人游戏"
    # 是真实按钮文字。这里用"组合抽象词"精确匹配，避免误伤真实 UI 文字。
    _ABSTRACT_TERMS = [
        # 组合抽象描述（屏幕不会直接显示这些字）
        "桌面图标", "游戏主菜单", "主菜单", "游戏界面", "游戏加载界面",
        "加载界面", "登录界面", "主界面", "进入游戏", "游戏菜单",
        "等待加载", "加载完成", "已进入", "已启动", "已打开", "加载中",
        "界面已", "窗口出现", "游戏窗口", "游戏启动",
    ]
    # 仅当整词就是这些短抽象词时才过滤（"游戏"单独出现是抽象，"开始游戏"不是）
    _ABSTRACT_EXACT = {"桌面", "图标", "界面", "主菜单", "游戏", "窗口",
                       "按钮", "元素", "状态", "icon", "desktop", "menu",
                       "screen", "app", "出现", "打开"}
    filtered_appear = []
    for item in (should_appear or []):
        s = str(item).strip()
        if not s:
            continue
        # 判断是否抽象描述：
        #   1. 命中组合抽象词（"游戏加载界面"、"桌面图标"等）
        #   2. 整词就是短抽象词（"游戏"、"界面"等单独出现）
        is_abstract = any(term in s for term in _ABSTRACT_TERMS)
        if not is_abstract and s in _ABSTRACT_EXACT:
            is_abstract = True
        if not is_abstract:
            filtered_appear.append(s)
    should_appear = filtered_appear

    # 建立预测
    pred = predictor.predict(
        action=tool_name,
        args=tool_args,
        predicted_state=predicted_state,
        should_appear=should_appear,
        timeout=timeout,
        source=source,
    )

    # 同步工作记忆
    working_memory.set_prediction(predicted_state, timeout)
    if should_appear:
        working_memory.set_expected_transition(
            predicted_state, should_appear, timeout
        )

    print(f"  [预测] {tool_name} → {predicted_state or '状态变化'} "
          f"(应出现: {', '.join(should_appear) or '无'})")

    # 预测有 should_appear → 等待并观察
    # 若无有效期望（全被抽象过滤），跳过观察避免空等
    observed_texts = []
    wait_result = None
    if should_appear:
        wait_result = predictor.wait_and_observe(pred)
        # ⚡ 复用 wait_and_observe 的末次观察结果，避免再全屏 OCR 一次（省 2-5s）
        observed_texts = wait_result.get("last_observation") or []
        print(f"  [观察] 匹配={wait_result.get('found', [])} "
              f"未匹配={wait_result.get('missed', [])} "
              f"耗时={wait_result.get('elapsed', 0)}s")
    else:
        # 无有效期望 → 视为观察通过（不阻塞）
        wait_result = {
            "matched": True, "found": [], "missed": [],
            "elapsed": 0.0, "observation_count": 0,
        }
        observed_texts = []

    # 对比预测 vs 观察（复用已观察文本，避免又一次全屏 OCR）
    compare_result = predictor.compare(
        pred,
        observed_phase=task_phase.current_phase,
        ocr_texts=(observed_texts or None),
    )
    classification = predictor.classify_failure(compare_result, tool_name)

    # 更新工作记忆
    if wait_result and wait_result.get("matched"):
        working_memory.update_state_after_observation(predicted_state, 0.8)
    else:
        working_memory.update(note=f"预测未匹配: {classification}")

    # 预测结果结构化输出（供日志）
    return {
        "prediction_made": True,
        "prediction": pred.to_dict(),
        "observe_result": wait_result,
        "result": compare_result,
        "classification": classification,
    }


def run_task(client, tools_schema, user_query: str):
    """执行单个任务（内部循环，接入状态感知机 + 异常处理 + 闭环智能体）"""
    # ============================================================
    # 初始化：状态感知机 + 异常处理器 + 知识库 + 闭环智能体模块
    # ============================================================
    state_machine = AgentStateMachine(max_steps=MAX_STEPS)
    exception_handler = ExceptionHandler(tools_registry=TOOL_REGISTRY)
    memory_manager = MemoryManager()

    # 闭环智能体新模块
    working_memory = StateManager()          # 增强版：WorkingMemory + 分层世界状态 + 失败计数
    task_phase = TaskPhaseMachine()          # 默认 EXPLORING，规划后替换
    verifier = GoalVerifier()                # 独立验收器
    tracker = ObjectTracker()                # 视觉对象记忆
    predictor = ActionPredictor()            # 动作预测
    exec_logger = ExecutionLogger()          # 结构化日志
    exp_store = AbstractExperienceStore()    # 抽象经验
    tool_manager = ToolManager()             # 工具调用策略管理器
    world_mgr = WorldStateManager()          # 世界状态理解模块
    world_before_dict = None                 # 状态 diff 快照
    ws_verifier = WorldStateVerifier()       # 世界状态驱动验收器
    budget = BudgetTracker()                 # 成本预算追踪（成本感知规划）
    replan_mgr = ReplanManager(              # 过期启动器 → 强制重新规划
        threshold=2, max_replans=2
    )

    # 关联工作记忆到异常处理器（供快照）
    exception_handler.working_memory = working_memory

    # 执行轨迹（供任务结束经验提取）
    execution_trace = []

    # ⚡ 动作门卫：坐标来源历史（按 step 分桶，仅保留最近若干步）
    coord_history: List[set] = []

    # ⚡ 假点击检测：重置点击状态（避免上一个任务的点击记录干扰本任务）
    try:
        from agent.tools import reset_fake_click_state
        reset_fake_click_state()
    except Exception:
        pass

    # 尝试接入知识库（自愈经验沉淀）
    try:
        from knowledge.base import KnowledgeBase
        kb = KnowledgeBase()
        exception_handler.knowledge_base = kb
        if exception_handler.healer is None:
            from self_healer import SelfHealer
            exception_handler.healer = SelfHealer(kb)
    except Exception as e:
        print(f"  [知识库] 初始化跳过: {e}")

    # ============================================================
    # 任务开始：Planning（理解任务 → 建立状态）
    # ============================================================
    print(f"\n开始执行任务: {user_query}\n")
    print("  [规划] 正在生成任务阶段与验收规格...")
    plan_result = plan_task(client, tools_schema, user_query)

    # 应用规划：工作记忆初始化
    application = plan_result.get("application", "")
    initial_state = plan_result.get("initial_state", "")
    working_memory.init_task(
        task=user_query,
        application=application,
        initial_state=initial_state,
        confidence=0.5,
    )

    # 应用规划：任务阶段机
    phases = plan_result.get("phases", [])
    if phases:
        task_phase = TaskPhaseMachine(phases=phases, start_phase=initial_state or "EXPLORING")
        print("  [阶段] 已应用任务阶段机")

    # 应用规划：验收规格
    goal_spec_dict = plan_result.get("goal_spec")
    if goal_spec_dict:
        verifier.set_spec_from_dict(goal_spec_dict)
        print(f"  [验收] 已设置独立验收规格: {goal_spec_dict.get('goal', '')}")

    # ⚡ 成本感知：用计划成本设定执行预算
    try:
        planned_cost = plan_result.get("est_total_cost", 0) or 0
        budget.set_from_plan(planned_cost)
        print(f"  [成本预算] 计划成本 {planned_cost} → 预算 {budget.budget:.1f} 成本点")
        if plan_result.get("selected_plan"):
            sp = plan_result["selected_plan"]
            print(f"  [成本预算] 采用方案: {sp.get('name')} "
                  f"(cost={sp.get('est_total_cost')}, risk={sp.get('risk')})")
    except Exception as e:
        print(f"  [成本预算] 初始化跳过: {e}")

    # ⚡ WorldState: 创建任务世界状态
    try:
        world_mgr.create(user_query)
        world_before_dict = world_mgr.to_dict()
        print("  [世界状态] 已创建任务世界模型")
    except Exception as e:
        print(f"  [世界状态] 初始化跳过: {e}")

    # 初始化结构化日志
    try:
        exec_logger.start_task(user_query, goal_spec=goal_spec_dict)
    except Exception as e:
        print(f"  [日志] 日志初始化跳过: {e}")

    state_machine.start()

    current_tier = 1
    tier_failures = 0
    max_tier_attempts = TIER_MAX_FAILURES

    system_prompt = build_system_prompt(
        current_tier, state_machine=state_machine,
        working_memory=working_memory, task_phase=task_phase,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # 检索注入历史经验（经验检索 + 抽象经验）
    try:
        experience_ctx = memory_manager.retrieve_context(user_query)
        if experience_ctx:
            messages.append({"role": "system", "content": experience_ctx})
            print("  [记忆] 已注入历史操作经验")
    except Exception as e:
        print(f"  [记忆] 经验检索跳过: {e}")

    # 注入抽象经验
    try:
        exp_ctx = exp_store.build_instruction(user_query)
        if exp_ctx:
            messages.append({"role": "system", "content": exp_ctx})
            print("  [记忆] 已注入抽象行为模式")
    except Exception as e:
        print(f"  [记忆] 抽象经验注入跳过: {e}")

    # 注入 Skill 技能指令（领域策略引导）
    active_skill = None
    try:
        skill_classes = get_base_skills()
        skill_ctx = build_skill_instruction(user_query, skill_classes)
        if skill_ctx:
            messages.append({"role": "system", "content": skill_ctx})
            from skills.base import select_skill
            skill_cls = select_skill(user_query, skill_classes)
            if skill_cls is not None:
                active_skill = skill_cls
                print(f"  [技能] 已匹配并注入 [{skill_cls.name}] 领域技能")
    except Exception as e:
        print(f"  [技能] 技能注入跳过: {e}")

    # ⚡ 重规划：注入已知失效路径清单（跨任务失效记忆，防止死磕过期启动器）
    try:
        _known_failed = replan_mgr.failed_paths()
        if _known_failed:
            messages.append({
                "role": "system",
                "content": (
                    "【失效路径记忆】以下启动方式历史上已确认失效，"
                    "**禁止再使用**，必须换用其他方式启动：\n  - "
                    + "\n  - ".join(_known_failed[:10])
                ),
            })
            print(f"  [重规划] 已注入失效路径记忆 {len(_known_failed)} 条")
    except Exception as e:
        print(f"  [重规划] 失效路径注入跳过: {e}")

    # ⚡ WorldState: 注入初始世界状态摘要（messages 已定义）
    try:
        ws_summary = world_mgr.to_summary()
        if ws_summary:
            messages.append({"role": "system", "content": ws_summary})
    except Exception as e:
        print(f"  [世界状态] 摘要注入跳过: {e}")

    # ⚡ Skill 约束：工具越权检查在工具调用时用 active_skill.is_tool_allowed()
    success = False
    final_result = ""
    steps_taken = 0
    all_tools_used = []
    recent_tool_results = []

    for step in range(1, MAX_STEPS + 1):
        steps_taken = step

        # ⚡ 动作门卫：准备当前 step 的坐标桶（供"禁止盲点"校验）
        if len(coord_history) < step:
            coord_history.append(set())
        current_coords = coord_history[step - 1]

        # ============================================================
        # 每轮前：状态注入提醒（卡住/恢复/无进展）
        # ============================================================
        state_payload = state_machine.get_status_payload()

        # 卡住状态 → 注入换策略指令
        if state_machine.state == AgentState.STUCK:
            messages.append({
                "role": "system",
                "content": (
                    "【状态警告】系统检测到你已卡住。"
                    "请立即改变策略：优先使用 visual_find_text() 定位目标文字，"
                    "或改用 run_powershell() 命令完成任务。"
                    "不要重复之前失败的调用。"
                ),
            })
            state_machine.enter_recovering("注入换策略指令")

        # 连续无进展提醒（替代旧 continuous_scan_count 逻辑）
        elif state_payload["steps_without_progress"] >= 2:
            messages.append({
                "role": "system",
                "content": (
                    "【警告】你已经连续多步没有实质进展。\n"
                    "**不要盲目点击。** 请按顺序确认后再动作：\n"
                    "  1. 用 visual_find_text(\"屏幕上的真实文字\") 或 visual_read_region() "
                    "确认目标**确实存在**并取得坐标；\n"
                    "  2. 对照 [任务阶段] 判断当前屏幕处于计划的哪个阶段；\n"
                    "  3. 再执行与该阶段匹配的操作（点击**已获得坐标**的目标，"
                    "或改用 run_powershell 直达）。\n"
                    "  若无法确认目标存在 → 禁止点击，改为继续观察或更换方案。"
                ),
            })

        # ============================================================
        # 状态动态注入：让 LLM 感知最新工作记忆 + 任务阶段
        # （仅在工作记忆或阶段发生变化时注入，避免无意义刷屏）
        # ============================================================
        state_inject = []
        wm_ins = working_memory.build_instruction()
        tp_ins = task_phase.build_instruction()
        if wm_ins:
            state_inject.append(wm_ins)
        if tp_ins:
            state_inject.append(tp_ins)
        # 只在本轮有状态变化时注入（简化：每轮都注入最新状态，但控制大小）
        if state_inject:
            msg_content = "\n\n".join(state_inject)
            messages.append({"role": "system", "content": msg_content})

        # ⚡ 成本感知：注入成本预算状态 + 超预算降级指令
        try:
            if budget.over_budget():
                hint = budget.take_downgrade_hint()
                if hint:
                    messages.append({"role": "system", "content": hint})
                    print("  [成本] ⚠️ 已超预算，注入降级指令")
            elif budget.should_warn():
                messages.append({"role": "system",
                                 "content": budget.build_instruction()})
        except Exception as e:
            print(f"  [成本] 注入跳过: {e}")

        # ============================================================
        # API 调用（带异常兜底：BadRequest / 网络错误 / 超时）
        # ============================================================
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL, messages=messages,
                tools=tools_schema, tool_choice="auto", temperature=0.1,
                timeout=30,
            )
        except Exception as e:
            # 阻止API异常让整个Agent崩溃 —— 记录错误并安全结束任务
            print(f"  [API错误] {type(e).__name__}: {e}")
            state_machine.mark_failed(f"API调用失败: {type(e).__name__}: {e}")
            success = False
            final_result = f"任务失败：API调用异常 ({type(e).__name__})"
            print(f"\n  [失败] API 调用异常，任务终止")
            break

        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            # ============================================================
            # 工具调用处理
            # ============================================================
            scan_tools = {"visual_scan", "visual_scan_region", "visual_scan_grid",
                          "visual_locate", "visual_locate_region"}
            action_tools = {"click_at", "drag_mouse", "type_text", "press_key",
                            "hotkey", "run_powershell"}
            has_scan_only = all(tc.function.name in scan_tools for tc in msg.tool_calls)
            has_action = any(tc.function.name in action_tools for tc in msg.tool_calls)

            tool_names_in_step = []
            has_step_failure = False
            has_step_progress = has_action
            pending_fix_instructions = []  # 修复指令延迟注入，保证 assistant→tool 协议顺序
            # ⚙ 观察串行化：一次 step 只允许一种观察工具
            #   观察类 = 扫描/定位 + OCR 读取（所有"只看不做"的工具）
            #   规则：本步已观察 -> 跳过后续观察；动作执行后 / 工具报错后 -> 解锁观察
            observation_tools = {
                "visual_scan", "visual_scan_region", "visual_scan_grid",
                "visual_locate", "visual_locate_region",
                "visual_read_text", "visual_read_region", "visual_find_text",
            }
            observation_done = False        # 本 step 是否已完成一次观察
            last_observation_tool = ""      # 已执行的观察工具名（用于提示）

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # ⚙ 观察串行化拦截：本步已观察过 -> 跳过后续观察工具
                #   仍需写入 tool 响应（OpenAI 协议要求每个 tool_call_id 必须响应）
                if tool_name in observation_tools and observation_done:
                    skip_msg = (
                        f"⏸ 已跳过观察工具 {tool_name}：本次 step 已完成一次观察"
                        f"（{last_observation_tool}）。\n"
                        f"请基于上一条观察结果直接决定下一步动作"
                        f"（click_at / type_text / press_key / hotkey / run_powershell），"
                        f"或需要新观察时等待下一轮。"
                    )
                    print(f"[Step {step}|观察串行] ⏸ 跳过 {tool_name}"
                          f"（本步已完成 {last_observation_tool} 观察）")
                    tool_names_in_step.append(tool_name)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": skip_msg})
                    recent_tool_results.append(skip_msg)
                    continue
                tier_label = {1: "T1-多模态", 2: "T2-PowerShell"}
                print(f"[Step {step}|{tier_label.get(current_tier, '?')}] "
                      f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                # ============================================================
                # ⚡ 动作门卫（阶段门控 + 禁止盲点）：不合法则拒绝执行
                # ============================================================
                guard_reject = _action_guard(
                    tool_name, tool_args, task_phase,
                    _recent_known_coords(coord_history, step),
                )
                if guard_reject:
                    print(f"  [门卫] 拦截 {tool_name}")
                    tool_result = guard_reject
                else:
                    tool_result = "未知工具"
                    for t in ALL_TOOLS:
                        if t.name == tool_name:
                            tool_result = t.invoke(tool_args)
                            break
                    else:
                        # 工具不在 ALL_TOOLS，尝试注册表
                        if tool_name in TOOL_REGISTRY:
                            tool_result = TOOL_REGISTRY[tool_name].invoke(tool_args)

                all_tools_used.append(tool_name)
                tool_names_in_step.append(tool_name)
                result_str = str(tool_result)

                # ⚡ 动作门卫：登记本次结果里的真实坐标（供后续点击校验）
                try:
                    _collect_coordinates(result_str, current_coords)
                except Exception:
                    pass

                # ⚡ 成本感知：据此工具调用记账
                try:
                    budget.add_tool(tool_name, tool_args)
                except Exception:
                    pass
                display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                safe_print(f"  <- {display}")
                # ⚙ 动作类工具会改变屏幕 → 让全屏 OCR 缓存立即失效
                if tool_name in action_tools:
                    try:
                        from vision.ocr import invalidate_ocr_cache
                        invalidate_ocr_cache()
                    except Exception:
                        pass
                # ⚙ 观察串行化：更新观察标志
                #   观察工具 -> 锁定（本步后续观察被跳过）
                #   动作工具 -> 解锁（动作后可重新观察）
                if tool_name in observation_tools:
                    observation_done = True
                    last_observation_tool = tool_name
                    # ⚡ 全屏观察刷新了整屏信息 → 更早的坐标视为过期，
                    #   避免点击"按钮曾经所在的位置"（假点击的常见来源）
                    if tool_name in ("visual_read_text", "visual_scan",
                                     "visual_scan_grid"):
                        try:
                            for _i in range(step - 1):
                                coord_history[_i] = set()
                        except Exception:
                            pass
                elif tool_name in action_tools:
                    observation_done = False

                # ⚡ 结构化日志：记录工具调用明细（此前缺失，导致 JSONL 无执行轨迹）
                try:
                    exec_logger.log_action(tool_name, tool_args, result_str,
                                           success=not exception_handler.is_success(result_str))
                except Exception:
                    pass

                # ⚡ 工作记忆：记录动作（动作前更新 last_action）
                working_memory.record_action(tool_name, tool_args)

                # ⚡ 动作预测闭环（仅动作类工具；感知工具跳过）
                prediction_ctx = _run_prediction_closure(
                    predictor, task_phase, working_memory,
                    tool_name, tool_args,
                    llm_content=msg.content or "",
                )
                prediction_mismatch = False
                if prediction_ctx.get("prediction_made"):
                    exec_logger.log_prediction(prediction_ctx.get("prediction", {}))
                    if prediction_ctx.get("classification", "ok") != "ok":
                        # 预测失败（动作"✅"但预期结果未出现）→ 属于事实上的假成功：
                        #   1) 记为失败，让异常处理/降级/重规划真正介入
                        #   2) 不再算作"实质进展"
                        classification = prediction_ctx.get("classification")
                        prediction_mismatch = True
                        has_step_progress = False
                        pending_fix_instructions.append(
                            f"【预测验证】动作 {tool_name} 已执行但预期结果未出现 "
                            f"(分类: {classification})，本次动作视为**未生效**。\n"
                            f"请不要原样重试同一动作："
                            f"先重新观察屏幕（visual_find_text / visual_read_region），"
                            f"确认目标是否存在、界面是否变化；"
                            f"再决定换坐标、换方式（press_key/hotkey/run_powershell）"
                            f"或关闭遮挡窗口。"
                        )

                # ⚡ 视觉对象记忆：点击动作失效被点击目标
                if tool_name == "click_at":
                    cx = tool_args.get("x")
                    cy = tool_args.get("y")
                    if cx is not None and cy is not None:
                        tracker.invalidate_by_click([int(cx), int(cy)])

                # 记录执行轨迹（供任务结束经验提取，不含坐标）
                execution_trace.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "is_failure": not exception_handler.is_success(result_str),
                })

                # 失败检测（统一失败检测器 + 预测不匹配 = 事实上的假成功）
                is_failure = ((not exception_handler.is_success(result_str))
                              or prediction_mismatch)
                if is_failure:
                    has_step_failure = True
                    tier_failures += 1
                    observation_done = False   # ⚙ 报错后允许重新观察
                    # ⚡ StateManager: 记录失败/重试计数
                    if hasattr(working_memory, 'record_failure'):
                        working_memory.record_failure()
                    print(f"  [!] 失败 ({tier_label.get(current_tier, '?')} "
                          f"失败 {tier_failures}/{max_tier_attempts})")

                    # ============================================================
                    # 异常处理：诊断（修复指令延迟注入，避免打断协议序列）
                    # ============================================================
                    failure_report = exception_handler.process_failure(
                        result_str, tool_name, user_query
                    )
                    if failure_report["fix_instruction"]:
                        pending_fix_instructions.append(failure_report["fix_instruction"])
                        print(f"  [异常处理] 错误类型: {failure_report['error_type']}")

                    if failure_report["should_degrade"]:
                        print(f"  [智能降级建议] {failure_report['degrade_suggestion']}")

                    # ⚡ 过期启动器检测：启动类失败 → 记录失效路径（供重新规划）
                    try:
                        if LaunchFailureDetector.is_launch_failure(
                                tool_name, tool_args, result_str):
                            _tgt = LaunchFailureDetector.extract_target(
                                tool_name, tool_args, result_str)
                            if _tgt:
                                _cnt = replan_mgr.record_launch_failure(
                                    _tgt, reason=str(result_str)[:120],
                                    tool_name=tool_name)
                                print(f"  [过期启动器] 记录失效目标: {_tgt} "
                                      f"（连续 {_cnt} 次）")
                    except Exception as e:
                        print(f"  [过期启动器] 检测跳过: {e}")
                else:
                    # ⚡ StateManager: 记录成功（重置失败/重试计数）
                    if hasattr(working_memory, 'record_success'):
                        working_memory.record_success()
                    # ⚡ 重规划：成功 → 重置连续失败计数
                    replan_mgr.record_success()
                    if tier_failures > 0:
                        tier_failures = 0
                        print(f"  [OK] 成功，重置失败计数")

                # ⚡ ToolManager: 工具使用建议 + 统计（延迟注入，保持协议有效）
                try:
                    advice = tool_manager.get_tool_advice(tool_name, user_query)
                    if advice:
                        pending_fix_instructions.append(advice)
                    # 记录工具成功率统计
                    from agent.tool_manager import infer_task_type
                    tt = infer_task_type(user_query, tool_name)
                    tool_manager.stats.record(tool_name, tt, not is_failure)
                    tool_manager._save_stats()
                except Exception:
                    pass

                # ⚡ WorldState: 工具结果 → 结构化状态 → 语义摘要 → 注入 LLM
                # （替代"原始结果全文直送"，大幅降低 token 消耗）
                llm_tool_content = result_str   # 默认完整结果（回退）
                try:
                    if world_mgr.active:
                        # 记录本次状态快照
                        prev_state = world_mgr.to_dict()

                        # 1. 提取并应用证据更新（含 autowrite 客观事实）
                        world_mgr.update_from_tool(tool_name, result_str, tool_args)
                        world_mgr.autowrite_from_tool(tool_name, result_str, tool_args)

                        # 2. 计算状态差异 → 语义摘要
                        cur_state = world_mgr.to_dict()
                        diff_result = StateDiff.diff(prev_state, cur_state)
                        semantic = diff_result.get("semantic_summary", "")

                        # 3. 产生压缩注入内容
                        if semantic and semantic != "世界状态无变化":
                            # 语义摘要有效 → 用它替换原始结果，节省 token
                            llm_tool_content = (
                                f"[工具执行结果已压缩为世界状态更新]\n"
                                f"工具: {tool_name}\n"
                                f"状态变化: {semantic}\n"
                                f"(如需完整输出请参考日志, 但通常不需要)"
                            )
                            print(f"  [世界状态压缩] {semantic[:80]}...")
                        # 保留 world_before_dict 供后续步骤使用
                        world_before_dict = prev_state
                except Exception as e:
                    # 压缩失败 → 回退原样直送（不崩溃、不丢信息）
                    print(f"  [世界状态] 压缩跳过: {e}")

                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": llm_tool_content})
                recent_tool_results.append(result_str)

            # ⚡ 所有 tool 结果消息追加完毕后，再注入修复指令（不打断 assistant→tool 协议序列）
            for fix_instr in pending_fix_instructions:
                messages.append({"role": "system", "content": fix_instr})

            # ============================================================
            # 阶段转换（观察证据驱动）
            # 从工具结果中提取文字证据，驱动 TaskPhaseMachine 状态推进
            # ============================================================
            observed_texts = []
            for r in recent_tool_results[-len(tool_names_in_step):]:
                observed_texts.extend(_extract_ocr_texts(r))
            if observed_texts:
                _drive_phase_transition(task_phase, working_memory, observed_texts)

            # ============================================================
            # 状态机记录本步
            # ============================================================
            state_machine.record_step(
                tool_names_in_step,
                has_progress=has_step_progress or not has_step_failure,
                is_failure=has_step_failure,
            )

            # ============================================================
            # 层级降级检测（连续失败）
            # ============================================================
            if tier_failures >= max_tier_attempts:
                if current_tier == 1:
                    current_tier = 2
                    tier_failures = 0
                    escalation_msg = (
                        f"多模态操作已连续失败 {max_tier_attempts} 次。\n"
                        f"请降级到 PowerShell 命令方式完成任务。\n"
                        f"使用 run_powershell 工具执行命令。"
                    )
                    print(f"\n  [降级] Tier 1 多模态操作失败 {max_tier_attempts} 次，降级到 Tier 2 PowerShell")
                    system_prompt = build_system_prompt(
                        current_tier, escalation_msg, state_machine,
                        working_memory=working_memory, task_phase=task_phase,
                    )
                    messages.insert(1, {"role": "system", "content": system_prompt})
                    state_machine.recover("降级到 Tier 2")
                elif current_tier == 2:
                    print(f"\n  [失败] Tier 2 PowerShell 也失败，任务无法完成")
                    state_machine.mark_failed(
                        "Tier 1 和 Tier 2 均无法完成",
                        condition=None,
                    )
                    success = False
                    final_result = "任务失败：Tier 1 和 Tier 2 均无法完成"
                    break

            # ============================================================
            # ⚡ 强制重新规划：过期启动器连续失败 → 调整计划（而非继续死磕）
            # ============================================================
            try:
                replan_reason = replan_mgr.should_replan()
                if replan_reason:
                    print(f"\n  [重规划] ⚡ 触发: {replan_reason}")
                    _failed = replan_mgr.failed_paths()
                    new_plan = plan_task(
                        client, tools_schema, user_query,
                        failed_paths=_failed,
                        replan_reason=replan_reason,
                        cost_feedback=budget.get_statistics(),
                    )
                    # 应用新阶段机
                    if new_plan.get("phases"):
                        task_phase = TaskPhaseMachine(
                            phases=new_plan["phases"],
                            start_phase=(new_plan.get("initial_state")
                                         or task_phase.current_phase),
                        )
                        print(f"  [重规划] 新计划阶段数: "
                              f"{len(new_plan['phases'])}")
                    # 应用新验收规格
                    if new_plan.get("goal_spec"):
                        verifier.set_spec_from_dict(new_plan["goal_spec"])
                    # 按新计划重设预算（含成本反馈约束）
                    budget.set_from_plan(
                        new_plan.get("est_total_cost", 0) or 0
                    )
                    budget.reset_hint()
                    replan_mgr.mark_replanned()
                    state_machine.recover("重新规划")
                    tier_failures = 0
                    messages.append({
                        "role": "system",
                        "content": (
                            "【重新规划】原计划已失效，"
                            f"原因: {replan_reason}\n"
                            "已切换为新计划执行。"
                            "失效的启动方式**禁止再用**，请严格按新阶段推进。"
                        ),
                    })
                    try:
                        exec_logger.log_info(
                            "replanned",
                            reason=replan_reason,
                            failed_paths=_failed,
                            stats=replan_mgr.get_statistics(),
                        )
                    except Exception:
                        pass
            except Exception as e:
                print(f"  [重规划] 跳过（不影响执行）: {e}")

        else:
            # ============================================================
            # 纯文本回复（AI 无工具调用）
            # ============================================================
            content = msg.content or ""
            print(f"[Step {step}] AI: {content[:300]}")

            # 状态机记录（无工具调用 = 无进展）
            state_machine.record_step([], has_progress=False, is_failure=False)
            recent_tool_results.append(content)

            completion_signals = ["任务完成", "已完成", "完成", "done", "finished", "已成功", "成功完成", "结束"]

            # ============================================================
            # 完成声明 → WorldState 驱动验收（主）+ GoalVerifier（fallback）
            # ============================================================
            if any(signal in content.lower() for signal in completion_signals):
                # 步骤A：WorldStateVerifier 基于客观证据验收
                ws_accept = False
                ws_conf = 0.0
                ws_missing = []
                ws_has_template = False   # WS 是否能匹配任务验收模板
                try:
                    if world_mgr.active and world_mgr.to_dict():
                        ws_result = ws_verifier.verify(user_query, world_mgr)
                        ws_accept = bool(ws_result.get("success"))
                        ws_conf = float(ws_result.get("confidence", 0.0))
                        # 有匹配模板才参与驳回（unknown → fallback GoalVerifier）
                        ws_has_template = ws_result.get("task_type") != "unknown"
                        ws_missing = [
                            m.get("description", m.get("condition", ""))
                            for m in ws_result.get("missing", [])
                        ]
                        if ws_accept:
                            print(f"  [世界状态验收] ✅ success "
                                  f"(confidence={ws_conf:.2f})")
                        elif ws_has_template:
                            print(f"  [世界状态验收] ❌ confidence={ws_conf:.2f} "
                                  f"missing={ws_missing}")
                except Exception as e:
                    print(f"  [世界状态验收] 异常: {e}")

                # 若 WorldState 客观验收通过 → 直接完成（不允许 LLM 否决）
                if ws_accept:
                    state_machine.mark_completed(content)
                    success = True
                    final_result = content
                    print(f"\n  [完成] 世界状态验收通过（confidence={ws_conf:.2f}）")
                    break

                # 若 WorldState 有缺失证据（且已匹配任务模板）→ 注入缺失项给 LLM 继续
                if ws_missing and not ws_accept and ws_has_template:
                    print(f"  [完成声明驳回] 世界状态缺失证据: {ws_missing}")
                    reject_msg = (
                        "【世界状态验收未通过】系统未找到以下客观证据：\n"
                        + "\n".join(f"  - {m}" for m in ws_missing)
                        + "\n请继续执行操作补全证据（如检查文件、启动进程、验证窗口），"
                          "不要直接宣告完成。"
                    )
                    messages.append({"role": "system", "content": reject_msg})
                    if state_machine.state in (AgentState.STUCK, AgentState.RECOVERING):
                        state_machine.recover("世界状态验收缺失证据")
                    continue

                # 步骤1：检查工具执行是否有失败残留（原始伪成功检测）
                validation = exception_handler.validate_completion(
                    content, recent_tool_results[-6:]
                )

                # 步骤2：GoalVerifier 独立验收（OCR 屏幕文字 + 最近工具结果文本）
                verification = None
                verification_failed = False
                if verifier.has_spec:
                    try:
                        # 证据源1：OCR 全屏文字
                        verify_texts = []
                        try:
                            from vision.ocr import ocr_screen
                            ocr_data = json.loads(ocr_screen())
                            verify_texts.extend(
                                t.get("text", "") for t in ocr_data.get("texts", [])
                            )
                        except Exception:
                            pass
                        # 证据源2：最近工具结果文本（PowerShell 输出等，
                        # 文件系统/命令类任务的证据在工具输出而非屏幕上）
                        for r in recent_tool_results[-6:]:
                            if isinstance(r, str) and r.strip():
                                verify_texts.append(r)

                        verification = verifier.verify(
                            ocr_texts=verify_texts if verify_texts else None
                        )
                        exec_logger.log_verification(verification)
                        if verification["verified"]:
                            print(f"  [独立验收] ✅ 通过: {verification['reasons']}")
                        else:
                            verification_failed = True
                            print(f"  [独立验收] ❌ 未通过: {verification['reasons']}")
                    except Exception as e:
                        print(f"  [独立验收] 验收异常跳过: {e}")
                        verification_failed = False

                # 判定：伪成功 或 独立验收未通过 → 驳回完成声明
                # ⚡ 驳回保护：连续验收失败 ≥3 次视为已尽力，强制接受完成声明，
                #    避免"文件整理已成功但 OCR 验证不到"导致的无限死循环
                reject_count = len(verifier.get_verify_history())
                if verification_failed and reject_count >= 3:
                    print(f"  [独立验收] ⚠️ 已连续驳回 {reject_count} 次，强制接受完成声明（防死循环）")
                    verification_failed = False

                is_rejected = validation["is_pseudo_success"] or verification_failed
                if is_rejected:
                    reasons = []
                    if validation["is_pseudo_success"]:
                        reasons.extend(validation["reasons"])
                    if verification_failed and verification:
                        reasons.extend(verification["reasons"])
                    if verification_failed and (not verification or not verification.get("reasons")):
                        reasons.append("独立验收未找到目标证据")

                    print(f"  [完成声明驳回] {reasons}")
                    reject_msg = "【完成声明审核】你的完成声明未通过："
                    if validation["is_pseudo_success"]:
                        reject_msg += f"执行过程存在异常（{'; '.join(validation['reasons'])}）；"
                    if verification_failed and verification:
                        reject_msg += f"独立验收未通过（{'; '.join(verification.get('reasons', []))}）。"
                        reject_msg += "请再执行一次验证步骤（如 visual_find_text / OCR），确认目标确实达成。"
                    else:
                        reject_msg += "请再执行一次验证步骤（如 visual_find_text / OCR / PowerShell 检查）。"
                    messages.append({
                        "role": "system",
                        "content": reject_msg,
                    })
                    # 状态机回到 RUNNING 继续
                    if state_machine.state in (AgentState.STUCK, AgentState.RECOVERING):
                        state_machine.recover("完成声明被驳回")
                    continue  # 不结束循环，要求继续验证

                # 两种验证都通过 → 完成
                state_machine.mark_completed(content)
                success = True
                final_result = content
                conf_text = (
                    f"，独立验收置信度 {verification['confidence']:.2f}"
                    if verification else ""
                )
                print(f"\n  [完成] AI 认为任务已完成（伪成功置信度 {validation['confidence']:.2f}{conf_text}）")
                break

            if step > 3:
                messages.append({
                    "role": "system",
                    "content": "你还没有调用任何工具，或只做了视觉扫描。"
                               "请先确认屏幕上的**真实文字/元素**，再决定动作："
                               "优先用 visual_find_text(\"真实文字\") 取得坐标后再 click_at。"
                               "**没有定位来源时禁止点击**"
                               "（系统会拦截无来源坐标；此时应先观察或改用 run_powershell）。"
                })

    # ============================================================
    # 任务结束：经验沉淀（Phase 3/4：学习成功工作流 + 失败案例）
    # ============================================================
    if execution_trace:
        try:
            memory_manager.learn_from_task(
                task=user_query,
                success=success,
                trace=execution_trace,
                error_history=exception_handler.error_history,
            )
            print(f"  [记忆] 已沉淀任务经验")
        except Exception as e:
            print(f"  [记忆] 经验沉淀跳过: {e}")

    # ============================================================
    # 任务结束：抽象经验沉淀（行为模式，不含坐标）
    # ============================================================
    try:
        if success and application:
            # 从追踪提取行为模式（复用 extractor，只存抽象描述）
            from agent.memory.extractor import ExperienceExtractor
            steps = ExperienceExtractor.extract_steps(execution_trace)
            if steps:
                exp_store.add(
                    application=application,
                    action_pattern="; ".join(steps[:5]),
                    detection_method=[
                        "OCR寻找提示文本",
                        "视觉扫描定位元素",
                    ],
                    state_transitions=task_phase.all_phase_names(),
                    failure_cases=[c.get("problem", "") for c in
                                   ExperienceExtractor.extract_failure_cases(
                                       exception_handler.error_history)],
                    confidence=0.5 if success else 0.3,
                )
                print(f"  [记忆] 已沉淀抽象经验（{application}）")
    except Exception as e:
        print(f"  [记忆] 抽象经验沉淀跳过: {e}")

    # ============================================================
    # 任务结束：结构化日志结束 + 工作记忆清理
    # ============================================================
    try:
        exec_logger.end_task(
            success=success,
            result=final_result,
            summary={
                "steps": steps_taken,
                "final_tier": current_tier,
                "final_phase": task_phase.current_phase,
                "final_state": str(state_machine.state),
                "tools": sorted(set(all_tools_used)),
                "prediction_count": len(predictor._active) if hasattr(predictor, '_active') else 0,
                "tracked_objects": len(tracker._objects) if hasattr(tracker, '_objects') else 0,
                "cost": budget.get_statistics(),
                "replan": replan_mgr.get_statistics(),
            },
        )
        print(f"  [日志] 执行日志已保存")
    except Exception as e:
        print(f"  [日志] 日志保存跳过: {e}")

    # 清理（任务结束）
    try:
        working_memory.clear()
        tracker.clear()
        verifier.clear()
    except Exception:
        pass

    # ⚡ WorldState: 任务结束 → 经验沉淀 + 世界状态销毁
    try:
        if world_mgr.active:
            world_mgr.destroy_with_lesson(succeeded=success)
            print("  [世界状态] 已销毁任务世界模型")
    except Exception as e:
        print(f"  [世界状态] 销毁跳过: {e}")

    # ============================================================
    # 任务结束汇总
    # ============================================================
    if not success and state_machine.state not in (AgentState.COMPLETED, AgentState.FAILED):
        state_machine.mark_failed("达到最大步数未完成")

    print(f"\n{'=' * 60}")
    print(f"任务: {user_query}")
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"步数: {steps_taken}/{MAX_STEPS}")
    print(f"最终层级: Tier {current_tier}")
    print(f"最终状态: {state_machine.state}")
    print(f"最终任务阶段: {task_phase.current_phase}")
    print(f"工具使用: {', '.join(sorted(set(all_tools_used)))}")
    exc_stats = exception_handler.get_statistics()
    print(f"异常处理: 共 {exc_stats['total_errors']} 次错误, "
          f"类型={exc_stats['error_types']}")
    # 成本 + 重规划统计（成本感知规划 / 过期启动器）
    _cost = budget.get_statistics()
    print(f"成本: 已用 {_cost['spent']} / 预算 {_cost['budget']} 成本点 "
          f"（计划 {_cost['planned_cost']}，超预算={_cost['over_budget']}）")
    _rp = replan_mgr.get_statistics()
    if _rp["replan_count"] or _rp["failed_paths"]:
        print(f"重规划: {_rp['replan_count']} 次, "
              f"失效路径 {len(_rp['failed_paths'])} 条")
    if verifier.get_verify_history():
        print(f"独立验收: 共 {len(verifier.get_verify_history())} 次验证")
    print(f"{'=' * 60}\n")

    return success, final_result


def run_agent(user_query: str = None):
    """
    运行Agent（持续模式）
    """
    print(f"\n{'=' * 70}")
    print(f"Computer Use Agent - 桌面自动化（持续运行）")
    print(f"主脑: DeepSeek（云端API） | 视觉: 本地模型 + OCR")
    print(f"{'=' * 70}\n")

    # 预加载视觉模型 + OCR
    print("正在初始化模型...")
    preload_models()
    print("视觉模型就绪！\n")

    try:
        from vision.ocr import ocr_screen
        import json; json.loads(ocr_screen())
        safe_print("  [OCR] [OK] 预热完成")
    except Exception as e:
        safe_print(f"  [OCR] 预热跳过: {e}")

    # ============================================================
    # 输入环境自检（键鼠模拟可用性）——避免"输入无效却无提示"
    # ============================================================
    try:
        from system.input_controller import describe_environment
        env = describe_environment()
        vs = env.get("virtual_screen", {})
        print(f"  [输入] DPI感知={env.get('dpi_aware')} "
              f"屏幕={vs.get('width')}x{vs.get('height')} "
              f"光标={env.get('cursor')}")
        if not env.get("self_elevated"):
            safe_print("  [输入] [注意] 当前以普通权限运行：")
            safe_print("         - 若目标窗口是管理员权限（安装程序/UAC 窗口），")
            safe_print("           Windows(UIPI) 会丢弃模拟键鼠输入（表现为'点了没反应'）")
            safe_print("         - 需要操作此类窗口时，请以【管理员身份】运行本程序")
        else:
            safe_print("  [输入] [OK] 已提权，可操作管理员窗口")
    except Exception as e:
        print(f"  [输入] 自检跳过: {e}")

    # 初始化 DeepSeek 客户端
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    tools_schema = build_tools_schema()

    if user_query:
        try:
            return run_task(client, tools_schema, user_query)
        except Exception as e:
            # 兜底：任何未捕获异常都不能让进程崩溃
            print(f"\n[致命错误] 任务执行异常: {e}")
            return False, f"任务失败：执行异常 ({type(e).__name__}: {e})"

    # 交互模式
    print("进入交互模式，输入任务开始使用，输入 'exit' 退出\n")
    while True:
        try:
            query = input(">>> 请输入任务: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("再见！")
                break

            run_task(client, tools_schema, query)

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] {e}\n")

    return True, "正常退出"