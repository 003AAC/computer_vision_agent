"""
SelfHealer - 自愈闭环系统
==========================
错误检测 → 特征提取 → 规则匹配/知识库查询 → 修复执行 → 效果验证 → 经验沉淀
"""
import os
import re
import json
from datetime import datetime


class SelfHealer:
    """自愈闭环引擎"""

    FIX_RULES = [
        {
            "name": "program_not_found",
            "patterns": ["不是内部或外部命令", "not recognized as",
                         "无法将 items", "不是可识别的",
                         "无法找到文件", "模式无法找到",
                         "不是内部", "where:"],
            "fix_type": "locate_program",
            "description": "程序未找到 → 搜索安装路径或提示替代程序"
        },
        {
            "name": "path_not_found",
            "patterns": ["找不到路径", "cannot find the path",
                         "the system cannot find the path",
                         "could not find path"],
            "fix_type": "create_parent_dir",
            "description": "路径不存在 → 自动创建父目录"
        },
        {
            "name": "encoding_error",
            "patterns": ["unicodedecoderror", "codec can't decode",
                         "gbk", "utf-8", "编码", "surrogate"],
            "fix_type": "set_utf8_env",
            "description": "编码错误 → 设置 UTF-8 环境变量"
        },
        {
            "name": "permission_denied",
            "patterns": ["拒绝访问", "permission denied", "access is denied",
                         "需要管理员权限", "unauthorized"],
            "fix_type": "check_permission",
            "description": "权限不足 → 检查权限并尝试替代路径"
        },
        {
            "name": "file_in_use",
            "patterns": ["正在使用", "being used by", "the process cannot access",
                         "占用", "locked", "另一个程序"],
            "fix_type": "wait_and_retry",
            "description": "文件被占用 → 等待释放后重试"
        },
        {
            "name": "disk_full",
            "patterns": ["空间不足", "not enough disk", "disk full", "没有可用空间"],
            "fix_type": "check_disk_space",
            "description": "磁盘空间不足 → 检查并清理"
        },
        {
            "name": "command_timeout",
            "patterns": ["命令执行超时", "command timeout", "timed out", "执行超时"],
            "fix_type": "simplify_command",
            "description": "命令超时 → 拆分为更简单的命令"
        },
        {
            "name": "network_error",
            "patterns": ["网络", "network", "connection refused",
                         "连接失败", "无法访问服务器", "winerror", "dns"],
            "fix_type": "retry_with_delay",
            "description": "网络错误 → 增加等待后重试"
        },
        {
            "name": "focus_lost",
            "patterns": ["焦点", "foreground", "活动窗口", "找不到窗口", "no active window"],
            "fix_type": "refocus_window",
            "description": "窗口焦点丢失 → 重新激活目标窗口"
        },
        {
            "name": "element_not_found",
            "patterns": ["未找到", "未识别", "找不到元素", "no element", "not visible", "不可见"],
            "fix_type": "retry_with_alt_description",
            "description": "元素未找到 → 换描述重试"
        },
        {
            "name": "command_syntax_error",
            "patterns": ["语法错误", "syntax error", "invalid syntax",
                         "unexpected token", "表达式不正确"],
            "fix_type": "simplify_command",
            "description": "命令语法错误 → 简化命令重试"
        },
    ]

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.heal_history = []
        self.heal_count = 0
        self.max_heal_per_task = 3

    def extract_error_features(self, error_result: str, tool_name: str) -> dict:
        features = {
            "raw_error": error_result,
            "error_lower": error_result.lower(),
            "tool_name": tool_name,
            "matched_rule": None,
            "error_type": "unknown",
            "extracted_paths": [],
            "extracted_commands": [],
        }
        for rule in self.FIX_RULES:
            if any(p in features["error_lower"] for p in rule["patterns"]):
                features["matched_rule"] = rule
                features["error_type"] = rule["name"]
                break
        path_patterns = [
            r'"([^"]+)"', r"'([^']+)'", r'([A-Za-z]:\\[^\s,;]+)',
        ]
        for pat in path_patterns:
            for m in re.finditer(pat, error_result):
                path = m.group(1)
                if '\\' in path or '/' in path:
                    features["extracted_paths"].append(path)
        cmd_patterns = [
            r'命令[：:]\s*(.+?)(?:\n|$)',
            r'command[：:]\s*(.+?)(?:\n|$)',
        ]
        for pat in cmd_patterns:
            m = re.search(pat, error_result, re.IGNORECASE)
            if m:
                features["extracted_commands"].append(m.group(1).strip())
        return features

    def diagnose(self, error_result: str, tool_name: str, task_context: dict = None) -> dict:
        if task_context is None:
            task_context = {}
        features = self.extract_error_features(error_result, tool_name)
        diagnosis = {
            "error_type": features["error_type"],
            "raw_error": error_result[:300],
            "tool_name": tool_name,
            "task": task_context.get("task", ""),
            "fix_actions": [],
            "fix_description": "",
            "confidence": 0.0,
            "source": "none",
            "features": features,
        }
        # 优先级 1：硬编码规则
        if features["matched_rule"]:
            rule = features["matched_rule"]
            fix_actions = self._generate_fix_from_rule(rule["fix_type"], features, task_context)
            if fix_actions:
                diagnosis["fix_actions"] = fix_actions
                diagnosis["fix_description"] = rule["description"]
                diagnosis["confidence"] = 0.95
                diagnosis["source"] = f"硬编码规则:{rule['name']}"
        # 优先级 2：知识库错误模式
        if not diagnosis["fix_actions"]:
            kb_solution = self.kb.get_error_solution(error_result)
            if kb_solution and kb_solution.get("actions"):
                diagnosis["fix_actions"] = [
                    {"tool": "info", "args": {},
                     "description": f"[知识库] {kb_solution.get('category', '')}: {kb_solution['actions']}"}
                ]
                diagnosis["fix_description"] = f"知识库方案: {kb_solution.get('category', '')}"
                diagnosis["confidence"] = 0.8
                diagnosis["source"] = f"知识库:{kb_solution.get('category', '未知')}"
        # 优先级 3：历史修复经验
        if not diagnosis["fix_actions"]:
            heal_exp = self._search_heal_experience(features)
            if heal_exp:
                diagnosis["fix_actions"] = heal_exp
                diagnosis["fix_description"] = "历史修复经验"
                diagnosis["confidence"] = 0.7
                diagnosis["source"] = "历史经验"
        return diagnosis

    def _generate_fix_from_rule(self, fix_type: str, features: dict, context: dict) -> list:
        actions = []
        raw = features["raw_error"]
        paths = features["extracted_paths"]

        if fix_type == "create_parent_dir":
            if paths:
                for p in paths[:2]:
                    parent = os.path.dirname(p)
                    if parent and not os.path.exists(parent):
                        actions.append({
                            "tool": "run_command",
                            "args": {"command": f'New-Item -Path "{parent}" -ItemType Directory -Force'},
                            "description": f"创建缺失目录: {parent}"
                        })
            if not actions and context.get("task"):
                desktop = self.kb.get_environment_info().get("desktop_path", r"C:\Users\华硕\Desktop")
                actions.append({
                    "tool": "run_command",
                    "args": {"command": f'dir "{desktop}"'},
                    "description": "验证桌面目录可访问性"
                })
        elif fix_type == "set_utf8_env":
            actions.append({
                "tool": "run_command",
                "args": {"command": "$env:PYTHONUTF8='1'; [Console]::OutputEncoding = [System.Text.Encoding]::UTF8"},
                "description": "设置 UTF-8 编码环境"
            })
        elif fix_type == "check_permission":
            actions.append({
                "tool": "run_command",
                "args": {"command": "whoami /priv | Select-String 'SeDebugPrivilege|SeTakeOwnershipPrivilege'"},
                "description": "检查当前权限级别"
            })
            if paths:
                for p in paths:
                    if "System32" in p or "Program Files" in p:
                        user_alt = p.replace(r"C:\Windows\System32", os.path.expanduser("~\\Documents"))
                        actions.append({
                            "tool": "info", "args": {},
                            "description": f"建议改用用户目录: {user_alt}"
                        })
        elif fix_type == "locate_program":
            program_name = context.get("program", "")
            if not program_name:
                task = context.get("task", "")
                for name in ["notepad", "chrome", "edge", "calc", "mspaint",
                             "explorer", "code", "python"]:
                    if name in task.lower():
                        program_name = name
                        break
            if program_name:
                actions.append({
                    "tool": "run_command",
                    "args": {"command": f"where {program_name} 2>$null; if (-not $?) {{ Get-ChildItem -Path 'C:\\Program Files','C:\\Program Files (x86)' -Recurse -Filter '{program_name}*' -ErrorAction SilentlyContinue | Select-Object -First 3 -ExpandProperty FullName }}"},
                    "description": f"搜索 {program_name} 安装路径"
                })
            else:
                actions.append({
                    "tool": "run_command",
                    "args": {"command": "Get-Command * | Where-Object {$_.Source} | Select-Object -First 5 Name,Source"},
                    "description": "列出可用命令"
                })
        elif fix_type == "wait_and_retry":
            actions.append({"tool": "wait", "args": {"seconds": 3.0}, "description": "等待资源释放 3 秒"})
        elif fix_type == "check_disk_space":
            actions.append({
                "tool": "run_command",
                "args": {"command": "Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}},@{N='UsedGB';E={[math]::Round($_.Used/1GB,2)}}"},
                "description": "检查磁盘空间"
            })
        elif fix_type == "retry_with_delay":
            actions.append({"tool": "wait", "args": {"seconds": 5.0}, "description": "等待 5 秒后重试"})
        elif fix_type == "refocus_window":
            actions.append({"tool": "hotkey", "args": {"keys": ["alt", "tab"]}, "description": "Alt+Tab 切换窗口焦点"})
            actions.append({"tool": "wait", "args": {"seconds": 1.0}, "description": "等待窗口切换"})
        elif fix_type == "retry_with_alt_description":
            actions.append({"tool": "info", "args": {}, "description": "[自愈] 请用不同的描述词重新查找元素，或先截图分析当前界面"})
        elif fix_type == "simplify_command":
            actions.append({"tool": "info", "args": {}, "description": "[自愈] 命令语法有误，请拆分为更简单的命令分步执行"})
        return actions

    def _search_heal_experience(self, features: dict) -> list:
        try:
            query = f"修复 {features['error_type']} {features['tool_name']}"
            results = self.kb.search_experience(query, n_results=1)
            if results:
                exp = results[0]
                meta = exp.get("metadata", {})
                if meta.get("result") == "成功":
                    fix_actions_raw = meta.get("fix_actions", "[]")
                    if isinstance(fix_actions_raw, str):
                        fix_actions = json.loads(fix_actions_raw)
                    else:
                        fix_actions = fix_actions_raw
                    if fix_actions:
                        return fix_actions
        except Exception:
            pass
        return []

    def apply_fix(self, diagnosis: dict, tools_registry: dict) -> dict:
        fix_results = []
        all_success = True
        for action in diagnosis["fix_actions"]:
            tool_name = action["tool"]
            args = action["args"]
            desc = action.get("description", "")
            print(f"    [自愈] 执行: {desc}")
            if tool_name == "info":
                fix_results.append({"tool": "info", "description": desc, "result": desc, "success": True})
                continue
            tool_fn = tools_registry.get(tool_name)
            if not tool_fn:
                fix_results.append({"tool": tool_name, "description": desc, "result": f"工具 {tool_name} 不存在", "success": False})
                all_success = False
                continue
            try:
                result = tool_fn.invoke(args) if hasattr(tool_fn, 'invoke') else tool_fn(**args)
                result_str = str(result)
                is_success = not self._is_failure(result_str)
                fix_results.append({"tool": tool_name, "description": desc, "result": result_str[:200], "success": is_success})
                if not is_success:
                    all_success = False
                    print(f"    [自愈] ✗ 修复失败: {result_str[:100]}")
                else:
                    print(f"    [自愈] ✓ 修复成功")
            except Exception as e:
                fix_results.append({"tool": tool_name, "description": desc, "result": f"异常: {str(e)}", "success": False})
                all_success = False
                print(f"    [自愈] ✗ 修复异常: {e}")
        return {"fix_results": fix_results, "all_success": all_success, "summary": self._summarize_fix(fix_results)}

    def record_result(self, diagnosis: dict, fix_result: dict, retry_success: bool):
        self.heal_count += 1
        self.heal_history.append({
            "time": datetime.now().isoformat(),
            "error_type": diagnosis["error_type"],
            "tool_name": diagnosis["tool_name"],
            "source": diagnosis["source"],
            "fix_description": diagnosis["fix_description"],
            "fix_actions": diagnosis["fix_actions"],
            "fix_result": fix_result,
            "retry_success": retry_success,
        })
        if retry_success and diagnosis["fix_actions"]:
            task = diagnosis.get("task", "")
            experience = {
                "task": f"自愈: {diagnosis['error_type']}({diagnosis['tool_name']})",
                "result": "成功",
                "lesson": (f"[{diagnosis['source']}] {diagnosis['fix_description']} → "
                           f"修复成功，错误类型: {diagnosis['error_type']}"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "steps": self.heal_count,
                "tools_used": json.dumps([diagnosis["tool_name"]], ensure_ascii=False),
                "fix_actions": json.dumps(diagnosis["fix_actions"], ensure_ascii=False),
                "tags": json.dumps(["自愈", "修复成功", diagnosis["error_type"], diagnosis["tool_name"]], ensure_ascii=False),
                "_type": "heal"
            }
            try:
                self.kb.add_experience(experience)
                print(f"    [知识库] 自愈经验已沉淀")
            except Exception as e:
                print(f"    [知识库] 沉淀失败: {e}")

    def can_heal(self) -> bool:
        return self.heal_count < self.max_heal_per_task

    def reset(self):
        self.heal_count = 0
        self.heal_history = []

    def get_stats(self) -> dict:
        total = len(self.heal_history)
        success = sum(1 for h in self.heal_history if h["retry_success"])
        return {
            "total_heals": total, "success_count": success,
            "success_rate": f"{success/total*100:.0f}%" if total > 0 else "N/A",
            "error_types": list(set(h["error_type"] for h in self.heal_history))
        }

    @staticmethod
    def _is_failure(result_str: str) -> bool:
        signals = ["失败", "错误", "异常", "超时", "未找到",
                   "not found", "error", "failed", "timeout",
                   "拒绝访问", "权限", "不存在"]
        return any(s in result_str.lower() for s in signals)

    @staticmethod
    def _summarize_fix(fix_results: list) -> str:
        total = len(fix_results)
        success = sum(1 for r in fix_results if r["success"])
        if success == total:
            return f"全部 {total} 项修复成功"
        elif success > 0:
            return f"{success}/{total} 项修复成功"
        else:
            return f"全部 {total} 项修复失败"
