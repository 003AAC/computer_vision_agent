"""
State Diff - 状态差异 + 语义摘要
================================
比较 state 快照的前后差异：
  - changes: 机器可读 diff
  - semantic_summary: 给 LLM 的状态变化描述（替代原始 tool result 注入）
"""
from datetime import datetime
from typing import Any, Dict


def _flatten(obj: Dict[str, Any], prefix: str = "",
             result: Dict[str, Any] = None) -> Dict[str, Any]:
    """将嵌套 dict 拍平成 "a.b.c" → 简化值

    仅提取 belief 的最终值 + 置信度，丢弃无用的中间结构。
    """
    if result is None:
        result = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            # belief 结构（含 value/confidence/status/supporting）
            if "value" in v and "confidence" in v:
                result[key] = {
                    "value": v["value"],
                    "confidence": v.get("confidence", 0.0),
                    "status": v.get("status", "unknown"),
                }
            else:
                _flatten(v, key, result)
        elif not isinstance(v, (dict, list)):
            result[key] = v
    return result


class StateDiff:
    """状态差异计算器"""

    @staticmethod
    def diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """比较前后状态，返回机器 diff + semantic_summary

        Args:
            before: 之前的状态 dict（如 manager.to_dict()）
            after: 之后的状态 dict

        Returns:
            {
              "changes": {路径: {"old":..., "new":...}},
              "semantic_summary": str,
            }
        """
        flat_before = _flatten(before)
        flat_after = _flatten(after)

        changes: Dict[str, Any] = {}
        all_keys = set(flat_before.keys()) | set(flat_after.keys())
        for key in sorted(all_keys):
            old = flat_before.get(key)
            new = flat_after.get(key)
            if old != new:
                # 提取实际 value 对比
                old_v = old["value"] if isinstance(old, dict) else old
                new_v = new["value"] if isinstance(new, dict) else new
                if old_v != new_v:
                    changes[key] = {
                        "old": old_v,
                        "new": new_v,
                        "old_confidence": old.get("confidence") if isinstance(old, dict) else None,
                        "new_confidence": new.get("confidence") if isinstance(new, dict) else None,
                    }

        semantic_summary = StateDiff._build_semantic_summary(changes)
        return {
            "changes": changes,
            "semantic_summary": semantic_summary,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _build_semantic_summary(changes: Dict[str, Any]) -> str:
        """将机器 diff 转成语义描述（给 LLM）"""
        if not changes:
            return "世界状态无变化"

        lines = []
        for path, change in changes.items():
            # 路径语义化
            parts = path.split(".")
            field = parts[-1] if parts else path
            old, new = change["old"], change["new"]
            new_conf = change.get("new_confidence")

            if old is None and new is not None:
                desc = f"新增：{field}={new}"
            elif new is None:
                desc = f"消失：{field}（原={old}）"
            else:
                desc = f"变化：{field} {old}→{new}"
            if new_conf is not None:
                desc += f" (置信{new_conf:.2f})"
            lines.append(desc)

        return "；".join(lines)