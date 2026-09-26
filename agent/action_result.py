"""Structured outcomes for desktop actions."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ActionResult:
    tool: str
    execution_status: str
    detail: str
    verification_status: str = "unverified"
    observed_texts: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def action_result(tool: str, execution_status: str, detail: str) -> str:
    return ActionResult(tool, execution_status, detail).to_json()


def parse_action_result(value: str) -> Dict[str, Any]:
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(result, dict) or "execution_status" not in result:
        return {}
    return result


def annotate_action_result(value: str, verification_status: str,
                           observed_texts: List[str]) -> str:
    result = parse_action_result(value)
    if not result:
        return value
    result["verification_status"] = verification_status
    result["observed_texts"] = observed_texts
    return json.dumps(result, ensure_ascii=False)
