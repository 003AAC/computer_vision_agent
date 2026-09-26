"""Execution policies for tool-call batches."""
from typing import Iterable, Set


def requires_sequential_execution(tool_names: Iterable[str],
                                  state_changing_tools: Set[str]) -> bool:
    """Do not run precomputed follow-up calls in a state-changing batch."""
    names = list(tool_names)
    return len(names) > 1 and any(name in state_changing_tools for name in names)
