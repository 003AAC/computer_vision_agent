"""
Execution Logger - 结构化执行日志
=================================
增强日志系统，不只记录 "Step 1 / Step 2"。

记录每步完整信息：
{
  timestamp,
  task_state,
  memory_before,
  action,
  observation,
  prediction,
  memory_after,
  verification
}

持久化到 logs/execution_YYYYMMDD_HHMMSS.jsonl（JSON Lines 格式）
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExecutionLogger:
    """结构化执行日志记录器"""

    def __init__(self, logs_dir: str = None):
        """
        Args:
            logs_dir: 日志目录（默认项目根/logs）
        """
        if logs_dir is None:
            logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "logs"
            )
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        # 每个任务一个文件
        self._current_file: Optional[str] = None
        self._session_id: str = ""
        self._step_records: list = []

    def start_task(self, task: str, goal_spec: Dict[str, Any] = None):
        """任务开始，创建会话"""
        timestamp = datetime.now()
        self._session_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        self._current_file = os.path.join(
            self.logs_dir, f"execution_{self._session_id}.jsonl"
        )
        self._step_records = []

        header = {
            "event": "task_start",
            "timestamp": timestamp.isoformat(),
            "task": task,
            "goal_spec": goal_spec,
        }
        self._write_record(header)

    def end_task(self, success: bool, result: str,
                 summary: Dict[str, Any] = None):
        """任务结束，写入总结"""
        record = {
            "event": "task_end",
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "result": result[:500],
            "summary": summary or {},
            "total_steps": len(self._step_records),
        }
        self._write_record(record)
        self._current_file = None

    def record_step(self, step_data: Dict[str, Any]):
        """记录一步执行

        Args:
            step_data: {
              "step": int,
              "task_state": {...},
              "memory_before": {...},
              "action": {...},
              "observation": {...},
              "prediction": {...},
              "memory_after": {...},
              "verification": {...},
            }
        """
        record = {
            "event": "step",
            "timestamp": datetime.now().isoformat(),
        }
        record.update(step_data)
        self._step_records.append(record)
        self._write_record(record)

    def log_action(self, tool_name: str, args: Dict[str, Any],
                   result_str: str, success: bool,
                   verification_status: str = "unknown"):
        """记录单个工具调用（作为 step 的子事件）"""
        record = {
            "event": "tool_call",
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "args": args,
            "result": result_str[:500],
            "success": success,
            "verification_status": verification_status,
        }
        self._write_record(record)

    def log_info(self, message: str, **kwargs):
        """记录任意信息"""
        record = {
            "event": "info",
            "timestamp": datetime.now().isoformat(),
            "message": message,
        }
        record.update(kwargs)
        self._write_record(record)

    def log_verification(self, verification: Dict[str, Any]):
        """记录独立验收结果"""
        record = {
            "event": "verification",
            "timestamp": datetime.now().isoformat(),
            "verification": verification,
        }
        self._write_record(record)

    def log_prediction(self, prediction: Dict[str, Any]):
        """记录动作预测"""
        record = {
            "event": "prediction",
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
        }
        self._write_record(record)

    # ============================================================
    # 内部
    # ============================================================

    def _write_record(self, record: Dict[str, Any]):
        """写入一条 JSON 记录"""
        if self._current_file is None:
            return
        try:
            with open(self._current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"日志写入失败: {e}")

    @property
    def current_file(self) -> Optional[str]:
        return self._current_file

    def get_step_count(self) -> int:
        return len(self._step_records)

    def get_step_records(self) -> list:
        return self._step_records