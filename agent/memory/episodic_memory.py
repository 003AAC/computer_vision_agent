"""
Episodic Memory - 任务经验（Episodic）
======================================
记录一次任务的完整流程：成功工作流、失败恢复、验证方法。
用于未来类似任务参考。不保存绝对坐标。
"""
from datetime import datetime
from typing import Dict, Any, List

from agent.memory.experience_store import ExperienceStore


class EpisodicMemory:
    """任务经验：按 task_pattern 分组存储"""

    def __init__(self, path: str):
        self.store = ExperienceStore(path)

    def add_success(self, task_type: str, task: str,
                    steps: List[str], verification: List[str],
                    failure_recoveries: List[Dict[str, str]]):
        """新增成功任务经验"""
        records = self.store.get(task_type, [])
        for rec in records:
            if rec.get("task") == task:
                rec["success"] = True
                rec["steps"] = steps
                rec["verification"] = verification
                if failure_recoveries:
                    rec["failure_recoveries"] = failure_recoveries
                rec["confidence"] = min(1.0, rec.get("confidence", 0.5) + 0.15)
                rec["timestamp"] = datetime.now().isoformat()
                self.store.set(task_type, records)
                return
        records.append({
            "task": task,
            "success": True,
            "steps": steps,
            "verification": verification,
            "failure_recoveries": failure_recoveries,
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat(),
        })
        self.store.set(task_type, records)

    def add_failure_recovery(self, task_type: str, step: str,
                             failure: str, recovery: str):
        """记录失败恢复经验"""
        records = self.store.get(task_type, [])
        if records:
            rec = records[-1]
        else:
            rec = {"task": "未完成任务", "success": False,
                   "steps": [], "verification": [], "failure_recoveries": [],
                   "confidence": 0.3, "timestamp": datetime.now().isoformat()}
            records.append(rec)
        if "failure_recoveries" not in rec:
            rec["failure_recoveries"] = []
        rec["failure_recoveries"].append({
            "step": step, "failure": failure, "recovery": recovery,
        })
        self.store.set(task_type, records)

    def search(self, keywords: List[str], top_n: int = 2) -> List[Dict[str, Any]]:
        """按关键词检索任务经验"""
        results = []
        for pattern, records in self.store.all().items():
            for rec in records:
                text = str(rec).lower()
                score = sum(1 for kw in keywords if kw.lower() in text)
                if score > 0:
                    results.append({
                        "key": pattern,
                        "score": score / max(len(keywords), 1),
                        "data": rec,
                        "type": "episodic",
                    })
        results.sort(key=lambda r: r["score"] * r["data"].get("confidence", 0.5),
                     reverse=True)
        return results[:top_n]