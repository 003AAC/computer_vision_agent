"""
Semantic Memory - 语义记忆（长期稳定知识）
保存"如何找到目标"、"如何验证目标"的稳定经验。
禁止保存：绝对坐标、固定窗口位置、单次 UI 布局。
"""
from typing import Dict, Any, List

from agent.memory.experience_store import ExperienceStore


class SemanticMemory:
    """语义记忆：生命周期长，任务间共享"""

    def __init__(self, path: str):
        self.store = ExperienceStore(path)

    def get_launch_strategy(self) -> List[str]:
        """获取应用启动经验"""
        item = self.store.get("app_launch")
        return item.get("knowledge", []) if item else []

    def get_general_strategy(self) -> List[str]:
        """获取通用操作策略"""
        item = self.store.get("general")
        return item.get("strategy", []) if item else []

    def search(self, keywords: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """按关键词检索语义记忆"""
        results = []
        candidates = []
        if self.store.get("app_launch"):
            candidates.append(("app_launch", self.store.get("app_launch")))
        if self.store.get("general"):
            candidates.append(("general", self.store.get("general")))
        for item in self.store.get("ui_elements", []):
            candidates.append((f"ui:{item.get('element','')}", item))

        for key, item in candidates:
            text = str(item).lower()
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > 0:
                results.append({
                    "key": key,
                    "score": score / max(len(keywords), 1),
                    "data": item,
                    "type": "semantic",
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]