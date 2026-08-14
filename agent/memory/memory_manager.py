"""
Memory Manager - 经验记忆统一管理器
====================================
整合三层记忆：
- Semantic Memory（语义层）：长期稳定知识
- Episodic Memory（任务层）：任务流程经验
- Runtime Memory（运行时）：当前任务状态
"""
import os
from typing import Dict, Any, List, Optional

from agent.memory.experience_store import ExperienceStore
from agent.memory.semantic_memory import SemanticMemory
from agent.memory.episodic_memory import EpisodicMemory
from agent.memory.runtime_memory import RuntimeMemory
from agent.memory.retriever import MemoryRetriever
from agent.memory.extractor import ExperienceExtractor


class MemoryManager:
    """经验记忆管理器（统一入口）"""

    def __init__(self, memories_dir: str = None):
        """初始化记忆管理器

        Args:
            memories_dir: 记忆存储目录（默认 agent/memories）
        """
        if memories_dir is None:
            memories_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memories"
            )
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)

        # 三层记忆
        self.semantic = SemanticMemory(
            os.path.join(self.memories_dir, "semantic.json")
        )
        self.episodic = EpisodicMemory(
            os.path.join(self.memories_dir, "episodic.json")
        )
        self.runtime = RuntimeMemory()

        # 检索器
        self.retriever = MemoryRetriever(self.semantic, self.episodic)

        # 提取器
        self.extractor = ExperienceExtractor()

    # ============================================================
    # 检索
    # ============================================================

    def retrieve_context(self, task: str) -> str:
        """检索并构建经验上下文（注入提示词）"""
        return self.retriever.build_experience_context(task)

    def build_prompt_section(self, task: str) -> str:
        """构建完整提示词段（检索经验 + 当前状态）"""
        parts = []
        exp_ctx = self.retriever.build_experience_context(task)
        if exp_ctx:
            parts.append(exp_ctx)
        rt = self.runtime.build_instruction()
        if rt:
            parts.append(rt)
        return "\n\n".join(parts)

    # ============================================================
    # 运行时状态
    # ============================================================

    def update_runtime(self, **kwargs):
        """更新运行时状态"""
        self.runtime.update(**kwargs)

    def get_runtime_instruction(self) -> str:
        """获取运行时状态指令"""
        return self.runtime.build_instruction()

    # ============================================================
    # 学习（任务结束后调用）
    # ============================================================

    def learn_from_task(self, task: str, success: bool,
                        trace: List[Dict[str, Any]],
                        error_history: List[Dict[str, Any]]):
        """从任务执行记录学习经验

        Args:
            task: 任务描述
            success: 是否成功
            trace: 执行轨迹（tool, args, result, is_failure）
            error_history: 异常历史
        """
        if not trace:
            return

        task_type = self.retriever.classify_task(task)
        card = self.extractor.build_experience_card(trace, error_history, task)

        if success:
            self.episodic.add_success(
                task_type=task_type,
                task=task,
                steps=card["strategy"],
                verification=card["verification"],
                failure_recoveries=card["failure_cases"],
            )
        else:
            # 失败也记录失败恢复经验（降低置信度）
            self.episodic.add_failure_recovery(
                task_type=task_type,
                step=task,
                failure=f"任务失败",
                recovery="参见失败案例",
            )

        # 清理运行时
        self.runtime.clear()

    # ============================================================
    # 统计
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        epi = self.episodic.store.all()
        return {
            "semantic_entries": len(self.semantic.store.all()),
            "episodic_patterns": len(epi),
            "episodic_records": sum(len(v) for v in epi.values()),
            "memories_dir": self.memories_dir,
        }