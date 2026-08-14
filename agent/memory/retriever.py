"""
Memory Retriever - 经验检索器
==============================
输入：用户任务描述
输出：相似历史经验（语义 + 任务层）
"""
import re
from typing import Dict, Any, List, Tuple

from agent.memory.semantic_memory import SemanticMemory
from agent.memory.episodic_memory import EpisodicMemory


# 任务类型关键词表（用于分类和检索）
TASK_TYPE_KEYWORDS = {
    "software_installation": ["install", "安装", "setup", "exe", "安装包"],
    "browser": ["打开浏览器", "浏览器", "chrome", "edge", "搜索", "播放视频"],
    "application_launch": ["打开", "启动", "运行", "launch", "启动应用"],
    "file_operation": ["创建文件", "删除", "移动", "复制", "保存文件", "file"],
    "window_operation": ["最小化", "最大化", "关闭窗口", "切换窗口", "window"],
}


class MemoryRetriever:
    """经验检索器"""

    def __init__(self, semantic: SemanticMemory, episodic: EpisodicMemory):
        self.semantic = semantic
        self.episodic = episodic

    def extract_keywords(self, task: str) -> List[str]:
        """从任务描述提取关键词"""
        keywords = []
        task_lower = task.lower()

        for task_type, kws in TASK_TYPE_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in task_lower:
                    keywords.append(kw)
                    keywords.append(task_type)

        for word in task_lower.split():
            clean = word.strip(",.()[]{}'\"")
            if clean and clean.isalpha() and len(clean) > 2:
                keywords.append(clean)

        chinese_chunks = re.findall(r'[\u4e00-\u9fff]{2,}', task)
        for chunk in chinese_chunks:
            if len(chunk) >= 2:
                keywords.append(chunk)

        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:20]

    def classify_task(self, task: str) -> str:
        """对任务进行分类"""
        task_lower = task.lower()
        for task_type, kws in TASK_TYPE_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in task_lower:
                    return task_type
        return "general"

    def retrieve(self, task: str, top_n: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """检索相关经验

        Returns:
            (语义经验列表, 任务经验列表)
        """
        keywords = self.extract_keywords(task)
        semantic_results = self.semantic.search(keywords, top_n=top_n)
        episodic_results = self.episodic.search(keywords, top_n=top_n)
        return semantic_results, episodic_results

    def build_experience_context(self, task: str) -> str:
        """构建经验上下文文本（注入提示词）"""
        semantic_results, episodic_results = self.retrieve(task)
        if not semantic_results and not episodic_results:
            return ""

        lines = ["## [历史经验] 检索到的相关操作经验"]
        lines.append("⚠️ 记忆是指导不是真理：执行前必须结合当前视觉观察验证。若界面与记忆不符，以当前观测为准。")

        for res in semantic_results:
            data = res["data"]
            conf = data.get("confidence", 0.5)
            if conf < 0.3:
                lines.append(f"- [语义经验·低置信] {str(res['key'])}")
            else:
                lines.append(f"- [语义经验] {str(data)[:150]}")

        for res in episodic_results:
            data = res["data"]
            conf = data.get("confidence", 0.5)
            prefix = "[任务经验·低置信]" if conf < 0.3 else "[任务经验]"
            lines.append(f"- {prefix} 任务[{data.get('task', '')}] "
                         f"步骤: {data.get('steps', [])[:6]} "
                         f"验证: {data.get('verification', [])[:3]}")
            failures = data.get("failure_recoveries", [])
            if failures:
                lines.append(f"  失败恢复: {failures[:3]}")

        return "\n".join(lines)