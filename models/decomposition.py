"""
任务分解结果数据模型 - Decomposition Result Data Model
===================================================
定义任务分解的结果数据结构
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from models.enums import DecompositionSource
from models.subtask import SubTask


@dataclass
class DecompositionResult:
    """任务分解结果数据类
    
    Attributes:
        task_id: 任务唯一标识
        original_description: 原始任务描述
        subtasks: 子任务列表
        decomposition_source: 分解来源
        confidence: 分配置信度 (0-1)
        dependencies: 依赖关系图 {subtask_id: [依赖的subtask_id列表]}
        created_at: 创建时间
    """
    task_id: str
    original_description: str
    subtasks: List[SubTask] = field(default_factory=list)
    decomposition_source: DecompositionSource = DecompositionSource.RULE_BASED
    confidence: float = 0.8
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """验证置信度范围"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"置信号必须在0-1之间, 当前: {self.confidence}")
        
        if not self.task_id:
            raise ValueError("任务ID不能为空")
        
        if not self.original_description:
            raise ValueError("原始任务描述不能为空")
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序 (拓扑排序)
        
        Returns:
            子任务ID列表, 按执行顺序排列
        """
        if not self.subtasks:
            return []
        
        in_degree: Dict[str, int] = defaultdict(int)
        graph: Dict[str, List[str]] = defaultdict(list)
        
        all_ids = [st.subtask_id for st in self.subtasks]
        
        for subtask in self.subtasks:
            in_degree[subtask.subtask_id] = 0
        
        for subtask_id, deps in self.dependencies.items():
            for dep in deps:
                if dep in all_ids and subtask_id in all_ids:
                    graph[dep].append(subtask_id)
                    in_degree[subtask_id] += 1
        
        queue = [sid for sid in all_ids if in_degree[sid] == 0]
        result = []
        
        while queue:
            queue.sort(key=lambda x: next(
                (st.priority for st in self.subtasks if st.subtask_id == x),
                5
            ))
            
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(all_ids):
            remaining = [sid for sid in all_ids if sid not in result]
            result.extend(remaining)
        
        return result
    
    def get_subtask_by_id(self, subtask_id: str) -> Optional[SubTask]:
        """根据ID获取子任务
        
        Args:
            subtask_id: 子任务ID
            
        Returns:
            子任务, 如果不存在返回None
        """
        for subtask in self.subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask
        return None
    
    def add_subtask(self, subtask: SubTask, dependencies: Optional[List[str]] = None):
        """添加子任务
        
        Args:
            subtask: 子任务
            dependencies: 依赖的子任务ID列表
        """
        self.subtasks.append(subtask)
        
        if dependencies:
            self.dependencies[subtask.subtask_id] = dependencies
            subtask.dependencies = dependencies
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "original_description": self.original_description,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "decomposition_source": str(self.decomposition_source),
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "execution_order": self.get_execution_order()
        }


@dataclass
class ExecutionResult:
    """执行结果数据类
    
    Attributes:
        success: 是否成功
        result: 执行结果内容
        tool_used: 使用的工具
        tier_used: 使用的层级
        heal_attempts: 自愈尝试次数
        execution_time: 执行时间 (秒)
        error: 错误信息
    """
    success: bool
    result: str = ""
    tool_used: str = ""
    tier_used: int = 1
    heal_attempts: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None
    
    def is_failure(self) -> bool:
        """检查是否失败
        
        Returns:
            是否失败
        """
        return not self.success
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "result": self.result,
            "tool_used": self.tool_used,
            "tier_used": self.tier_used,
            "heal_attempts": self.heal_attempts,
            "execution_time": self.execution_time,
            "error": self.error
        }


@dataclass
class TaskExecutionSummary:
    """任务执行摘要数据类
    
    Attributes:
        task_id: 任务ID
        total_subtasks: 子任务总数
        completed_subtasks: 已完成子任务数
        failed_subtasks: 失败子任务数
        total_time: 总执行时间
        total_heal_attempts: 总自愈尝试次数
        tier_usage: 各层级使用次数
        tool_usage: 各工具使用次数
    """
    task_id: str
    total_subtasks: int = 0
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    total_time: float = 0.0
    total_heal_attempts: int = 0
    tier_usage: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    tool_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """计算成功率
        
        Returns:
            成功率 (0-1)
        """
        if self.total_subtasks == 0:
            return 0.0
        return self.completed_subtasks / self.total_subtasks
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "failed_subtasks": self.failed_subtasks,
            "total_time": self.total_time,
            "total_heal_attempts": self.total_heal_attempts,
            "tier_usage": self.tier_usage,
            "tool_usage": self.tool_usage,
            "success_rate": self.success_rate
        }