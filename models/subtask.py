"""
子任务数据模型 - SubTask Data Model
==================================
定义子任务的数据结构
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from models.enums import SubTaskStatus


@dataclass
class SubTask:
    """子任务数据类
    
    表示复杂任务分解后的单个子任务
    
    Attributes:
        subtask_id: 子任务唯一标识 (格式: task_{timestamp}_{index})
        description: 子任务描述 (长度≤100字符)
        goal: 子任务目标
        preconditions: 前置条件列表
        verification_criteria: 验收标准
        strategy_context_id: 策略上下文ID
        status: 子任务状态
        dependencies: 依赖的子任务ID列表
        priority: 优先级 (0-10, 数字越小优先级越高)
        timeout: 超时时间 (秒)
        retry_count: 重试次数
        max_retries: 最大重试次数
    """
    subtask_id: str
    description: str
    goal: str = ""
    preconditions: List[str] = field(default_factory=list)
    verification_criteria: List[str] = field(default_factory=list)
    strategy_context_id: Optional[str] = None
    status: SubTaskStatus = SubTaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """验证字段合法性"""
        if not self.description:
            raise ValueError("子任务描述不能为空")
        
        if len(self.description) > 100:
            raise ValueError(f"子任务描述长度不能超过100字符, 当前: {len(self.description)}")
        
        if not self.subtask_id:
            raise ValueError("子任务ID不能为空")
        
        if self.priority < 0 or self.priority > 10:
            raise ValueError(f"优先级必须在0-10之间, 当前: {self.priority}")
    
    def is_ready(self, completed_subtask_ids: List[str]) -> bool:
        """检查子任务是否就绪 (所有依赖已完成)
        
        Args:
            completed_subtask_ids: 已完成的子任务ID列表
            
        Returns:
            是否就绪
        """
        if self.status != SubTaskStatus.PENDING:
            return False
        
        for dep_id in self.dependencies:
            if dep_id not in completed_subtask_ids:
                return False
        
        return True
    
    def mark_running(self):
        """标记子任务开始执行"""
        self.status = SubTaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self):
        """标记子任务完成"""
        self.status = SubTaskStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self):
        """标记子任务失败"""
        self.status = SubTaskStatus.FAILED
        self.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """增加重试计数"""
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "subtask_id": self.subtask_id,
            "description": self.description,
            "goal": self.goal,
            "preconditions": self.preconditions,
            "verification_criteria": self.verification_criteria,
            "strategy_context_id": self.strategy_context_id,
            "status": str(self.status),
            "dependencies": self.dependencies,
            "priority": self.priority,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubTask':
        """从字典创建"""
        return cls(
            subtask_id=data["subtask_id"],
            description=data["description"],
            goal=data.get("goal", ""),
            preconditions=data.get("preconditions", []),
            verification_criteria=data.get("verification_criteria", []),
            strategy_context_id=data.get("strategy_context_id"),
            status=SubTaskStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", 5),
            timeout=data.get("timeout", 30),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )