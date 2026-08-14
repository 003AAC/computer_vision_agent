"""
执行策略数据模型 - Strategy Context Data Model
============================================
定义子任务的执行策略上下文
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.enums import TaskTier


@dataclass
class StrategyContext:
    """执行策略上下文数据类
    
    为每个子任务提供独立的执行策略
    
    Attributes:
        strategy_id: 策略唯一标识
        initial_tier: 初始执行层级 (1=CLI, 2=快捷键, 3=UI Automation, 4=视觉模型)
        current_tier: 当前执行层级
        tier_failures: 各层级失败次数
        max_tier_attempts: 每层最大尝试次数
        tools_used: 已使用的工具列表
        heal_quota: 自愈配额
        heal_used: 已使用的自愈次数
        fallback_path: 降级路径
        timeout_config: 超时配置
        created_at: 创建时间
    """
    strategy_id: str
    initial_tier: int = 1
    current_tier: int = 1
    tier_failures: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    max_tier_attempts: int = 2
    tools_used: List[str] = field(default_factory=list)
    heal_quota: int = 3
    heal_used: int = 0
    fallback_path: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    timeout_config: Dict[str, int] = field(default_factory=lambda: {
        "cli": 10,
        "hotkey": 5,
        "ui_automation": 15,
        "vision": 30
    })
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """验证字段合法性"""
        if self.initial_tier < 1 or self.initial_tier > 4:
            raise ValueError(f"初始层级必须在1-4之间, 当前: {self.initial_tier}")
        
        if self.current_tier < 1 or self.current_tier > 4:
            raise ValueError(f"当前层级必须在1-4之间, 当前: {self.current_tier}")
        
        if self.heal_used > self.heal_quota:
            raise ValueError(f"已使用自愈次数不能超过配额, 已用: {self.heal_used}, 配额: {self.heal_quota}")
    
    def can_downgrade(self) -> bool:
        """检查是否可以降级
        
        Returns:
            是否可以降级
        """
        current_index = self.fallback_path.index(self.current_tier)
        return current_index < len(self.fallback_path) - 1
    
    def next_tier(self) -> Optional[int]:
        """获取下一层级
        
        Returns:
            下一层级, 如果无法降级返回None
        """
        if not self.can_downgrade():
            return None
        
        current_index = self.fallback_path.index(self.current_tier)
        return self.fallback_path[current_index + 1]
    
    def downgrade(self) -> bool:
        """执行降级
        
        Returns:
            是否降级成功
        """
        next_tier = self.next_tier()
        if next_tier is None:
            return False
        
        self.current_tier = next_tier
        return True
    
    def record_tool_use(self, tool_name: str):
        """记录工具使用
        
        Args:
            tool_name: 工具名称
        """
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
    
    def record_failure(self, tier: Optional[int] = None):
        """记录失败
        
        Args:
            tier: 失败的层级, 默认为当前层级
        """
        if tier is None:
            tier = self.current_tier
        
        if tier not in self.tier_failures:
            self.tier_failures[tier] = 0
        
        self.tier_failures[tier] += 1
    
    def should_downgrade(self) -> bool:
        """检查是否应该降级
        
        Returns:
            是否应该降级
        """
        current_failures = self.tier_failures.get(self.current_tier, 0)
        return current_failures >= self.max_tier_attempts
    
    def reset_for_new_tier(self):
        """为新层级重置状态"""
        self.tier_failures[self.current_tier] = 0
    
    def can_heal(self) -> bool:
        """检查是否可以使用自愈
        
        Returns:
            是否可以使用自愈
        """
        return self.heal_used < self.heal_quota
    
    def use_heal(self) -> bool:
        """使用自愈配额
        
        Returns:
            是否成功使用
        """
        if not self.can_heal():
            return False
        
        self.heal_used += 1
        return True
    
    def get_tier_name(self) -> str:
        """获取当前层级名称
        
        Returns:
            层级名称
        """
        tier_names = {
            1: "CLI",
            2: "Hotkey",
            3: "UI Automation",
            4: "Vision"
        }
        return tier_names.get(self.current_tier, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_id": self.strategy_id,
            "initial_tier": self.initial_tier,
            "current_tier": self.current_tier,
            "tier_failures": self.tier_failures,
            "max_tier_attempts": self.max_tier_attempts,
            "tools_used": self.tools_used,
            "heal_quota": self.heal_quota,
            "heal_used": self.heal_used,
            "fallback_path": self.fallback_path,
            "timeout_config": self.timeout_config,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyContext':
        """从字典创建"""
        return cls(
            strategy_id=data["strategy_id"],
            initial_tier=data.get("initial_tier", 1),
            current_tier=data.get("current_tier", 1),
            tier_failures=data.get("tier_failures", {1: 0, 2: 0, 3: 0, 4: 0}),
            max_tier_attempts=data.get("max_tier_attempts", 2),
            tools_used=data.get("tools_used", []),
            heal_quota=data.get("heal_quota", 3),
            heal_used=data.get("heal_used", 0),
            fallback_path=data.get("fallback_path", [1, 2, 3, 4]),
            timeout_config=data.get("timeout_config", {})
        )
    
    @classmethod
    def create_for_subtask(cls, subtask_id: str, initial_tier: int = 1) -> 'StrategyContext':
        """为子任务创建策略上下文
        
        Args:
            subtask_id: 子任务ID
            initial_tier: 初始层级
            
        Returns:
            策略上下文
        """
        return cls(
            strategy_id=f"strategy_{subtask_id}",
            initial_tier=initial_tier,
            current_tier=initial_tier
        )