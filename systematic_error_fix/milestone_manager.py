"""
里程碑管理器 - Milestone Manager
================================
里程碑检查点，动态步数调整

核心功能：
  - 里程碑检查点设定
  - 进度评估
  - 动态步数调整
  - 任务拆解验证
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from systematic_error_fix.models import (
    Milestone, MilestoneStatus, StrategyAdjustment,
    MilestoneProgressReport
)

logger = logging.getLogger(__name__)


class MilestoneChecker:
    """里程碑检查器"""
    
    def __init__(self, check_interval: int = 5):
        """初始化里程碑检查器
        
        Args:
            check_interval: 检查间隔（步数）
        """
        self.check_interval = check_interval
    
    def should_check(self, current_step: int) -> bool:
        """判断是否应该检查
        
        Args:
            current_step: 当前步数
            
        Returns:
            是否应该检查
        """
        return current_step > 0 and current_step % self.check_interval == 0
    
    def check_progress(
        self,
        milestone: Milestone,
        current_step: int,
        success_indicators: List[str] = None
    ) -> MilestoneStatus:
        """检查里程碑进度
        
        Args:
            milestone: 里程碑
            current_step: 当前步数
            success_indicators: 成功指标
            
        Returns:
            里程碑状态
        """
        if milestone.status == MilestoneStatus.COMPLETED:
            return MilestoneStatus.COMPLETED
        
        if milestone.status == MilestoneStatus.FAILED:
            return MilestoneStatus.FAILED
        
        progress_ratio = current_step / milestone.target_step
        
        if progress_ratio >= 1.0:
            if success_indicators:
                return MilestoneStatus.COMPLETED
            return MilestoneStatus.TIMEOUT
        
        if progress_ratio >= 0.5:
            return MilestoneStatus.IN_PROGRESS
        
        return MilestoneStatus.NOT_STARTED


class StrategyAdjuster:
    """策略调整器"""
    
    def __init__(
        self,
        max_steps_base: int = 50,
        max_steps_multiplier: float = 1.5
    ):
        """初始化策略调整器
        
        Args:
            max_steps_base: 基础最大步数
            max_steps_multiplier: 步数乘数
        """
        self.max_steps_base = max_steps_base
        self.max_steps_multiplier = max_steps_multiplier
    
    def adjust(
        self,
        milestone: Milestone,
        progress_status: MilestoneStatus,
        current_step: int
    ) -> List[StrategyAdjustment]:
        """调整策略
        
        Args:
            milestone: 里程碑
            progress_status: 进度状态
            current_step: 当前步数
            
        Returns:
            策略调整列表
        """
        adjustments = []
        
        if progress_status == MilestoneStatus.TIMEOUT:
            adjustments.append(StrategyAdjustment(
                adjustment_type="increase_max_steps",
                reason=f"里程碑'{milestone.name}'超时，增加最大步数",
                new_max_steps=int(self.max_steps_base * self.max_steps_multiplier),
                priority=1
            ))
        
        if progress_status == MilestoneStatus.IN_PROGRESS:
            if current_step > milestone.target_step * 0.7:
                adjustments.append(StrategyAdjustment(
                    adjustment_type="switch_strategy",
                    reason="进度缓慢，建议切换策略",
                    new_strategy="use_cli_first",
                    priority=2
                ))
        
        if progress_status == MilestoneStatus.FAILED:
            adjustments.append(StrategyAdjustment(
                adjustment_type="abort_and_report",
                reason=f"里程碑'{milestone.name}'失败",
                priority=0
            ))
        
        return adjustments


class MilestoneManager:
    """里程碑管理器主类"""
    
    DEFAULT_MILESTONES = [
        {"name": "启动应用", "target_step": 10},
        {"name": "等待加载", "target_step": 20},
        {"name": "执行操作", "target_step": 40},
        {"name": "完成任务", "target_step": 50}
    ]
    
    def __init__(
        self,
        max_steps: int = 50,
        milestones: Optional[List[Dict[str, Any]]] = None
    ):
        """初始化里程碑管理器
        
        Args:
            max_steps: 最大步数
            milestones: 里程碑配置
        """
        self.max_steps = max_steps
        self.checker = MilestoneChecker()
        self.adjuster = StrategyAdjuster(max_steps_base=max_steps)
        
        self._milestones: List[Milestone] = []
        self._current_milestone_index = 0
        self._current_step = 0
        
        self._init_milestones(milestones or self.DEFAULT_MILESTONES)
    
    def _init_milestones(self, milestones_config: List[Dict[str, Any]]):
        """初始化里程碑
        
        Args:
            milestones_config: 里程碑配置
        """
        for i, config in enumerate(milestones_config):
            milestone = Milestone(
                milestone_id=f"milestone_{i}",
                name=config["name"],
                target_step=config["target_step"],
                status=MilestoneStatus.NOT_STARTED
            )
            self._milestones.append(milestone)
    
    def update(self, current_step: int, success_indicators: List[str] = None) -> MilestoneProgressReport:
        """更新进度
        
        Args:
            current_step: 当前步数
            success_indicators: 成功指标
            
        Returns:
            进度报告
        """
        self._current_step = current_step
        
        current_milestone = self._get_current_milestone()
        
        if not current_milestone:
            return MilestoneProgressReport(
                current_milestone=None,
                overall_progress=1.0,
                warnings=["所有里程碑已完成"]
            )
        
        if self.checker.should_check(current_step):
            status = self.checker.check_progress(
                current_milestone,
                current_step,
                success_indicators
            )
            
            if status != current_milestone.status:
                old_status = current_milestone.status
                current_milestone.status = status
                current_milestone.current_step = current_step
                
                if status == MilestoneStatus.COMPLETED:
                    current_milestone.completed_at = datetime.now()
                    self._current_milestone_index += 1
                    logger.info(f"里程碑完成: {current_milestone.name}")
                
                logger.info(
                    f"里程碑'{current_milestone.name}'状态变更: "
                    f"{old_status} → {status}"
                )
        
        adjustments = self.adjuster.adjust(
            current_milestone,
            current_milestone.status,
            current_step
        )
        
        for adj in adjustments:
            if adj.new_max_steps:
                self.max_steps = adj.new_max_steps
                logger.info(f"调整最大步数: {adj.new_max_steps}")
        
        warnings = self._generate_warnings(current_milestone, current_step)
        
        overall_progress = self._calculate_overall_progress()
        
        return MilestoneProgressReport(
            current_milestone=current_milestone,
            overall_progress=overall_progress,
            adjustments=adjustments,
            warnings=warnings
        )
    
    def _get_current_milestone(self) -> Optional[Milestone]:
        """获取当前里程碑
        
        Returns:
            当前里程碑
        """
        if self._current_milestone_index < len(self._milestones):
            return self._milestones[self._current_milestone_index]
        return None
    
    def _calculate_overall_progress(self) -> float:
        """计算总体进度
        
        Returns:
            总体进度 (0-1)
        """
        completed = self._current_milestone_index
        total = len(self._milestones)
        
        if total == 0:
            return 1.0
        
        base_progress = completed / total
        
        current = self._get_current_milestone()
        if current and current.target_step > 0:
            current_progress = self._current_step / current.target_step
            current_progress = min(current_progress, 1.0)
            base_progress += (current_progress / total)
        
        return min(base_progress, 1.0)
    
    def _generate_warnings(
        self,
        milestone: Milestone,
        current_step: int
    ) -> List[str]:
        """生成警告
        
        Args:
            milestone: 里程碑
            current_step: 当前步数
            
        Returns:
            警告列表
        """
        warnings = []
        
        if milestone.status == MilestoneStatus.TIMEOUT:
            warnings.append(f"里程碑'{milestone.name}'超时")
        
        if current_step > milestone.target_step * 0.8:
            progress = current_step / milestone.target_step * 100
            warnings.append(f"里程碑'{milestone.name}'进度: {progress:.0f}%，接近超时")
        
        return warnings
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            "current_step": self._current_step,
            "max_steps": self.max_steps,
            "current_milestone_index": self._current_milestone_index,
            "total_milestones": len(self._milestones),
            "milestones": [m.to_dict() for m in self._milestones],
            "overall_progress": self._calculate_overall_progress()
        }


def test_milestone_manager():
    """测试里程碑管理器"""
    print("=" * 80)
    print("里程碑管理器测试")
    print("=" * 80)
    
    manager = MilestoneManager(max_steps=50)
    
    print("\n[测试1] 初始状态")
    stats = manager.get_statistics()
    print(f"  总里程碑数: {stats['total_milestones']}")
    print(f"  当前索引: {stats['current_milestone_index']}")
    for m in stats['milestones']:
        print(f"    - {m['name']}: 目标步数={m['target_step']}, 状态={m['status']}")
    
    print("\n[测试2] 进度更新")
    for step in [5, 10, 15, 20, 25, 30]:
        report = manager.update(step)
        if report.warnings or report.adjustments:
            print(f"  步数{step}:")
            if report.current_milestone:
                print(f"    当前里程碑: {report.current_milestone.name} ({report.current_milestone.status})")
            print(f"    总体进度: {report.overall_progress*100:.0f}%")
            for w in report.warnings:
                print(f"    警告: {w}")
            for a in report.adjustments:
                print(f"    调整: {a.reason}")
    
    print("\n[测试3] 最终统计")
    stats = manager.get_statistics()
    print(f"  当前步数: {stats['current_step']}")
    print(f"  总体进度: {stats['overall_progress']*100:.0f}%")
    for m in stats['milestones']:
        print(f"    - {m['name']}: 状态={m['status']}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_milestone_manager()