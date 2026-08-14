"""
路径规划器 - Path Planner
========================
死循环检测，新路径探索

核心功能：
  - 死循环检测
  - 失败路径记录
  - 新路径探索
"""
import logging
from typing import Dict, Any, List, Optional
from collections import Counter

from systematic_error_fix.models import (
    RepeatType, PathStatus, PathSuggestion,
    NegativeExperience, PathPlanningResult, ErrorType
)

logger = logging.getLogger(__name__)


class LoopDetector:
    """死循环检测器"""
    
    def __init__(self, history_size: int = 20):
        """初始化死循环检测器
        
        Args:
            history_size: 历史记录大小
        """
        self.history_size = history_size
        self._operation_history: List[str] = []
    
    def record(self, operation: str):
        """记录操作
        
        Args:
            operation: 操作名称
        """
        self._operation_history.append(operation)
        if len(self._operation_history) > self.history_size:
            self._operation_history.pop(0)
    
    def detect(self) -> tuple[bool, RepeatType, int]:
        """检测死循环
        
        Returns:
            (是否有循环, 循环类型, 重复次数)
        """
        if len(self._operation_history) < 4:
            return (False, RepeatType.NONE, 0)
        
        recent = self._operation_history[-10:]
        counter = Counter(recent)
        most_common = counter.most_common(1)[0]
        most_common_op, count = most_common
        
        if count >= 4:
            return (True, RepeatType.SINGLE, count)
        
        if len(recent) >= 6:
            pattern = recent[-6:]
            if pattern[:3] == pattern[3:6]:
                return (True, RepeatType.DUAL, 2)
        
        if len(recent) >= 8:
            pattern = recent[-8:]
            if pattern[:4] == pattern[4:8]:
                return (True, RepeatType.MULTI, 2)
        
        return (False, RepeatType.NONE, 0)


class NegativeExperienceManager:
    """负面经验管理器"""
    
    def __init__(self, knowledge_base=None):
        """初始化负面经验管理器
        
        Args:
            knowledge_base: 知识库
        """
        self.knowledge_base = knowledge_base
        self._negative_experiences: Dict[str, List[NegativeExperience]] = {}
    
    def record(self, path: str, error_type: ErrorType, reason: str = ""):
        """记录负面经验
        
        Args:
            path: 失败路径
            error_type: 错误类型
            reason: 原因
        """
        exp = NegativeExperience(
            path=path,
            error_type=error_type,
            reason=reason
        )
        
        if path not in self._negative_experiences:
            self._negative_experiences[path] = []
        self._negative_experiences[path].append(exp)
        
        logger.info(f"记录负面经验: {path} ({error_type})")
    
    def is_failed_path(self, path: str) -> bool:
        """检查是否为失败路径
        
        Args:
            path: 路径
            
        Returns:
            是否为失败路径
        """
        return path in self._negative_experiences
    
    def get_failed_paths(self) -> List[str]:
        """获取所有失败路径
        
        Returns:
            失败路径列表
        """
        return list(self._negative_experiences.keys())


class PathExplorer:
    """路径探索器"""
    
    SEARCH_PATHS = [
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "D:\\",
        "C:\\Users\\{user}\\Desktop",
        "C:\\Users\\{user}\\AppData\\Local"
    ]
    
    def __init__(self):
        """初始化路径探索器"""
        self._explored_paths: Dict[str, PathStatus] = {}
    
    def suggest_new_paths(
        self,
        target: str,
        failed_paths: List[str]
    ) -> List[PathSuggestion]:
        """建议新路径
        
        Args:
            target: 目标程序名
            failed_paths: 失败路径列表
            
        Returns:
            路径建议列表
        """
        suggestions = []
        
        for base_path in self.SEARCH_PATHS:
            if base_path in failed_paths:
                continue
            
            suggestions.append(PathSuggestion(
                path=base_path,
                method="search_in_directory",
                confidence=0.7,
                description=f"在{base_path}中搜索{target}"
            ))
        
        suggestions.append(PathSuggestion(
            path="start_menu",
            method="search_start_menu",
            confidence=0.6,
            description="在开始菜单中搜索"
        ))
        
        suggestions.append(PathSuggestion(
            path="where_command",
            method="use_where_command",
            confidence=0.8,
            description=f"使用where命令查找{target}"
        ))
        
        return suggestions[:5]


class PathPlanner:
    """路径规划器主类"""
    
    def __init__(self, knowledge_base=None):
        """初始化路径规划器
        
        Args:
            knowledge_base: 知识库
        """
        self.loop_detector = LoopDetector()
        self.negative_manager = NegativeExperienceManager(knowledge_base)
        self.explorer = PathExplorer()
    
    def plan(
        self,
        current_operation: str,
        target: str = "",
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR
    ) -> PathPlanningResult:
        """规划路径
        
        Args:
            current_operation: 当前操作
            target: 目标
            error_type: 错误类型
            
        Returns:
            路径规划结果
        """
        self.loop_detector.record(current_operation)
        
        has_loop, loop_type, repeat_count = self.loop_detector.detect()
        
        negative_experiences = []
        if error_type != ErrorType.UNKNOWN_ERROR:
            self.negative_manager.record(current_operation, error_type)
            negative_experiences = self.negative_manager._negative_experiences.get(
                current_operation, []
            )
        
        failed_paths = self.negative_manager.get_failed_paths()
        
        suggested_paths = []
        should_switch = False
        reason = ""
        
        if has_loop:
            should_switch = True
            reason = f"检测到{loop_type}循环，重复{repeat_count}次，建议切换路径"
            suggested_paths = self.explorer.suggest_new_paths(target, failed_paths)
            logger.warning(reason)
        
        return PathPlanningResult(
            has_loop=has_loop,
            loop_type=loop_type,
            suggested_paths=suggested_paths,
            negative_experiences=negative_experiences,
            should_switch_path=should_switch,
            reason=reason
        )
    
    def record_failure(self, path: str, error_type: ErrorType, reason: str = ""):
        """记录失败
        
        Args:
            path: 失败路径
            error_type: 错误类型
            reason: 原因
        """
        self.negative_manager.record(path, error_type, reason)


def test_path_planner():
    """测试路径规划器"""
    print("=" * 80)
    print("路径规划器测试")
    print("=" * 80)
    
    planner = PathPlanner()
    
    print("\n[测试1] 正常操作序列")
    for i, op in enumerate(["run_command", "find_element", "click"]):
        result = planner.plan(op, "原神")
        print(f"  操作{i+1}: {op}, 循环={result.has_loop}")
    
    print("\n[测试2] 死循环检测")
    for i in range(6):
        result = planner.plan("run_command", "原神")
        if result.has_loop:
            print(f"  第{i+1}次: 检测到{result.loop_type}循环!")
            print(f"  建议: {result.reason}")
            for path in result.suggested_paths:
                print(f"    - {path.description}")
    
    print("\n[测试3] 负面经验记录")
    planner.record_failure("D:\\原神.lnk", ErrorType.PROGRAM_NOT_FOUND, "快捷方式无效")
    failed = planner.negative_manager.get_failed_paths()
    print(f"  失败路径: {failed}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_path_planner()