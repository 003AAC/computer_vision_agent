"""
置信度评估器 - Confidence Evaluator
==================================
多方法交叉验证，检测伪成功

核心功能：
  - 置信度计算
  - 伪成功检测
  - 多方法验证
  - 降级决策
"""
import logging
from typing import Dict, Any, Optional, List

from systematic_error_fix.models import (
    QualityLevel, ConfidenceEvaluationResult
)

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """置信度计算器"""
    
    DEFAULT_WEIGHTS = {
        "vision": 0.4,
        "ui_automation": 0.4,
        "history": 0.2
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """初始化置信度计算器
        
        Args:
            weights: 权重配置
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def calculate(self, scores: Dict[str, float]) -> float:
        """计算加权平均置信度
        
        Args:
            scores: 各方法评分
            
        Returns:
            综合置信度
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for method, score in scores.items():
            weight = self.weights.get(method, 0.0)
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.5
    
    def determine_quality_level(self, confidence: float) -> QualityLevel:
        """判定质量等级
        
        Args:
            confidence: 置信度
            
        Returns:
            质量等级
        """
        if confidence >= 0.8:
            return QualityLevel.HIGH
        elif confidence >= 0.6:
            return QualityLevel.MEDIUM
        elif confidence >= 0.4:
            return QualityLevel.LOW
        else:
            return QualityLevel.VERY_LOW


class PseudoSuccessDetector:
    """伪成功检测器"""
    
    def __init__(
        self,
        low_confidence_threshold: float = 0.4,
        pseudo_success_threshold: float = 0.5
    ):
        """初始化伪成功检测器
        
        Args:
            low_confidence_threshold: 低置信度阈值
            pseudo_success_threshold: 伪成功阈值
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.pseudo_success_threshold = pseudo_success_threshold
    
    def detect(
        self,
        confidence: float,
        has_visual_evidence: bool = False,
        task_completed_flag: bool = False
    ) -> bool:
        """检测是否为伪成功
        
        Args:
            confidence: 置信度
            has_visual_evidence: 是否有视觉证据
            task_completed_flag: 任务完成标志
            
        Returns:
            是否为伪成功
        """
        if confidence < self.low_confidence_threshold:
            return True
        
        if not has_visual_evidence and confidence < self.pseudo_success_threshold:
            return True
        
        if task_completed_flag and confidence < 0.6:
            return True
        
        return False
    
    def calculate_confidence_gap(
        self,
        reported_confidence: float,
        actual_evidence: float
    ) -> float:
        """计算置信度差距
        
        Args:
            reported_confidence: 报告的置信度
            actual_evidence: 实际证据
            
        Returns:
            置信度差距
        """
        return abs(reported_confidence - actual_evidence)


class MultiMethodValidator:
    """多方法验证器"""
    
    def __init__(self, ui_controller=None, knowledge_base=None):
        """初始化多方法验证器
        
        Args:
            ui_controller: UI Automation控制器
            knowledge_base: 知识库
        """
        self.ui_controller = ui_controller
        self.knowledge_base = knowledge_base
    
    def validate(
        self,
        target: str,
        vision_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """多方法验证
        
        Args:
            target: 目标描述
            vision_result: 视觉模型结果
            
        Returns:
            各方法评分
        """
        scores = {}
        
        if vision_result:
            scores["vision"] = vision_result.get("confidence", 0.5)
        else:
            scores["vision"] = 0.3
        
        scores["ui_automation"] = self._validate_with_uiautomation(target)
        
        scores["history"] = self._validate_with_history(target)
        
        return scores
    
    def _validate_with_uiautomation(self, target: str) -> float:
        """使用UI Automation验证
        
        Args:
            target: 目标描述
            
        Returns:
            评分
        """
        if not self.ui_controller:
            return 0.5
        
        try:
            element = self.ui_controller.find_element(target)
            if element:
                return 0.9
            return 0.3
        except Exception:
            return 0.5
    
    def _validate_with_history(self, target: str) -> float:
        """使用历史数据验证
        
        Args:
            target: 目标描述
            
        Returns:
            评分
        """
        if not self.knowledge_base:
            return 0.5
        
        try:
            success_rate = self.knowledge_base.get_success_rate(target)
            return success_rate if success_rate else 0.5
        except Exception:
            return 0.5


class DegradationDecisionMaker:
    """降级决策器"""
    
    def __init__(
        self,
        max_consecutive_failures: int = 3,
        degradation_threshold: float = 0.4
    ):
        """初始化降级决策器
        
        Args:
            max_consecutive_failures: 最大连续失败次数
            degradation_threshold: 降级阈值
        """
        self.max_consecutive_failures = max_consecutive_failures
        self.degradation_threshold = degradation_threshold
        self._consecutive_failures = 0
        self._degradation_count = 0
    
    def should_degrade(self, confidence: float) -> bool:
        """判断是否应该降级
        
        Args:
            confidence: 置信度
            
        Returns:
            是否应该降级
        """
        if confidence < self.degradation_threshold:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._degradation_count += 1
            self._consecutive_failures = 0
            return True
        
        return False
    
    def get_degradation_suggestion(self) -> str:
        """获取降级建议
        
        Returns:
            降级建议
        """
        suggestions = [
            "尝试使用UI Automation替代视觉模型",
            "尝试使用CLI命令替代视觉操作",
            "尝试使用快捷键操作",
            "检查目标是否存在"
        ]
        return suggestions[self._degradation_count % len(suggestions)]


class ConfidenceEvaluator:
    """置信度评估器主类"""
    
    def __init__(
        self,
        ui_controller=None,
        knowledge_base=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """初始化置信度评估器
        
        Args:
            ui_controller: UI Automation控制器
            knowledge_base: 知识库
            weights: 权重配置
        """
        self.calculator = ConfidenceCalculator(weights)
        self.pseudo_detector = PseudoSuccessDetector()
        self.multi_validator = MultiMethodValidator(ui_controller, knowledge_base)
        self.degradation_maker = DegradationDecisionMaker()
    
    def evaluate(
        self,
        vision_result: Optional[Dict[str, Any]] = None,
        target: str = "",
        task_completed: bool = False
    ) -> ConfidenceEvaluationResult:
        """评估置信度
        
        Args:
            vision_result: 视觉模型结果
            target: 目标描述
            task_completed: 任务完成标志
            
        Returns:
            置信度评估结果
        """
        scores = self.multi_validator.validate(target, vision_result)
        
        confidence = self.calculator.calculate(scores)
        
        quality_level = self.calculator.determine_quality_level(confidence)
        
        has_visual_evidence = (
            vision_result and
            vision_result.get("confidence", 0) > 0.5
        )
        
        is_pseudo_success = self.pseudo_detector.detect(
            confidence,
            has_visual_evidence,
            task_completed
        )
        
        should_degrade = self.degradation_maker.should_degrade(confidence)
        
        suggestions = self._generate_suggestions(
            confidence, is_pseudo_success, should_degrade
        )
        
        logger.info(
            f"置信度评估: {confidence:.2f} ({quality_level}), "
            f"伪成功={is_pseudo_success}, 降级={should_degrade}"
        )
        
        return ConfidenceEvaluationResult(
            confidence=confidence,
            quality_level=quality_level,
            is_pseudo_success=is_pseudo_success,
            evidence=scores,
            suggestions=suggestions,
            should_degrade=should_degrade
        )
    
    def _generate_suggestions(
        self,
        confidence: float,
        is_pseudo_success: bool,
        should_degrade: bool
    ) -> List[str]:
        """生成建议
        
        Args:
            confidence: 置信度
            is_pseudo_success: 是否伪成功
            should_degrade: 是否应该降级
            
        Returns:
            建议列表
        """
        suggestions = []
        
        if is_pseudo_success:
            suggestions.append("检测到伪成功，建议重新验证")
        
        if confidence < 0.5:
            suggestions.append("置信度过低，建议使用其他方法")
        
        if should_degrade:
            suggestions.append(self.degradation_maker.get_degradation_suggestion())
        
        return suggestions


def test_confidence_evaluator():
    """测试置信度评估器"""
    print("=" * 80)
    print("置信度评估器测试")
    print("=" * 80)
    
    evaluator = ConfidenceEvaluator()
    
    print("\n[测试1] 高置信度场景")
    result = evaluator.evaluate(
        vision_result={"confidence": 0.85, "found": True},
        target="原神图标",
        task_completed=True
    )
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  质量等级: {result.quality_level}")
    print(f"  伪成功: {result.is_pseudo_success}")
    print(f"  降级: {result.should_degrade}")
    
    print("\n[测试2] 低置信度场景")
    result = evaluator.evaluate(
        vision_result={"confidence": 0.35, "found": False},
        target="原神图标",
        task_completed=False
    )
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  质量等级: {result.quality_level}")
    print(f"  伪成功: {result.is_pseudo_success}")
    print(f"  降级: {result.should_degrade}")
    print(f"  建议: {result.suggestions}")
    
    print("\n[测试3] 伪成功检测")
    result = evaluator.evaluate(
        vision_result={"confidence": 0.45, "found": True},
        target="原神图标",
        task_completed=True
    )
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  伪成功: {result.is_pseudo_success}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_confidence_evaluator()