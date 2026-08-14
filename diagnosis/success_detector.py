"""
成功检测器 - Success Detector
=============================
优先检测执行结果是否为成功

核心功能：
  - detect(): 检测执行结果是否为成功
  - _check_exit_code(): 检查退出码
  - _match_success_signals(): 匹配成功标志
  - _calculate_confidence(): 计算综合置信度

特点：
  - 优先检测成功标志（避免误判）
  - 多维度判断（退出码、输出、执行时间）
  - 计算成功置信度
"""
import re
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnosis.models import ExecutionResult, SuccessDiagnosis


class SuccessDetector:
    """成功检测器"""
    
    # 成功标志列表（高置信度）
    SUCCESS_SIGNALS_HIGH = [
        "成功", "完成", "已创建", "已删除", "已移动",
        "已启动", "已关闭", "已保存", "已打开",
        "success", "completed", "done", "finished",
        "✓", "✅", "[OK]", "执行成功"
    ]
    
    # 成功标志列表（中置信度）
    SUCCESS_SIGNALS_MEDIUM = [
        "正常", "就绪", "可用", "存在",
        "ok", "ready", "available"
    ]
    
    # 成功正则表达式列表
    SUCCESS_PATTERNS = [
        r"退出码[:：]\s*0",
        r"return\s+code[:：]\s*0",
        r"exit\s+code[:：]\s*0",
        r"已\s*\S+\s*成功",
    ]
    
    def __init__(self):
        """初始化成功检测器"""
        # 编译正则表达式
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SUCCESS_PATTERNS
        ]
    
    def detect(self, result: ExecutionResult) -> SuccessDiagnosis:
        """
        检测执行结果是否为成功
        
        Args:
            result: 执行结果
        
        Returns:
            SuccessDiagnosis: 成功诊断结果
        """
        reasons = []
        matched_signals = []
        confidence = 0.0
        
        # 1. 检查退出码
        exit_code_confidence = self._check_exit_code(result)
        if exit_code_confidence > 0:
            confidence += exit_code_confidence
            reasons.append(f"退出码为{result.exit_code}")
        
        # 2. 匹配成功标志
        signal_confidence, signals = self._match_success_signals(result)
        if signal_confidence > 0:
            confidence += signal_confidence
            matched_signals.extend(signals)
            reasons.append(f"输出包含成功标志: {signals[:3]}")
        
        # 3. 计算综合置信度
        confidence = self._calculate_confidence(confidence, result)
        
        # 4. 判断是否成功（置信度 >= 0.8）
        is_success = confidence >= 0.8
        
        if is_success:
            reasons.append(f"综合置信度{confidence:.2f} >= 0.8，判定为成功")
        else:
            reasons.append(f"综合置信度{confidence:.2f} < 0.8，不确定是否成功")
        
        return SuccessDiagnosis(
            is_success=is_success,
            confidence=confidence,
            reasons=reasons,
            matched_signals=matched_signals
        )
    
    def _check_exit_code(self, result: ExecutionResult) -> float:
        """
        检查退出码
        
        Args:
            result: 执行结果
        
        Returns:
            float: 置信度贡献值
        """
        if result.exit_code == 0:
            return 0.4  # 退出码为0，贡献0.4置信度
        return 0.0
    
    def _match_success_signals(self, result: ExecutionResult) -> tuple:
        """
        匹配成功标志
        
        Args:
            result: 执行结果
        
        Returns:
            tuple: (置信度贡献值, 匹配的信号列表)
        """
        output = result.output.lower()
        matched_signals = []
        confidence = 0.0
        
        # 匹配高置信度信号
        for signal in self.SUCCESS_SIGNALS_HIGH:
            if signal.lower() in output:
                matched_signals.append(signal)
                confidence = max(confidence, 0.5)
        
        # 匹配中置信度信号
        if confidence == 0:
            for signal in self.SUCCESS_SIGNALS_MEDIUM:
                if signal.lower() in output:
                    matched_signals.append(signal)
                    confidence = max(confidence, 0.3)
        
        # 匹配正则表达式
        for pattern in self.compiled_patterns:
            if pattern.search(output):
                matched_signals.append(f"pattern:{pattern.pattern}")
                confidence = max(confidence, 0.4)
        
        return confidence, matched_signals
    
    def _calculate_confidence(self, base_confidence: float, result: ExecutionResult) -> float:
        """
        计算综合置信度
        
        Args:
            base_confidence: 基础置信度
            result: 执行结果
        
        Returns:
            float: 综合置信度
        """
        # 限制最大置信度为1.0
        confidence = min(base_confidence, 1.0)
        
        # 如果执行时间过短（<10ms），可能是假成功，降低置信度
        if result.execution_time_ms < 10:
            confidence *= 0.8
        
        return confidence


