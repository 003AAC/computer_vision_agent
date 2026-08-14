"""
诊断模块 - Diagnosis Models
===========================
定义自愈引擎错误诊断修复所需的核心数据结构

包含：
  - ExecutionResult: 执行结果数据类
  - SuccessDiagnosis: 成功诊断结果
  - FailureDiagnosis: 失败诊断结果
  - ConfirmationResult: 二次确认结果
  - ValidationResult: 经验验证结果
  - Experience: 经验数据类
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DiagnosisType(Enum):
    """诊断类型枚举"""
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Confidence(Enum):
    """置信度级别枚举"""
    HIGH = 0.95
    MEDIUM = 0.7
    LOW = 0.5


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    tool_name: str  # 工具名称
    tool_args: Dict[str, Any]  # 工具参数
    exit_code: int = 0  # 退出码（0=成功）
    stdout: str = ""  # 标准输出
    stderr: str = ""  # 标准错误
    execution_time_ms: float = 0.0  # 执行时间（毫秒）
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    
    @property
    def output(self) -> str:
        """获取完整输出（stdout + stderr）"""
        return f"{self.stdout}\n{self.stderr}".strip()
    
    @property
    def is_success_by_exit_code(self) -> bool:
        """根据退出码判断是否成功"""
        return self.exit_code == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:500],  # 截断长输出
            "stderr": self.stderr[:500],
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SuccessDiagnosis:
    """成功诊断结果"""
    is_success: bool  # 是否成功
    confidence: float  # 置信度 (0-1)
    reasons: List[str] = field(default_factory=list)  # 判断依据列表
    matched_signals: List[str] = field(default_factory=list)  # 匹配的成功标志
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_success": self.is_success,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "matched_signals": self.matched_signals
        }


@dataclass
class FailureDiagnosis:
    """失败诊断结果"""
    is_failure: bool  # 是否失败
    confidence: float  # 置信度 (0-1)
    failure_reason: str = ""  # 失败原因
    matched_signals: List[str] = field(default_factory=list)  # 匹配的失败信号
    severity: str = "medium"  # 严重程度 (high/medium/low)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_failure": self.is_failure,
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "matched_signals": self.matched_signals,
            "severity": self.severity
        }


@dataclass
class ConfirmationResult:
    """二次确认结果"""
    confirmed: bool  # 是否确认
    original_diagnosis: DiagnosisType  # 原始诊断
    final_diagnosis: DiagnosisType  # 最终诊断
    is_misjudgment: bool = False  # 是否误判
    misjudgment_reason: str = ""  # 误判原因
    confidence: float = 0.0  # 最终置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "confirmed": self.confirmed,
            "original_diagnosis": self.original_diagnosis.value,
            "final_diagnosis": self.final_diagnosis.value,
            "is_misjudgment": self.is_misjudgment,
            "misjudgment_reason": self.misjudgment_reason,
            "confidence": self.confidence
        }


@dataclass
class ValidationResult:
    """经验验证结果"""
    is_valid: bool  # 是否有效
    confidence: float  # 置信度 (0-1)
    rejection_reason: str = ""  # 拒绝原因
    validation_checks: List[str] = field(default_factory=list)  # 验证检查项
    should_persist: bool = False  # 是否应该沉淀
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "rejection_reason": self.rejection_reason,
            "validation_checks": self.validation_checks,
            "should_persist": self.should_persist
        }


@dataclass
class Experience:
    """经验数据类"""
    experience_id: str  # 经验ID
    error_pattern: str  # 错误特征
    failure_reason: str  # 失败原因
    fix_solution: str  # 修复方案
    validation_status: str = "pending"  # 验证状态 (pending/validated/rejected)
    confidence: float = 0.0  # 置信度
    success_count: int = 0  # 成功次数
    failure_count: int = 0  # 失败次数
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experience_id": self.experience_id,
            "error_pattern": self.error_pattern,
            "failure_reason": self.failure_reason,
            "fix_solution": self.fix_solution,
            "validation_status": self.validation_status,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


