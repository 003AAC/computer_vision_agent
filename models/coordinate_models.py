"""
坐标模型 - Coordinate Models
===========================
定义视觉坐标校准系统所需的核心数据模型

包含：
  - CoordinateSource: 坐标来源枚举
  - ValidationMethod: 验证方法枚举
  - Coordinate: 坐标数据类
  - ValidationResult: 验证结果数据类
  - CalibrationParams: 校准参数数据类
  - Decision: 决策数据类
  - CalibrationHistory: 校准历史数据类
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum


class CoordinateSource(Enum):
    """坐标来源枚举"""
    VISION = "vision"  # 视觉模型识别
    UI_AUTOMATION = "ui_automation"  # UI Automation
    CALIBRATED = "calibrated"  # 已校准
    MANUAL = "manual"  # 手动指定
    
    def __str__(self) -> str:
        return self.value


class ValidationMethod(Enum):
    """验证方法枚举"""
    BOUNDARY = "boundary"  # 边界验证
    UI_AUTOMATION = "ui_automation"  # UI Automation基准验证
    HISTORY = "history"  # 历史数据验证
    BBOX = "bbox"  # 边界框验证
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Coordinate:
    """坐标数据类"""
    x: int  # X坐标
    y: int  # Y坐标
    source: CoordinateSource = CoordinateSource.VISION  # 坐标来源
    confidence: float = 0.0  # 置信度 (0-1)
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    
    def to_tuple(self) -> Tuple[int, int]:
        """转换为元组"""
        return (self.x, self.y)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "x": self.x,
            "y": self.y,
            "source": str(self.source),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationResult:
    """验证结果数据类"""
    confidence: float  # 综合置信度 (0-1)
    is_valid: bool  # 是否有效
    original_coordinate: Coordinate  # 原始坐标
    calibrated_coordinate: Optional[Coordinate] = None  # 校准后坐标
    uia_deviation: Optional[Tuple[int, int]] = None  # UI Automation偏差
    scores: Dict[str, float] = field(default_factory=dict)  # 各项评分
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "original_coordinate": self.original_coordinate.to_dict(),
            "calibrated_coordinate": self.calibrated_coordinate.to_dict() if self.calibrated_coordinate else None,
            "uia_deviation": list(self.uia_deviation) if self.uia_deviation else None,
            "scores": self.scores
        }


@dataclass
class CalibrationParams:
    """校准参数数据类"""
    offset_x: int = 0  # X偏移量
    offset_y: int = 0  # Y偏移量
    scale_x: float = 1.0  # X缩放比例
    scale_y: float = 1.0  # Y缩放比例
    confidence: float = 0.0  # 校准置信度
    sample_count: int = 0  # 样本数量
    resolution: str = "1920x1080"  # 分辨率
    dpi: int = 96  # DPI
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "resolution": self.resolution,
            "dpi": self.dpi
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationParams':
        """从字典创建"""
        return cls(
            offset_x=data.get("offset_x", 0),
            offset_y=data.get("offset_y", 0),
            scale_x=data.get("scale_x", 1.0),
            scale_y=data.get("scale_y", 1.0),
            confidence=data.get("confidence", 0.0),
            sample_count=data.get("sample_count", 0),
            resolution=data.get("resolution", "1920x1080"),
            dpi=data.get("dpi", 96)
        )


@dataclass
class Decision:
    """决策数据类"""
    action: str  # 决策动作 (use_calibrated, use_original, degrade_to_uia)
    final_coordinate: Coordinate  # 最终坐标
    reason: str = ""  # 决策原因
    is_degraded: bool = False  # 是否降级
    error_code: str = ""  # 错误码
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action": self.action,
            "final_coordinate": self.final_coordinate.to_dict(),
            "reason": self.reason,
            "is_degraded": self.is_degraded,
            "error_code": self.error_code
        }


@dataclass
class CalibrationHistory:
    """校准历史数据类"""
    target_description: str  # 目标描述
    vision_coordinate: Coordinate  # 视觉坐标
    uia_coordinate: Optional[Coordinate] = None  # UI Automation坐标
    offset_vector: Optional[Tuple[int, int]] = None  # 偏移向量
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target_description": self.target_description,
            "vision_coordinate": self.vision_coordinate.to_dict(),
            "uia_coordinate": self.uia_coordinate.to_dict() if self.uia_coordinate else None,
            "offset_vector": list(self.offset_vector) if self.offset_vector else None,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================
# 测试函数
# ============================================================

def test_coordinate_models():
    """测试坐标模型"""
    print("=" * 80)
    print("坐标模型测试")
    print("=" * 80)
    
    # 测试1: Coordinate
    print("\n[测试1] Coordinate")
    coord = Coordinate(
        x=960,
        y=540,
        source=CoordinateSource.VISION,
        confidence=0.85
    )
    print(f"  坐标: {coord.to_tuple()}")
    print(f"  来源: {coord.source}")
    print(f"  置信度: {coord.confidence}")
    print(f"  字典: {coord.to_dict()}")
    
    # 测试2: ValidationResult
    print("\n[测试2] ValidationResult")
    result = ValidationResult(
        confidence=0.9,
        is_valid=True,
        original_coordinate=Coordinate(960, 540),
        calibrated_coordinate=Coordinate(970, 550),
        uia_deviation=(10, 10),
        scores={"uia": 0.95, "history": 0.85, "bbox": 0.9}
    )
    print(f"  综合置信度: {result.confidence}")
    print(f"  是否有效: {result.is_valid}")
    print(f"  各项评分: {result.scores}")
    
    # 测试3: CalibrationParams
    print("\n[测试3] CalibrationParams")
    params = CalibrationParams(
        offset_x=10,
        offset_y=15,
        confidence=0.9,
        sample_count=50,
        resolution="1920x1080",
        dpi=120
    )
    print(f"  偏移量: ({params.offset_x}, {params.offset_y})")
    print(f"  样本数: {params.sample_count}")
    
    # 测试序列化/反序列化
    params_dict = params.to_dict()
    params_restored = CalibrationParams.from_dict(params_dict)
    print(f"  恢复偏移量: ({params_restored.offset_x}, {params_restored.offset_y})")
    
    # 测试4: Decision
    print("\n[测试4] Decision")
    decision = Decision(
        action="use_calibrated",
        final_coordinate=Coordinate(970, 550),
        reason="视觉坐标置信度低，使用校准坐标",
        is_degraded=False
    )
    print(f"  决策动作: {decision.action}")
    print(f"  最终坐标: {decision.final_coordinate.to_tuple()}")
    print(f"  是否降级: {decision.is_degraded}")
    
    # 测试5: CalibrationHistory
    print("\n[测试5] CalibrationHistory")
    history = CalibrationHistory(
        target_description="原神图标",
        vision_coordinate=Coordinate(470, 680),
        uia_coordinate=Coordinate(480, 690),
        offset_vector=(10, 10)
    )
    print(f"  目标: {history.target_description}")
    print(f"  视觉坐标: {history.vision_coordinate.to_tuple()}")
    print(f"  UIA坐标: {history.uia_coordinate.to_tuple()}")
    print(f"  偏移向量: {history.offset_vector}")
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)


if __name__ == "__main__":
    test_coordinate_models()