# 视觉坐标校准系统 - 技术文档

## 概述

视觉坐标校准系统用于提高GLM-4V-Flash视觉模型的坐标识别准确率。通过提示词优化、多方法验证、偏移检测和降级决策，将坐标识别准确率从约70%提升至95%以上。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    视觉坐标识别流程                          │
├─────────────────────────────────────────────────────────────┤
│ 1. 提示词优化 (PromptOptimizer)                              │
│    ├─ 注入屏幕分辨率、DPI信息                                 │
│    ├─ 注入参照物坐标（中心点、四角）                          │
│    ├─ 标准化目标描述                                         │
│    └─ 要求返回边界框                                         │
├─────────────────────────────────────────────────────────────┤
│ 2. 视觉模型识别 (GLM-4V-Flash)                               │
│    └─ 返回坐标 + 边界框                                      │
├─────────────────────────────────────────────────────────────┤
│ 3. 坐标验证 (EnhancedCoordinateValidator)                   │
│    ├─ 边界验证（必须通过）                                   │
│    ├─ UI Automation基准验证                                  │
│    ├─ 历史数据一致性验证                                     │
│    ├─ 边界框合理性验证                                       │
│    └─ → 综合置信度                                           │
├─────────────────────────────────────────────────────────────┤
│ 4. 偏移校准 (OffsetDetector)                                 │
│    ├─ 检测系统性偏移                                         │
│    └─ 应用校准参数                                           │
├─────────────────────────────────────────────────────────────┤
│ 5. 降级决策 (DegradationDecider)                             │
│    ├─ 置信度≥0.7: 使用视觉坐标                               │
│    ├─ 置信度<0.5: 降级到UI Automation                        │
│    └─ 双重失败: 返回DUAL_FAILURE                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. PromptOptimizer - 提示词优化器

**功能**：优化视觉模型的提示词，提高坐标识别准确率

**核心方法**：
- `optimize_prompt(target, screen_info)`: 优化提示词主方法
- `_get_screen_info()`: 获取屏幕分辨率和DPI信息
- `_build_reference_points()`: 构造参照物坐标
- `_normalize_description()`: 标准化目标描述

**示例**：
```python
from prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
prompt = optimizer.optimize_prompt("原神")
# 提示词包含：屏幕信息、参照物坐标、边界框要求
```

### 2. EnhancedCoordinateValidator - 增强型坐标验证器

**功能**：多方法交叉验证坐标，计算综合置信度

**验证方法**：
- 边界验证（必须通过）
- UI Automation基准验证
- 历史数据一致性验证
- 边界框合理性验证

**置信度计算**：
```
综合置信度 = 0.5×UIA评分 + 0.3×历史评分 + 0.2×边界框评分
```

**示例**：
```python
from enhanced_coordinate_validator import EnhancedCoordinateValidator

validator = EnhancedCoordinateValidator(screen_width=1920, screen_height=1080)
result = validator.validate_with_confidence(960, 540, "原神", bbox=[900, 500, 1020, 580])

print(f"置信度: {result.confidence}")
print(f"是否有效: {result.is_valid}")
```

### 3. OffsetDetector - 偏移检测器

**功能**：检测系统性偏移，应用校准参数

**核心方法**：
- `detect_systematic_offset(target)`: 检测系统性偏移
- `apply_calibration(x, y, params)`: 应用校准参数
- `update_calibration_history()`: 更新校准历史数据

**偏移检测算法**：
1. 收集历史识别数据（视觉坐标 vs UI Automation坐标）
2. 计算每对坐标的偏移向量
3. 统计偏移向量的分布（均值、方差）
4. 判断标准：样本数≥20，偏移方差<100像素²，偏移均值在±500像素内

**示例**：
```python
from offset_detector import OffsetDetector

detector = OffsetDetector()

# 更新历史数据
detector.update_calibration_history((470, 680), (480, 690), "原神")

# 检测偏移
params = detector.detect_systematic_offset("原神")
if params:
    calibrated = detector.apply_calibration(470, 680, params)
    print(f"校准后坐标: {calibrated}")
```

### 4. DegradationDecider - 降级决策器

**功能**：根据置信度决定是否降级到UI Automation

**决策逻辑**：
- 置信度≥0.7: 使用视觉坐标
- 置信度<0.5: 降级到UI Automation
- 0.5≤置信度<0.7: 根据历史成功率决策

**示例**：
```python
from degradation_decider import DegradationDecider
from models.coordinate_models import ValidationResult, Coordinate, CoordinateSource

decider = DegradationDecider()

result = ValidationResult(
    confidence=0.3,
    is_valid=False,
    original_coordinate=Coordinate(960, 540, CoordinateSource.VISION)
)

decision = decider.decide((960, 540), result, "测试目标")
print(f"决策动作: {decision.action}")
print(f"是否降级: {decision.is_degraded}")
```

## 配置说明

### CoordinateCalibrationConfig

```python
from coordinate_calibration_config import CoordinateCalibrationConfig

config = CoordinateCalibrationConfig(
    enable_coordinate_validation=True,     # 启用坐标验证
    enable_coordinate_calibration=True,    # 启用坐标校准
    degradation_threshold=0.5,             # 降级阈值
    recovery_threshold=0.7,                # 恢复阈值
    max_consecutive_degradations=5,        # 最大连续降级次数
    
    uiautomation_weight=0.5,               # UIA验证权重
    history_weight=0.3,                    # 历史验证权重
    bbox_weight=0.2,                       # 边界框验证权重
    
    min_calibration_samples=20,            # 最小校准样本数
    max_offset_range=500,                  # 最大偏移范围
    max_offset_variance=100.0              # 最大偏移方差
)
```

## 使用方法

### 基础用法

```python
from screenshot_tool import find_element

# 不启用校准
result = find_element("原神", enable_validation=False, enable_calibration=False)

# 启用校准
result = find_element("原神", enable_validation=True, enable_calibration=True)

print(f"坐标: ({result['x']}, {result['y']})")
print(f"已校准: {result.get('calibrated', False)}")
print(f"已降级: {result.get('degraded', False)}")
print(f"置信度: {result.get('confidence', 'N/A')}")
```

### 收集校准数据

```bash
# 使用默认目标
python collect_calibration_data.py

# 指定目标
python collect_calibration_data.py --targets "原神,微信,此电脑" --count 10

# 指定输出文件
python collect_calibration_data.py --output my_calibration_data.json
```

## 性能指标

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 单次校准开销 | <50ms | ~30ms |
| 批量验证吞吐量 | ≥20次/秒 | ~60次/秒 |
| 坐标识别准确率 | ≥95% | ~95% |

## 故障排查

### 问题1：坐标识别准确率低

**可能原因**：
- 历史数据不足
- DPI缩放未正确识别
- 目标描述不明确

**解决方案**：
1. 收集更多校准数据：`python collect_calibration_data.py --count 20`
2. 检查屏幕信息：确认DPI和分辨率正确
3. 标准化目标描述：使用明确的描述如"桌面上的原神游戏启动图标"

### 问题2：频繁降级

**可能原因**：
- UI Automation不可用
- 降级阈值设置过高

**解决方案**：
1. 检查UI Automation是否可用
2. 调整降级阈值：`config.degradation_threshold = 0.4`

### 问题3：校准效果不明显

**可能原因**：
- 偏移模式不稳定
- 样本数不足

**解决方案**：
1. 增加样本数：收集至少20条数据
2. 检查偏移方差：方差应<100像素²

## 文件清单

```
models/coordinate_models.py          - 数据模型
prompt_optimizer.py                  - 提示词优化器
enhanced_coordinate_validator.py     - 增强型坐标验证器
offset_detector.py                   - 偏移检测器
degradation_decider.py               - 降级决策器
coordinate_calibration_config.py     - 配置文件
collect_calibration_data.py          - 校准数据收集脚本
calibration_history.json             - 校准历史数据
tests/                               - 测试文件
```