"""
演示学习模块 - 包装 demonstration_learner
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径，以便导入 demonstration_learner
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from demonstration_learner import DemonstrationLearner

__all__ = ["DemonstrationLearner"]
