"""
视觉模型加载配置
================
模型默认位于项目根目录下的 vismodels/ 文件夹。
基于文件位置计算路径，支持在任意目录/磁盘部署（git clone 后可直接使用）。
"""
import os

# 模型根目录（基于本文件位置计算 → 项目根的 vismodels/）
_VIS_DIR = os.path.dirname(os.path.abspath(__file__))  # vision/
VIS_MODELS_ROOT = os.path.normpath(os.path.join(_VIS_DIR, "..", "vismodels"))

# 三个模型的本地路径
F2_BASE_PATH = os.path.join(VIS_MODELS_ROOT, "base")     # Florence-2-base
GD_PATH = os.path.join(VIS_MODELS_ROOT, "dino")          # Grounding DINO
F2_LARGE_PATH = os.path.join(VIS_MODELS_ROOT, "large")   # Florence-2-large