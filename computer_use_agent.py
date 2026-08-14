"""
Computer Use Agent - 桌面自动化（持续运行）
============================================
架构：
  主脑（DeepSeek，云端API）：任务拆解 + 执行决策
  视觉模型（本地部署）：看屏幕、找坐标（只负责看，不做决策）
  执行层：pyautogui 鼠标键盘 + PowerShell 命令

  迭代式视觉：
    - 全屏扫描 → 分块扫描 → 局部放大 → 精确定位
    - 由 DeepSeek 决定何时放大、何时分块

  两层降级：
    Tier 1 - 多模态操作（视觉+键鼠，优先使用）
    Tier 2 - PowerShell 命令（视觉无法完成时回退）

用法：
  python computer_use_agent.py "帮我打开记事本"    # 单次任务
  python computer_use_agent.py                      # 持续交互模式
"""
import sys

from agent.core import run_agent


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Computer Use Agent - 桌面自动化（持续运行）")
    parser.add_argument("task", nargs="?", help="要完成的任务描述（可选，不提供则进入交互模式）")

    args = parser.parse_args()

    success, result = run_agent(args.task)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
