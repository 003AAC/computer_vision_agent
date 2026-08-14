"""
Computer Use Agent - 主入口
===========================
用法：
  python -m agent.main "帮我打开记事本并输入hello"
"""
import sys

from agent.core import run_agent


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Computer Use Agent - 桌面自动化")
    parser.add_argument("task", nargs="?", help="要完成的任务描述")

    args = parser.parse_args()

    if not args.task:
        parser.print_help()
        return

    success, result = run_agent(args.task)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
