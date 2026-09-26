"""
Computer Use Agent - Windows 风格桌面 UI
========================================
基于 tkinter 的图形界面，替代终端交互方式。

功能：
  - 任务输入 + 开始/停止
  - 实时日志输出（重定向 stdout/stderr）
  - 模型预加载进度提示
  - 多任务连续运行

用法：
  python agent_ui.py
"""
import io
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk, scrolledtext

# 将项目根目录加入 sys.path（支持任意工作目录启动）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from agent.core import run_task, build_tools_schema
from vision.engine import preload_models
from vision.ocr import ocr_screen, _get_reader


class TextRedirector(io.TextIOBase):
    """将 print 输出重定向到 UI 日志队列

    替代 sys.stdout/sys.stderr，所有 print 内容经 queue 送入 UI 线程。
    """

    def __init__(self, log_queue: queue.Queue, tag: str = "stdout"):
        super().__init__()
        self._queue = log_queue
        self._tag = tag

    def write(self, s: str):
        if s:
            self._queue.put((self._tag, s))
        return len(s)

    def flush(self):
        pass


class AgentUI:
    """主窗口"""

    WINDOW_TITLE = "Computer Use Agent - 桌面自动化"
    WINDOW_SIZE = "980x720"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(760, 540)

        # 运行状态
        self._running = False
        self._stop_flag = threading.Event()
        self._worker: threading.Thread = None
        self._model_ready = False

        # 日志队列（后台线程 → UI 线程）
        self._log_queue = queue.Queue()

        # DeepSeek 客户端（延迟到首次任务初始化）
        self._client = None
        self._tools_schema = None

        self._build_ui()
        self._poll_log_queue()
        self._start_model_preload()

    # ============================================================
    # UI 构建
    # ============================================================

    def _build_ui(self):
        self.root.configure(bg="#0b1020")
        self.root.option_add("*Font", "Arial 10")

        self.title_font = ("Arial", 20, "bold")
        self.subtitle_font = ("Arial", 10)
        self.card_title_font = ("Arial", 10, "bold")
        self.pill_font = ("Arial", 9, "bold")
        self.input_font = ("Arial", 11)
        self.status_font = ("Arial", 9)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Root.TFrame", background="#0b1020")
        style.configure("Card.TFrame", background="#121a2b")
        style.configure("Panel.TFrame", background="#0f172a")
        style.configure("TEntry",
                        fieldbackground="#111827",
                        foreground="#edf2ff",
                        borderwidth=0,
                        lightcolor="#7c5cff",
                        darkcolor="#7c5cff",
                        insertcolor="#edf2ff",
                        padding=10)
        style.map("TEntry",
                  fieldbackground=[("active", "#151f31"), ("!disabled", "#111827")],
                  bordercolor=[("focus", "#7c5cff")])
        style.configure("Primary.TButton",
                        background="#7c5cff",
                        foreground="#f8f8ff",
                        borderwidth=0,
                        padding=(18, 10),
                        relief="flat")
        style.map("Primary.TButton",
                  background=[("active", "#8b6cff"), ("pressed", "#6a4ee3")],
                  foreground=[("disabled", "#dfe8ff")])
        style.configure("Secondary.TButton",
                        background="#1b2436",
                        foreground="#e7ebff",
                        borderwidth=0,
                        padding=(16, 10),
                        relief="flat")
        style.map("Secondary.TButton",
                  background=[("active", "#232f46"), ("pressed", "#182338")],
                  foreground=[("disabled", "#a9b7d4")])

        main = tk.Frame(self.root, bg="#0b1020")
        main.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        header = tk.Frame(main, bg="#121a2b", padx=18, pady=16, highlightthickness=0)
        header.pack(fill=tk.X)

        title_row = tk.Frame(header, bg="#121a2b")
        title_row.pack(fill=tk.X)

        brand = tk.Label(
            title_row, text="Computer Use Agent",
            bg="#121a2b", fg="#edf3ff",
            font=self.title_font,
            anchor="w"
        )
        brand.pack(side=tk.LEFT)

        self.runtime_badge = tk.Label(
            title_row, text=" Desktop Automation ",
            bg="#1b2436", fg="#a8c6ff",
            font=self.pill_font,
            padx=10, pady=4, bd=0, relief=tk.FLAT
        )
        self.runtime_badge.pack(side=tk.RIGHT)

        subtitle = tk.Label(
            header, text="多模态桌面智能体 · 规划 / 视觉 / 执行闭环",
            bg="#121a2b", fg="#94a3b8",
            font=self.subtitle_font,
            anchor="w"
        )
        subtitle.pack(fill=tk.X, pady=(10, 0))

        # --- 顶部：任务输入 ---
        input_card = tk.Frame(main, bg="#121a2b", padx=16, pady=14, highlightthickness=0)
        input_card.pack(fill=tk.X, pady=(14, 12))

        task_label = tk.Label(
            input_card, text="任务描述",
            bg="#121a2b", fg="#c7d2fe",
            font=self.card_title_font,
            anchor="w"
        )
        task_label.pack(anchor="w", pady=(0, 8))

        task_row = tk.Frame(input_card, bg="#121a2b")
        task_row.pack(fill=tk.X)

        self.task_var = tk.StringVar()
        self.task_entry = ttk.Entry(
            task_row, textvariable=self.task_var, font=self.input_font, style="TEntry"
        )
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.task_entry.bind("<Return>", lambda e: self.start_task())

        self.start_btn = ttk.Button(task_row, text="开始任务", command=self.start_task, style="Primary.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=(12, 8))

        self.stop_btn = ttk.Button(task_row, text="停止", command=self.stop_task, state=tk.DISABLED, style="Secondary.TButton")
        self.stop_btn.pack(side=tk.LEFT)

        # --- 中部：日志输出 ---
        log_card = tk.Frame(main, bg="#121a2b", padx=14, pady=14, highlightthickness=0)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_card, bg="#121a2b")
        log_header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(log_header, text="执行日志", bg="#121a2b", fg="#edf2ff",
                 font=self.card_title_font).pack(side=tk.LEFT)

        self.log_text = scrolledtext.ScrolledText(
            log_card, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 10), bg="#0d1728", fg="#e5edf8",
            insertbackground="#e5edf8",
            highlightthickness=0, borderwidth=0, padx=10, pady=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 日志颜色 tag
        self.log_text.tag_configure("info", foreground="#e5edf8")
        self.log_text.tag_configure("success", foreground="#61e88a")
        self.log_text.tag_configure("error", foreground="#ff7d7d")
        self.log_text.tag_configure("step", foreground="#7dc2ff", font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("tool", foreground="#d7a9ff")
        self.log_text.tag_configure("result", foreground="#b7c6dd")

        # --- 底部：状态栏 ---
        status_frame = tk.Frame(main, bg="#0b1020")
        status_frame.pack(fill=tk.X, pady=(12, 0))

        self.model_status = tk.StringVar(value="⏳ 模型初始化中...")
        self.task_status = tk.StringVar(value="就绪")

        self.model_status_label = tk.Label(
            status_frame, textvariable=self.model_status,
            bg="#111827", fg="#d5ddf6",
            font=self.status_font,
            padx=12, pady=6, bd=0, relief=tk.FLAT
        )
        self.model_status_label.pack(side=tk.LEFT)

        self.task_status_label = tk.Label(
            status_frame, textvariable=self.task_status,
            bg="#1d2a3d", fg="#7dd3fc",
            font=self.pill_font,
            padx=12, pady=6, bd=0, relief=tk.FLAT
        )
        self.task_status_label.pack(side=tk.RIGHT)

    # ============================================================
    # 日志显示
    # ============================================================

    def _poll_log_queue(self):
        """定期检查日志队列并刷新 UI"""
        try:
            while True:
                tag, msg = self._log_queue.get_nowait()
                self._append_log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log_queue)

    def _append_log(self, msg: str, tag: str = "info"):
        """向日志区追加一行（带颜色标记）"""
        # 行首简单着色
        text_tag = "info"
        stripped = msg.strip()
        if not stripped:
            text_tag = "info"
        elif stripped.startswith("[Step"):
            text_tag = "step"
        elif "✅" in stripped or "成功" in stripped or "✓" in stripped:
            text_tag = "success"
        elif "❌" in stripped or "失败" in stripped or "错误" in stripped or "异常" in stripped:
            text_tag = "error"
        elif "|T1-" in stripped or "|T2-" in stripped:
            text_tag = "tool"
        elif stripped.startswith("<-") or "坐标" in stripped:
            text_tag = "result"

        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg, text_tag)
        # 自动滚动到底部
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _safe_log(self, msg: str, tag: str = "info"):
        """从后台线程安全打印（通过队列）"""
        self._log_queue.put((tag, msg + "\n"))

    # ============================================================
    # 模型预加载（后台线程，一次性）
    # ============================================================

    def _start_model_preload(self):
        def _preload():
            try:
                self._safe_log("[初始化] 正在预加载视觉模型（首次约 30-60 秒）...")
                preload_models()
                self._safe_log("[初始化] ✓ Florence-2 / DINO 已就绪")
                try:
                    _get_reader()  # 加载 EasyOCR
                    ocr_screen()
                    self._safe_log("[初始化] ✓ OCR 已就绪")
                except Exception as e:
                    self._safe_log(f"[初始化] OCR 预热跳过: {e}", "error")
                self._model_ready = True
                self.root.after(0, lambda: self.model_status.set("✅ 模型就绪"))
            except Exception as e:
                self._model_ready = False
                self.root.after(0, lambda: self.model_status.set("⚠️ 模型加载失败"))
                self._safe_log(f"[初始化] 模型加载失败: {e}", "error")

        threading.Thread(target=_preload, daemon=True).start()

    # ============================================================
    # 任务控制
    # ============================================================

    def start_task(self):
        """开始执行任务"""
        task = self.task_var.get().strip()
        if not task:
            self._safe_log("[提示] 请输入任务描述", "error")
            return
        if self._running:
            self._safe_log("[提示] 当前任务仍在运行，请先停止", "error")
            return

        # 初始化客户端（懒加载）
        try:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
                )
                self._tools_schema = build_tools_schema()
        except Exception as e:
            self._safe_log(f"[错误] DeepSeek 客户端初始化失败: {e}", "error")
            return

        # 重置停止标志 + 启动后台线程
        self._stop_flag.clear()
        self._running = True
        self._set_controls_running(True)
        self.task_status.set("▶ 执行中")

        self._worker = threading.Thread(
            target=self._run_task_worker, args=(task,), daemon=True
        )
        self._worker.start()

    def _run_task_worker(self, task: str):
        """后台线程：执行 run_task，捕获输出"""
        # 重定向 stdout/stderr 到日志队列
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TextRedirector(self._log_queue, "stdout")
        sys.stderr = TextRedirector(self._log_queue, "stderr")
        try:
            success, result = run_task(
                self._client, self._tools_schema, task,
                confirm_powershell=self._confirm_powershell_command,
            )
        except Exception as e:
            success, result = False, f"任务异常: {type(e).__name__}: {e}"
            self._safe_log(f"[致命错误] {e}", "error")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        self._safe_log(f"\n{'=' * 60}")
        self._safe_log(f"任务结果: {'✅ 成功' if success else '❌ 失败'}")
        self._safe_log(f"{'=' * 60}\n")

        # 更新 UI（主线程）
        self.root.after(0, lambda: self._on_task_finished(success, result))

    def _confirm_powershell_command(self, command: str) -> bool:
        """Ask for approval on the UI thread before a risky command executes."""
        response = []
        completed = threading.Event()

        def ask():
            try:
                response.append(messagebox.askyesno(
                    "确认高风险 PowerShell 命令",
                    "该命令可能修改或删除数据/系统设置。\n\n"
                    f"{command[:1500]}\n\n仍要执行吗？",
                    parent=self.root,
                ))
            finally:
                completed.set()

        self.root.after(0, ask)
        completed.wait()
        return bool(response and response[0])

    def stop_task(self):
        """请求停止当前任务"""
        if not self._running:
            return
        self._stop_flag.set()
        self._safe_log("[停止] 已请求停止，等待当前步骤完成...", "error")
        self.task_status.set("⏹ 停止中")

    def _on_task_finished(self, success: bool, result: str):
        """任务结束回调（UI 线程）"""
        self._running = False
        self._stop_flag.clear()
        self._set_controls_running(False)
        self.task_status.set("✅ 已完成" if success else "❌ 已结束")

    def _set_controls_running(self, running: bool):
        """切换控件可用状态"""
        if running:
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.task_entry.configure(state=tk.DISABLED)
        else:
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.task_entry.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = AgentUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
