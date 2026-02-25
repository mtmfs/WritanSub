"""通用 GUI 控件：日志区、进度条、滚动容器、气泡提示、参数面板"""

import io
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

from writansub.config import (
    PP_DEFAULTS, PARAM_DEFS,
    load_pp_config, save_pp_config,
)


class TextRedirector(io.TextIOBase):
    """将 print 输出重定向到 Tkinter Text 控件"""

    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, s: str):
        self.text_widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, s)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


def make_log_area(parent, height: int = 8) -> Tuple[tk.Text, ttk.Scrollbar]:
    """创建日志文本区域 + 滚动条，返回 (text_widget, scrollbar)"""
    log_text = tk.Text(parent, height=height, state="disabled", wrap="word")
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=scrollbar.set)
    return log_text, scrollbar


def make_progress_area(parent) -> Tuple[ttk.Progressbar, ttk.Label]:
    """创建进度条 + 状态标签，返回 (progressbar, status_label)"""
    frame = ttk.Frame(parent, padding=(12, 4))
    frame.pack(fill="x")
    status = ttk.Label(frame, text="就绪")
    status.pack(side="left")
    pct_label = ttk.Label(frame, text="")
    pct_label.pack(side="right")
    bar = ttk.Progressbar(parent, mode="determinate", maximum=100)
    bar.pack(fill="x", padx=12, pady=(0, 4))
    bar._pct_label = pct_label  # type: ignore[attr-defined]
    return bar, status


def update_progress(widget: tk.Widget, bar: ttk.Progressbar,
                    status_label: ttk.Label, pct: float, msg: str):
    """线程安全地更新进度条和状态标签"""
    def _do():
        bar['value'] = pct * 100
        status_label.configure(text=msg)
        pct_label = getattr(bar, '_pct_label', None)
        if pct_label:
            pct_label.configure(text=f"{int(pct * 100)}%" if pct > 0 else "")
    widget.after(0, _do)


def reset_progress(widget: tk.Widget, bar: ttk.Progressbar,
                   status_label: ttk.Label):
    """重置进度条"""
    def _do():
        bar['value'] = 0
        status_label.configure(text="就绪")
        pct_label = getattr(bar, '_pct_label', None)
        if pct_label:
            pct_label.configure(text="")
    widget.after(0, _do)


def gui_log(text_widget: tk.Text, msg: str):
    """线程安全地向 Text 控件追加一行日志"""
    def _append():
        text_widget.configure(state="normal")
        text_widget.insert(tk.END, msg + "\n")
        text_widget.see(tk.END)
        text_widget.configure(state="disabled")
    text_widget.after(0, _append)


def clear_log(text_widget: tk.Text):
    """清空日志区域"""
    text_widget.configure(state="normal")
    text_widget.delete("1.0", tk.END)
    text_widget.configure(state="disabled")


class ScrollableFrame(ttk.Frame):
    """可垂直滚动的容器，内容放入 self.inner"""

    def __init__(self, parent):
        super().__init__(parent)
        self._canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._canvas_window = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._canvas_window, width=event.width)

    def _bind_mousewheel(self, _event):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(-1 * (event.delta // 120), "units")


class ToolTip:
    """鼠标悬停提示框"""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9),
        ).pack()

    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def build_params_grid(
    parent, keys: List[str],
) -> Dict[str, tk.DoubleVar]:
    """
    在 parent 中创建 2 列参数网格。

    Args:
        parent: 父控件 (LabelFrame 等)
        keys: 要显示的参数 key 列表 (对应 PARAM_DEFS)

    Returns:
        {key: DoubleVar} 变量字典，值变化时自动保存到配置文件
    """
    cfg = load_pp_config()
    pp: Dict[str, tk.DoubleVar] = {}

    for i, key in enumerate(keys):
        defn = PARAM_DEFS[key]
        r = i // 2
        c = (i % 2) * 2

        lf = ttk.Frame(parent)
        lf.grid(row=r, column=c, sticky="e", padx=(0, 4), pady=3)
        ttk.Label(lf, text=defn["label"]).pack(side="left")
        if defn.get("tip"):
            q = tk.Label(
                lf, text="?", fg="#4a86c8", cursor="question_arrow",
                font=("TkDefaultFont", 9, "bold"),
            )
            q.pack(side="left", padx=(2, 0))
            ToolTip(q, defn["tip"])

        pp[key] = tk.DoubleVar(value=cfg.get(key, PP_DEFAULTS[key]))
        ttk.Spinbox(
            parent, textvariable=pp[key],
            from_=defn["from"], to=defn["to"], increment=defn["inc"],
            width=8, format="%.2f",
        ).grid(row=r, column=c + 1, sticky="w", padx=(0, 16), pady=3)

    def _on_change(*_):
        try:
            current = load_pp_config()
            current.update({k: round(v.get(), 2) for k, v in pp.items()})
            save_pp_config(current)
        except (tk.TclError, ValueError):
            pass

    for var in pp.values():
        var.trace_add("write", _on_change)

    return pp
