"""单独 AI 翻译页"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict

from writansub.core.types import SRT_FILETYPES, TRANSLATE_TARGETS
from writansub.core.translate import translate_srt
from writansub.config import load_translate_config, save_translate_config
from writansub.gui.widgets import (
    make_log_area, make_progress_area,
    update_progress, reset_progress,
    gui_log, clear_log,
)


class TranslateTab:
    """Tab 4: 独立 AI 翻译"""

    def __init__(self, parent: ttk.Frame):
        self.parent = parent

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        settings = ttk.Frame(container, padding=12)
        settings.pack(fill="x")

        card_file = ttk.LabelFrame(settings, text="文件", padding=8)
        card_file.pack(fill="x", pady=(0, 8))

        r = 0
        ttk.Label(card_file, text="字幕文件").grid(row=r, column=0, sticky="w", pady=3)
        self.srt_var = tk.StringVar()
        self.srt_var.trace_add("write", self._on_srt_changed)
        ttk.Entry(card_file, textvariable=self.srt_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(card_file, text="浏览", width=6, command=self._browse_srt).grid(
            row=r, column=2, pady=3)
        r += 1

        ttk.Label(card_file, text="输出路径").grid(row=r, column=0, sticky="w", pady=3)
        self.out_var = tk.StringVar()
        ttk.Entry(card_file, textvariable=self.out_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(card_file, text="浏览", width=6, command=self._browse_output).grid(
            row=r, column=2, pady=3)
        card_file.columnconfigure(1, weight=1)

        card_cfg = ttk.LabelFrame(settings, text="翻译设置", padding=8)
        card_cfg.pack(fill="x", pady=(0, 8))

        t_cfg = load_translate_config()

        r = 0
        row1 = ttk.Frame(card_cfg)
        row1.grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(row1, text="目标语言").pack(side="left")
        self._target_var = tk.StringVar(value=t_cfg["target_lang"])
        ttk.Combobox(
            row1, textvariable=self._target_var,
            values=TRANSLATE_TARGETS, width=10,
        ).pack(side="left", padx=(4, 16))
        ttk.Label(row1, text="模型").pack(side="left")
        self._model_var = tk.StringVar(value=t_cfg["model"])
        ttk.Entry(row1, textvariable=self._model_var, width=20).pack(
            side="left", padx=(4, 0))
        r += 1

        ttk.Label(card_cfg, text="API 地址").grid(row=r, column=0, sticky="w", pady=3)
        self._base_var = tk.StringVar(value=t_cfg["api_base"])
        ttk.Entry(card_cfg, textvariable=self._base_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        r += 1

        ttk.Label(card_cfg, text="API Key").grid(row=r, column=0, sticky="w", pady=3)
        self._key_var = tk.StringVar(value=t_cfg["api_key"])
        ttk.Entry(card_cfg, textvariable=self._key_var, show="*").grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        card_cfg.columnconfigure(1, weight=1)

        action_bar = ttk.Frame(container, padding=(12, 6))
        action_bar.pack(fill="x")

        btn_row = ttk.Frame(action_bar)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(
            btn_row, text="开始翻译", command=self._start,
        )
        self.start_btn.pack(side="right")

        self._progress_bar, self._status_label = make_progress_area(container)
        ttk.Separator(container, orient="horizontal").pack(fill="x")

        log_frame = ttk.Frame(container, padding=(12, 4, 12, 12))
        log_frame.pack(fill="both", expand=True)
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill="both", expand=True)
        self.log_text, scrollbar = make_log_area(log_container)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _browse_srt(self):
        path = filedialog.askopenfilename(
            title="选择 SRT 字幕", filetypes=SRT_FILETYPES,
        )
        if path:
            self.srt_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="保存翻译 SRT",
            defaultextension=".srt", filetypes=SRT_FILETYPES,
        )
        if path:
            self.out_var.set(path)

    def _on_srt_changed(self, *_args):
        srt = self.srt_var.get().strip()
        if srt and not self.out_var.get().strip():
            base = os.path.splitext(srt)[0]
            self.out_var.set(f"{base}_translated.srt")

    def _log(self, msg: str):
        gui_log(self.log_text, msg)

    def _progress(self, pct: float, msg: str):
        update_progress(self.parent, self._progress_bar,
                        self._status_label, pct, msg)

    def _start(self):
        srt = self.srt_var.get().strip()
        if not srt or not os.path.isfile(srt):
            self._log("请选择有效的 SRT 文件")
            return

        output = self.out_var.get().strip()
        if not output:
            output = os.path.splitext(srt)[0] + "_translated.srt"
            self.out_var.set(output)

        cfg = {
            "target_lang": self._target_var.get(),
            "api_base": self._base_var.get(),
            "api_key": self._key_var.get(),
            "model": self._model_var.get(),
        }
        save_translate_config(cfg)

        self.start_btn.configure(state="disabled")
        clear_log(self.log_text)
        reset_progress(self.parent, self._progress_bar, self._status_label)

        thread = threading.Thread(
            target=self._run_translate,
            args=(srt, output, cfg),
            daemon=True,
        )
        thread.start()

    def _run_translate(self, srt: str, output: str, cfg: Dict[str, str]):
        try:
            result_path = translate_srt(
                srt,
                target_lang=cfg["target_lang"],
                api_base=cfg["api_base"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                log_callback=self._log,
                progress_callback=self._progress,
            )
            if os.path.abspath(result_path) != os.path.abspath(output):
                import shutil
                shutil.move(result_path, output)
            self._progress(1.0, "翻译完成")
        except Exception as e:
            self._log(f"出错: {e}")
        finally:
            self.parent.after(
                0, lambda: self.start_btn.configure(state="normal"),
            )
