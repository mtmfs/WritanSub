"""单独强制打轴页"""

import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict

from writansub.core.types import AUDIO_FILETYPES, SRT_FILETYPES, LANGUAGES
from writansub.core.srt_io import parse_srt, write_srt
from writansub.core.alignment import load_audio, run_alignment, post_process
from writansub.gui.widgets import (
    TextRedirector,
    make_log_area, make_progress_area,
    update_progress, reset_progress,
    gui_log, clear_log, build_params_grid,
)


class AlignmentTab:
    """Tab 3: 独立 MMS_FA 强制打轴"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "min_duration",
    ]

    def __init__(self, parent: ttk.Frame):
        self.parent = parent

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        settings = ttk.Frame(container, padding=12)
        settings.pack(fill="x")

        card_file = ttk.LabelFrame(settings, text="文件", padding=8)
        card_file.pack(fill="x", pady=(0, 8))

        r = 0
        ttk.Label(card_file, text="音频文件").grid(row=r, column=0, sticky="w", pady=3)
        self.audio_var = tk.StringVar()
        ttk.Entry(card_file, textvariable=self.audio_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(card_file, text="浏览", width=6, command=self._browse_audio).grid(
            row=r, column=2, pady=3)
        r += 1

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

        card_param = ttk.LabelFrame(settings, text="参数", padding=8)
        card_param.pack(fill="x", pady=(0, 8))

        basic_row = ttk.Frame(card_param)
        basic_row.pack(anchor="w", pady=(0, 6))
        ttk.Label(basic_row, text="语言").pack(side="left")
        self.lang_var = tk.StringVar(value="ja")
        ttk.Combobox(
            basic_row, textvariable=self.lang_var, values=LANGUAGES,
            state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))
        ttk.Label(basic_row, text="设备").pack(side="left")
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            basic_row, textvariable=self.device_var,
            values=["cuda", "cpu"], state="readonly", width=6,
        ).pack(side="left", padx=(4, 0))

        pp_frame = ttk.Frame(card_param)
        pp_frame.pack(fill="x")
        self._pp_vars = build_params_grid(pp_frame, self._PP_KEYS)

        action_bar = ttk.Frame(container, padding=(12, 6))
        action_bar.pack(fill="x")

        btn_row = ttk.Frame(action_bar)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(
            btn_row, text="开始对齐", command=self._start,
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

    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="选择音频文件", filetypes=AUDIO_FILETYPES,
        )
        if path:
            self.audio_var.set(path)

    def _browse_srt(self):
        path = filedialog.askopenfilename(
            title="选择 SRT 字幕", filetypes=SRT_FILETYPES,
        )
        if path:
            self.srt_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="保存输出 SRT",
            defaultextension=".srt", filetypes=SRT_FILETYPES,
        )
        if path:
            self.out_var.set(path)

    def _on_srt_changed(self, *_args):
        srt = self.srt_var.get().strip()
        if srt and not self.out_var.get().strip():
            base = srt.rsplit(".", 1)[0]
            self.out_var.set(f"{base}_aligned.srt")

    def _log(self, msg: str):
        gui_log(self.log_text, msg)

    def _progress(self, pct: float, msg: str):
        update_progress(self.parent, self._progress_bar,
                        self._status_label, pct, msg)

    def _start(self):
        audio = self.audio_var.get().strip()
        srt = self.srt_var.get().strip()
        output = self.out_var.get().strip()

        if not audio:
            self._log("请选择音频文件")
            return
        if not srt:
            self._log("请选择字幕文件")
            return
        if not output:
            base = srt.rsplit(".", 1)[0]
            output = f"{base}_aligned.srt"
            self.out_var.set(output)

        self.start_btn.configure(state="disabled")
        clear_log(self.log_text)
        reset_progress(self.parent, self._progress_bar, self._status_label)

        pp = {k: v.get() for k, v in self._pp_vars.items()}
        lang = self.lang_var.get()
        thread = threading.Thread(
            target=self._run_alignment,
            args=(audio, srt, output, self.device_var.get(), pp, lang),
            daemon=True,
        )
        thread.start()

    def _run_alignment(self, audio: str, srt: str, output: str,
                       device: str, pp: Dict[str, float],
                       lang: str = "ja"):
        import torch

        redirector = TextRedirector(self.log_text)
        old_stdout = sys.stdout
        sys.stdout = redirector

        try:
            if device == "cuda" and not torch.cuda.is_available():
                self._log("CUDA 不可用，回退到 CPU")
                device = "cpu"

            self._progress(0.0, "加载音频...")
            waveform = load_audio(audio)

            self._progress(0.05, "解析字幕...")
            subs = parse_srt(srt, lang=lang)
            self._log(f"字幕条数: {len(subs)}")

            self._progress(0.1, "对齐中...")
            aligned = run_alignment(waveform, subs, device=device,
                                    progress_callback=lambda p, m: self._progress(0.1 + p * 0.85, m))

            self._progress(0.95, "后处理...")
            pp.pop("align_conf_threshold", None)
            final = post_process(aligned, **pp)

            write_srt(final, output)
            self._progress(1.0, "对齐完成")
            self._log(f"完成! 输出: {output}")
        except Exception as e:
            self._log(f"出错: {e}")
        finally:
            sys.stdout = old_stdout
            self.parent.after(
                0, lambda: self.start_btn.configure(state="normal"),
            )
