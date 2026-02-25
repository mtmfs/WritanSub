"""单独语音识别页"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from writansub.core.types import MEDIA_FILETYPES, SRT_FILETYPES, LANGUAGES
from writansub.core.whisper import transcribe_to_srt
from writansub.config import PP_DEFAULTS, PARAM_DEFS, load_pp_config, save_pp_config
from writansub.gui.widgets import (
    ToolTip,
    make_log_area, make_progress_area,
    update_progress, reset_progress,
    gui_log, clear_log,
)


class WhisperTab:
    """Tab 2: 独立 Whisper 语音识别"""

    def __init__(self, parent: ttk.Frame):
        self.parent = parent

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        settings = ttk.Frame(container, padding=12)
        settings.pack(fill="x")

        card_file = ttk.LabelFrame(settings, text="文件", padding=8)
        card_file.pack(fill="x", pady=(0, 8))

        r = 0
        ttk.Label(card_file, text="媒体文件").grid(row=r, column=0, sticky="w", pady=3)
        self.media_var = tk.StringVar()
        self.media_var.trace_add("write", self._on_media_changed)
        ttk.Entry(card_file, textvariable=self.media_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(card_file, text="浏览", width=6, command=self._browse_media).grid(
            row=r, column=2, pady=3)
        r += 1

        ttk.Label(card_file, text="输出 SRT").grid(row=r, column=0, sticky="w", pady=3)
        self.output_var = tk.StringVar()
        ttk.Entry(card_file, textvariable=self.output_var).grid(
            row=r, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(card_file, text="浏览", width=6, command=self._browse_output).grid(
            row=r, column=2, pady=3)
        card_file.columnconfigure(1, weight=1)

        card_param = ttk.LabelFrame(settings, text="参数", padding=8)
        card_param.pack(fill="x", pady=(0, 8))

        param_row = ttk.Frame(card_param)
        param_row.pack(anchor="w")
        ttk.Label(param_row, text="语言").pack(side="left")
        self.lang_var = tk.StringVar(value="ja")
        ttk.Combobox(
            param_row, textvariable=self.lang_var, values=LANGUAGES,
            state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))
        ttk.Label(param_row, text="设备").pack(side="left")
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            param_row, textvariable=self.device_var,
            values=["cuda", "cpu"], state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))
        lbl_wc = ttk.Label(param_row, text="置信阈值")
        lbl_wc.pack(side="left")
        tip_text = PARAM_DEFS["word_conf_threshold"].get("tip", "")
        if tip_text:
            q = tk.Label(
                param_row, text="?", fg="#4a86c8", cursor="question_arrow",
                font=("TkDefaultFont", 9, "bold"),
            )
            q.pack(side="left", padx=(2, 0))
            ToolTip(q, tip_text)
        cfg = load_pp_config()
        wc_var = tk.DoubleVar(
            value=cfg.get("word_conf_threshold",
                          PP_DEFAULTS["word_conf_threshold"]),
        )
        ttk.Spinbox(
            param_row, textvariable=wc_var,
            from_=0.0, to=1.0, increment=0.05, width=6, format="%.2f",
        ).pack(side="left", padx=(4, 0))

        def _on_wc_change(*_):
            try:
                current = load_pp_config()
                current["word_conf_threshold"] = round(wc_var.get(), 2)
                save_pp_config(current)
            except (tk.TclError, ValueError):
                pass

        wc_var.trace_add("write", _on_wc_change)

        self._cond_prev_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            param_row, text="上文关联",
            variable=self._cond_prev_var,
        ).pack(side="left", padx=(16, 0))
        cond_tip = tk.Label(
            param_row, text="?", fg="#4a86c8", cursor="question_arrow",
            font=("TkDefaultFont", 9, "bold"),
        )
        cond_tip.pack(side="left", padx=(2, 0))
        ToolTip(cond_tip, "开启时，前一句识别结果作为下一句的上下文，\n"
                          "提高连贯性但可能传播错误。\n"
                          "关闭可防止幻觉扩散。")

        self._wc_vars = {"word_conf_threshold": wc_var}

        action_bar = ttk.Frame(container, padding=(12, 6))
        action_bar.pack(fill="x")

        btn_row = ttk.Frame(action_bar)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(
            btn_row, text="开始识别", command=self._start,
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

    def _browse_media(self):
        path = filedialog.askopenfilename(filetypes=MEDIA_FILETYPES)
        if path:
            self.media_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".srt", filetypes=SRT_FILETYPES,
        )
        if path:
            self.output_var.set(path)

    def _on_media_changed(self, *_args):
        media = self.media_var.get().strip()
        if media and not self.output_var.get().strip():
            self.output_var.set(os.path.splitext(media)[0] + ".srt")

    def _log(self, msg: str):
        gui_log(self.log_text, msg)

    def _progress(self, pct: float, msg: str):
        update_progress(self.parent, self._progress_bar,
                        self._status_label, pct, msg)

    def _start(self):
        media = self.media_var.get().strip()
        if not media or not os.path.isfile(media):
            self._log("请先选择有效的媒体文件")
            return

        output = self.output_var.get().strip()
        if not output:
            output = os.path.splitext(media)[0] + ".srt"
            self.output_var.set(output)

        self.start_btn.configure(state="disabled")
        clear_log(self.log_text)
        reset_progress(self.parent, self._progress_bar, self._status_label)

        lang = self.lang_var.get()
        device = self.device_var.get()
        wc = self._wc_vars["word_conf_threshold"].get()
        cond_prev = self._cond_prev_var.get()

        thread = threading.Thread(
            target=self._run_whisper,
            args=(media, output, lang, device, wc, cond_prev),
            daemon=True,
        )
        thread.start()

    def _run_whisper(self, media: str, output: str, lang: str,
                     device: str, wc_threshold: float,
                     cond_prev: bool = True):
        try:
            srt_path = transcribe_to_srt(
                media, lang=lang, device=device, log_callback=self._log,
                progress_callback=self._progress,
                word_conf_threshold=wc_threshold,
                condition_on_previous_text=cond_prev,
            )
            if os.path.abspath(srt_path) != os.path.abspath(output):
                import shutil
                shutil.move(srt_path, output)
            self._progress(1.0, "识别完成")
        except Exception as e:
            self._log(f"出错: {e}")
        finally:
            self.parent.after(
                0, lambda: self.start_btn.configure(state="normal"),
            )
