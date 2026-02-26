"""一键流水线页：批量处理，两阶段模型管理"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, List, Optional

from writansub.core.types import MEDIA_FILETYPES, LANGUAGES
from writansub.core.srt_io import parse_srt, write_srt, mark_low_align_in_review
from writansub.core.whisper import transcribe_to_srt, _keep_alive
from writansub.core.alignment import load_audio, run_alignment, post_process, init_model
from writansub.pipeline import PipelineOrchestrator
from writansub.gui.widgets import (
    TextRedirector, ScrollableFrame, ToolTip,
    make_log_area, make_progress_area,
    update_progress, reset_progress,
    gui_log, clear_log, build_params_grid,
)


class PipelineTab:
    """Tab 1: 一键流水线 (Whisper → ForceAlign)"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "word_conf_threshold", "align_conf_threshold",
        "min_duration",
    ]

    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        self._orchestrator: Optional[PipelineOrchestrator] = None
        self._running = False
        self._cancelled = False
        self._media_files: List[str] = []

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        scrollable = ScrollableFrame(container)
        scrollable.pack(fill="both", expand=True)

        inner = ttk.Frame(scrollable.inner, padding=12)
        inner.pack(fill="x")

        card_file = ttk.LabelFrame(inner, text="输入文件", padding=8)
        card_file.pack(fill="x", pady=(0, 8))

        file_body = ttk.Frame(card_file)
        file_body.pack(fill="x")
        self.file_listbox = tk.Listbox(file_body, height=4, selectmode="extended")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        list_scroll = ttk.Scrollbar(
            file_body, orient="vertical", command=self.file_listbox.yview,
        )
        list_scroll.pack(side="left", fill="y")
        self.file_listbox.configure(yscrollcommand=list_scroll.set)

        btn_col = ttk.Frame(file_body)
        btn_col.pack(side="left", padx=(6, 0))
        ttk.Button(btn_col, text="添加", width=6, command=self._add_files).pack(pady=1)
        ttk.Button(btn_col, text="移除", width=6, command=self._remove_files).pack(pady=1)
        ttk.Button(btn_col, text="清空", width=6, command=self._clear_files).pack(pady=1)

        card_retention = ttk.LabelFrame(inner, text="输出中间文件", padding=8)
        card_retention.pack(fill="x", pady=(0, 8))

        ret_row = ttk.Frame(card_retention)
        ret_row.pack(anchor="w")
        self._retention_vars: Dict[str, tk.BooleanVar] = {}
        v1 = tk.BooleanVar(value=False)
        self._retention_vars["whisper"] = v1
        ttk.Checkbutton(ret_row, text="Whisper 原始 SRT", variable=v1).pack(
            side="left", padx=(0, 16))
        v2 = tk.BooleanVar(value=False)
        self._retention_vars["force_align"] = v2
        ttk.Checkbutton(ret_row, text="打轴 SRT", variable=v2).pack(side="left")

        card_basic = ttk.LabelFrame(inner, text="基本设置", padding=8)
        card_basic.pack(fill="x", pady=(0, 8))

        basic_row = ttk.Frame(card_basic)
        basic_row.pack(anchor="w")
        ttk.Label(basic_row, text="语言").pack(side="left")
        self.lang_var = tk.StringVar(value="ja")
        ttk.Combobox(
            basic_row, textvariable=self.lang_var, values=LANGUAGES,
            state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))
        ttk.Label(basic_row, text="Whisper 设备").pack(side="left")
        self.whisper_device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            basic_row, textvariable=self.whisper_device_var,
            values=["cuda", "cpu"], state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))
        ttk.Label(basic_row, text="打轴设备").pack(side="left")
        self.align_device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            basic_row, textvariable=self.align_device_var,
            values=["cuda", "cpu"], state="readonly", width=6,
        ).pack(side="left", padx=(4, 16))

        self._cond_prev_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            basic_row, text="上文关联",
            variable=self._cond_prev_var,
        ).pack(side="left")
        cond_tip = tk.Label(
            basic_row, text="?", fg="#4a86c8", cursor="question_arrow",
            font=("TkDefaultFont", 9, "bold"),
        )
        cond_tip.pack(side="left", padx=(2, 0))
        ToolTip(cond_tip, "开启时，前一句识别结果作为下一句的上下文，\n"
                          "提高连贯性但可能传播错误。\n"
                          "关闭可防止幻觉扩散。")

        card_pp = ttk.LabelFrame(inner, text="后处理参数", padding=8)
        card_pp.pack(fill="x", pady=(0, 8))
        self._pp_vars = build_params_grid(card_pp, self._PP_KEYS)

        action_bar = ttk.Frame(container, padding=(12, 6))
        action_bar.pack(fill="x")

        btn_row = ttk.Frame(action_bar)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(
            btn_row, text="开始处理", command=self._start,
        )
        self.start_btn.pack(side="right", padx=(4, 0))
        self.cancel_btn = ttk.Button(
            btn_row, text="取消", command=self._cancel, state="disabled",
        )
        self.cancel_btn.pack(side="right")

        self._progress_bar, self._status_label = make_progress_area(container)
        ttk.Separator(container, orient="horizontal").pack(fill="x")

        log_frame = ttk.Frame(container, padding=(12, 4, 12, 12))
        log_frame.pack(fill="both", expand=True)
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill="both", expand=True)
        self.log_text, scrollbar = make_log_area(log_container)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def set_media_path(self, path: str):
        """供外部（如 sys.argv）设置媒体路径"""
        if path and path not in self._media_files:
            self._media_files.append(path)
            self.file_listbox.insert(tk.END, os.path.basename(path))

    def _add_files(self):
        paths = filedialog.askopenfilenames(filetypes=MEDIA_FILETYPES)
        for p in paths:
            if p not in self._media_files:
                self._media_files.append(p)
                self.file_listbox.insert(tk.END, os.path.basename(p))

    def _remove_files(self):
        for i in reversed(self.file_listbox.curselection()):
            self.file_listbox.delete(i)
            del self._media_files[i]

    def _clear_files(self):
        self.file_listbox.delete(0, tk.END)
        self._media_files.clear()

    def _log(self, msg: str):
        gui_log(self.log_text, msg)

    def _progress(self, pct: float, msg: str):
        update_progress(self.parent, self._progress_bar,
                        self._status_label, pct, msg)

    def _start(self):
        if not self._media_files:
            self._log("请先添加媒体文件")
            return

        self._running = True
        self._cancelled = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        clear_log(self.log_text)
        reset_progress(self.parent, self._progress_bar, self._status_label)

        lang = self.lang_var.get()
        whisper_device = self.whisper_device_var.get()
        align_device = self.align_device_var.get()
        retention = {k: v.get() for k, v in self._retention_vars.items()}
        pp = {k: v.get() for k, v in self._pp_vars.items()}
        cond_prev = self._cond_prev_var.get()

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(list(self._media_files), lang, whisper_device,
                  align_device, retention, pp, cond_prev),
            daemon=True,
        )
        thread.start()

    def _cancel(self):
        self._cancelled = True
        if self._orchestrator:
            self._orchestrator.cancel()
        self._log("正在取消...")

    def _run_pipeline(
        self, media_files: List[str], lang: str,
        whisper_device: str, align_device: str,
        retention: Dict[str, bool], pp: Dict[str, float],
        cond_prev: bool = True,
    ):
        redirector = TextRedirector(self.log_text)
        old_stdout = sys.stdout
        sys.stdout = redirector

        import gc
        import torch

        wc_threshold = pp.pop("word_conf_threshold", 0.50)
        ac_threshold = pp.pop("align_conf_threshold", 0.50)

        try:
            total = len(media_files)
            srt_results: Dict[str, str] = {}  # media_path -> srt_path

            # ── 阶段 1/2: Whisper 语音识别 ──────────────────
            self._log("── 阶段 1/2: Whisper 语音识别 ──")

            w_device = whisper_device
            if w_device == "cuda":
                try:
                    import ctranslate2
                    ctranslate2.get_supported_compute_types("cuda")
                except Exception:
                    self._log("CUDA 不可用，Whisper 回退到 CPU")
                    w_device = "cpu"

            from faster_whisper import WhisperModel
            self._log("加载 Whisper 模型...")
            whisper_model = WhisperModel("large-v3", device=w_device,
                                         compute_type="int8")

            for idx, media in enumerate(media_files, 1):
                if self._cancelled:
                    self._log("已取消")
                    break

                self._log(f"[{idx}/{total}] {os.path.basename(media)}")

                def _w_progress(pct, msg, _idx=idx):
                    overall = ((_idx - 1) + pct) / (total * 2)
                    self._progress(overall, f"[Whisper {_idx}/{total}] {msg}")

                try:
                    srt_path = transcribe_to_srt(
                        media,
                        lang=lang,
                        device=w_device,
                        log_callback=self._log,
                        progress_callback=_w_progress,
                        word_conf_threshold=wc_threshold,
                        condition_on_previous_text=cond_prev,
                        model=whisper_model,
                    )
                    srt_results[media] = srt_path
                except Exception as e:
                    self._log(f"文件处理出错: {e}")

            # 释放 Whisper 模型 — 保留引用以规避 CTranslate2 析构崩溃
            _keep_alive.append(whisper_model)
            del whisper_model
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # ── 阶段 2/2: MMS_FA 强制打轴 ──────────────────
            if srt_results and not self._cancelled:
                self._log("── 阶段 2/2: MMS_FA 强制打轴 ──")

                a_device = align_device
                if a_device == "cuda" and not torch.cuda.is_available():
                    self._log("CUDA 不可用，回退到 CPU")
                    a_device = "cpu"

                self._log("加载 MMS_FA 模型...")
                mms_bundle = init_model(a_device)

                for idx, media in enumerate(media_files, 1):
                    if self._cancelled:
                        self._log("已取消")
                        break

                    if media not in srt_results:
                        continue

                    srt_path = srt_results[media]
                    self._log(f"[{idx}/{total}] {os.path.basename(media)}")

                    def _a_progress(pct, msg, _idx=idx):
                        overall = (total + (_idx - 1) + pct) / (total * 2)
                        self._progress(overall, f"[打轴 {_idx}/{total}] {msg}")

                    try:
                        waveform = load_audio(media)
                        subs = parse_srt(srt_path, lang=lang)
                        self._log(f"字幕 {len(subs)} 条，设备: {a_device}")

                        aligned = run_alignment(
                            waveform, subs, device=a_device,
                            progress_callback=_a_progress,
                            model_bundle=mms_bundle,
                        )

                        final = post_process(aligned, **pp)

                        base = srt_path.rsplit('.', 1)[0]
                        output_path = f"{base}_aligned.srt"
                        write_srt(final, output_path)

                        if ac_threshold > 0:
                            low_align = {
                                s.index for s in final
                                if s.score < ac_threshold
                            }
                            if low_align:
                                mark_low_align_in_review(base, low_align)
                                self._log(
                                    f"低置信对齐 {len(low_align)} 句，已标记"
                                )

                        # 清理中间文件
                        if not retention.get("whisper", False):
                            if srt_path and os.path.isfile(srt_path):
                                try:
                                    os.remove(srt_path)
                                    self._log(
                                        f"已清理: {os.path.basename(srt_path)}"
                                    )
                                except OSError:
                                    pass

                    except Exception as e:
                        self._log(f"文件处理出错: {e}")

                # 释放 MMS_FA 模型
                del mms_bundle
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if not self._cancelled:
                self._progress(1.0, "全部完成")
                self._log(f"全部完成! 共处理 {total} 个文件")
        except Exception as e:
            self._log(f"出错: {e}")
        finally:
            sys.stdout = old_stdout
            self._running = False
            self._orchestrator = None
            self.parent.after(0, lambda: self.start_btn.configure(state="normal"))
            self.parent.after(0, lambda: self.cancel_btn.configure(state="disabled"))
