"""一键流水线页：批量处理，两阶段模型管理"""

import gc
import os
import sys
import threading
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QListWidget, QPushButton, QLabel, QComboBox, QCheckBox,
    QFileDialog, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Signal, QObject

from writansub.core.types import MEDIA_FILETYPES, LANGUAGES
from writansub.core.srt_io import parse_srt, write_srt, mark_low_align_in_review
from writansub.core.whisper import transcribe_to_srt, _keep_alive
from writansub.core.alignment import load_audio, run_alignment, post_process, init_model
from writansub.pipeline import PipelineOrchestrator
from writansub.gui.widgets import (
    TextRedirector, ScrollableFrame, LogWidget, ProgressWidget, build_params_grid,
)


class _PipelineSignals(QObject):
    """线程安全的信号"""
    finished = Signal()
    enable_start = Signal(bool)


class PipelineTab(QWidget):
    """Tab 1: 一键流水线 (Whisper → ForceAlign)"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "word_conf_threshold", "align_conf_threshold",
        "min_duration",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orchestrator: Optional[PipelineOrchestrator] = None
        self._running = False
        self._cancelled = False
        self._media_files: List[str] = []
        self._signals = _PipelineSignals()
        self._signals.finished.connect(self._on_finished)
        self._signals.enable_start.connect(self._set_buttons_state)

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 可滚动区域
        scrollable = ScrollableFrame()
        main_layout.addWidget(scrollable, 1)

        inner_layout = QVBoxLayout(scrollable.inner)
        inner_layout.setContentsMargins(12, 12, 12, 12)

        # 输入文件
        card_file = QGroupBox("输入文件")
        inner_layout.addWidget(card_file)
        file_layout = QHBoxLayout(card_file)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setMaximumHeight(100)
        file_layout.addWidget(self._file_list, 1)

        btn_col = QVBoxLayout()
        file_layout.addLayout(btn_col)
        btn_add = QPushButton("添加")
        btn_add.clicked.connect(self._add_files)
        btn_col.addWidget(btn_add)
        btn_remove = QPushButton("移除")
        btn_remove.clicked.connect(self._remove_files)
        btn_col.addWidget(btn_remove)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._clear_files)
        btn_col.addWidget(btn_clear)
        btn_col.addStretch()

        # 输出中间文件
        card_retention = QGroupBox("输出中间文件")
        inner_layout.addWidget(card_retention)
        ret_layout = QHBoxLayout(card_retention)
        self._chk_whisper = QCheckBox("Whisper 原始 SRT")
        ret_layout.addWidget(self._chk_whisper)
        self._chk_force_align = QCheckBox("打轴 SRT")
        ret_layout.addWidget(self._chk_force_align)
        ret_layout.addStretch()

        # 基本设置
        card_basic = QGroupBox("基本设置")
        inner_layout.addWidget(card_basic)
        basic_layout = QHBoxLayout(card_basic)

        basic_layout.addWidget(QLabel("语言"))
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_layout.addWidget(self._lang_combo)

        basic_layout.addWidget(QLabel("Whisper 设备"))
        self._whisper_device_combo = QComboBox()
        self._whisper_device_combo.addItems(["cuda", "cpu"])
        basic_layout.addWidget(self._whisper_device_combo)

        basic_layout.addWidget(QLabel("打轴设备"))
        self._align_device_combo = QComboBox()
        self._align_device_combo.addItems(["cuda", "cpu"])
        basic_layout.addWidget(self._align_device_combo)

        self._chk_cond_prev = QCheckBox("上文关联")
        self._chk_cond_prev.setChecked(True)
        self._chk_cond_prev.setToolTip(
            "开启时，前一句识别结果作为下一句的上下文，\n"
            "提高连贯性但可能传播错误。\n"
            "关闭可防止幻觉扩散。"
        )
        basic_layout.addWidget(self._chk_cond_prev)
        basic_layout.addStretch()

        # 后处理参数
        card_pp = QGroupBox("后处理参数")
        inner_layout.addWidget(card_pp)
        self._pp_vars = build_params_grid(card_pp, self._PP_KEYS)

        inner_layout.addStretch()

        # 操作按钮
        action_bar = QWidget()
        main_layout.addWidget(action_bar)
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 6, 12, 6)
        action_layout.addStretch()
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setEnabled(False)
        action_layout.addWidget(self._cancel_btn)
        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._start)
        action_layout.addWidget(self._start_btn)

        # 进度条
        self._progress = ProgressWidget()
        main_layout.addWidget(self._progress)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep)

        # 日志区
        self._log = LogWidget()
        self._log.setMinimumHeight(120)
        main_layout.addWidget(self._log, 1)

    def set_media_path(self, path: str):
        """供外部（如 sys.argv）设置媒体路径"""
        if path and path not in self._media_files:
            self._media_files.append(path)
            self._file_list.addItem(os.path.basename(path))

    def _add_files(self):
        filter_str = "媒体文件 (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.aac *.ogg *.m4a);;所有文件 (*.*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "选择媒体文件", "", filter_str)
        for p in paths:
            if p not in self._media_files:
                self._media_files.append(p)
                self._file_list.addItem(os.path.basename(p))

    def _remove_files(self):
        for item in reversed(self._file_list.selectedItems()):
            row = self._file_list.row(item)
            self._file_list.takeItem(row)
            del self._media_files[row]

    def _clear_files(self):
        self._file_list.clear()
        self._media_files.clear()

    def _log_msg(self, msg: str):
        self._log.log(msg)

    def _update_progress(self, pct: float, msg: str):
        self._progress.update_progress(pct, msg)

    def _set_buttons_state(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)

    def _on_finished(self):
        self._running = False
        self._set_buttons_state(False)

    def _start(self):
        if not self._media_files:
            self._log_msg("请先添加媒体文件")
            return

        self._running = True
        self._cancelled = False
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        lang = self._lang_combo.currentText()
        whisper_device = self._whisper_device_combo.currentText()
        align_device = self._align_device_combo.currentText()
        retention = {
            "whisper": self._chk_whisper.isChecked(),
            "force_align": self._chk_force_align.isChecked(),
        }
        pp = {k: v.value() for k, v in self._pp_vars.items()}
        cond_prev = self._chk_cond_prev.isChecked()

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
        self._log_msg("正在取消...")

    def _run_pipeline(
        self, media_files: List[str], lang: str,
        whisper_device: str, align_device: str,
        retention: Dict[str, bool], pp: Dict[str, float],
        cond_prev: bool = True,
    ):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        wc_threshold = pp.pop("word_conf_threshold", 0.50)
        ac_threshold = pp.pop("align_conf_threshold", 0.50)

        try:
            total = len(media_files)
            srt_results: Dict[str, str] = {}

            # ── 阶段 1/2: Whisper 语音识别 ──
            self._log_msg("── 阶段 1/2: Whisper 语音识别 ──")

            w_device = whisper_device
            if w_device == "cuda":
                try:
                    import ctranslate2
                    ctranslate2.get_supported_compute_types("cuda")
                except Exception:
                    self._log_msg("CUDA 不可用，Whisper 回退到 CPU")
                    w_device = "cpu"

            from faster_whisper import WhisperModel
            self._log_msg("加载 Whisper 模型...")
            whisper_model = WhisperModel("large-v3", device=w_device,
                                         compute_type="int8")

            for idx, media in enumerate(media_files, 1):
                if self._cancelled:
                    self._log_msg("已取消")
                    break

                self._log_msg(f"[{idx}/{total}] {os.path.basename(media)}")

                def _w_progress(pct, msg, _idx=idx):
                    overall = ((_idx - 1) + pct) / (total * 2)
                    self._update_progress(overall, f"[Whisper {_idx}/{total}] {msg}")

                try:
                    srt_path = transcribe_to_srt(
                        media,
                        lang=lang,
                        device=w_device,
                        log_callback=self._log_msg,
                        progress_callback=_w_progress,
                        word_conf_threshold=wc_threshold,
                        condition_on_previous_text=cond_prev,
                        model=whisper_model,
                    )
                    srt_results[media] = srt_path
                except Exception as e:
                    self._log_msg(f"文件处理出错: {e}")

            # 释放 Whisper 模型
            _keep_alive.append(whisper_model)
            del whisper_model
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # ── 阶段 2/2: MMS_FA 强制打轴 ──
            if srt_results and not self._cancelled:
                self._log_msg("── 阶段 2/2: MMS_FA 强制打轴 ──")

                a_device = align_device
                if a_device == "cuda" and not torch.cuda.is_available():
                    self._log_msg("CUDA 不可用，回退到 CPU")
                    a_device = "cpu"

                self._log_msg("加载 MMS_FA 模型...")
                mms_bundle = init_model(a_device)

                for idx, media in enumerate(media_files, 1):
                    if self._cancelled:
                        self._log_msg("已取消")
                        break

                    if media not in srt_results:
                        continue

                    srt_path = srt_results[media]
                    self._log_msg(f"[{idx}/{total}] {os.path.basename(media)}")

                    def _a_progress(pct, msg, _idx=idx):
                        overall = (total + (_idx - 1) + pct) / (total * 2)
                        self._update_progress(overall, f"[打轴 {_idx}/{total}] {msg}")

                    try:
                        waveform = load_audio(media)
                        subs = parse_srt(srt_path, lang=lang)
                        self._log_msg(f"字幕 {len(subs)} 条，设备: {a_device}")

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
                                self._log_msg(
                                    f"低置信对齐 {len(low_align)} 句，已标记"
                                )

                        # 清理中间文件
                        if not retention.get("whisper", False):
                            if srt_path and os.path.isfile(srt_path):
                                try:
                                    os.remove(srt_path)
                                    self._log_msg(
                                        f"已清理: {os.path.basename(srt_path)}"
                                    )
                                except OSError:
                                    pass

                    except Exception as e:
                        self._log_msg(f"文件处理出错: {e}")

                # 释放 MMS_FA 模型
                del mms_bundle
                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            if not self._cancelled:
                self._update_progress(1.0, "全部完成")
                self._log_msg(f"全部完成! 共处理 {total} 个文件")
        except Exception as e:
            self._log_msg(f"出错: {e}")
        finally:
            sys.stdout = old_stdout
            self._orchestrator = None
            self._signals.finished.emit()
