"""一键流水线页：批量处理，内存数据流 + 模型池"""

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

from writansub.core.types import MEDIA_FILETYPES, LANGUAGES, Sub, WordInfo
from writansub.core.srt_io import write_srt, populate_romaji, merge_bilingual
from writansub.core.whisper import transcribe
from writansub.core.review import generate_review, write_review_files, mark_low_align_in_review
from writansub.core.alignment import load_audio, run_alignment, post_process, init_model
from writansub.core.translate import translate_subs
from writansub.pipeline import PipelineOrchestrator
from writansub.config import load_gui_state, save_gui_state, load_translate_config
from writansub.registry import ResourceRegistry
from writansub.gui.widgets import (
    TextRedirector, ScrollableFrame, LogWidget, ProgressWidget, build_params_grid,
)


class _PipelineSignals(QObject):
    """线程安全的信号"""
    finished = Signal()
    enable_start = Signal(bool)


class PipelineTab(QWidget):
    """Tab 1: 一键流水线 (Whisper → ForceAlign → Translate)"""

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
        self._connect_state_signals()
        self.restore_state(load_gui_state())

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
        self._chk_translate = QCheckBox("AI 翻译")
        action_layout.addWidget(self._chk_translate)
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

    def _connect_state_signals(self):
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._whisper_device_combo.currentTextChanged.connect(self._auto_save)
        self._align_device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)
        self._chk_whisper.stateChanged.connect(self._auto_save)
        self._chk_force_align.stateChanged.connect(self._auto_save)
        self._chk_translate.stateChanged.connect(self._auto_save)

    def _auto_save(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def save_state(self) -> dict:
        return {
            "pipeline.lang": self._lang_combo.currentText(),
            "pipeline.whisper_device": self._whisper_device_combo.currentText(),
            "pipeline.align_device": self._align_device_combo.currentText(),
            "pipeline.cond_prev": self._chk_cond_prev.isChecked(),
            "pipeline.retain_whisper": self._chk_whisper.isChecked(),
            "pipeline.retain_align": self._chk_force_align.isChecked(),
            "pipeline.enable_translate": self._chk_translate.isChecked(),
            "pipeline.media_files": list(self._media_files),
        }

    def restore_state(self, state: dict):
        if "pipeline.lang" in state:
            self._lang_combo.setCurrentText(state["pipeline.lang"])
        if "pipeline.whisper_device" in state:
            self._whisper_device_combo.setCurrentText(state["pipeline.whisper_device"])
        if "pipeline.align_device" in state:
            self._align_device_combo.setCurrentText(state["pipeline.align_device"])
        if "pipeline.cond_prev" in state:
            self._chk_cond_prev.setChecked(state["pipeline.cond_prev"])
        if "pipeline.retain_whisper" in state:
            self._chk_whisper.setChecked(state["pipeline.retain_whisper"])
        if "pipeline.retain_align" in state:
            self._chk_force_align.setChecked(state["pipeline.retain_align"])
        if "pipeline.enable_translate" in state:
            self._chk_translate.setChecked(state["pipeline.enable_translate"])
        if "pipeline.media_files" in state:
            for p in state["pipeline.media_files"]:
                if p not in self._media_files and os.path.isfile(p):
                    self._media_files.append(p)
                    self._file_list.addItem(os.path.basename(p))

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
        enable_translate = self._chk_translate.isChecked()
        translate_cfg = load_translate_config() if enable_translate else None

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(list(self._media_files), lang, whisper_device,
                  align_device, retention, pp, cond_prev, translate_cfg),
            daemon=True,
        )
        reg = ResourceRegistry.instance()
        self._thread_handle = reg.register_thread(thread)
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
        translate_cfg: Optional[Dict[str, str]] = None,
    ):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        wc_threshold = pp.pop("word_conf_threshold", 0.50)
        ac_threshold = pp.pop("align_conf_threshold", 0.50)

        num_phases = 3 if translate_cfg else 2
        reg = ResourceRegistry.instance()

        try:
            total = len(media_files)
            sub_results: Dict[str, List[Sub]] = {}
            word_results: Dict[str, List[List[WordInfo]]] = {}

            # ── 阶段 1: Whisper 语音识别 ──
            self._log_msg(f"── 阶段 1/{num_phases}: Whisper 语音识别 ──")

            w_device = whisper_device
            if w_device == "cuda":
                try:
                    import ctranslate2
                    ctranslate2.get_supported_compute_types("cuda")
                except Exception:
                    self._log_msg("CUDA 不可用，Whisper 回退到 CPU")
                    w_device = "cpu"

            def _whisper_factory():
                from faster_whisper import WhisperModel
                self._log_msg("加载 Whisper 模型...")
                return WhisperModel("large-v3", device=w_device, compute_type="int8")

            wh = reg.acquire_model("whisper", w_device, _whisper_factory)
            whisper_model = reg.get_model(wh)

            for idx, media in enumerate(media_files, 1):
                if self._cancelled:
                    self._log_msg("已取消")
                    break

                self._log_msg(f"[{idx}/{total}] {os.path.basename(media)}")

                def _w_progress(pct, msg, _idx=idx):
                    overall = ((_idx - 1) + pct) / (total * num_phases)
                    self._update_progress(overall, f"[Whisper {_idx}/{total}] {msg}")

                try:
                    subs, word_data = transcribe(
                        media,
                        lang=lang,
                        device=w_device,
                        log_callback=self._log_msg,
                        progress_callback=_w_progress,
                        condition_on_previous_text=cond_prev,
                        model=whisper_model,
                    )
                    sub_results[media] = subs
                    word_results[media] = word_data

                    # review 文件
                    if wc_threshold > 0.0:
                        srt_content, ass_content, low_count, total_words = generate_review(
                            subs, word_data, wc_threshold,
                        )
                        if low_count > 0:
                            base = os.path.splitext(media)[0]
                            write_review_files(base, srt_content, ass_content)
                            self._log_msg(f"低置信词 {low_count}/{total_words}，已生成标记版")

                    # 保留中间文件
                    if retention.get("whisper", False):
                        whisper_srt = os.path.splitext(media)[0] + ".srt"
                        write_srt(subs, whisper_srt)
                        self._log_msg(f"保留: {os.path.basename(whisper_srt)}")

                except Exception as e:
                    self._log_msg(f"文件处理出错: {e}")

            reg.release_model(wh)

            # ── 阶段 2: MMS_FA 强制打轴 ──
            aligned_results: Dict[str, List[Sub]] = {}
            if sub_results and not self._cancelled:
                self._log_msg(f"── 阶段 2/{num_phases}: MMS_FA 强制打轴 ──")

                a_device = align_device
                if a_device == "cuda" and not torch.cuda.is_available():
                    self._log_msg("CUDA 不可用，回退到 CPU")
                    a_device = "cpu"

                def _mms_factory():
                    self._log_msg("加载 MMS_FA 模型...")
                    return init_model(a_device)

                mh = reg.acquire_model("mms_fa", a_device, _mms_factory)
                mms_bundle = reg.get_model(mh)

                for idx, media in enumerate(media_files, 1):
                    if self._cancelled:
                        self._log_msg("已取消")
                        break

                    if media not in sub_results:
                        continue

                    subs = sub_results[media]
                    self._log_msg(f"[{idx}/{total}] {os.path.basename(media)}")

                    def _a_progress(pct, msg, _idx=idx):
                        overall = (total + (_idx - 1) + pct) / (total * num_phases)
                        self._update_progress(overall, f"[打轴 {_idx}/{total}] {msg}")

                    try:
                        waveform = load_audio(media)
                        populate_romaji(subs, lang)
                        self._log_msg(f"字幕 {len(subs)} 条，设备: {a_device}")

                        aligned = run_alignment(
                            waveform, subs, device=a_device,
                            progress_callback=_a_progress,
                            model_bundle=mms_bundle,
                        )

                        final = post_process(aligned, **pp)
                        aligned_results[media] = final

                        # review 低置信对齐标记
                        if ac_threshold > 0:
                            low_align = {
                                s.index for s in final
                                if s.score < ac_threshold
                            }
                            if low_align:
                                base = os.path.splitext(media)[0]
                                mark_low_align_in_review(base, low_align)
                                self._log_msg(
                                    f"低置信对齐 {len(low_align)} 句，已标记"
                                )

                        # 保留中间文件
                        if retention.get("force_align", False):
                            align_srt = os.path.splitext(media)[0] + "_aligned.srt"
                            write_srt(final, align_srt)
                            self._log_msg(f"保留: {os.path.basename(align_srt)}")

                    except Exception as e:
                        self._log_msg(f"文件处理出错: {e}")

                reg.release_model(mh)

            # Phase 1+2 结束，flush 所有模型释放 GPU
            reg.flush_models()

            # ── 阶段 3: AI 翻译（可选）──
            if translate_cfg and aligned_results and not self._cancelled:
                self._log_msg(f"── 阶段 3/{num_phases}: AI 翻译 ──")
                for idx, media in enumerate(media_files, 1):
                    if self._cancelled:
                        self._log_msg("已取消")
                        break
                    if media not in aligned_results:
                        continue

                    final_subs = aligned_results[media]
                    self._log_msg(f"[{idx}/{total}] 翻译 {os.path.basename(media)}")

                    def _t_progress(pct, msg, _idx=idx):
                        overall = (total * 2 + (_idx - 1) + pct) / (total * num_phases)
                        self._update_progress(overall, f"[翻译 {_idx}/{total}] {msg}")

                    try:
                        translate_subs(
                            final_subs,
                            target_lang=translate_cfg["target_lang"],
                            api_base=translate_cfg["api_base"],
                            api_key=translate_cfg["api_key"],
                            model=translate_cfg["model"],
                            log_callback=self._log_msg,
                            progress_callback=_t_progress,
                        )
                        bilingual = merge_bilingual(final_subs)
                        bilingual_path = os.path.splitext(media)[0] + ".srt"
                        write_srt(bilingual, bilingual_path)
                        self._log_msg(f"双语字幕: {os.path.basename(bilingual_path)}")
                    except Exception as e:
                        self._log_msg(f"翻译出错: {e}")

            # 无翻译时直接输出
            if not translate_cfg and aligned_results and not self._cancelled:
                for media, final_subs in aligned_results.items():
                    output_path = os.path.splitext(media)[0] + ".srt"
                    write_srt(final_subs, output_path)

            if not self._cancelled:
                self._update_progress(1.0, "全部完成")
                self._log_msg(f"全部完成! 共处理 {total} 个文件")
        except Exception as e:
            self._log_msg(f"出错: {e}")
        finally:
            sys.stdout = old_stdout
            self._orchestrator = None
            ResourceRegistry.instance().unregister_thread(self._thread_handle)
            self._signals.finished.emit()
