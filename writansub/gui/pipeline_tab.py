"""一键流水线页：批量处理，内存数据流 + 模型池"""

import os
import sys
import threading
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QListWidget, QPushButton, QLabel, QCheckBox,
    QFileDialog, QFrame, QAbstractItemView, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

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
    NoScrollComboBox,
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
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # ── 上半部分：设置 + 操作 + 进度 ──
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 可滚动区域
        scrollable = ScrollableFrame()
        top_layout.addWidget(scrollable, 1)

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
        self._chk_review = QCheckBox("校对标记文件")
        self._chk_review.setToolTip("输出低置信词/低置信对齐的 SRT+ASS 标记文件")
        ret_layout.addWidget(self._chk_review)
        ret_layout.addStretch()

        # 基本设置
        card_basic = QGroupBox("基本设置")
        inner_layout.addWidget(card_basic)
        basic_layout = QHBoxLayout(card_basic)

        basic_layout.addWidget(QLabel("语言"))
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_layout.addWidget(self._lang_combo)

        basic_layout.addWidget(QLabel("Whisper 设备"))
        self._whisper_device_combo = NoScrollComboBox()
        self._whisper_device_combo.addItems(["cuda", "cpu"])
        basic_layout.addWidget(self._whisper_device_combo)

        basic_layout.addWidget(QLabel("打轴设备"))
        self._align_device_combo = NoScrollComboBox()
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

        # TIGER 音频分离
        card_tiger = QGroupBox("TIGER 音频分离（可选）")
        inner_layout.addWidget(card_tiger)
        tiger_layout = QHBoxLayout(card_tiger)

        self._chk_tiger_denoise = QCheckBox("降噪")
        self._chk_tiger_denoise.setToolTip(
            "DnR 分离出纯对话轨（去除音效和音乐）"
        )
        tiger_layout.addWidget(self._chk_tiger_denoise)

        self._chk_tiger_separate = QCheckBox("对话分轨")
        self._chk_tiger_separate.setToolTip(
            "分离说话人 + VAD 重叠段检测\n"
            "（自动启用降噪作为前置步骤）"
        )
        self._chk_tiger_separate.stateChanged.connect(self._on_tiger_separate_changed)
        tiger_layout.addWidget(self._chk_tiger_separate)

        tiger_layout.addWidget(QLabel("设备"))
        self._tiger_device_combo = NoScrollComboBox()
        self._tiger_device_combo.addItems(["cuda", "cpu"])
        tiger_layout.addWidget(self._tiger_device_combo)

        self._chk_tiger_save = QCheckBox("保留分离音频")
        self._chk_tiger_save.setToolTip("保存 TIGER 分离的中间 WAV 文件")
        tiger_layout.addWidget(self._chk_tiger_save)
        tiger_layout.addStretch()

        # 后处理参数
        card_pp = QGroupBox("后处理参数")
        inner_layout.addWidget(card_pp)
        self._pp_vars = build_params_grid(card_pp, self._PP_KEYS)

        inner_layout.addStretch()

        splitter.addWidget(top_widget)

        # ── 下半部分：操作 + 进度 + 日志 ──
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        action_bar = QWidget()
        bottom_layout.addWidget(action_bar)
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

        self._progress = ProgressWidget()
        bottom_layout.addWidget(self._progress)

        self._log = LogWidget()
        bottom_layout.addWidget(self._log, 1)

        splitter.addWidget(bottom_widget)

        # 默认比例：上 3 下 1
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _connect_state_signals(self):
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._whisper_device_combo.currentTextChanged.connect(self._auto_save)
        self._align_device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)
        self._chk_whisper.stateChanged.connect(self._auto_save)
        self._chk_force_align.stateChanged.connect(self._auto_save)
        self._chk_review.stateChanged.connect(self._auto_save)
        self._chk_translate.stateChanged.connect(self._auto_save)
        self._chk_tiger_denoise.stateChanged.connect(self._auto_save)
        self._chk_tiger_separate.stateChanged.connect(self._auto_save)
        self._tiger_device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_tiger_save.stateChanged.connect(self._auto_save)

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
            "pipeline.retain_review": self._chk_review.isChecked(),
            "pipeline.enable_translate": self._chk_translate.isChecked(),
            "pipeline.tiger_denoise": self._chk_tiger_denoise.isChecked(),
            "pipeline.tiger_separate": self._chk_tiger_separate.isChecked(),
            "pipeline.tiger_device": self._tiger_device_combo.currentText(),
            "pipeline.tiger_save": self._chk_tiger_save.isChecked(),
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
        if "pipeline.retain_review" in state:
            self._chk_review.setChecked(state["pipeline.retain_review"])
        if "pipeline.enable_translate" in state:
            self._chk_translate.setChecked(state["pipeline.enable_translate"])
        if "pipeline.tiger_denoise" in state:
            self._chk_tiger_denoise.setChecked(state["pipeline.tiger_denoise"])
        if "pipeline.tiger_separate" in state:
            self._chk_tiger_separate.setChecked(state["pipeline.tiger_separate"])
        if "pipeline.tiger_device" in state:
            self._tiger_device_combo.setCurrentText(state["pipeline.tiger_device"])
        if "pipeline.tiger_save" in state:
            self._chk_tiger_save.setChecked(state["pipeline.tiger_save"])
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

    def _on_tiger_separate_changed(self, state):
        if state:
            self._chk_tiger_denoise.setChecked(True)

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
            "review": self._chk_review.isChecked(),
        }
        pp = {k: v.value() for k, v in self._pp_vars.items()}
        cond_prev = self._chk_cond_prev.isChecked()
        enable_translate = self._chk_translate.isChecked()
        translate_cfg = load_translate_config() if enable_translate else None

        tiger_denoise = self._chk_tiger_denoise.isChecked()
        tiger_separate = self._chk_tiger_separate.isChecked()
        if tiger_separate:
            tiger_mode = "separate"
        elif tiger_denoise:
            tiger_mode = "denoise"
        else:
            tiger_mode = None
        tiger_save = self._chk_tiger_save.isChecked()
        tiger_device = self._tiger_device_combo.currentText()

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(list(self._media_files), lang, whisper_device,
                  align_device, retention, pp, cond_prev, translate_cfg,
                  tiger_mode, tiger_save, tiger_device),
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
        tiger_mode: Optional[str] = None,
        tiger_save: bool = False,
        tiger_device: str = "cuda",
    ):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        wc_threshold = pp.pop("word_conf_threshold", 0.50)
        ac_threshold = pp.pop("align_conf_threshold", 0.50)

        base_phases = 2  # Whisper + Align
        if translate_cfg:
            base_phases += 1
        if tiger_mode:
            base_phases += 1
        num_phases = base_phases
        phase_offset = 1 if tiger_mode else 0

        reg = ResourceRegistry.instance()

        try:
            total = len(media_files)
            sub_results: Dict[str, List[Sub]] = {}
            word_results: Dict[str, List[List[WordInfo]]] = {}
            tiger_results: Dict[str, dict] = {}  # media -> tiger output

            # ── 阶段 0 (可选): TIGER 音频分离 ──
            if tiger_mode and not self._cancelled:
                self._log_msg(f"── 阶段 1/{num_phases}: TIGER 音频分离 ──")
                from writansub.core.tiger import run_dnr_batch, run_speech_batch

                # DnR 降噪（所有文件共享一次模型加载）
                def _dnr_progress(pct, msg):
                    self._update_progress(
                        pct * 0.5 / num_phases, f"[DnR] {msg}")

                try:
                    tiger_results = run_dnr_batch(
                        media_files,
                        device=tiger_device,
                        save_intermediate=tiger_save,
                        log_callback=self._log_msg,
                        progress_callback=_dnr_progress,
                    )
                except Exception as e:
                    self._log_msg(f"DnR 处理出错: {e}")

                # 说话人分轨 + VAD（仅 separate 模式）
                if tiger_mode == "separate" and tiger_results and not self._cancelled:
                    def _spk_progress(pct, msg):
                        self._update_progress(
                            (0.5 + pct * 0.5) / num_phases, f"[Speech] {msg}")

                    try:
                        run_speech_batch(
                            tiger_results,
                            device=tiger_device,
                            save_intermediate=tiger_save,
                            log_callback=self._log_msg,
                            progress_callback=_spk_progress,
                        )
                    except Exception as e:
                        self._log_msg(f"Speech 处理出错: {e}")

            # ── 阶段 1 (或 2): Whisper 语音识别 ──
            whisper_phase = 1 + phase_offset
            self._log_msg(f"── 阶段 {whisper_phase}/{num_phases}: Whisper 语音识别 ──")

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
                    phase_start = phase_offset * total
                    overall = (phase_start + (_idx - 1) + pct) / (total * num_phases)
                    self._update_progress(overall, f"[Whisper {_idx}/{total}] {msg}")

                try:
                    tiger_data = tiger_results.get(media)

                    # 有降噪结果时，用降噪后的对话轨做 Whisper 输入
                    whisper_input = media
                    _tmp_dialog = None
                    if tiger_data and "dialog_wav" in tiger_data:
                        import tempfile
                        from writansub.core.tiger import save_wav
                        _tmp_dialog = tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False)
                        _tmp_dialog.close()
                        save_wav(tiger_data["dialog_wav"], _tmp_dialog.name,
                                 tiger_data["dialog_sr"])
                        whisper_input = _tmp_dialog.name
                        self._log_msg("使用降噪后音频")

                    overlap_regions = tiger_data.get("overlap_regions") if tiger_data else None
                    separated_tracks = None
                    if tiger_data and "spk1_wav" in tiger_data:
                        separated_tracks = (tiger_data["spk1_wav"], tiger_data["spk2_wav"])

                    try:
                        if overlap_regions and separated_tracks:
                            # TIGER 分轨模式：完整 Whisper + 重叠段替换
                            subs, word_data = self._whisper_with_overlap(
                                whisper_input, overlap_regions, separated_tracks,
                                tiger_data.get("spk_sr", 16000),
                                lang, w_device, cond_prev, whisper_model,
                                _w_progress,
                            )
                        else:
                            subs, word_data = transcribe(
                                whisper_input,
                                lang=lang,
                                device=w_device,
                                log_callback=self._log_msg,
                                progress_callback=_w_progress,
                                condition_on_previous_text=cond_prev,
                                model=whisper_model,
                            )
                    finally:
                        if _tmp_dialog:
                            try:
                                os.unlink(_tmp_dialog.name)
                            except OSError:
                                pass

                    sub_results[media] = subs
                    word_results[media] = word_data

                    # review 文件（可选）
                    if retention.get("review") and wc_threshold > 0.0:
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

            # ── MMS_FA 强制打轴 ──
            align_phase = 2 + phase_offset
            aligned_results: Dict[str, List[Sub]] = {}
            if sub_results and not self._cancelled:
                self._log_msg(f"── 阶段 {align_phase}/{num_phases}: MMS_FA 强制打轴 ──")

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
                        phase_start = (1 + phase_offset) * total
                        overall = (phase_start + (_idx - 1) + pct) / (total * num_phases)
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

                        # review 低置信对齐标记（可选）
                        if retention.get("review") and ac_threshold > 0:
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

            # ── AI 翻译（可选）──
            translate_phase = num_phases  # 翻译始终是最后一个阶段
            if translate_cfg and aligned_results and not self._cancelled:
                self._log_msg(f"── 阶段 {translate_phase}/{num_phases}: AI 翻译 ──")
                for idx, media in enumerate(media_files, 1):
                    if self._cancelled:
                        self._log_msg("已取消")
                        break
                    if media not in aligned_results:
                        continue

                    final_subs = aligned_results[media]
                    self._log_msg(f"[{idx}/{total}] 翻译 {os.path.basename(media)}")

                    def _t_progress(pct, msg, _idx=idx):
                        phase_start = (translate_phase - 1) * total
                        overall = (phase_start + (_idx - 1) + pct) / (total * num_phases)
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

    def _whisper_with_overlap(
        self, media, overlap_regions, separated_tracks, spk_sr,
        lang, device, cond_prev, whisper_model, progress_callback,
    ):
        """完整 Whisper + 重叠段局部替换，返回 (subs, word_data)"""
        import tempfile
        from writansub.core.tiger import save_wav

        self._log_msg("完整音频 Whisper 识别...")
        full_subs, full_word_data = transcribe(
            media, lang=lang, device=device,
            log_callback=self._log_msg,
            progress_callback=progress_callback,
            condition_on_previous_text=cond_prev,
            model=whisper_model,
        )

        if not overlap_regions:
            return full_subs, full_word_data

        self._log_msg(f"处理 {len(overlap_regions)} 个重叠段...")
        spk1_wav, spk2_wav = separated_tracks

        overlap_subs = []
        for region in overlap_regions:
            if self._cancelled:
                return full_subs, full_word_data

            start_sample = int(region.start * spk_sr)
            end_sample = int(region.end * spk_sr)

            for spk_wav in [spk1_wav, spk2_wav]:
                chunk = spk_wav[:, start_sample:end_sample]
                if chunk.shape[1] < 1600:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    save_wav(chunk, tmp_path, spk_sr)
                    local_subs, _ = transcribe(
                        tmp_path, lang=lang, device=device,
                        condition_on_previous_text=False,
                        model=whisper_model,
                    )
                    for s in local_subs:
                        s.start += region.start
                        s.end += region.start
                    overlap_subs.extend(local_subs)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # 标记重叠段的完整结果为 commented
        for sub in full_subs:
            mid = (sub.start + sub.end) / 2
            for region in overlap_regions:
                if region.start <= mid <= region.end:
                    sub.commented = True
                    break

        active = [s for s in full_subs if not s.commented]
        active.extend(overlap_subs)
        active.sort(key=lambda s: s.start)
        for i, s in enumerate(active, 1):
            s.index = i

        commented_count = sum(1 for s in full_subs if s.commented)
        self._log_msg(
            f"合并完成: {len(active)} 条有效, {commented_count} 条被替换"
        )

        # word_data 对应 active subs（局部结果没有词级数据）
        word_data = [[] for _ in active]
        return active, word_data
