import os
import sys
import threading
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QListWidget, QPushButton, QLabel, QCheckBox,
    QFileDialog, QAbstractItemView,
)
from PySide6.QtCore import Signal, QObject

from writansub.types import LANGUAGES
from writansub.subtitle.srt_io import write_srt, populate_romaji, merge_bilingual
from writansub.transcribe.core import transcribe
from writansub.subtitle.review import generate_review, write_review_files, mark_low_align_in_review
from writansub.align.core import load_audio, run_alignment, post_process, init_model
from writansub.translate.core import translate_subs
from writansub.config import load_gui_state, save_gui_state, load_translate_config
from writansub.bridge import ResourceRegistry
from writansub.gui.widgets import (
    TextRedirector, LogWidget, ProgressWidget, build_params_grid,
    NoScrollComboBox,
)


class _PipelineSignals(QObject):
    """线程间信号传输"""
    finished = Signal()
    enable_start = Signal(bool)
    log_requested = Signal(str)
    progress_requested = Signal(float, str)


class PipelineTab(QWidget):
    """Tab 1: 核心流水线 - 纯布局调整版"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "word_conf_threshold", "align_conf_threshold",
        "min_duration",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orchestrator = None
        self._running = False
        self._cancelled = False
        self._media_files: List[str] = []

        self._signals = _PipelineSignals()
        self._signals.finished.connect(self._on_finished)
        self._signals.enable_start.connect(self._set_buttons_state)
        self._signals.log_requested.connect(self._log_msg)
        self._signals.progress_requested.connect(self._update_progress)

        self._setup_ui()
        self._connect_state_signals()
        self.restore_state(load_gui_state())

    # ── UI 构建 ──

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        self._build_file_section()
        self._build_config_section()
        self._build_pp_section()
        self._build_action_section()
        self._build_log_section()

    def _build_file_section(self):
        """1. 媒体文件区"""
        self.card_file = QGroupBox("媒体文件列表")
        file_layout = QHBoxLayout(self.card_file)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setMaximumHeight(120)
        file_layout.addWidget(self._file_list, 1)

        file_btns = QVBoxLayout()
        btn_add = QPushButton("添加文件")
        btn_add.clicked.connect(self._add_files)
        file_btns.addWidget(btn_add)
        btn_remove = QPushButton("移除选中")
        btn_remove.clicked.connect(self._remove_files)
        file_btns.addWidget(btn_remove)
        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(self._clear_files)
        file_btns.addWidget(btn_clear)
        file_btns.addStretch()
        file_layout.addLayout(file_btns)

        self.layout.addWidget(self.card_file)

    def _build_config_section(self):
        """2. 中间三列配置区"""
        config_layout = QHBoxLayout()

        # 模型配置
        self.card_basic = QGroupBox("模型配置")
        basic_layout = QGridLayout(self.card_basic)
        basic_layout.addWidget(QLabel("识别语言:"), 0, 0)
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_layout.addWidget(self._lang_combo, 0, 1)
        basic_layout.addWidget(QLabel("推理设备:"), 1, 0)
        self._whisper_device_combo = NoScrollComboBox()
        self._whisper_device_combo.addItems(["cuda", "cpu"])
        basic_layout.addWidget(self._whisper_device_combo, 1, 1)
        self._chk_cond_prev = QCheckBox("启用前文调优")
        basic_layout.addWidget(self._chk_cond_prev, 2, 0, 1, 2)
        config_layout.addWidget(self.card_basic)

        # TIGER 配置
        self.card_tiger = QGroupBox("TIGER 增强")
        tiger_layout = QVBoxLayout(self.card_tiger)
        self._chk_tiger_denoise = QCheckBox("人声去噪 (DnR)")
        tiger_layout.addWidget(self._chk_tiger_denoise)
        self._chk_tiger_separate = QCheckBox("重叠分离 (Separate)")
        self._chk_tiger_separate.stateChanged.connect(self._on_tiger_separate_changed)
        tiger_layout.addWidget(self._chk_tiger_separate)
        self._chk_tiger_save = QCheckBox("保存中间音轨")
        tiger_layout.addWidget(self._chk_tiger_save)
        tiger_layout.addStretch()
        config_layout.addWidget(self.card_tiger)

        # 输出配置
        self.card_retention = QGroupBox("输出保留")
        ret_layout = QVBoxLayout(self.card_retention)
        self._chk_whisper = QCheckBox("保留 Whisper SRT")
        ret_layout.addWidget(self._chk_whisper)
        self._chk_force_align = QCheckBox("保留对齐 SRT")
        ret_layout.addWidget(self._chk_force_align)
        self._chk_review = QCheckBox("生成 Review 文件")
        ret_layout.addWidget(self._chk_review)
        ret_layout.addStretch()
        config_layout.addWidget(self.card_retention)

        self.layout.addLayout(config_layout)

    def _build_pp_section(self):
        """3. 后处理参数 (横向排列)"""
        self.card_pp = QGroupBox("后处理参数")
        QGridLayout(self.card_pp)
        self._pp_vars = build_params_grid(self.card_pp, self._PP_KEYS)
        self.layout.addWidget(self.card_pp)

    def _build_action_section(self):
        """4. 底部控制区"""
        action_layout = QHBoxLayout()
        self._chk_translate = QCheckBox("AI 翻译")
        action_layout.addWidget(self._chk_translate)

        self._progress = ProgressWidget()
        action_layout.addWidget(self._progress, 1)

        self._cancel_btn = QPushButton("取消任务")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        action_layout.addWidget(self._cancel_btn)

        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._start)
        action_layout.addWidget(self._start_btn)

        self.layout.addLayout(action_layout)
        self.layout.addSpacing(10)

    def _build_log_section(self):
        """5. 可折叠日志区"""
        toggle_layout = QHBoxLayout()
        self._btn_toggle_log = QPushButton("▶ 查看日志")
        self._btn_toggle_log.setFlat(True)
        self._btn_toggle_log.clicked.connect(self._toggle_log)
        toggle_layout.addWidget(self._btn_toggle_log)
        toggle_layout.addStretch()
        self.layout.addLayout(toggle_layout)

        self._log = LogWidget()
        self._log.setVisible(False)
        self.layout.addWidget(self._log, 1)

    # ── 状态保存 / 恢复 ──

    def _connect_state_signals(self):
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._whisper_device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)
        self._chk_whisper.stateChanged.connect(self._auto_save)
        self._chk_force_align.stateChanged.connect(self._auto_save)
        self._chk_review.stateChanged.connect(self._auto_save)
        self._chk_translate.stateChanged.connect(self._auto_save)
        self._chk_tiger_denoise.stateChanged.connect(self._auto_save)
        self._chk_tiger_separate.stateChanged.connect(self._auto_save)
        self._chk_tiger_save.stateChanged.connect(self._auto_save)

    def _auto_save(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def save_state(self) -> dict:
        return {
            "pipeline.lang": self._lang_combo.currentText(),
            "pipeline.whisper_device": self._whisper_device_combo.currentText(),
            "pipeline.cond_prev": self._chk_cond_prev.isChecked(),
            "pipeline.retain_whisper": self._chk_whisper.isChecked(),
            "pipeline.retain_align": self._chk_force_align.isChecked(),
            "pipeline.retain_review": self._chk_review.isChecked(),
            "pipeline.enable_translate": self._chk_translate.isChecked(),
            "pipeline.tiger_denoise": self._chk_tiger_denoise.isChecked(),
            "pipeline.tiger_separate": self._chk_tiger_separate.isChecked(),
            "pipeline.tiger_save": self._chk_tiger_save.isChecked(),
            "pipeline.media_files": list(self._media_files),
        }

    def restore_state(self, state: dict):
        if "pipeline.lang" in state:
            self._lang_combo.setCurrentText(state["pipeline.lang"])
        if "pipeline.whisper_device" in state:
            self._whisper_device_combo.setCurrentText(state["pipeline.whisper_device"])
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
        if "pipeline.tiger_save" in state:
            self._chk_tiger_save.setChecked(state["pipeline.tiger_save"])
        if "pipeline.media_files" in state:
            for p in state["pipeline.media_files"]:
                if p not in self._media_files and os.path.isfile(p):
                    self._media_files.append(p)
                    self._file_list.addItem(os.path.basename(p))

    # ── 文件管理 ──

    def set_media_path(self, path: str):
        if path and path not in self._media_files:
            self._media_files.append(path)
            self._file_list.addItem(os.path.basename(path))

    def _on_tiger_separate_changed(self, state):
        if state:
            self._chk_tiger_denoise.setChecked(True)

    def _add_files(self):
        filter_str = (
            "媒体文件 (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.aac *.ogg *.m4a)"
            ";;全部 (*.*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "添加媒体文件", "", filter_str)
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

    # ── UI 回调 ──

    def _toggle_log(self):
        visible = self._log.isVisible()
        self._log.setVisible(not visible)
        self._btn_toggle_log.setText("▼ 隐藏日志" if not visible else "▶ 查看日志")

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

    def _cancel(self):
        self._cancelled = True
        self._signals.log_requested.emit("正在取消任务...")

    # ── 流水线启动 ──

    def _resolve_tiger_mode(self) -> str | None:
        if self._chk_tiger_separate.isChecked():
            return "separate"
        if self._chk_tiger_denoise.isChecked():
            return "denoise"
        return None

    def _start(self):
        if not self._media_files:
            self._signals.log_requested.emit("请先添加媒体文件")
            return

        self._running = True
        self._cancelled = False
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        device = self._whisper_device_combo.currentText()
        enable_translate = self._chk_translate.isChecked()

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(
                list(self._media_files),
                self._lang_combo.currentText(),
                device,
                {
                    "whisper": self._chk_whisper.isChecked(),
                    "force_align": self._chk_force_align.isChecked(),
                    "review": self._chk_review.isChecked(),
                },
                {k: v.value() for k, v in self._pp_vars.items()},
                self._chk_cond_prev.isChecked(),
                load_translate_config() if enable_translate else None,
                self._resolve_tiger_mode(),
                self._chk_tiger_save.isChecked(),
            ),
            daemon=True,
        )
        reg = ResourceRegistry.instance()
        self._thread_handle = reg.register_thread(thread)
        thread.start()

    # ── 流水线执行 (后台线程) ──

    def _run_pipeline(self, media_files, lang, device, retention, pp,
                      cond_prev, translate_cfg, tiger_mode, tiger_save):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        wc_threshold = pp.pop("word_conf_threshold", 0.50)
        ac_threshold = pp.pop("align_conf_threshold", 0.50)

        num_phases = 2 + (1 if translate_cfg else 0) + (1 if tiger_mode else 0)
        phase_offset = 1 if tiger_mode else 0
        reg = ResourceRegistry.instance()
        total = len(media_files)
        log_emit = self._signals.log_requested.emit
        prog_emit = self._signals.progress_requested.emit

        try:
            sub_results, word_results, tiger_results = {}, {}, {}

            # Phase: TIGER 增强
            if tiger_mode and not self._cancelled:
                tiger_results = self._run_tiger_phase(
                    media_files, tiger_mode, tiger_save, device,
                    num_phases, log_emit, prog_emit,
                )

            # Phase: Whisper 转录
            w_phase = 1 + phase_offset
            log_emit(f">>> Phase {w_phase}/{num_phases}: Whisper 转录")

            def _w_factory():
                return __import__("faster_whisper").WhisperModel(
                    "large-v3", device=device, compute_type="int8",
                )

            wh = reg.acquire_model("whisper", device, _w_factory)
            whisper_model = reg.get_model(wh)

            for idx, media in enumerate(media_files, 1):
                if self._cancelled:
                    break

                def _w_p(pct, msg):
                    prog_emit(
                        (phase_offset * total + (idx - 1) + pct) / (total * num_phases),
                        f"[Whisper {idx}/{total}] {msg}",
                    )

                subs, word_data = self._transcribe_single(
                    media, tiger_results.get(media), lang, device,
                    cond_prev, whisper_model, _w_p,
                )
                sub_results[media] = subs
                word_results[media] = word_data

                base = os.path.splitext(media)[0]
                if retention.get("review") and wc_threshold > 0:
                    srt_c, ass_c, low_c, tot_w = generate_review(subs, word_data, wc_threshold)
                    if low_c > 0:
                        write_review_files(base, srt_c, ass_c)
                if retention.get("whisper"):
                    write_srt(subs, base + ".srt")

            reg.release_model(wh)

            # Phase: MMS 对齐
            a_phase = 2 + phase_offset
            aligned_results = {}
            if sub_results and not self._cancelled:
                log_emit(f">>> Phase {a_phase}/{num_phases}: MMS 对齐")
                mh = reg.acquire_model("mms_fa", device, lambda: init_model(device))
                mms_bundle = reg.get_model(mh)

                for idx, media in enumerate(media_files, 1):
                    if self._cancelled or media not in sub_results:
                        continue

                    def _a_p(pct, msg):
                        prog_emit(
                            ((1 + phase_offset) * total + (idx - 1) + pct) / (total * num_phases),
                            f"[对齐 {idx}/{total}] {msg}",
                        )

                    waveform = load_audio(media)
                    populate_romaji(sub_results[media], lang)
                    aligned = run_alignment(
                        waveform, sub_results[media],
                        device=device, progress_callback=_a_p, model_bundle=mms_bundle,
                    )
                    final = post_process(aligned, **pp)
                    aligned_results[media] = final

                    base = os.path.splitext(media)[0]
                    if retention.get("review") and ac_threshold > 0:
                        low_a = {s.index for s in final if s.score < ac_threshold}
                        if low_a:
                            mark_low_align_in_review(base, low_a)
                    if retention.get("force_align"):
                        write_srt(final, base + "_aligned.srt")

                reg.release_model(mh)

            # Phase: AI 翻译
            if translate_cfg and aligned_results and not self._cancelled:
                log_emit(f">>> Phase {num_phases}/{num_phases}: AI 翻译")
                for idx, media in enumerate(media_files, 1):
                    if self._cancelled or media not in aligned_results:
                        continue

                    def _t_p(pct, msg):
                        prog_emit(
                            ((num_phases - 1) * total + (idx - 1) + pct) / (total * num_phases),
                            f"[翻译 {idx}/{total}] {msg}",
                        )

                    translate_subs(
                        aligned_results[media], **translate_cfg,
                        log_callback=log_emit, progress_callback=_t_p,
                    )
                    write_srt(
                        merge_bilingual(aligned_results[media]),
                        os.path.splitext(media)[0] + ".srt",
                    )
            elif aligned_results and not self._cancelled:
                for media, subs in aligned_results.items():
                    write_srt(subs, os.path.splitext(media)[0] + ".srt")

            if not self._cancelled:
                prog_emit(1.0, "任务完成")
                log_emit(f"全部完成! 已处理 {total} 个文件")

        except Exception as e:
            log_emit(f"发生错误: {e}")
        finally:
            sys.stdout = old_stdout
            reg.unregister_thread(self._thread_handle)
            self._signals.finished.emit()

    def _run_tiger_phase(self, media_files, tiger_mode, tiger_save, device,
                         num_phases, log_emit, prog_emit) -> dict:
        """执行 TIGER 增强阶段，返回 tiger_results"""
        log_emit(f">>> Phase 1/{num_phases}: TIGER 增强")
        from writansub.preprocess.core import run_dnr_batch, run_speech_batch

        do_speech = (tiger_mode == "separate")
        dnr_weight = 0.5 if do_speech else 1.0

        def _dnr_p(pct, msg):
            prog_emit(pct * dnr_weight / num_phases, f"[TIGER] {msg}")

        tiger_results = run_dnr_batch(
            media_files, device=device, save_intermediate=tiger_save,
            log_callback=log_emit, progress_callback=_dnr_p,
        )

        if do_speech and tiger_results and not self._cancelled:
            def _spk_p(pct, msg):
                prog_emit((0.5 + pct * 0.5) / num_phases, f"[TIGER] {msg}")

            run_speech_batch(
                tiger_results, device=device, save_intermediate=tiger_save,
                log_callback=log_emit, progress_callback=_spk_p,
            )

        return tiger_results

    def _transcribe_single(self, media, tiger_data, lang, device,
                           cond_prev, whisper_model, progress_callback):
        """对单个媒体文件执行 Whisper 转录，处理 TIGER 增强数据"""
        whisper_input = media
        tmp_dialog = None

        if tiger_data and "dialog_wav" in tiger_data:
            from writansub.preprocess.core import save_wav
            import tempfile
            tmp_dialog = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_dialog.close()
            save_wav(tiger_data["dialog_wav"], tmp_dialog.name, tiger_data["dialog_sr"])
            whisper_input = tmp_dialog.name

        try:
            overlap_r = tiger_data.get("overlap_regions") if tiger_data else None
            separated = None
            if tiger_data and "spk1_wav" in tiger_data:
                separated = (tiger_data["spk1_wav"], tiger_data["spk2_wav"])

            if overlap_r and separated:
                return self._whisper_with_overlap(
                    whisper_input, overlap_r, separated,
                    tiger_data.get("spk_sr", 16000),
                    lang, device, cond_prev, whisper_model, progress_callback,
                )

            return transcribe(
                whisper_input, lang=lang, device=device,
                log_callback=self._signals.log_requested.emit,
                progress_callback=progress_callback,
                condition_on_previous_text=cond_prev, model=whisper_model,
            )
        finally:
            if tmp_dialog:
                try:
                    os.unlink(tmp_dialog.name)
                except OSError:
                    pass

    def _whisper_with_overlap(self, media, overlap_regions, separated_tracks,
                              spk_sr, lang, device, cond_prev, whisper_model,
                              progress_callback):
        """重叠区域分轨转录并合并结果"""
        import tempfile
        from writansub.preprocess.core import save_wav

        full_subs, full_word_data = transcribe(
            media, lang=lang, device=device,
            log_callback=self._signals.log_requested.emit,
            progress_callback=progress_callback,
            condition_on_previous_text=cond_prev, model=whisper_model,
        )
        if not overlap_regions:
            return full_subs, full_word_data

        overlap_subs = []
        spk1_wav, spk2_wav = separated_tracks

        for region in overlap_regions:
            if self._cancelled:
                break
            start = int(region.start * spk_sr)
            end = int(region.end * spk_sr)

            for spk_wav in [spk1_wav, spk2_wav]:
                chunk = spk_wav[:, start:end]
                if chunk.shape[1] < 1600:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    save_wav(chunk, tmp_path, spk_sr)
                    local_subs, _ = transcribe(
                        tmp_path, lang=lang, device=device,
                        condition_on_previous_text=False, model=whisper_model,
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

        # 标记落在重叠区域内的原始字幕
        for sub in full_subs:
            mid = (sub.start + sub.end) / 2
            if any(r.start <= mid <= r.end for r in overlap_regions):
                sub.commented = True

        active = [s for s in full_subs if not s.commented] + overlap_subs
        active.sort(key=lambda s: s.start)
        for i, s in enumerate(active, 1):
            s.index = i

        return active, [[] for _ in active]
