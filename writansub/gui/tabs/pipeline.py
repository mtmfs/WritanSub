import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QListWidget, QPushButton, QLabel, QCheckBox, QLineEdit,
    QFileDialog, QAbstractItemView,
)
from PySide6.QtCore import Signal, QObject

from writansub.types import LANGUAGES, WHISPER_MODELS, ALIGN_MODELS, MSS_MODELS, SS_MODELS
from writansub.config import load_gui_state, load_translate_config
from writansub.bridge import ResourceRegistry, CancelledError
from writansub.gui.widgets import (
    LogWidget, ProgressWidget, build_params_grid,
    NoScrollComboBox, GroupedComboBox, StateMixin,
)


class _PipelineSignals(QObject):
    finished = Signal()
    enable_start = Signal(bool)
    log_requested = Signal(str)
    progress_requested = Signal(float, str)


class PipelineTab(StateMixin, QWidget):

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "word_conf_threshold", "align_conf_threshold",
        "min_duration", "pad_sec",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orchestrator = None
        self._running = False
        self._media_files: list[str] = []

        self._signals = _PipelineSignals()
        self._signals.finished.connect(self._on_finished)
        self._signals.enable_start.connect(self._set_buttons_state)
        self._signals.log_requested.connect(lambda msg: self._log.log(msg))
        self._signals.progress_requested.connect(lambda pct, msg: self._progress.update_progress(pct, msg))

        self._setup_ui()
        self._connect_state_signals()
        self.restore_state(load_gui_state())

    # ── UI 构建 ──

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self._main_layout = layout

        self._build_file_section()
        self._build_config_section()
        self._build_pp_section()
        self._build_action_section()
        self._build_log_section()

    def _build_file_section(self):
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

        self._main_layout.addWidget(self.card_file)

    def _build_config_section(self):
        config_layout = QHBoxLayout()

        self.card_basic = QGroupBox("模型配置")
        basic_layout = QGridLayout(self.card_basic)
        basic_layout.addWidget(QLabel("识别语言:"), 0, 0)
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_layout.addWidget(self._lang_combo, 0, 1)
        basic_layout.addWidget(QLabel("听写模型:"), 1, 0)
        self._model_combo = GroupedComboBox()
        self._model_combo.set_grouped_items(WHISPER_MODELS)
        self._model_combo.setCurrentName("large-v3")
        basic_layout.addWidget(self._model_combo, 1, 1)
        basic_layout.addWidget(QLabel("推理设备:"), 2, 0)
        self._whisper_device_combo = NoScrollComboBox()
        self._whisper_device_combo.addItems(["cuda", "cpu"])
        basic_layout.addWidget(self._whisper_device_combo, 2, 1)
        basic_layout.addWidget(QLabel("对齐模型:"), 3, 0)
        self._align_model_combo = GroupedComboBox()
        self._align_model_combo.set_grouped_items(ALIGN_MODELS)
        self._align_model_combo.setCurrentName("mms_fa")
        basic_layout.addWidget(self._align_model_combo, 3, 1)
        self._chk_cond_prev = QCheckBox("启用前文调优")
        basic_layout.addWidget(self._chk_cond_prev, 4, 0)
        self._chk_vad = QCheckBox("跳过静音 (VAD)")
        self._chk_vad.setToolTip("启用 Silero VAD 跳过静音段，加速识别\n（启用 TIGER 预处理时建议关闭）")
        basic_layout.addWidget(self._chk_vad, 4, 1)
        basic_layout.addWidget(QLabel("初始提示:"), 5, 0)
        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("人名、术语（≤224 token）")
        self._prompt_edit.setToolTip(
            "Whisper initial_prompt：传给模型的上下文偏好，\n"
            "用于让角色名/专有名词保持一致写法。\n"
            "例如：瀬尾拓也、伊地知琴子、天音ケイ、キラモン\n"
            "上限约 224 token，过长会被截断。"
        )
        basic_layout.addWidget(self._prompt_edit, 5, 1)
        config_layout.addWidget(self.card_basic)

        self.card_tiger = QGroupBox("预处理增强")
        tiger_layout = QVBoxLayout(self.card_tiger)
        self._chk_tiger_denoise = QCheckBox("人声去噪 (DnR)")
        tiger_layout.addWidget(self._chk_tiger_denoise)

        mss_row = QHBoxLayout()
        mss_row.addWidget(QLabel("降噪模型:"))
        self._mss_model_combo = GroupedComboBox()
        self._mss_model_combo.set_grouped_items(MSS_MODELS)
        self._mss_model_combo.setCurrentName("tiger-dnr")
        mss_row.addWidget(self._mss_model_combo)
        tiger_layout.addLayout(mss_row)

        self._chk_tiger_separate = QCheckBox("重叠分离 (Separate)")
        self._chk_tiger_separate.stateChanged.connect(self._on_tiger_separate_changed)
        tiger_layout.addWidget(self._chk_tiger_separate)

        ss_row = QHBoxLayout()
        ss_row.addWidget(QLabel("分轨模型:"))
        self._ss_model_combo = GroupedComboBox()
        self._ss_model_combo.set_grouped_items(SS_MODELS)
        self._ss_model_combo.setCurrentName("tiger-speech")
        ss_row.addWidget(self._ss_model_combo)
        tiger_layout.addLayout(ss_row)

        self._chk_tiger_save = QCheckBox("保存中间音轨")
        tiger_layout.addWidget(self._chk_tiger_save)
        tiger_layout.addStretch()
        config_layout.addWidget(self.card_tiger)

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

        self.card_ref = QGroupBox("参考字幕")
        ref_layout = QVBoxLayout(self.card_ref)
        self._chk_ref_sub = QCheckBox("参考内嵌字幕")
        self._chk_ref_sub.setToolTip(
            "从视频中提取内嵌字幕的时间轴作为参考\n"
            "Whisper 文本会映射到内嵌字幕的时间窗口上"
        )
        ref_layout.addWidget(self._chk_ref_sub)

        ref_srt_row = QHBoxLayout()
        self._ref_srt_label = QLabel("无")
        self._ref_srt_label.setToolTip("指定外部 SRT 文件后，优先使用外部文件而非内嵌字幕")
        self._ref_srt_path: str = ""
        btn_ref_srt = QPushButton("外部SRT")
        btn_ref_srt.clicked.connect(self._browse_ref_srt)
        btn_ref_clear = QPushButton("清除")
        btn_ref_clear.clicked.connect(self._clear_ref_srt)
        ref_srt_row.addWidget(self._ref_srt_label, 1)
        ref_srt_row.addWidget(btn_ref_srt)
        ref_srt_row.addWidget(btn_ref_clear)
        ref_layout.addLayout(ref_srt_row)

        self._chk_ref_direct = QCheckBox("直接使用参考轴")
        self._chk_ref_direct.setToolTip("跳过强制对齐，直接使用参考字幕的时间轴")
        ref_layout.addWidget(self._chk_ref_direct)
        ref_layout.addStretch()
        config_layout.addWidget(self.card_ref)

        self._main_layout.addLayout(config_layout)

    def _build_pp_section(self):
        self.card_pp = QGroupBox("后处理参数")
        QGridLayout(self.card_pp)
        self._pp_vars = build_params_grid(self.card_pp, self._PP_KEYS)
        self._main_layout.addWidget(self.card_pp)

    def _build_action_section(self):
        action_layout = QHBoxLayout()
        self._chk_translate = QCheckBox("AI 翻译")
        action_layout.addWidget(self._chk_translate)

        self._progress = ProgressWidget()
        action_layout.addWidget(self._progress, 1)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        action_layout.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消任务")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        action_layout.addWidget(self._cancel_btn)

        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._start)
        action_layout.addWidget(self._start_btn)

        self._main_layout.addLayout(action_layout)
        self._main_layout.addSpacing(10)

    def _build_log_section(self):
        toggle_layout = QHBoxLayout()
        self._btn_toggle_log = QPushButton("▶ 查看日志")
        self._btn_toggle_log.setFlat(True)
        self._btn_toggle_log.clicked.connect(self._toggle_log)
        toggle_layout.addWidget(self._btn_toggle_log)
        toggle_layout.addStretch()
        self._main_layout.addLayout(toggle_layout)

        self._log = LogWidget()
        self._log.setVisible(False)
        self._main_layout.addWidget(self._log, 1)

    # ── 状态保存 / 恢复 ──

    def _connect_state_signals(self):
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._model_combo.currentTextChanged.connect(self._auto_save)
        self._whisper_device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)
        self._chk_vad.stateChanged.connect(self._auto_save)
        self._prompt_edit.editingFinished.connect(self._auto_save)
        self._align_model_combo.currentTextChanged.connect(self._auto_save)
        self._chk_whisper.stateChanged.connect(self._auto_save)
        self._chk_force_align.stateChanged.connect(self._auto_save)
        self._chk_review.stateChanged.connect(self._auto_save)
        self._chk_translate.stateChanged.connect(self._auto_save)
        self._chk_tiger_denoise.stateChanged.connect(self._auto_save)
        self._mss_model_combo.currentTextChanged.connect(self._auto_save)
        self._chk_tiger_separate.stateChanged.connect(self._auto_save)
        self._ss_model_combo.currentTextChanged.connect(self._auto_save)
        self._chk_tiger_save.stateChanged.connect(self._auto_save)
        self._chk_ref_sub.stateChanged.connect(self._auto_save)
        self._chk_ref_direct.stateChanged.connect(self._auto_save)

    def save_state(self) -> dict:
        return {
            "pipeline.lang": self._lang_combo.currentText(),
            "pipeline.whisper_model": self._model_combo.currentName(),
            "pipeline.whisper_device": self._whisper_device_combo.currentText(),
            "pipeline.cond_prev": self._chk_cond_prev.isChecked(),
            "pipeline.vad_filter": self._chk_vad.isChecked(),
            "pipeline.initial_prompt": self._prompt_edit.text(),
            "pipeline.align_model": self._align_model_combo.currentName(),
            "pipeline.retain_whisper": self._chk_whisper.isChecked(),
            "pipeline.retain_align": self._chk_force_align.isChecked(),
            "pipeline.retain_review": self._chk_review.isChecked(),
            "pipeline.enable_translate": self._chk_translate.isChecked(),
            "pipeline.tiger_denoise": self._chk_tiger_denoise.isChecked(),
            "pipeline.mss_model": self._mss_model_combo.currentName(),
            "pipeline.tiger_separate": self._chk_tiger_separate.isChecked(),
            "pipeline.ss_model": self._ss_model_combo.currentName(),
            "pipeline.tiger_save": self._chk_tiger_save.isChecked(),
            "pipeline.ref_sub": self._chk_ref_sub.isChecked(),
            "pipeline.ref_srt_path": self._ref_srt_path,
            "pipeline.ref_direct": self._chk_ref_direct.isChecked(),
            "pipeline.media_files": list(self._media_files),
        }

    def restore_state(self, state: dict):
        if "pipeline.lang" in state:
            self._lang_combo.setCurrentText(state["pipeline.lang"])
        if "pipeline.whisper_model" in state:
            self._model_combo.setCurrentName(state["pipeline.whisper_model"])
        if "pipeline.whisper_device" in state:
            self._whisper_device_combo.setCurrentText(state["pipeline.whisper_device"])
        if "pipeline.cond_prev" in state:
            self._chk_cond_prev.setChecked(state["pipeline.cond_prev"])
        if "pipeline.vad_filter" in state:
            self._chk_vad.setChecked(state["pipeline.vad_filter"])
        if "pipeline.initial_prompt" in state:
            self._prompt_edit.setText(state["pipeline.initial_prompt"])
        if "pipeline.align_model" in state:
            self._align_model_combo.setCurrentName(state["pipeline.align_model"])
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
        if "pipeline.mss_model" in state:
            self._mss_model_combo.setCurrentName(state["pipeline.mss_model"])
        if "pipeline.tiger_separate" in state:
            self._chk_tiger_separate.setChecked(state["pipeline.tiger_separate"])
        if "pipeline.ss_model" in state:
            self._ss_model_combo.setCurrentName(state["pipeline.ss_model"])
        if "pipeline.tiger_save" in state:
            self._chk_tiger_save.setChecked(state["pipeline.tiger_save"])
        if "pipeline.ref_sub" in state:
            self._chk_ref_sub.setChecked(state["pipeline.ref_sub"])
        if "pipeline.ref_srt_path" in state and state["pipeline.ref_srt_path"]:
            p = state["pipeline.ref_srt_path"]
            if os.path.isfile(p):
                self._ref_srt_path = p
                self._ref_srt_label.setText(os.path.basename(p))
                self._ref_srt_label.setToolTip(p)
        if "pipeline.ref_direct" in state:
            self._chk_ref_direct.setChecked(state["pipeline.ref_direct"])
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

    def _browse_ref_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考 SRT 文件", "",
            "SRT 字幕 (*.srt);;全部 (*.*)",
        )
        if path:
            self._ref_srt_path = path
            self._ref_srt_label.setText(os.path.basename(path))
            self._ref_srt_label.setToolTip(path)
            self._auto_save()

    def _clear_ref_srt(self):
        self._ref_srt_path = ""
        self._ref_srt_label.setText("无")
        self._ref_srt_label.setToolTip("指定外部 SRT 文件后，优先使用外部文件而非内嵌字幕")
        self._auto_save()

    # ── UI 回调 ──

    def _toggle_log(self):
        visible = self._log.isVisible()
        self._log.setVisible(not visible)
        self._btn_toggle_log.setText("▼ 隐藏日志" if not visible else "▶ 查看日志")

    def _set_buttons_state(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        if not running:
            self._pause_btn.setText("暂停")

    def _on_finished(self):
        self._running = False
        self._set_buttons_state(False)

    def _toggle_pause(self):
        reg = ResourceRegistry.instance()
        if reg.paused:
            reg.resume()
            self._pause_btn.setText("暂停")
            self._signals.log_requested.emit("已恢复执行")
        else:
            reg.pause()
            self._pause_btn.setText("继续")
            self._signals.log_requested.emit("已暂停，点击「继续」恢复")

    def _cancel(self):
        reg = ResourceRegistry.instance()
        reg.cancelled = True
        reg.resume()
        self._signals.log_requested.emit("取消请求已发送，将在当前段结束后停止")

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

        self._save_now()
        self._running = True
        ResourceRegistry.instance().reset_controls()
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        from writansub.pipeline.runner import PipelineConfig

        translate_cfg = load_translate_config() if self._chk_translate.isChecked() else None
        pp = {k: v.value() for k, v in self._pp_vars.items()}

        cfg = PipelineConfig(
            media_files=list(self._media_files),
            lang=self._lang_combo.currentText(),
            device=self._whisper_device_combo.currentText(),
            whisper_model=self._model_combo.currentName(),
            align_model=self._align_model_combo.currentName(),
            condition_on_prev=self._chk_cond_prev.isChecked(),
            vad_filter=self._chk_vad.isChecked(),
            initial_prompt=self._prompt_edit.text().strip() or None,
            tiger_mode=self._resolve_tiger_mode(),
            mss_model=self._mss_model_combo.currentName(),
            ss_model=self._ss_model_combo.currentName(),
            save_intermediate=self._chk_tiger_save.isChecked(),
            ref_srt=self._ref_srt_path or None,
            use_ref_sub=self._chk_ref_sub.isChecked(),
            ref_direct=self._chk_ref_direct.isChecked(),
            keep_whisper_srt=self._chk_whisper.isChecked(),
            keep_aligned_srt=self._chk_force_align.isChecked(),
            generate_review=self._chk_review.isChecked(),
            translate=bool(translate_cfg),
            extend_end=pp.get("extend_end", 0.30),
            extend_start=pp.get("extend_start", 0.00),
            gap_threshold=pp.get("gap_threshold", 0.50),
            min_gap=pp.get("min_gap", 0.30),
            word_conf_threshold=pp.get("word_conf_threshold", 0.50),
            align_conf_threshold=pp.get("align_conf_threshold", 0.50),
            min_duration=pp.get("min_duration", 0.30),
            pad_sec=pp.get("pad_sec", 0.50),
        )
        if translate_cfg:
            cfg.api_base = translate_cfg.get("api_base", cfg.api_base)
            cfg.api_key = translate_cfg.get("api_key", cfg.api_key)
            cfg.llm_model = translate_cfg.get("model", cfg.llm_model)
            cfg.target_lang = translate_cfg.get("target_lang", cfg.target_lang)
            cfg.batch_size = translate_cfg.get("batch_size", cfg.batch_size)

        thread = threading.Thread(
            target=self._run_pipeline, args=(cfg,), daemon=True,
        )
        thread.start()

    # ── 流水线执行 (后台线程) ──

    def _run_pipeline(self, cfg):
        from writansub.pipeline.runner import run_pipeline

        log_emit = self._signals.log_requested.emit
        prog_emit = self._signals.progress_requested.emit

        try:
            run_pipeline(cfg, log=log_emit, progress=prog_emit)
        except CancelledError:
            log_emit("任务已取消")
        except Exception as e:
            log_emit(f"发生错误: {e}")
        finally:
            self._signals.finished.emit()
