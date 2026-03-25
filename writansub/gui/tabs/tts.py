"""[废案模块] 语音合成页：多模型选择（MMS_FA / Style-Bert-VITS2）
未接入主程序，依赖的 TTS_MODELS / load_tts_config / register_thread 等尚未实现。
"""

import os
import sys
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox,
    QFileDialog, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.types import (
    TTS_MODELS, LANGUAGES, MEDIA_FILETYPES, SRT_FILETYPES, AUDIO_FILETYPES,
)
from writansub.config import PARAM_DEFS, load_gui_state, load_tts_config, save_tts_config
from writansub.bridge import ResourceRegistry
from writansub.gui.widgets import (
    TextRedirector, LogWidget, ProgressWidget,
    GroupedComboBox, NoScrollComboBox, NoScrollSpinBox,
    build_params_grid, StateMixin,
)


class _TTSSignals(QObject):
    finished = Signal()


class TTSTab(StateMixin, QWidget):
    """Tab: 语音合成（多模型）"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "min_duration",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _TTSSignals()
        self._signals.finished.connect(self._on_finished)
        self._setup_ui()
        self._on_model_changed()
        self._connect_state_signals()
        self.restore_state(load_gui_state())

    # ── UI 构建 ────────────────────────────────────────

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # ── 上半部分：设置 ──
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        settings = QWidget()
        top_layout.addWidget(settings, 1)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(12, 12, 12, 12)

        # 模型选择
        card_model = QGroupBox("模型")
        settings_layout.addWidget(card_model)
        model_layout = QHBoxLayout(card_model)

        model_layout.addWidget(QLabel("模型"))
        self._model_combo = GroupedComboBox()
        self._model_combo.set_grouped_items(TTS_MODELS)
        self._model_combo.setCurrentName("sbv2-jp-extra")
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self._model_combo)

        model_layout.addWidget(QLabel("设备"))
        self._device_combo = NoScrollComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        model_layout.addWidget(self._device_combo)
        model_layout.addStretch()

        # 共用：SRT 输入
        card_file = QGroupBox("文件")
        settings_layout.addWidget(card_file)
        file_layout = QGridLayout(card_file)

        file_layout.addWidget(QLabel("字幕文件"), 0, 0)
        self._srt_edit = QLineEdit()
        self._srt_edit.textChanged.connect(self._on_srt_changed)
        file_layout.addWidget(self._srt_edit, 0, 1)
        btn_srt = QPushButton("浏览")
        btn_srt.clicked.connect(self._browse_srt)
        file_layout.addWidget(btn_srt, 0, 2)

        # MMS_FA 专属：音频输入 + 输出 SRT
        self._lbl_audio = QLabel("音频文件")
        file_layout.addWidget(self._lbl_audio, 1, 0)
        self._audio_edit = QLineEdit()
        file_layout.addWidget(self._audio_edit, 1, 1)
        self._btn_audio = QPushButton("浏览")
        self._btn_audio.clicked.connect(self._browse_audio)
        file_layout.addWidget(self._btn_audio, 1, 2)

        self._lbl_out_srt = QLabel("输出 SRT")
        file_layout.addWidget(self._lbl_out_srt, 2, 0)
        self._out_srt_edit = QLineEdit()
        file_layout.addWidget(self._out_srt_edit, 2, 1)
        self._btn_out_srt = QPushButton("浏览")
        self._btn_out_srt.clicked.connect(self._browse_out_srt)
        file_layout.addWidget(self._btn_out_srt, 2, 2)

        # SBV2 专属：输出 WAV
        self._lbl_out_wav = QLabel("输出 WAV")
        file_layout.addWidget(self._lbl_out_wav, 3, 0)
        self._out_wav_edit = QLineEdit()
        file_layout.addWidget(self._out_wav_edit, 3, 1)
        self._btn_out_wav = QPushButton("浏览")
        self._btn_out_wav.clicked.connect(self._browse_out_wav)
        file_layout.addWidget(self._btn_out_wav, 3, 2)

        file_layout.setColumnStretch(1, 1)

        # ── MMS_FA 专属参数面板 ──
        self._mms_panel = QGroupBox("对齐参数")
        settings_layout.addWidget(self._mms_panel)
        mms_layout = QVBoxLayout(self._mms_panel)

        lang_row = QHBoxLayout()
        mms_layout.addLayout(lang_row)
        lang_row.addWidget(QLabel("语言"))
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch()

        pp_frame = QWidget()
        mms_layout.addWidget(pp_frame)
        self._pp_vars = build_params_grid(pp_frame, self._PP_KEYS)

        # ── SBV2 专属参数面板 ──
        self._sbv2_panel = QGroupBox("合成参数")
        settings_layout.addWidget(self._sbv2_panel)
        sbv2_layout = QVBoxLayout(self._sbv2_panel)

        # 模型目录
        dir_row = QHBoxLayout()
        sbv2_layout.addLayout(dir_row)
        dir_row.addWidget(QLabel("模型目录"))
        self._model_dir_edit = QLineEdit()
        dir_row.addWidget(self._model_dir_edit, 1)
        btn_dir = QPushButton("浏览")
        btn_dir.clicked.connect(self._browse_model_dir)
        dir_row.addWidget(btn_dir)
        btn_load = QPushButton("加载模型信息")
        btn_load.clicked.connect(self._load_model_info)
        dir_row.addWidget(btn_load)

        # Speaker / Style / Speed
        param_row = QHBoxLayout()
        sbv2_layout.addLayout(param_row)

        param_row.addWidget(QLabel("说话人"))
        self._speaker_combo = NoScrollComboBox()
        param_row.addWidget(self._speaker_combo)

        param_row.addWidget(QLabel("风格"))
        self._style_combo = NoScrollComboBox()
        param_row.addWidget(self._style_combo)

        param_row.addWidget(QLabel("语速"))
        self._speed_spin = NoScrollSpinBox()
        self._speed_spin.setRange(0.5, 2.0)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setFixedWidth(80)
        param_row.addWidget(self._speed_spin)

        self._chk_translated = QCheckBox("使用译文合成")
        param_row.addWidget(self._chk_translated)
        param_row.addStretch()

        settings_layout.addStretch()
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
        action_layout.addStretch()
        self._start_btn = QPushButton("开始")
        self._start_btn.clicked.connect(self._start)
        action_layout.addWidget(self._start_btn)

        self._progress = ProgressWidget()
        bottom_layout.addWidget(self._progress)

        self._log = LogWidget()
        bottom_layout.addWidget(self._log, 1)

        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    # ── 模型切换：显示/隐藏专属面板 ──────────────────────

    def _on_model_changed(self):
        model = self._model_combo.currentName()
        is_mms = (model == "mms_fa")
        is_sbv2 = (model == "sbv2-jp-extra")

        # MMS_FA 专属
        for w in (self._lbl_audio, self._audio_edit, self._btn_audio,
                  self._lbl_out_srt, self._out_srt_edit, self._btn_out_srt):
            w.setVisible(is_mms)
        self._mms_panel.setVisible(is_mms)

        # SBV2 专属
        for w in (self._lbl_out_wav, self._out_wav_edit, self._btn_out_wav):
            w.setVisible(is_sbv2)
        self._sbv2_panel.setVisible(is_sbv2)

        # 按钮文字
        self._start_btn.setText("开始对齐" if is_mms else "开始合成")

    # ── 状态持久化 ─────────────────────────────────────

    def _connect_state_signals(self):
        self._model_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)
        self._srt_edit.editingFinished.connect(self._auto_save)
        self._audio_edit.editingFinished.connect(self._auto_save)
        self._out_srt_edit.editingFinished.connect(self._auto_save)
        self._out_wav_edit.editingFinished.connect(self._auto_save)
        self._model_dir_edit.editingFinished.connect(self._auto_save)
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._speaker_combo.currentTextChanged.connect(self._auto_save)
        self._style_combo.currentTextChanged.connect(self._auto_save)
        self._speed_spin.valueChanged.connect(self._auto_save)
        self._chk_translated.stateChanged.connect(self._auto_save)

    def save_state(self) -> dict:
        return {
            "tts.model": self._model_combo.currentName(),
            "tts.device": self._device_combo.currentText(),
            "tts.srt": self._srt_edit.text(),
            "tts.audio": self._audio_edit.text(),
            "tts.out_srt": self._out_srt_edit.text(),
            "tts.out_wav": self._out_wav_edit.text(),
            "tts.model_dir": self._model_dir_edit.text(),
            "tts.lang": self._lang_combo.currentText(),
            "tts.speaker": self._speaker_combo.currentText(),
            "tts.style": self._style_combo.currentText(),
            "tts.speed": self._speed_spin.value(),
            "tts.use_translated": self._chk_translated.isChecked(),
        }

    def restore_state(self, state: dict):
        if "tts.model" in state:
            self._model_combo.setCurrentName(state["tts.model"])
        if "tts.device" in state:
            self._device_combo.setCurrentText(state["tts.device"])
        if "tts.srt" in state:
            self._srt_edit.setText(state["tts.srt"])
        if "tts.audio" in state:
            self._audio_edit.setText(state["tts.audio"])
        if "tts.out_srt" in state:
            self._out_srt_edit.setText(state["tts.out_srt"])
        if "tts.out_wav" in state:
            self._out_wav_edit.setText(state["tts.out_wav"])
        if "tts.model_dir" in state:
            self._model_dir_edit.setText(state["tts.model_dir"])
        if "tts.lang" in state:
            self._lang_combo.setCurrentText(state["tts.lang"])
        if "tts.speaker" in state:
            self._speaker_combo.setCurrentText(state["tts.speaker"])
        if "tts.style" in state:
            self._style_combo.setCurrentText(state["tts.style"])
        if "tts.speed" in state:
            self._speed_spin.setValue(state["tts.speed"])
        if "tts.use_translated" in state:
            self._chk_translated.setChecked(state["tts.use_translated"])
        self._on_model_changed()

    # ── 浏览按钮 ──────────────────────────────────────

    def _browse_srt(self):
        filter_str = ";;".join(f"{l} ({e})" for l, e in SRT_FILETYPES)
        path, _ = QFileDialog.getOpenFileName(self, "选择 SRT 字幕", "", filter_str)
        if path:
            self._srt_edit.setText(path)

    def _browse_audio(self):
        filter_str = ";;".join(f"{l} ({e})" for l, e in MEDIA_FILETYPES)
        path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", filter_str)
        if path:
            self._audio_edit.setText(path)

    def _browse_out_srt(self):
        filter_str = ";;".join(f"{l} ({e})" for l, e in SRT_FILETYPES)
        path, _ = QFileDialog.getSaveFileName(self, "保存 SRT", "", filter_str)
        if path:
            self._out_srt_edit.setText(path)

    def _browse_out_wav(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 WAV", "", "WAV 音频 (*.wav);;所有文件 (*.*)"
        )
        if path:
            self._out_wav_edit.setText(path)

    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self._model_dir_edit.setText(path)

    def _on_srt_changed(self, text: str):
        srt = text.strip()
        if not srt:
            return
        base = os.path.splitext(srt)[0]
        model = self._model_combo.currentName()
        if model == "mms_fa" and not self._out_srt_edit.text().strip():
            self._out_srt_edit.setText(f"{base}_aligned.srt")
        elif model == "sbv2-jp-extra" and not self._out_wav_edit.text().strip():
            self._out_wav_edit.setText(f"{base}_tts.wav")

    # ── SBV2 模型信息加载 ─────────────────────────────

    def _load_model_info(self):
        model_dir = self._model_dir_edit.text().strip()
        if not model_dir:
            self._log.log("请先选择模型目录")
            return
        try:
            from writansub.tts.core import load_model_meta
            meta = load_model_meta("sbv2-jp-extra", model_dir)
            self._speaker_combo.clear()
            self._speaker_combo.addItems(meta["speakers"])
            self._style_combo.clear()
            self._style_combo.addItems(meta["styles"])
            self._log.log(
                f"模型信息加载完成: {len(meta['speakers'])} 说话人, "
                f"{len(meta['styles'])} 风格, {meta['sample_rate']}Hz"
            )
        except Exception as e:
            self._log.log(f"加载模型信息失败: {e}")

    # ── 执行 ──────────────────────────────────────────

    def _on_finished(self):
        self._start_btn.setEnabled(True)

    def _start(self):
        srt = self._srt_edit.text().strip()
        if not srt or not os.path.isfile(srt):
            self._log.log("请选择有效的字幕文件")
            return

        model_name = self._model_combo.currentName()

        if model_name == "mms_fa":
            audio = self._audio_edit.text().strip()
            if not audio or not os.path.isfile(audio):
                self._log.log("请选择有效的音频文件")
                return
            output = self._out_srt_edit.text().strip()
            if not output:
                output = os.path.splitext(srt)[0] + "_aligned.srt"
                self._out_srt_edit.setText(output)
        else:
            model_dir = self._model_dir_edit.text().strip()
            if not model_dir or not os.path.isdir(model_dir):
                self._log.log("请选择有效的模型目录")
                return
            output = self._out_wav_edit.text().strip()
            if not output:
                output = os.path.splitext(srt)[0] + "_tts.wav"
                self._out_wav_edit.setText(output)

        self._start_btn.setEnabled(False)
        self._log.clear_log()
        self._progress.reset()

        thread = threading.Thread(
            target=self._run,
            args=(model_name, srt, output),
            daemon=True,
        )
        reg = ResourceRegistry.instance()
        self._thread_handle = reg.register_thread(thread)
        thread.start()

    def _run(self, model_name: str, srt: str, output: str):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        reg = ResourceRegistry.instance()
        model_handle = None
        try:
            device = self._device_combo.currentText()
            if device == "cuda" and not torch.cuda.is_available():
                self._log.log("CUDA 不可用，回退到 CPU")
                device = "cpu"

            from writansub.subtitle.srt_io import parse_srt
            from writansub.tts.core import init_model

            if model_name == "mms_fa":
                self._run_mms_fa(srt, output, device, reg)
            elif model_name == "sbv2-jp-extra":
                self._run_sbv2(srt, output, device, reg)

        except Exception as e:
            self._log.log(f"出错: {e}")
        finally:
            sys.stdout = old_stdout
            reg.unregister_thread(self._thread_handle)
            self._signals.finished.emit()

    def _run_mms_fa(self, srt: str, output: str, device: str, reg: ResourceRegistry):
        from writansub.subtitle.srt_io import parse_srt, write_srt
        from writansub.tts.core import init_model, run_mms_fa

        lang = self._lang_combo.currentText()

        self._progress.update_progress(0.0, "加载模型...")
        model_bundle = init_model("mms_fa", device)
        handle = reg.register_model("mms_fa", model_bundle, device)

        try:
            self._progress.update_progress(0.05, "解析字幕...")
            subs = parse_srt(srt, lang=lang)

            pp = {k: v.value() for k, v in self._pp_vars.items()}
            final = run_mms_fa(
                audio_path=self._audio_edit.text().strip(),
                subs=subs, device=device, model_bundle=model_bundle,
                pp_params=pp, lang=lang,
                log_callback=self._log.log,
                progress_callback=self._progress.update_progress,
            )
            write_srt(final, output)
            self._log.log(f"完成! 输出: {output}")
        finally:
            reg.unload_model(handle)

    def _run_sbv2(self, srt: str, output: str, device: str, reg: ResourceRegistry):
        import soundfile as sf
        from writansub.subtitle.srt_io import parse_srt
        from writansub.tts.core import init_model, run_sbv2

        model_dir = self._model_dir_edit.text().strip()

        self._progress.update_progress(0.0, "加载模型...")
        model = init_model("sbv2-jp-extra", device, model_dir)
        handle = reg.register_model(f"sbv2:{model_dir}", model, device)

        try:
            self._progress.update_progress(0.05, "解析字幕...")
            subs = parse_srt(srt)

            audio, sr = run_sbv2(
                subs=subs, model=model,
                speaker=self._speaker_combo.currentText(),
                style=self._style_combo.currentText(),
                speed=self._speed_spin.value(),
                use_translated=self._chk_translated.isChecked(),
                log_callback=self._log.log,
                progress_callback=lambda p, m: self._progress.update_progress(
                    0.05 + p * 0.95, m
                ),
            )

            sf.write(output, audio, sr, subtype="PCM_16")
            self._progress.update_progress(1.0, "合成完成")
            self._log.log(f"完成! 输出: {output}")
        finally:
            reg.unload_model(handle)
