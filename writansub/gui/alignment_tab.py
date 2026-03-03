"""单独强制打轴页"""

import sys
import threading
from typing import Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QFileDialog, QFrame,
)
from PySide6.QtCore import Signal, QObject

from writansub.core.types import AUDIO_FILETYPES, SRT_FILETYPES, LANGUAGES
from writansub.core.srt_io import parse_srt, write_srt
from writansub.core.alignment import load_audio, run_alignment, post_process, init_model
from writansub.config import load_gui_state, save_gui_state
from writansub.registry import ResourceRegistry
from writansub.gui.widgets import TextRedirector, LogWidget, ProgressWidget, build_params_grid


class _AlignSignals(QObject):
    """线程安全的信号"""
    finished = Signal()


class AlignmentTab(QWidget):
    """Tab 3: 独立 MMS_FA 强制打轴"""

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "min_duration",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _AlignSignals()
        self._signals.finished.connect(self._on_finished)
        self._setup_ui()
        self._connect_state_signals()
        self.restore_state(load_gui_state())

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        settings = QWidget()
        main_layout.addWidget(settings)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(12, 12, 12, 12)

        # 文件
        card_file = QGroupBox("文件")
        settings_layout.addWidget(card_file)
        file_layout = QGridLayout(card_file)

        file_layout.addWidget(QLabel("音频文件"), 0, 0)
        self._audio_edit = QLineEdit()
        file_layout.addWidget(self._audio_edit, 0, 1)
        btn_audio = QPushButton("浏览")
        btn_audio.clicked.connect(self._browse_audio)
        file_layout.addWidget(btn_audio, 0, 2)

        file_layout.addWidget(QLabel("字幕文件"), 1, 0)
        self._srt_edit = QLineEdit()
        self._srt_edit.textChanged.connect(self._on_srt_changed)
        file_layout.addWidget(self._srt_edit, 1, 1)
        btn_srt = QPushButton("浏览")
        btn_srt.clicked.connect(self._browse_srt)
        file_layout.addWidget(btn_srt, 1, 2)

        file_layout.addWidget(QLabel("输出路径"), 2, 0)
        self._out_edit = QLineEdit()
        file_layout.addWidget(self._out_edit, 2, 1)
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(self._browse_output)
        file_layout.addWidget(btn_out, 2, 2)

        file_layout.setColumnStretch(1, 1)

        # 参数
        card_param = QGroupBox("参数")
        settings_layout.addWidget(card_param)
        param_main = QVBoxLayout(card_param)

        basic_row = QHBoxLayout()
        param_main.addLayout(basic_row)
        basic_row.addWidget(QLabel("语言"))
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_row.addWidget(self._lang_combo)

        basic_row.addWidget(QLabel("设备"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        basic_row.addWidget(self._device_combo)
        basic_row.addStretch()

        pp_frame = QWidget()
        param_main.addWidget(pp_frame)
        self._pp_vars = build_params_grid(pp_frame, self._PP_KEYS)

        settings_layout.addStretch()

        # 操作按钮
        action_bar = QWidget()
        main_layout.addWidget(action_bar)
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 6, 12, 6)
        action_layout.addStretch()
        self._start_btn = QPushButton("开始对齐")
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
        self._audio_edit.editingFinished.connect(self._auto_save)
        self._srt_edit.editingFinished.connect(self._auto_save)
        self._out_edit.editingFinished.connect(self._auto_save)
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)

    def _auto_save(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def save_state(self) -> dict:
        return {
            "alignment.audio": self._audio_edit.text(),
            "alignment.srt": self._srt_edit.text(),
            "alignment.output": self._out_edit.text(),
            "alignment.lang": self._lang_combo.currentText(),
            "alignment.device": self._device_combo.currentText(),
        }

    def restore_state(self, state: dict):
        if "alignment.audio" in state:
            self._audio_edit.setText(state["alignment.audio"])
        if "alignment.srt" in state:
            self._srt_edit.setText(state["alignment.srt"])
        if "alignment.output" in state:
            self._out_edit.setText(state["alignment.output"])
        if "alignment.lang" in state:
            self._lang_combo.setCurrentText(state["alignment.lang"])
        if "alignment.device" in state:
            self._device_combo.setCurrentText(state["alignment.device"])

    def _browse_audio(self):
        filter_str = "音频文件 (*.wav *.mp3 *.flac *.ogg *.aac *.m4a);;所有文件 (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", filter_str)
        if path:
            self._audio_edit.setText(path)

    def _browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SRT 字幕", "", "SRT 字幕 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self._srt_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存输出 SRT", "", "SRT 字幕 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self._out_edit.setText(path)

    def _on_srt_changed(self, text: str):
        srt = text.strip()
        if srt and not self._out_edit.text().strip():
            base = srt.rsplit(".", 1)[0]
            self._out_edit.setText(f"{base}_aligned.srt")

    def _log_msg(self, msg: str):
        self._log.log(msg)

    def _update_progress(self, pct: float, msg: str):
        self._progress.update_progress(pct, msg)

    def _on_finished(self):
        self._start_btn.setEnabled(True)

    def _start(self):
        audio = self._audio_edit.text().strip()
        srt = self._srt_edit.text().strip()
        output = self._out_edit.text().strip()

        if not audio:
            self._log_msg("请选择音频文件")
            return
        if not srt:
            self._log_msg("请选择字幕文件")
            return
        if not output:
            base = srt.rsplit(".", 1)[0]
            output = f"{base}_aligned.srt"
            self._out_edit.setText(output)

        self._start_btn.setEnabled(False)
        self._log.clear_log()
        self._progress.reset()

        pp = {k: v.value() for k, v in self._pp_vars.items()}
        lang = self._lang_combo.currentText()
        device = self._device_combo.currentText()

        thread = threading.Thread(
            target=self._run_alignment,
            args=(audio, srt, output, device, pp, lang),
            daemon=True,
        )
        reg = ResourceRegistry.instance()
        self._thread_handle = reg.register_thread(thread)
        thread.start()

    def _run_alignment(self, audio: str, srt: str, output: str,
                       device: str, pp: Dict[str, float],
                       lang: str = "ja"):
        import torch

        redirector = TextRedirector(self._log)
        old_stdout = sys.stdout
        sys.stdout = redirector

        reg = ResourceRegistry.instance()
        model_handle = None
        try:
            if device == "cuda" and not torch.cuda.is_available():
                self._log_msg("CUDA 不可用，回退到 CPU")
                device = "cpu"

            self._update_progress(0.0, "加载音频...")
            waveform = load_audio(audio)

            self._update_progress(0.05, "解析字幕...")
            subs = parse_srt(srt, lang=lang)
            self._log_msg(f"字幕条数: {len(subs)}")

            self._update_progress(0.1, "加载模型...")
            model_bundle = init_model(device)
            model_handle = reg.register_model("mms_fa", model_bundle, device)

            aligned = run_alignment(
                waveform, subs, device=device,
                progress_callback=lambda p, m: self._update_progress(0.1 + p * 0.85, m),
                model_bundle=model_bundle,
            )

            self._update_progress(0.95, "后处理...")
            pp.pop("align_conf_threshold", None)
            final = post_process(aligned, **pp)

            write_srt(final, output)
            self._update_progress(1.0, "对齐完成")
            self._log_msg(f"完成! 输出: {output}")
        except Exception as e:
            self._log_msg(f"出错: {e}")
        finally:
            if model_handle is not None:
                reg.unload_model(model_handle)
            sys.stdout = old_stdout
            reg.unregister_thread(self._thread_handle)
            self._signals.finished.emit()
