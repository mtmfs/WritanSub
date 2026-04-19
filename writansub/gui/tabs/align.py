import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.types import LANGUAGES, ALIGN_MODELS
from writansub.subtitle.srt_io import parse_srt, write_srt
from writansub.align.core import (
    load_audio, run_alignment, post_process, init_model,
    init_qwen3_model, run_qwen3_alignment,
)
from writansub.config import load_gui_state
from writansub.bridge import ResourceRegistry, CancelledError
from writansub.gui.widgets import (
    LogWidget, ProgressWidget, build_params_grid,
    NoScrollComboBox, GroupedComboBox, StateMixin,
)


class _AlignSignals(QObject):
    finished = Signal()


class AlignmentTab(StateMixin, QWidget):

    _PP_KEYS = [
        "extend_end", "extend_start", "gap_threshold",
        "min_gap", "min_duration", "pad_sec",
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
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, 1)

        # ── 上半部分：设置 + 操作 + 进度 ──
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        settings = QWidget()
        top_layout.addWidget(settings, 1)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(12, 12, 12, 12)

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

        card_param = QGroupBox("参数")
        settings_layout.addWidget(card_param)
        param_main = QVBoxLayout(card_param)

        basic_row = QHBoxLayout()
        param_main.addLayout(basic_row)
        basic_row.addWidget(QLabel("语言"))
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        basic_row.addWidget(self._lang_combo)

        basic_row.addWidget(QLabel("对齐模型"))
        self._align_model_combo = GroupedComboBox()
        self._align_model_combo.set_grouped_items(ALIGN_MODELS)
        self._align_model_combo.setCurrentName("mms_fa")
        basic_row.addWidget(self._align_model_combo)

        basic_row.addWidget(QLabel("设备"))
        self._device_combo = NoScrollComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        basic_row.addWidget(self._device_combo)
        basic_row.addStretch()

        pp_frame = QWidget()
        param_main.addWidget(pp_frame)
        self._pp_vars = build_params_grid(pp_frame, self._PP_KEYS)

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

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        action_layout.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        action_layout.addWidget(self._cancel_btn)

        self._start_btn = QPushButton("开始对齐")
        self._start_btn.clicked.connect(self._start)
        action_layout.addWidget(self._start_btn)

        self._progress = ProgressWidget()
        bottom_layout.addWidget(self._progress)

        self._log = LogWidget()
        bottom_layout.addWidget(self._log, 1)

        splitter.addWidget(bottom_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _connect_state_signals(self):
        self._audio_edit.editingFinished.connect(self._auto_save)
        self._srt_edit.editingFinished.connect(self._auto_save)
        self._out_edit.editingFinished.connect(self._auto_save)
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._align_model_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)

    def save_state(self) -> dict:
        return {
            "alignment.audio": self._audio_edit.text(),
            "alignment.srt": self._srt_edit.text(),
            "alignment.output": self._out_edit.text(),
            "alignment.lang": self._lang_combo.currentText(),
            "alignment.model": self._align_model_combo.currentName(),
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
        if "alignment.model" in state:
            self._align_model_combo.setCurrentName(state["alignment.model"])
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

    def _set_buttons_state(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        if not running:
            self._pause_btn.setText("暂停")

    def _on_finished(self):
        self._set_buttons_state(False)

    def _toggle_pause(self):
        reg = ResourceRegistry.instance()
        if reg.paused:
            reg.resume()
            self._pause_btn.setText("暂停")
            self._log.log("已恢复执行")
        else:
            reg.pause()
            self._pause_btn.setText("继续")
            self._log.log("已暂停，点击「继续」恢复")

    def _cancel(self):
        reg = ResourceRegistry.instance()
        reg.cancelled = True
        reg.resume()
        self._log.log("正在取消...")

    def _start(self):
        audio = self._audio_edit.text().strip()
        srt = self._srt_edit.text().strip()
        output = self._out_edit.text().strip()

        if not audio:
            self._log.log("请选择音频文件")
            return
        if not srt:
            self._log.log("请选择字幕文件")
            return
        if not output:
            base = srt.rsplit(".", 1)[0]
            output = f"{base}_aligned.srt"
            self._out_edit.setText(output)

        self._save_now()
        ResourceRegistry.instance().reset_controls()
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        pp = {k: v.value() for k, v in self._pp_vars.items()}
        lang = self._lang_combo.currentText()
        align_model = self._align_model_combo.currentName()
        device = self._device_combo.currentText()

        thread = threading.Thread(
            target=self._run_alignment,
            args=(audio, srt, output, device, pp, lang, align_model),
            daemon=True,
        )
        thread.start()

    def _run_alignment(self, audio: str, srt: str, output: str,
                       device: str, pp: dict[str, float],
                       lang: str = "ja", align_model: str = "mms_fa"):
        import torch

        reg = ResourceRegistry.instance()
        model_handle = None
        try:
            if device == "cuda" and not torch.cuda.is_available():
                self._log.log("CUDA 不可用，回退到 CPU")
                device = "cpu"

            self._progress.update_progress(0.0, "加载音频...")
            waveform = load_audio(audio)

            self._progress.update_progress(0.05, "解析字幕...")
            subs = parse_srt(srt, lang=lang)
            self._log.log(f"字幕条数: {len(subs)}")

            self._progress.update_progress(0.1, "加载模型...")
            pad_sec = pp.pop("pad_sec", 0.5)

            if align_model == "qwen3-fa-0.6b":
                qwen3_model = init_qwen3_model(device)
                model_handle = reg.register_model("qwen3_fa", qwen3_model, device)

                aligned = run_qwen3_alignment(
                    waveform, subs, device=device, pad_sec=pad_sec,
                    progress_callback=lambda p, m: self._progress.update_progress(0.1 + p * 0.85, m),
                    model=qwen3_model, lang=lang,
                    log_callback=self._log.log,
                )
            else:
                model_bundle = init_model(device)
                model_handle = reg.register_model("mms_fa", model_bundle, device)

                aligned = run_alignment(
                    waveform, subs, device=device, pad_sec=pad_sec,
                    progress_callback=lambda p, m: self._progress.update_progress(0.1 + p * 0.85, m),
                    model_bundle=model_bundle,
                    log_callback=self._log.log,
                )

            self._progress.update_progress(0.95, "后处理...")
            pp.pop("align_conf_threshold", None)
            final = post_process(aligned, **pp)

            write_srt(final, output)
            self._progress.update_progress(1.0, "对齐完成")
            self._log.log(f"完成! 输出: {output}")
        except CancelledError:
            self._log.log("对齐已取消")
        except Exception as e:
            from writansub.logger import log_exception, session_log_path
            log_exception("align._run_alignment", e)
            self._log.log(f"出错: {e}")
            path = session_log_path()
            if path:
                self._log.log(f"详细日志已写入: {path}")
        finally:
            if model_handle is not None:
                reg.unload_model(model_handle)
            self._signals.finished.emit()
