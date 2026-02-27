"""单独语音识别页"""

import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QCheckBox,
    QFileDialog, QFrame, QDoubleSpinBox,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.core.types import MEDIA_FILETYPES, SRT_FILETYPES, LANGUAGES
from writansub.core.whisper import transcribe_to_srt
from writansub.config import PP_DEFAULTS, PARAM_DEFS, load_pp_config, save_pp_config, load_gui_state, save_gui_state
from writansub.gui.widgets import LogWidget, ProgressWidget


class _WhisperSignals(QObject):
    """线程安全的信号"""
    finished = Signal()


class WhisperTab(QWidget):
    """Tab 2: 独立 Whisper 语音识别"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _WhisperSignals()
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

        file_layout.addWidget(QLabel("媒体文件"), 0, 0)
        self._media_edit = QLineEdit()
        self._media_edit.textChanged.connect(self._on_media_changed)
        file_layout.addWidget(self._media_edit, 0, 1)
        btn_media = QPushButton("浏览")
        btn_media.clicked.connect(self._browse_media)
        file_layout.addWidget(btn_media, 0, 2)

        file_layout.addWidget(QLabel("输出 SRT"), 1, 0)
        self._output_edit = QLineEdit()
        file_layout.addWidget(self._output_edit, 1, 1)
        btn_output = QPushButton("浏览")
        btn_output.clicked.connect(self._browse_output)
        file_layout.addWidget(btn_output, 1, 2)

        file_layout.setColumnStretch(1, 1)

        # 参数
        card_param = QGroupBox("参数")
        settings_layout.addWidget(card_param)
        param_layout = QHBoxLayout(card_param)

        param_layout.addWidget(QLabel("语言"))
        self._lang_combo = QComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        param_layout.addWidget(self._lang_combo)

        param_layout.addWidget(QLabel("设备"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        param_layout.addWidget(self._device_combo)

        lbl_wc = QLabel("置信阈值")
        tip_text = PARAM_DEFS["word_conf_threshold"].get("tip", "")
        if tip_text:
            lbl_wc.setText("置信阈值 ⓘ")
            lbl_wc.setToolTip(tip_text)
            lbl_wc.setCursor(Qt.WhatsThisCursor)
        param_layout.addWidget(lbl_wc)

        cfg = load_pp_config()
        self._wc_spin = QDoubleSpinBox()
        self._wc_spin.setRange(0.0, 1.0)
        self._wc_spin.setSingleStep(0.05)
        self._wc_spin.setDecimals(2)
        self._wc_spin.setValue(cfg.get("word_conf_threshold", PP_DEFAULTS["word_conf_threshold"]))
        self._wc_spin.valueChanged.connect(self._on_wc_change)
        param_layout.addWidget(self._wc_spin)

        self._chk_cond_prev = QCheckBox("上文关联")
        self._chk_cond_prev.setChecked(True)
        self._chk_cond_prev.setToolTip(
            "开启时，前一句识别结果作为下一句的上下文，\n"
            "提高连贯性但可能传播错误。\n"
            "关闭可防止幻觉扩散。"
        )
        param_layout.addWidget(self._chk_cond_prev)
        param_layout.addStretch()

        settings_layout.addStretch()

        # 操作按钮
        action_bar = QWidget()
        main_layout.addWidget(action_bar)
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(12, 6, 12, 6)
        action_layout.addStretch()
        self._start_btn = QPushButton("开始识别")
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
        self._media_edit.editingFinished.connect(self._auto_save)
        self._output_edit.editingFinished.connect(self._auto_save)
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)

    def _auto_save(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def save_state(self) -> dict:
        return {
            "whisper.media": self._media_edit.text(),
            "whisper.output": self._output_edit.text(),
            "whisper.lang": self._lang_combo.currentText(),
            "whisper.device": self._device_combo.currentText(),
            "whisper.cond_prev": self._chk_cond_prev.isChecked(),
        }

    def restore_state(self, state: dict):
        if "whisper.media" in state:
            self._media_edit.setText(state["whisper.media"])
        if "whisper.output" in state:
            self._output_edit.setText(state["whisper.output"])
        if "whisper.lang" in state:
            self._lang_combo.setCurrentText(state["whisper.lang"])
        if "whisper.device" in state:
            self._device_combo.setCurrentText(state["whisper.device"])
        if "whisper.cond_prev" in state:
            self._chk_cond_prev.setChecked(state["whisper.cond_prev"])

    def _browse_media(self):
        filter_str = "媒体文件 (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.aac *.ogg *.m4a);;所有文件 (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "选择媒体文件", "", filter_str)
        if path:
            self._media_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 SRT", "", "SRT 字幕 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self._output_edit.setText(path)

    def _on_media_changed(self, text: str):
        media = text.strip()
        if media and not self._output_edit.text().strip():
            self._output_edit.setText(os.path.splitext(media)[0] + ".srt")

    def _on_wc_change(self, value: float):
        try:
            current = load_pp_config()
            current["word_conf_threshold"] = round(value, 2)
            save_pp_config(current)
        except Exception:
            pass

    def _log_msg(self, msg: str):
        self._log.log(msg)

    def _update_progress(self, pct: float, msg: str):
        self._progress.update_progress(pct, msg)

    def _on_finished(self):
        self._start_btn.setEnabled(True)

    def _start(self):
        media = self._media_edit.text().strip()
        if not media or not os.path.isfile(media):
            self._log_msg("请先选择有效的媒体文件")
            return

        output = self._output_edit.text().strip()
        if not output:
            output = os.path.splitext(media)[0] + ".srt"
            self._output_edit.setText(output)

        self._start_btn.setEnabled(False)
        self._log.clear_log()
        self._progress.reset()

        lang = self._lang_combo.currentText()
        device = self._device_combo.currentText()
        wc = self._wc_spin.value()
        cond_prev = self._chk_cond_prev.isChecked()

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
                media, lang=lang, device=device, log_callback=self._log_msg,
                progress_callback=self._update_progress,
                word_conf_threshold=wc_threshold,
                condition_on_previous_text=cond_prev,
            )
            if os.path.abspath(srt_path) != os.path.abspath(output):
                import shutil
                shutil.move(srt_path, output)
            self._update_progress(1.0, "识别完成")
        except Exception as e:
            self._log_msg(f"出错: {e}")
        finally:
            self._signals.finished.emit()
