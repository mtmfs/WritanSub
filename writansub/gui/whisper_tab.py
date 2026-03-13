"""单独语音识别页"""

import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox,
    QFileDialog, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.core.types import MEDIA_FILETYPES, SRT_FILETYPES, LANGUAGES
from writansub.core.whisper import transcribe
from writansub.core.review import generate_review, write_review_files
from writansub.core.srt_io import write_srt
from writansub.config import PARAM_DEFS, load_gui_state, save_gui_state
from writansub.registry import ResourceRegistry
from writansub.gui.widgets import LogWidget, ProgressWidget, NoScrollComboBox, ParamSpinBox


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
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        param_layout.addWidget(self._lang_combo)

        param_layout.addWidget(QLabel("设备"))
        self._device_combo = NoScrollComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        param_layout.addWidget(self._device_combo)

        wc_def = PARAM_DEFS["word_conf_threshold"]
        lbl_wc = QLabel(wc_def["label"] + " ⓘ")
        lbl_wc.setToolTip(wc_def.get("tip", ""))
        lbl_wc.setCursor(Qt.WhatsThisCursor)
        param_layout.addWidget(lbl_wc)

        self._wc_spin = ParamSpinBox("word_conf_threshold")
        self._wc_spin.setRange(wc_def["from"], wc_def["to"])
        self._wc_spin.setSingleStep(wc_def["inc"])
        self._wc_spin.setDecimals(2)
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
        self._start_btn = QPushButton("开始识别")
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
        filter_str = ";;".join(f"{label} ({exts})" for label, exts in MEDIA_FILETYPES)
        path, _ = QFileDialog.getOpenFileName(self, "选择媒体文件", "", filter_str)
        if path:
            self._media_edit.setText(path)

    def _browse_output(self):
        filter_str = ";;".join(f"{label} ({exts})" for label, exts in SRT_FILETYPES)
        path, _ = QFileDialog.getSaveFileName(self, "保存 SRT", "", filter_str)
        if path:
            self._output_edit.setText(path)

    def _on_media_changed(self, text: str):
        media = text.strip()
        if media and not self._output_edit.text().strip():
            self._output_edit.setText(os.path.splitext(media)[0] + ".srt")

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
        reg = ResourceRegistry.instance()
        self._thread_handle = reg.register_thread(thread)
        thread.start()

    def _run_whisper(self, media: str, output: str, lang: str,
                     device: str, wc_threshold: float,
                     cond_prev: bool = True):
        try:
            subs, word_data = transcribe(
                media, lang=lang, device=device, log_callback=self._log_msg,
                progress_callback=self._update_progress,
                condition_on_previous_text=cond_prev,
            )

            # 生成 review 文件
            if wc_threshold > 0.0:
                srt_content, ass_content, low_count, total_words = generate_review(
                    subs, word_data, wc_threshold,
                )
                if low_count > 0:
                    base = os.path.splitext(media)[0]
                    write_review_files(base, srt_content, ass_content)
                    self._log_msg(f"低置信词 {low_count}/{total_words}，已生成标记版")

            # 直接写到用户指定路径
            write_srt(subs, output)
            self._update_progress(1.0, "识别完成")
        except Exception as e:
            self._log_msg(f"出错: {e}")
        finally:
            ResourceRegistry.instance().unregister_thread(self._thread_handle)
            self._signals.finished.emit()
