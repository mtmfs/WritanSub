import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox,
    QFileDialog, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.types import MEDIA_FILETYPES, SRT_FILETYPES, LANGUAGES, WHISPER_MODELS
from writansub.transcribe.core import transcribe
from writansub.subtitle.review import generate_review, write_review_files
from writansub.subtitle.srt_io import write_srt
from writansub.config import PARAM_DEFS, load_gui_state
from writansub.bridge import ResourceRegistry, CancelledError
from writansub.gui.widgets import LogWidget, ProgressWidget, NoScrollComboBox, GroupedComboBox, ParamSpinBox, StateMixin


class _WhisperSignals(QObject):
    finished = Signal()


class WhisperTab(StateMixin, QWidget):

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

        card_param = QGroupBox("参数")
        settings_layout.addWidget(card_param)
        param_layout = QHBoxLayout(card_param)

        param_layout.addWidget(QLabel("语言"))
        self._lang_combo = NoScrollComboBox()
        self._lang_combo.addItems(LANGUAGES)
        self._lang_combo.setCurrentText("ja")
        param_layout.addWidget(self._lang_combo)

        param_layout.addWidget(QLabel("听写模型"))
        self._model_combo = GroupedComboBox()
        self._model_combo.set_grouped_items(WHISPER_MODELS)
        self._model_combo.setCurrentName("large-v3")
        param_layout.addWidget(self._model_combo)

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

        self._chk_vad = QCheckBox("跳过静音")
        self._chk_vad.setChecked(False)
        self._chk_vad.setToolTip(
            "启用 Silero VAD 自动跳过静音段，\n"
            "可显著加速长音频的识别。"
        )
        param_layout.addWidget(self._chk_vad)
        param_layout.addStretch()

        card_prompt = QGroupBox("初始提示")
        settings_layout.addWidget(card_prompt)
        prompt_layout = QVBoxLayout(card_prompt)
        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("人名、专有名词，空格或顿号分隔（≤224 token）")
        self._prompt_edit.setToolTip(
            "Whisper initial_prompt：作为上下文偏好传给模型，\n"
            "用于让角色名、作品名、术语保持一致的写法。\n"
            "例如：瀬尾拓也、伊地知琴子、天音ケイ、キラモン\n"
            "注意：faster-whisper 上限约 224 token，过长会被截断。"
        )
        prompt_layout.addWidget(self._prompt_edit)

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

        self._start_btn = QPushButton("开始识别")
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
        self._media_edit.editingFinished.connect(self._auto_save)
        self._output_edit.editingFinished.connect(self._auto_save)
        self._lang_combo.currentTextChanged.connect(self._auto_save)
        self._model_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_cond_prev.stateChanged.connect(self._auto_save)
        self._chk_vad.stateChanged.connect(self._auto_save)
        self._prompt_edit.editingFinished.connect(self._auto_save)

    def save_state(self) -> dict:
        return {
            "whisper.media": self._media_edit.text(),
            "whisper.output": self._output_edit.text(),
            "whisper.lang": self._lang_combo.currentText(),
            "whisper.model": self._model_combo.currentName(),
            "whisper.device": self._device_combo.currentText(),
            "whisper.cond_prev": self._chk_cond_prev.isChecked(),
            "whisper.vad_filter": self._chk_vad.isChecked(),
            "whisper.initial_prompt": self._prompt_edit.text(),
        }

    def restore_state(self, state: dict):
        if "whisper.media" in state:
            self._media_edit.setText(state["whisper.media"])
        if "whisper.output" in state:
            self._output_edit.setText(state["whisper.output"])
        if "whisper.lang" in state:
            self._lang_combo.setCurrentText(state["whisper.lang"])
        if "whisper.model" in state:
            self._model_combo.setCurrentName(state["whisper.model"])
        if "whisper.device" in state:
            self._device_combo.setCurrentText(state["whisper.device"])
        if "whisper.cond_prev" in state:
            self._chk_cond_prev.setChecked(state["whisper.cond_prev"])
        if "whisper.vad_filter" in state:
            self._chk_vad.setChecked(state["whisper.vad_filter"])
        if "whisper.initial_prompt" in state:
            self._prompt_edit.setText(state["whisper.initial_prompt"])

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
        self._log.log("取消请求已发送，将在当前段结束后停止")

    def _start(self):
        media = self._media_edit.text().strip()
        if not media or not os.path.isfile(media):
            self._log.log("请先选择有效的媒体文件")
            return

        output = self._output_edit.text().strip()
        if not output:
            output = os.path.splitext(media)[0] + ".srt"
            self._output_edit.setText(output)

        self._save_now()
        ResourceRegistry.instance().reset_controls()
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        lang = self._lang_combo.currentText()
        model_size = self._model_combo.currentName()
        device = self._device_combo.currentText()
        wc = self._wc_spin.value()
        cond_prev = self._chk_cond_prev.isChecked()
        vad_filter = self._chk_vad.isChecked()
        initial_prompt = self._prompt_edit.text().strip() or None

        thread = threading.Thread(
            target=self._run_whisper,
            args=(media, output, lang, model_size, device, wc, cond_prev, vad_filter, initial_prompt),
            daemon=True,
        )
        thread.start()

    def _run_whisper(self, media: str, output: str, lang: str,
                     model_size: str, device: str, wc_threshold: float,
                     cond_prev: bool = True, vad_filter: bool = False,
                     initial_prompt: str | None = None):
        reg = ResourceRegistry.instance()

        def _w_factory():
            from faster_whisper import WhisperModel
            return WhisperModel(model_size, device=device, compute_type="int8")

        wh = None
        try:
            wh = reg.acquire_model(f"whisper:{model_size}", device, _w_factory)
            whisper_model = reg.get_model(wh)

            subs, word_data = transcribe(
                media, lang=lang, device=device, log_callback=self._log.log,
                progress_callback=self._progress.update_progress,
                condition_on_previous_text=cond_prev, model=whisper_model,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )

            if wc_threshold > 0.0:
                srt_content, ass_content, low_count, total_words = generate_review(
                    subs, word_data, wc_threshold,
                )
                if low_count > 0:
                    base = os.path.splitext(media)[0]
                    write_review_files(base, srt_content, ass_content)
                    self._log.log(f"低置信词 {low_count}/{total_words}，已生成标记版")

            write_srt(subs, output)
            self._progress.update_progress(1.0, "识别完成")
        except CancelledError:
            self._log.log("识别已取消")
        except Exception as e:
            self._log.log(f"出错: {e}")
        finally:
            if wh is not None:
                reg.release_model(wh)
            self._signals.finished.emit()
