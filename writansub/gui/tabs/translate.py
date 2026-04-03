"""单独 AI 翻译页"""

import os
import threading
from dataclasses import replace

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLineEdit, QPushButton, QLabel, QCheckBox, QFileDialog, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.types import TRANSLATE_TARGETS
from writansub.subtitle.srt_io import parse_srt, write_srt, merge_bilingual
from writansub.translate.core import translate_subs
from writansub.config import load_translate_config, save_translate_config, load_gui_state
from writansub.bridge import ResourceRegistry, CancelledError
from writansub.gui.widgets import LogWidget, ProgressWidget, NoScrollComboBox, StateMixin

class _TranslateSignals(QObject):
    """线程安全的信号"""
    finished = Signal()


class TranslateTab(StateMixin, QWidget):
    """Tab 4: 独立 AI 翻译"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = _TranslateSignals()
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

        # ── 上半部分：设置 ──
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

        file_layout.addWidget(QLabel("字幕文件"), 0, 0)
        self._srt_edit = QLineEdit()
        self._srt_edit.textChanged.connect(self._on_srt_changed)
        file_layout.addWidget(self._srt_edit, 0, 1)
        btn_srt = QPushButton("浏览")
        btn_srt.clicked.connect(self._browse_srt)
        file_layout.addWidget(btn_srt, 0, 2)

        file_layout.addWidget(QLabel("输出路径"), 1, 0)
        self._out_edit = QLineEdit()
        file_layout.addWidget(self._out_edit, 1, 1)
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(self._browse_output)
        file_layout.addWidget(btn_out, 1, 2)

        file_layout.setColumnStretch(1, 1)

        # 翻译设置
        card_cfg = QGroupBox("翻译设置")
        settings_layout.addWidget(card_cfg)
        cfg_layout = QGridLayout(card_cfg)

        t_cfg = load_translate_config()

        row1 = QHBoxLayout()
        cfg_layout.addLayout(row1, 0, 0, 1, 2)
        row1.addWidget(QLabel("目标语言"))
        self._target_combo = NoScrollComboBox()
        self._target_combo.addItems(TRANSLATE_TARGETS)
        self._target_combo.setCurrentText(t_cfg["target_lang"])
        row1.addWidget(self._target_combo)

        row1.addWidget(QLabel("模型"))
        self._model_edit = QLineEdit(t_cfg["model"])
        self._model_edit.setMinimumWidth(150)
        row1.addWidget(self._model_edit)
        row1.addStretch()

        cfg_layout.addWidget(QLabel("API 地址"), 1, 0)
        self._base_edit = QLineEdit(t_cfg["api_base"])
        cfg_layout.addWidget(self._base_edit, 1, 1)

        cfg_layout.addWidget(QLabel("API Key"), 2, 0)
        self._key_edit = QLineEdit(t_cfg["api_key"])
        self._key_edit.setEchoMode(QLineEdit.Password)
        cfg_layout.addWidget(self._key_edit, 2, 1)

        cfg_layout.setColumnStretch(1, 1)

        self._chk_bilingual = QCheckBox("双语输出（原文 + 译文）")
        self._chk_bilingual.setToolTip("勾选后输出双语字幕，每条包含原文和译文两行\n不勾选则只输出译文")
        settings_layout.addWidget(self._chk_bilingual)

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

        self._start_btn = QPushButton("开始翻译")
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
        self._srt_edit.editingFinished.connect(self._auto_save)
        self._out_edit.editingFinished.connect(self._auto_save)
        self._target_combo.currentTextChanged.connect(self._auto_save)
        self._model_edit.editingFinished.connect(self._auto_save)
        self._base_edit.editingFinished.connect(self._auto_save)
        self._key_edit.editingFinished.connect(self._auto_save)
        self._chk_bilingual.stateChanged.connect(self._auto_save)

    def _get_translate_config(self) -> dict:
        """当前翻译配置（用于保存和执行）"""
        return {
            "target_lang": self._target_combo.currentText(),
            "api_base": self._base_edit.text(),
            "api_key": self._key_edit.text(),
            "model": self._model_edit.text(),
        }

    def save_state(self) -> dict:
        # Also sync translate config so pipeline can read it
        save_translate_config(self._get_translate_config())
        return {
            "translate.srt": self._srt_edit.text(),
            "translate.output": self._out_edit.text(),
            "translate.target_lang": self._target_combo.currentText(),
            "translate.model": self._model_edit.text(),
            "translate.api_base": self._base_edit.text(),
            "translate.api_key": self._key_edit.text(),
            "translate.bilingual": self._chk_bilingual.isChecked(),
        }

    def restore_state(self, state: dict):
        if "translate.srt" in state:
            self._srt_edit.setText(state["translate.srt"])
        if "translate.output" in state:
            self._out_edit.setText(state["translate.output"])
        if "translate.target_lang" in state:
            self._target_combo.setCurrentText(state["translate.target_lang"])
        if "translate.model" in state:
            self._model_edit.setText(state["translate.model"])
        if "translate.api_base" in state:
            self._base_edit.setText(state["translate.api_base"])
        if "translate.api_key" in state:
            self._key_edit.setText(state["translate.api_key"])
        if "translate.bilingual" in state:
            self._chk_bilingual.setChecked(state["translate.bilingual"])

    def _browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SRT 字幕", "", "SRT 字幕 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self._srt_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存翻译 SRT", "", "SRT 字幕 (*.srt);;所有文件 (*.*)"
        )
        if path:
            self._out_edit.setText(path)

    def _on_srt_changed(self, text: str):
        srt = text.strip()
        if srt and not self._out_edit.text().strip():
            base = os.path.splitext(srt)[0]
            self._out_edit.setText(f"{base}_translated.srt")

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
        srt = self._srt_edit.text().strip()
        if not srt or not os.path.isfile(srt):
            self._log.log("请选择有效的 SRT 文件")
            return

        output = self._out_edit.text().strip()
        if not output:
            output = os.path.splitext(srt)[0] + "_translated.srt"
            self._out_edit.setText(output)

        cfg = self._get_translate_config()
        save_translate_config(cfg)
        self._save_now()

        ResourceRegistry.instance().reset_controls()
        self._set_buttons_state(True)
        self._log.clear_log()
        self._progress.reset()

        bilingual = self._chk_bilingual.isChecked()

        thread = threading.Thread(
            target=self._run_translate,
            args=(srt, output, cfg, bilingual),
            daemon=True,
        )
        thread.start()

    def _run_translate(self, srt: str, output: str, cfg: dict, bilingual: bool):
        try:
            subs = parse_srt(srt)
            translate_subs(
                subs,
                target_lang=cfg["target_lang"],
                api_base=cfg["api_base"],
                api_key=cfg["api_key"],
                model=cfg["model"],
                log_callback=self._log.log,
                progress_callback=self._progress.update_progress,
            )

            if bilingual:
                output_subs = merge_bilingual(subs)
            else:
                output_subs = [replace(s, text=s.translated or s.text) for s in subs]
            write_srt(output_subs, output)
            self._progress.update_progress(1.0, "翻译完成")
        except CancelledError:
            self._log.log("翻译已取消")
        except Exception as e:
            self._log.log(f"出错: {e}")
        finally:
            self._signals.finished.emit()
