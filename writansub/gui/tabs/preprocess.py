"""预处理页：TIGER 降噪 / 说话人分轨"""

import os
import threading

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QListWidget, QPushButton, QLabel, QCheckBox,
    QFileDialog, QAbstractItemView, QSplitter,
)
from PySide6.QtCore import Signal, QObject, Qt

from writansub.types import MEDIA_FILETYPES, MSS_MODELS, SS_MODELS
from writansub.config import load_gui_state
from writansub.bridge import ResourceRegistry, CancelledError
from writansub.gui.widgets import LogWidget, ProgressWidget, NoScrollComboBox, GroupedComboBox, StateMixin


class _TigerSignals(QObject):
    """线程安全的信号"""
    finished = Signal()


class TigerTab(StateMixin, QWidget):
    """Tab 2: TIGER 音频预处理（降噪 / 说话人分轨）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._media_files: list[str] = []
        self._signals = _TigerSignals()
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
        top_layout.setContentsMargins(12, 12, 12, 12)

        # 输入文件
        card_file = QGroupBox("输入文件")
        top_layout.addWidget(card_file)
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

        # TIGER 设置
        card_tiger = QGroupBox("TIGER 设置")
        top_layout.addWidget(card_tiger)
        tiger_layout = QHBoxLayout(card_tiger)

        self._chk_denoise = QCheckBox("降噪")
        self._chk_denoise.setChecked(True)
        self._chk_denoise.setToolTip(
            "分离出纯对话轨（去除音效和音乐）"
        )
        tiger_layout.addWidget(self._chk_denoise)

        tiger_layout.addWidget(QLabel("降噪模型"))
        self._mss_model_combo = GroupedComboBox()
        self._mss_model_combo.set_grouped_items(MSS_MODELS)
        self._mss_model_combo.setCurrentName("tiger-dnr")
        tiger_layout.addWidget(self._mss_model_combo)

        self._chk_separate = QCheckBox("对话分轨")
        self._chk_separate.setToolTip(
            "分离说话人 + VAD 重叠段检测\n"
            "（自动启用降噪作为前置步骤）"
        )
        self._chk_separate.stateChanged.connect(self._on_separate_changed)
        tiger_layout.addWidget(self._chk_separate)

        tiger_layout.addWidget(QLabel("分轨模型"))
        self._ss_model_combo = GroupedComboBox()
        self._ss_model_combo.set_grouped_items(SS_MODELS)
        self._ss_model_combo.setCurrentName("tiger-speech")
        tiger_layout.addWidget(self._ss_model_combo)

        tiger_layout.addWidget(QLabel("设备"))
        self._device_combo = NoScrollComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        tiger_layout.addWidget(self._device_combo)

        self._chk_save = QCheckBox("保留分离音频")
        self._chk_save.setChecked(True)
        self._chk_save.setToolTip("保存 TIGER 分离的中间 WAV 文件")
        tiger_layout.addWidget(self._chk_save)
        tiger_layout.addStretch()

        top_layout.addStretch()

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
        self._chk_denoise.stateChanged.connect(self._auto_save)
        self._mss_model_combo.currentTextChanged.connect(self._auto_save)
        self._chk_separate.stateChanged.connect(self._auto_save)
        self._ss_model_combo.currentTextChanged.connect(self._auto_save)
        self._device_combo.currentTextChanged.connect(self._auto_save)
        self._chk_save.stateChanged.connect(self._auto_save)

    def save_state(self) -> dict:
        return {
            "tiger.denoise": self._chk_denoise.isChecked(),
            "tiger.mss_model": self._mss_model_combo.currentName(),
            "tiger.separate": self._chk_separate.isChecked(),
            "tiger.ss_model": self._ss_model_combo.currentName(),
            "tiger.device": self._device_combo.currentText(),
            "tiger.save": self._chk_save.isChecked(),
            "tiger.files": list(self._media_files),
        }

    def restore_state(self, state: dict):
        if "tiger.denoise" in state:
            self._chk_denoise.setChecked(state["tiger.denoise"])
        if "tiger.mss_model" in state:
            self._mss_model_combo.setCurrentName(state["tiger.mss_model"])
        if "tiger.separate" in state:
            self._chk_separate.setChecked(state["tiger.separate"])
        if "tiger.ss_model" in state:
            self._ss_model_combo.setCurrentName(state["tiger.ss_model"])
        if "tiger.device" in state:
            self._device_combo.setCurrentText(state["tiger.device"])
        if "tiger.save" in state:
            self._chk_save.setChecked(state["tiger.save"])
        if "tiger.files" in state:
            files = [f for f in state["tiger.files"] if os.path.isfile(f)]
            self._media_files = files
            self._file_list.clear()
            self._file_list.addItems(files)

    # ── 文件管理 ──

    def _add_files(self):
        filter_str = ";;".join(f"{label} ({exts})" for label, exts in MEDIA_FILETYPES)
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择媒体文件", "", filter_str)
        for f in files:
            if f not in self._media_files:
                self._media_files.append(f)
                self._file_list.addItem(f)
        self._auto_save()

    def _remove_files(self):
        for item in reversed(self._file_list.selectedItems()):
            row = self._file_list.row(item)
            self._file_list.takeItem(row)
            self._media_files.pop(row)
        self._auto_save()

    def _clear_files(self):
        self._media_files.clear()
        self._file_list.clear()
        self._auto_save()

    # ── TIGER 联动 ──

    def _on_separate_changed(self, state):
        if state:
            self._chk_denoise.setChecked(True)

    # ── 执行 ──

    def _set_buttons_state(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._pause_btn.setEnabled(running)
        if not running:
            self._pause_btn.setText("暂停")

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
        if not self._media_files:
            self._log.log("请先添加媒体文件")
            return
        if not self._chk_denoise.isChecked() and not self._chk_separate.isChecked():
            self._log.log("请至少选择一项处理（降噪或对话分轨）")
            return

        self._save_now()
        ResourceRegistry.instance().reset_controls()
        self._set_buttons_state(True)
        self._progress.reset()
        self._log.clear_log()

        mss_model = self._mss_model_combo.currentName()
        do_separate = self._chk_separate.isChecked()
        ss_model = self._ss_model_combo.currentName()
        save_intermediate = self._chk_save.isChecked()
        device = self._device_combo.currentText()

        thread = threading.Thread(
            target=self._run_tiger,
            args=(list(self._media_files), mss_model, do_separate, ss_model, save_intermediate, device),
            daemon=True,
        )
        thread.start()

    def _run_tiger(self, media_files: list[str], mss_model: str, do_separate: bool,
                   ss_model: str, save_intermediate: bool, device: str):
        from writansub.preprocess.core import run_dnr_batch, run_speech_batch

        log = self._log.log
        progress = self._progress.update_progress

        try:
            total_phases = 2 if do_separate else 1

            # DnR 降噪
            log(f"── 阶段 1/{total_phases}: DnR 降噪 ──")

            def _dnr_progress(pct, msg):
                progress(pct / total_phases, f"[DnR] {msg}")

            tiger_results = run_dnr_batch(
                media_files,
                device=device,
                save_intermediate=save_intermediate,
                mss_model=mss_model,
                log_callback=log,
                progress_callback=_dnr_progress,
            )

            # 说话人分轨
            if do_separate and tiger_results:
                log(f"── 阶段 2/{total_phases}: 说话人分轨 ──")

                def _spk_progress(pct, msg):
                    progress((1 + pct) / total_phases, f"[Speech] {msg}")

                run_speech_batch(
                    tiger_results,
                    device=device,
                    save_intermediate=save_intermediate,
                    ss_model=ss_model,
                    log_callback=log,
                    progress_callback=_spk_progress,
                )

            log("全部处理完成")
            progress(1.0, "完成")

        except CancelledError:
            log("处理已取消")
        except Exception as e:
            log(f"处理出错: {e}")
        finally:
            self._signals.finished.emit()

    def _on_finished(self):
        self._set_buttons_state(False)
