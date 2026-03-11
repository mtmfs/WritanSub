"""通用 PySide6 控件：日志区、进度条、滚动容器、参数面板"""

import io
from typing import Any, Dict, List

from PySide6.QtWidgets import (
    QTextEdit, QProgressBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QGridLayout, QFrame,
    QScrollArea, QSizePolicy, QComboBox,
)
from PySide6.QtCore import Qt, Signal, QObject

from writansub.config import (
    PP_DEFAULTS, PARAM_DEFS,
    load_pp_config, save_pp_config,
)


class _LogSignal(QObject):
    """线程安全的日志信号"""
    message = Signal(str)


class LogWidget(QTextEdit):
    """日志显示区域，支持线程安全追加"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._signal = _LogSignal()
        self._signal.message.connect(self._append)

    def log(self, msg: str):
        """线程安全地追加日志（带换行）"""
        self._signal.message.emit(msg)

    def _append(self, msg: str):
        self.append(msg)
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )

    def clear_log(self):
        self.clear()


class TextRedirector(io.TextIOBase):
    """将 print 输出重定向到 LogWidget"""

    def __init__(self, log_widget: LogWidget):
        super().__init__()
        self._widget = log_widget

    def write(self, s: str):
        if s and s.strip():
            self._widget.log(s.rstrip('\n'))
        return len(s)

    def flush(self):
        pass


class _ProgressSignal(QObject):
    """线程安全的进度信号"""
    progress = Signal(float, str)


class ProgressWidget(QWidget):
    """进度条 + 状态标签"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)

        self._status = QLabel("就绪")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._pct = QLabel("")
        self._pct.setMinimumWidth(40)
        self._pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._status)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._pct)

        self._signal = _ProgressSignal()
        self._signal.progress.connect(self._update)

    def update_progress(self, pct: float, msg: str):
        """线程安全地更新进度"""
        self._signal.progress.emit(pct, msg)

    def _update(self, pct: float, msg: str):
        self._bar.setValue(int(pct * 100))
        self._status.setText(msg)
        self._pct.setText(f"{int(pct * 100)}%" if pct > 0 else "")

    def reset(self):
        self._bar.setValue(0)
        self._status.setText("就绪")
        self._pct.setText("")


class ScrollableFrame(QScrollArea):
    """可垂直滚动的容器，内容放入 self.inner"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.inner = QWidget()
        self.setWidget(self.inner)


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores wheel events unless focused."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class NoScrollSpinBox(QDoubleSpinBox):
    """SpinBox that ignores wheel events unless focused."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class ParamSpinBox(QDoubleSpinBox):
    """参数 SpinBox，值变化时自动保存配置。
    滚轮仅在获得焦点时生效，防止滚动页面时误触。"""

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key = key
        self.setFocusPolicy(Qt.StrongFocus)
        self.valueChanged.connect(self._on_change)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)

    def _on_change(self, value: float):
        try:
            current = load_pp_config()
            current[self._key] = round(value, 2)
            save_pp_config(current)
        except Exception:
            pass


def build_params_grid(parent: QWidget, keys: List[str]) -> Dict[str, QDoubleSpinBox]:
    """
    在 parent 中创建 2 列参数网格。

    Args:
        parent: 父控件
        keys: 要显示的参数 key 列表 (对应 PARAM_DEFS)

    Returns:
        {key: QDoubleSpinBox} 字典
    """
    cfg = load_pp_config()

    layout = parent.layout()
    if layout is None:
        layout = QGridLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)

    spinboxes: Dict[str, QDoubleSpinBox] = {}

    for i, key in enumerate(keys):
        defn = PARAM_DEFS[key]
        row = i // 2
        col = (i % 2) * 2

        label = QLabel(defn["label"])
        if defn.get("tip"):
            label.setToolTip(defn["tip"])
            label.setText(defn["label"] + " ⓘ")
            label.setCursor(Qt.WhatsThisCursor)
        layout.addWidget(label, row, col)

        spin = ParamSpinBox(key)
        spin.setRange(defn["from"], defn["to"])
        spin.setSingleStep(defn["inc"])
        spin.setDecimals(2)
        spin.setValue(cfg.get(key, PP_DEFAULTS[key]))
        spin.setFixedWidth(80)
        layout.addWidget(spin, row, col + 1)

        spinboxes[key] = spin

    return spinboxes
