import io

from PySide6.QtWidgets import (
    QTextEdit, QProgressBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QGridLayout, QFrame,
    QScrollArea, QComboBox, QStyledItemDelegate,
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPalette

from writansub.config import (
    PP_DEFAULTS, PARAM_DEFS,
    load_pp_config, save_pp_config,
    load_gui_state, save_gui_state,
)



class StateMixin:

    def _save_now(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def _auto_save(self):
        pass



class _NoScrollMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)



class _LogSignal(QObject):
    message = Signal(str)


class LogWidget(QTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._signal = _LogSignal()
        self._signal.message.connect(self._append)

    def log(self, msg: str) -> None:
        self._signal.message.emit(msg)

    def _append(self, msg: str) -> None:
        self.append(msg)
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )

    def clear_log(self) -> None:
        self.clear()


class TextRedirector(io.TextIOBase):

    def __init__(self, log_widget: LogWidget):
        super().__init__()
        self._widget = log_widget

    def write(self, s: str) -> int:
        if s and s.strip():
            self._widget.log(s.rstrip('\n'))
        return len(s)

    def flush(self) -> None:
        pass



class _ProgressSignal(QObject):
    progress = Signal(float, str)


class ProgressWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)

        self._status = QLabel("就绪")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._pct = QLabel("")
        self._pct.setMinimumWidth(40)
        self._pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._status)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._pct)

        self._signal = _ProgressSignal()
        self._signal.progress.connect(self._update)

    def update_progress(self, pct: float, msg: str) -> None:
        self._signal.progress.emit(pct, msg)

    def _update(self, pct: float, msg: str) -> None:
        self._bar.setValue(int(pct * 100))
        self._status.setText(msg)
        self._pct.setText(f"{int(pct * 100)}%" if pct > 0 else "")

    def reset(self) -> None:
        self._bar.setValue(0)
        self._status.setText("就绪")
        self._pct.setText("")



class ScrollableFrame(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.inner = QWidget()
        self.setWidget(self.inner)



class NoScrollComboBox(_NoScrollMixin, QComboBox):
    pass


class _InfoDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        info = index.data(Qt.UserRole + 1)
        if not info:
            return
        painter.save()
        rect = option.rect.adjusted(0, 0, -8, 0)
        painter.setPen(option.palette.color(QPalette.ColorRole.Text))
        painter.drawText(rect, Qt.AlignRight | Qt.AlignVCenter, info)
        painter.restore()


class GroupedComboBox(NoScrollComboBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(_InfoDelegate(self))

    def set_grouped_items(self, groups: list[tuple[str, list[tuple[str, str]]]]) -> None:
        self.clear()
        model = self.model()
        for series, items in groups:
            self.addItem(f"── {series} ──")
            model.item(self.count() - 1).setEnabled(False)
            for name, info in items:
                self.addItem(name)
                self.setItemData(self.count() - 1, name, Qt.UserRole)
                self.setItemData(self.count() - 1, info, Qt.UserRole + 1)

    def currentName(self) -> str:
        return self.currentData(Qt.UserRole) or ""

    def setCurrentName(self, name: str) -> None:
        for i in range(self.count()):
            if self.itemData(i, Qt.UserRole) == name:
                self.setCurrentIndex(i)
                return


class NoScrollSpinBox(_NoScrollMixin, QDoubleSpinBox):
    pass


class ParamSpinBox(_NoScrollMixin, QDoubleSpinBox):

    def __init__(self, key: str, parent=None):
        self._key = key
        super().__init__(parent)



def build_params_grid(
    parent: QWidget,
    keys: list[str],
) -> dict[str, QDoubleSpinBox]:
    cfg = load_pp_config()

    layout = parent.layout()
    if layout is None:
        layout = QGridLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)

    spinboxes: dict[str, QDoubleSpinBox] = {}

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
