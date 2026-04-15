import io

from PySide6.QtWidgets import (
    QTextEdit, QProgressBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QGridLayout, QFrame,
    QScrollArea, QComboBox, QStyledItemDelegate, QStyleOptionViewItem,
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPalette

from writansub.config import (
    PP_DEFAULTS, PARAM_DEFS,
    load_pp_config, save_pp_config,
    load_gui_state, save_gui_state,
)


# ── State persistence mixin ───────────────────────────────────────────


class StateMixin:
    """Mixin: state persistence at task start + program close.

    Subclass must implement save_state() -> dict and restore_state(dict).
    """

    def _save_now(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def _auto_save(self):
        """向后兼容：信号连接仍在但不再触发 I/O"""
        pass


# ── Wheel-scroll guard mixin ──────────────────────────────────────────


class _NoScrollMixin:
    """Mixin: ignore wheel events unless the widget has focus.

    Prevents accidental value changes while scrolling the page.
    Must be listed *before* the Qt base class in the MRO so that
    ``wheelEvent`` is resolved to this implementation first.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


# ── Logging ────────────────────────────────────────────────────────────


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


# ── Progress ───────────────────────────────────────────────────────────


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


# ── Scrollable container ──────────────────────────────────────────────


class ScrollableFrame(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.inner = QWidget()
        self.setWidget(self.inner)


# ── No-scroll input widgets ──────────────────────────────────────────


class NoScrollComboBox(_NoScrollMixin, QComboBox):
    """ComboBox that ignores wheel events unless focused."""


class _InfoDelegate(QStyledItemDelegate):

    _INFO_GAP = 16          # name 与 info 之间最小间距
    _RIGHT_PAD = 8          # info 右侧留白
    _NAME_MAX_PX = 240      # name 区最大显示宽度，超出 elide

    def paint(self, painter, option, index):
        info = index.data(Qt.UserRole + 1) or ""
        info_width = option.fontMetrics.horizontalAdvance(info) if info else 0
        reserved = info_width + self._INFO_GAP + self._RIGHT_PAD if info else 0

        # 给基类绘制名字的区域：原 rect 右边扣掉 info 占位
        opt = QStyleOptionViewItem(option)
        opt.rect = option.rect.adjusted(0, 0, -reserved, 0)
        name = index.data(Qt.DisplayRole) or ""
        max_name_width = min(opt.rect.width(), self._NAME_MAX_PX)
        elided = option.fontMetrics.elidedText(name, Qt.ElideMiddle, max_name_width)
        if elided != name:
            opt.text = elided
            # 用 model index 的 displayText override 不方便，直接靠 super().paint 用 opt.text
        super().paint(painter, opt, index)

        if info:
            painter.save()
            painter.setPen(option.palette.color(QPalette.ColorRole.Text))
            painter.drawText(
                option.rect.adjusted(0, 0, -self._RIGHT_PAD, 0),
                Qt.AlignRight | Qt.AlignVCenter,
                info,
            )
            painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        name = index.data(Qt.DisplayRole) or ""
        info = index.data(Qt.UserRole + 1) or ""
        name_w = min(option.fontMetrics.horizontalAdvance(name), self._NAME_MAX_PX)
        info_w = option.fontMetrics.horizontalAdvance(info) if info else 0
        gap = self._INFO_GAP + self._RIGHT_PAD if info else self._RIGHT_PAD
        size.setWidth(name_w + info_w + gap)
        return size


class GroupedComboBox(NoScrollComboBox):
    """带分组标题的 ComboBox。

    调用 set_grouped_items(groups) 填充，
    groups 为 [(系列名, [(模型名, 附加信息), ...])] 列表。
    分组标题不可选，仅作为视觉分隔。
    """

    _POPUP_MIN_WIDTH = 320  # 弹出列表最小宽度，保证名字 + info 同行不挤

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(_InfoDelegate(self))
        self.view().setMinimumWidth(self._POPUP_MIN_WIDTH)
        # 避免 combobox 关闭态被父布局压得过窄
        self.setMinimumContentsLength(16)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)

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
    """SpinBox that ignores wheel events unless focused."""


class ParamSpinBox(_NoScrollMixin, QDoubleSpinBox):
    """参数 SpinBox，滚轮仅在获得焦点时生效，防止滚动页面时误触。
    值由 MainWindow.closeEvent() 统一持久化。"""

    def __init__(self, key: str, parent=None):
        self._key = key
        super().__init__(parent)


# ── Parameter grid builder ────────────────────────────────────────────


def build_params_grid(
    parent: QWidget,
    keys: list[str],
) -> dict[str, QDoubleSpinBox]:
    """在 parent 中创建 2 列参数网格。

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
