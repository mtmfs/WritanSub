import io

from PySide6.QtWidgets import (
    QTextEdit, QProgressBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QGridLayout, QFrame,
    QScrollArea, QComboBox, QStyledItemDelegate, QStyleOptionViewItem,
    QStyle, QApplication,
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QPalette

from writansub.config import (
    PP_DEFAULTS, PARAM_DEFS,
    load_pp_config, save_pp_config,
    load_gui_state, save_gui_state,
)



class StateMixin:

    def is_running(self) -> bool:
        """本 tab 是否有任务在跑。取消按钮可用态即运行态（_set_buttons_state 维护）。"""
        btn = getattr(self, "_cancel_btn", None)
        return bool(btn is not None and btn.isEnabled())

    def _save_now(self):
        state = load_gui_state()
        state.update(self.save_state())
        save_gui_state(state)

    def _auto_save(self, *args, **kwargs):
        self._save_now()


def remove_selected_rows(file_list, backing_list) -> None:
    """按行号降序删除 QListWidget 选中行，并同步删除后备列表对应项。

    必须先取行号排序再删：selectedItems() 的顺序不保证与行序一致，
    非连续多选时按 item 顺序删会因行号变动而删错行。
    """
    rows = sorted((file_list.row(it) for it in file_list.selectedItems()), reverse=True)
    for row in rows:
        file_list.takeItem(row)
        del backing_list[row]



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

    _INFO_GAP = 16
    _RIGHT_PAD = 8
    _NAME_MAX_PX = 240

    def paint(self, painter, option, index):
        # 手动展开基类 paint = initStyleOption + drawControl：
        # 直接 super().paint(opt) 会在内部重新 initStyleOption，
        # 用 DisplayRole 覆盖我们改过的 opt.text，elide 永远不生效（N4 根因）
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        info = index.data(Qt.UserRole + 1) or ""
        info_width = opt.fontMetrics.horizontalAdvance(info) if info else 0
        reserved = info_width + self._INFO_GAP + self._RIGHT_PAD if info else 0

        # 给名字的区域：原 rect 右边扣掉 info 占位；过长中段省略
        opt.rect = opt.rect.adjusted(0, 0, -reserved, 0)
        max_name_width = min(opt.rect.width(), self._NAME_MAX_PX)
        opt.text = opt.fontMetrics.elidedText(opt.text, Qt.ElideMiddle, max_name_width)

        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

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
    pass


class ParamSpinBox(_NoScrollMixin, QDoubleSpinBox):

    def __init__(self, key: str, parent=None):
        self._key = key
        super().__init__(parent)


def bind_pp_autosave(spin: ParamSpinBox) -> None:
    """编辑即单键合并写盘（T01 根修）。

    多页同键 spinbox 是同一份后处理配置的多个视图，最后编辑者赢；
    未触碰的 spinbox 永不写盘，未初始化值在机制上进不了配置文件。
    keyboardTracking 关闭后 valueChanged 只在箭头点击/回车/失焦时触发，
    键入中间态（如输 0.35 过程中的 0）不会落盘。
    必须在初始 setValue 之后调用，否则播种值会触发一次写盘。
    """
    spin.setKeyboardTracking(False)
    spin.valueChanged.connect(
        lambda _v, s=spin: save_pp_config(
            load_pp_config() | {s._key: round(s.value(), 2)})
    )



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
        bind_pp_autosave(spin)
        layout.addWidget(spin, row, col + 1)

        spinboxes[key] = spin

    return spinboxes
