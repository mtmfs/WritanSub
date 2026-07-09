"""T32：remove_selected_rows —— 非连续多选删除的共享实现（离屏 Qt）。"""
import pytest

from writansub.gui.widgets import remove_selected_rows


@pytest.fixture
def file_list(qapp):
    from PySide6.QtWidgets import QListWidget
    lw = QListWidget()
    lw.setSelectionMode(QListWidget.ExtendedSelection)
    yield lw
    lw.deleteLater()


def _fill(lw, names):
    for n in names:
        lw.addItem(n)
    return list(names)


def _texts(lw):
    return [lw.item(i).text() for i in range(lw.count())]


def test_non_contiguous_removal(file_list):
    backing = _fill(file_list, ["a", "b", "c", "d", "e", "f"])
    for row in (1, 3, 4):  # 非连续多选：b, d, e
        file_list.item(row).setSelected(True)
    remove_selected_rows(file_list, backing)
    assert _texts(file_list) == ["a", "c", "f"]
    assert backing == ["a", "c", "f"]  # 控件与后备列表保持同步


def test_selection_order_irrelevant(file_list):
    """选中顺序打乱也必须删对（原 bug 的触发形态）。"""
    backing = _fill(file_list, ["a", "b", "c", "d", "e"])
    for row in (4, 0, 2):  # 倒着选、跳着选
        file_list.item(row).setSelected(True)
    remove_selected_rows(file_list, backing)
    assert _texts(file_list) == ["b", "d"]
    assert backing == ["b", "d"]


def test_empty_selection_noop(file_list):
    backing = _fill(file_list, ["a", "b"])
    remove_selected_rows(file_list, backing)
    assert _texts(file_list) == ["a", "b"]
    assert backing == ["a", "b"]


def test_remove_all(file_list):
    backing = _fill(file_list, ["a", "b", "c"])
    file_list.selectAll()
    remove_selected_rows(file_list, backing)
    assert _texts(file_list) == []
    assert backing == []
