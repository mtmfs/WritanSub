"""T32：remove_selected_rows；T01：bind_pp_autosave 编辑即写（离屏 Qt）。"""
import pytest

from writansub.gui.widgets import remove_selected_rows, ParamSpinBox, bind_pp_autosave


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


def _make_pp_spin(key, value):
    spin = ParamSpinBox(key)
    spin.setRange(0.0, 10.0)
    spin.setSingleStep(0.1)
    spin.setDecimals(2)
    spin.setValue(value)      # 播种在绑定之前，不触发写盘
    bind_pp_autosave(spin)
    return spin


def test_pp_autosave_writes_single_key_preserves_others(qapp, isolated_pp_config):
    from writansub.config import load_pp_config, save_pp_config, PP_DEFAULTS
    save_pp_config(dict(PP_DEFAULTS))          # 现有配置就位
    spin = _make_pp_spin("pad_sec", 0.5)
    spin.stepUp()                              # 模拟点上箭头 → 0.6
    cfg = load_pp_config()
    assert cfg["pad_sec"] == 0.6
    # 其余 7 键原样保留，未被整文件覆写冲掉
    for k in PP_DEFAULTS:
        if k != "pad_sec":
            assert cfg[k] == PP_DEFAULTS[k]
    spin.deleteLater()


def test_pp_untouched_spin_never_writes(qapp, isolated_pp_config):
    """T01 根因断言：未触碰的 spinbox（哪怕值是 0）永不写盘。"""
    spin = _make_pp_spin("pad_sec", 0.0)
    assert not isolated_pp_config.exists()
    spin.deleteLater()


def test_pp_two_views_last_editor_wins(qapp, isolated_pp_config):
    """同键双视图（pipeline/align 页各一份）：最后编辑者赢，无撞 key 覆盖。"""
    from writansub.config import load_pp_config
    a = _make_pp_spin("min_gap", 0.3)
    b = _make_pp_spin("min_gap", 0.3)
    a.stepUp()          # a 视图改到 0.4
    b.stepUp()
    b.stepUp()          # b 视图改到 0.5，后编辑
    assert load_pp_config()["min_gap"] == 0.5
    a.deleteLater()
    b.deleteLater()
