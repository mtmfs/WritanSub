"""T36：关窗确认——运行中弹框，拒绝则不退出（离屏 Qt，helper 级）。"""
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from writansub.gui import app as app_mod
from writansub.gui.widgets import StateMixin


class _FakeTab(StateMixin, QWidget):
    def __init__(self, running: bool):
        super().__init__()
        self._cancel_btn = QPushButton(self)
        self._cancel_btn.setEnabled(running)


def test_state_mixin_is_running(qapp):
    assert _FakeTab(True).is_running() is True
    assert _FakeTab(False).is_running() is False


def test_idle_quits_without_dialog(qapp, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("空闲时不应弹确认框")
    monkeypatch.setattr(QMessageBox, "question", boom)
    assert app_mod._confirm_quit([_FakeTab(False), _FakeTab(False)], None) is True


def test_running_declined_blocks_quit(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert app_mod._confirm_quit([_FakeTab(True), _FakeTab(False)], None) is False


def test_running_confirmed_quits(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert app_mod._confirm_quit([_FakeTab(True)], None) is True
