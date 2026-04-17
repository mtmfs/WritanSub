import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from writansub.config import load_gui_state, save_gui_state, save_pp_config
from writansub.gui.tabs.align import AlignmentTab
from writansub.gui.tabs.pipeline import PipelineTab
from writansub.gui.tabs.preprocess import TigerTab
from writansub.gui.tabs.translate import TranslateTab
from writansub.gui.tabs.transcribe import WhisperTab

# (标签名, Tab 类) — 顺序即 UI 中的显示顺序
_TAB_DEFS = [
    ("一键流水线", PipelineTab),
    ("预处理", TigerTab),
    ("语音识别", WhisperTab),
    ("强制打轴", AlignmentTab),
    ("AI 翻译", TranslateTab),
]


class MainWindow(QMainWindow):

    def __init__(self, initial_media: str = ""):
        super().__init__()
        self.setWindowTitle("WritanSub - AI 字幕处理流水线")
        self.setMinimumSize(640, 520)
        self.resize(720, 700)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self._tabs = []
        for label, cls in _TAB_DEFS:
            tab = cls()
            tabs.addTab(tab, label)
            self._tabs.append(tab)

        self.pipeline_tab: PipelineTab = self._tabs[0]

        if initial_media:
            self.pipeline_tab.set_media_path(initial_media)

    def closeEvent(self, event) -> None:
        from writansub.bridge import ResourceRegistry

        ResourceRegistry.instance().shutdown()

        state = load_gui_state()
        for tab in self._tabs:
            state.update(tab.save_state())
        save_gui_state(state)

        from writansub.gui.widgets import ParamSpinBox
        pp = {}
        for spin in self.findChildren(ParamSpinBox):
            pp[spin._key] = round(spin.value(), 2)
        if pp:
            save_pp_config(pp)

        super().closeEvent(event)


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def main() -> None:
    _enable_windows_dpi_awareness()

    from writansub.network import setup_hf_mirror
    setup_hf_mirror()

    from writansub.gui.driver_check import check_driver
    check_driver()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    initial_media = ""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        initial_media = sys.argv[1]

    app = QApplication(sys.argv)
    window = MainWindow(initial_media=initial_media)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
