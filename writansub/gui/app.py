"""主窗口：组装五个 Tab，程序入口 main()"""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from writansub.config import load_gui_state, save_gui_state
from writansub.gui.alignment_tab import AlignmentTab
from writansub.gui.pipeline_tab import PipelineTab
from writansub.gui.tiger_tab import TigerTab
from writansub.gui.translate_tab import TranslateTab
from writansub.gui.whisper_tab import WhisperTab

# (标签名, Tab 类) — 顺序即 UI 中的显示顺序
_TAB_DEFS = [
    ("一键流水线", PipelineTab),
    ("预处理", TigerTab),
    ("语音识别", WhisperTab),
    ("强制打轴", AlignmentTab),
    ("AI 翻译", TranslateTab),
]


class MainWindow(QMainWindow):
    """主窗口：五个选项卡"""

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
        """关闭时：停止所有资源 → 保存 GUI 状态"""
        from writansub.registry import ResourceRegistry

        ResourceRegistry.instance().shutdown()

        state = load_gui_state()
        for tab in self._tabs:
            state.update(tab.save_state())
        save_gui_state(state)

        super().closeEvent(event)


def _enable_windows_dpi_awareness() -> None:
    """Windows 4K/高 DPI 适配（非 Windows 平台静默跳过）"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def main() -> None:
    _enable_windows_dpi_awareness()

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
