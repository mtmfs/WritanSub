"""主窗口：组装四个 Tab，程序入口 main()"""

import os
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt

from writansub.config import load_gui_state, save_gui_state

from writansub.gui.pipeline_tab import PipelineTab
from writansub.gui.tiger_tab import TigerTab
from writansub.gui.whisper_tab import WhisperTab
from writansub.gui.alignment_tab import AlignmentTab
from writansub.gui.translate_tab import TranslateTab


class MainWindow(QMainWindow):
    """主窗口：五个选项卡"""

    def __init__(self, initial_media: str = ""):
        super().__init__()
        self.setWindowTitle("WritanSub - AI 字幕处理流水线")
        self.setMinimumSize(640, 520)
        self.resize(720, 700)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.pipeline_tab = PipelineTab()
        tabs.addTab(self.pipeline_tab, "一键流水线")

        self.tiger_tab = TigerTab()
        tabs.addTab(self.tiger_tab, "预处理")

        self.whisper_tab = WhisperTab()
        tabs.addTab(self.whisper_tab, "语音识别")

        self.alignment_tab = AlignmentTab()
        tabs.addTab(self.alignment_tab, "强制打轴")

        self.translate_tab = TranslateTab()
        tabs.addTab(self.translate_tab, "AI 翻译")

        if initial_media:
            self.pipeline_tab.set_media_path(initial_media)

    def closeEvent(self, event):
        """关闭时：停止所有资源 → 保存 GUI 状态"""
        from writansub.registry import ResourceRegistry
        ResourceRegistry.instance().shutdown()

        state = load_gui_state()
        state.update(self.pipeline_tab.save_state())
        state.update(self.tiger_tab.save_state())
        state.update(self.whisper_tab.save_state())
        state.update(self.alignment_tab.save_state())
        state.update(self.translate_tab.save_state())
        save_gui_state(state)
        super().closeEvent(event)


def main():
    # Windows 4K/高 DPI 适配
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    # Qt 高 DPI 支持
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
