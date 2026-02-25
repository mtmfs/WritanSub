"""主窗口：组装四个 Tab，程序入口 main()"""

import os
import sys
import tkinter as tk
from tkinter import ttk

from writansub.gui.pipeline_tab import PipelineTab
from writansub.gui.whisper_tab import WhisperTab
from writansub.gui.alignment_tab import AlignmentTab
from writansub.gui.translate_tab import TranslateTab


class AItransApp:
    """主窗口：四个选项卡"""

    def __init__(self, root: tk.Tk, initial_media: str = ""):
        self.root = root
        root.title("AItrans - 字幕处理流水线")
        root.unbind_class("TSpinbox", "<MouseWheel>")
        root.geometry("720x700")
        root.minsize(640, 520)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

        pipeline_frame = ttk.Frame(notebook)
        notebook.add(pipeline_frame, text="一键流水线")
        self.pipeline_tab = PipelineTab(pipeline_frame)

        whisper_frame = ttk.Frame(notebook)
        notebook.add(whisper_frame, text="语音识别")
        self.whisper_tab = WhisperTab(whisper_frame)

        align_frame = ttk.Frame(notebook)
        notebook.add(align_frame, text="强制打轴")
        self.align_tab = AlignmentTab(align_frame)

        translate_frame = ttk.Frame(notebook)
        notebook.add(translate_frame, text="AI 翻译")
        self.translate_tab = TranslateTab(translate_frame)

        if initial_media:
            self.pipeline_tab.set_media_path(initial_media)


def main():
    initial_media = ""
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        initial_media = sys.argv[1]

    root = tk.Tk()
    AItransApp(root, initial_media=initial_media)
    root.mainloop()


if __name__ == "__main__":
    main()
