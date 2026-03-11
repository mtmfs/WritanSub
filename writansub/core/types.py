"""公共数据类型：Sub 字幕结构体、语言列表、文件类型常量、时间格式化"""

from dataclasses import dataclass
from typing import NamedTuple

MEDIA_FILETYPES = [
    ("媒体文件", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
    ("所有文件", "*.*"),
]

AUDIO_FILETYPES = [
    ("音频文件", "*.wav *.mp3 *.flac *.ogg *.aac *.m4a"),
    ("所有文件", "*.*"),
]

SRT_FILETYPES = [
    ("SRT 字幕", "*.srt"),
    ("所有文件", "*.*"),
]

LANGUAGES = ["ja", "zh", "en", "ko", "fr", "de", "es", "ru"]

TRANSLATE_TARGETS = [
    "简体中文", "繁體中文", "English", "日本語", "한국어",
    "Français", "Deutsch", "Español", "Русский", "Tiếng Việt",
]


class WordInfo(NamedTuple):
    """词级别识别信息，用于 review 标记"""
    word: str
    probability: float


@dataclass
class Sub:
    """一条字幕"""
    index: int
    start: float       # 秒
    end: float          # 秒
    text: str           # 原始文本
    romaji: str = ""    # 罗马音（用于 alignment）
    score: float = 0.0  # 对齐置信度
    translated: str = ""  # 翻译文本
    commented: bool = False  # 被重叠替换后标记为"注释"


def fmt_srt_time(seconds: float) -> str:
    """秒 → SRT 时间格式 HH:MM:SS,mmm"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ass_time(seconds: float) -> str:
    """秒 → ASS 时间格式 H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_ASS_REVIEW_HEADER = """\
[Script Info]
Title: AItrans Review
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
