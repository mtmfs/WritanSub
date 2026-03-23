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

WHISPER_MODELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Whisper", [
        ("large-v3", "~10 GB"),
        ("large-v2", "~10 GB"),
        ("medium", "~5 GB"),
        ("small", "~2 GB"),
        ("base", "~1 GB"),
        ("tiny", "~1 GB"),
    ]),
    ("Distil-Whisper", [
        ("distil-large-v3", "~6 GB"),
    ]),
]

MSS_MODELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("TIGER", [
        ("tiger-dnr", "DnR ~1 GB"),
    ]),
    ("Demucs", [
        ("htdemucs_ft", "Fine-tuned ~1 GB"),
        ("htdemucs", "~1 GB"),
    ]),
]

SS_MODELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("TIGER", [
        ("tiger-speech", "~1 GB"),
    ]),
    # TODO: TF-GridNet spatialized 模型不兼容单声道输入，需换用单声道兼容模型
]

ALIGN_MODELS: list[tuple[str, list[tuple[str, str]]]] = [
    ("torchaudio", [
        ("mms_fa", "~2 GB"),
    ]),
    ("Qwen3-ForcedAligner", [
        ("qwen3-fa-0.6b", "~2 GB VRAM"),
    ]),
]

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
    start: float          # 秒
    end: float            # 秒
    text: str             # 原始文本
    romaji: str = ""      # 罗马音（用于 alignment）
    score: float = 0.0    # 对齐置信度
    translated: str = ""  # 翻译文本
    commented: bool = False  # 被重叠替换后标记为"注释"


def _hms_parts(seconds: float) -> tuple[int, int, int, float]:
    """将秒拆分为 (时, 分, 秒, 小数秒) 四元组。"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    frac = seconds % 1
    return h, m, s, frac


def fmt_srt_time(seconds: float) -> str:
    """秒 → SRT 时间格式 HH:MM:SS,mmm"""
    h, m, s, frac = _hms_parts(seconds)
    ms = min(int(round(frac * 1000)), 999)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ass_time(seconds: float) -> str:
    """秒 → ASS 时间格式 H:MM:SS.cc"""
    h, m, s, frac = _hms_parts(seconds)
    cs = min(int(round(frac * 100)), 99)
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
