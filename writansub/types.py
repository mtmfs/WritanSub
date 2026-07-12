from dataclasses import dataclass, field
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

LANGUAGES = ["ja", "zh", "en", "ko", "fr", "de", "es", "ru", "vi"]

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
    word: str
    probability: float


@dataclass
class Sub:
    index: int
    start: float
    end: float
    text: str
    romaji: str = ""      # 罗马音（用于 alignment）
    score: float = 0.0
    translated: str = ""
    # 低置信词随本体携带：重编号/合并等索引变换后仍可在流程末尾单点生成 review（T06）
    low_words: list[str] = field(default_factory=list)
    # 说话人标签：0=无（单轨/未知），1/2=separate 模式的两条分离轨（T10）
    speaker: int = 0


def fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ass_time(seconds: float) -> str:
    total_cs = max(0, round(seconds * 100))
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

