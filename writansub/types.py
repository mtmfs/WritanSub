from dataclasses import dataclass, field
from typing import Literal, NamedTuple

class WordInfo(NamedTuple):
    word: str
    probability: float | None
    start: float | None = None
    end: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    time_kind: Literal["estimate", "window", "aligned"] = "estimate"
    speaker: str | None = None
    words: list[WordInfo] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float
    backend: str


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

