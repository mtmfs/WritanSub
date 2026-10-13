from typing import Any, Callable, Protocol

from writansub.types import Sub, Transcript, WordInfo


class Recognizer(Protocol):
    name: str

    def tag(self) -> str: ...

    def cache_key(self) -> str: ...

    def load(self, device: str) -> Any: ...

    def transcribe(
        self,
        model: Any,
        path: str,
        lang: str,
        progress: Callable[[float, str], None] | None = None,
        *,
        short_clip: bool = False,
    ) -> Transcript: ...


def is_aligned(transcript: Transcript) -> bool:
    return all(seg.time_kind == "aligned" for seg in transcript.segments)


def to_subs(transcript: Transcript) -> tuple[list[Sub], list[list[WordInfo]]]:
    subs = [
        Sub(index=i, start=seg.start, end=seg.end, text=seg.text)
        for i, seg in enumerate(transcript.segments, 1)
    ]
    word_data = [list(seg.words) for seg in transcript.segments]
    return subs, word_data
