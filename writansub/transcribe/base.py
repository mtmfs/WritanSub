import re
from typing import Any, Callable, Protocol

from writansub.subtitle.srt_io import safe_name
from writansub.types import Sub, Transcript, WordInfo


def model_tag(spec: str) -> str:
    """模型标识 → 产物文件名片段（T49）：裸名 / HF repo id / 本地路径一律只取末段再安全化。

    `/` 与 `\\` 在任何平台都当分隔符（不用 os.path.basename：POSIX 上它不切反斜杠，测试会跨平台漂移）；
    忽略空段与 `.`/`..`；取不到末段时退化为 "local"。只管命名，加载模型与缓存键仍用原始 spec。
    """
    parts = [p for p in re.split(r"[\\/]+", spec.strip()) if p not in ("", ".", "..")]
    return (safe_name(parts[-1]) if parts else "") or "local"


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
