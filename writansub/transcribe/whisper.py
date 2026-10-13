from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from writansub.types import Segment, Transcript, WordInfo


@dataclass(frozen=True)
class WhisperOptions:
    model: str = "large-v3"
    compute_type: str = "int8"
    condition_on_previous_text: bool = True
    vad_filter: bool = False
    initial_prompt: str | None = None


@dataclass
class WhisperRecognizer:
    options: WhisperOptions = field(default_factory=WhisperOptions)
    name: ClassVar[str] = "whisper"

    def tag(self) -> str:
        return f"whisper-{self.options.model}"

    def cache_key(self) -> str:
        return f"whisper:{self.options.model}:{self.options.compute_type}"

    def load(self, device: str) -> Any:
        from faster_whisper import WhisperModel
        return WhisperModel(self.options.model, device=device, compute_type=self.options.compute_type)

    def transcribe(
        self,
        model: Any,
        path: str,
        lang: str,
        progress: Callable[[float, str], None] | None = None,
        *,
        short_clip: bool = False,
    ) -> Transcript:
        from writansub.bridge import ResourceRegistry

        reg = ResourceRegistry.instance()
        opts = self.options

        def _progress(pct: float, msg: str) -> None:
            if progress:
                progress(min(pct, 1.0), msg)

        _progress(0.0, "开始识别...")
        _progress(0.02, "识别中...")
        segments, info = model.transcribe(
            path,
            language=lang,
            word_timestamps=True,
            condition_on_previous_text=False if short_clip else opts.condition_on_previous_text,
            vad_filter=False if short_clip else opts.vad_filter,
            initial_prompt=opts.initial_prompt or None,
        )

        out: list[Segment] = []
        for i, seg in enumerate(segments, 1):
            reg.checkpoint()
            out.append(Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=[WordInfo(w.word, w.probability, w.start, w.end) for w in (seg.words or [])],
            ))
            _progress(seg.end / info.duration if info.duration else 0, f"识别中... {i} 条")

        _progress(1.0, "识别完成")
        return Transcript(segments=out, duration=info.duration, backend=self.name)
