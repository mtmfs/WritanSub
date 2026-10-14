from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from writansub.align.core import LANG_MAP
from writansub.transcribe.base import model_tag
from writansub.transcribe.segment import attach_punctuation, split_tokens
from writansub.types import Segment, Transcript

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class QwenOptions:
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    context: str = ""
    chunk_seconds: float = 60.0
    max_gap: float = 0.8
    max_chars: int = 40
    max_new_tokens: int = 1024
    max_word_seconds: float = 3.0
    batch_size: int = 1


@dataclass
class QwenRecognizer:
    options: QwenOptions = field(default_factory=QwenOptions)
    name: ClassVar[str] = "qwen3"

    def tag(self) -> str:
        return model_tag(self.options.model).lower()

    def cache_key(self) -> str:
        return f"qwen3:{self.options.model}:{self.options.aligner}"

    def load(self, device: str) -> Any:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        device_map = "cuda:0" if device == "cuda" else device
        return Qwen3ASRModel.from_pretrained(
            self.options.model,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=self.options.batch_size,
            max_new_tokens=self.options.max_new_tokens,
            forced_aligner=self.options.aligner,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
        )

    def transcribe(
        self,
        model: Any,
        path: str,
        lang: str,
        progress: Callable[[float, str], None] | None = None,
        *,
        short_clip: bool = False,
    ) -> Transcript:
        from qwen_asr.inference.utils import split_audio_into_chunks
        from writansub.bridge import ResourceRegistry

        reg = ResourceRegistry.instance()
        opts = self.options
        language = LANG_MAP.get(lang, "Japanese")

        def _progress(pct: float, msg: str) -> None:
            if progress:
                progress(min(pct, 1.0), msg)

        _progress(0.0, "开始识别...")
        waveform, _ = reg.decode_audio(path, sample_rate=SAMPLE_RATE)
        audio = waveform.squeeze(0).numpy()
        duration = audio.shape[0] / SAMPLE_RATE
        parts = split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=opts.chunk_seconds)

        segments: list[Segment] = []
        for i, (chunk, offset) in enumerate(parts, 1):
            reg.checkpoint()
            _progress(0.02 + 0.96 * (i - 1) / len(parts), f"识别中... {i}/{len(parts)}")
            result = model.transcribe(
                audio=(chunk, SAMPLE_RATE),
                context=opts.context,
                language=language,
                return_time_stamps=True,
            )[0]
            text = result.text.strip()
            tokens = []
            for it in (result.time_stamps or []):
                start = float(it.start_time)
                end = min(float(it.end_time), start + opts.max_word_seconds)
                tokens.append((it.text, start + offset, end + offset))
            if tokens:
                segments += split_tokens(attach_punctuation(text, tokens), opts.max_gap, opts.max_chars,
                                         time_kind="aligned")
            elif text:
                end = offset + chunk.shape[0] / SAMPLE_RATE
                segments.append(Segment(start=offset, end=end, text=text, time_kind="window"))

        reg.checkpoint()
        segments.sort(key=lambda s: s.start)
        _progress(1.0, "识别完成")
        return Transcript(segments=segments, duration=duration, backend=self.name)
