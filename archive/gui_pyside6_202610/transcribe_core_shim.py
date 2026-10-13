from typing import Any, Callable

from writansub.transcribe.base import to_subs
from writansub.transcribe.whisper import WhisperOptions, WhisperRecognizer
from writansub.types import Sub, WordInfo


def transcribe(
    file_path: str,
    lang: str = "ja",
    device: str = "cuda",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    condition_on_previous_text: bool = True,
    model: Any | None = None,
    model_size: str = "large-v3",
    cancelled: Callable[[], bool] | None = None,
    vad_filter: bool = False,
    initial_prompt: str | None = None,
) -> tuple[list[Sub], list[list[WordInfo]]]:
    rec = WhisperRecognizer(WhisperOptions(
        model=model_size,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    ))
    if model is None:
        model = rec.load(device)
    return to_subs(rec.transcribe(model, file_path, lang, progress_callback))
