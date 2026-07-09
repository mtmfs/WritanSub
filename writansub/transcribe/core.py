from typing import Any, Callable

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
    from writansub.bridge import ResourceRegistry, CancelledError

    _log = log_callback or (lambda msg: None)
    reg = ResourceRegistry.instance()

    def _progress(pct: float, msg: str = "") -> None:
        if progress_callback:
            progress_callback(min(pct, 1.0), msg)

    if model is None:
        _progress(0.0, "加载模型...")
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type="int8")
    else:
        _progress(0.0, "开始识别...")

    _progress(0.02, "识别中...")
    segments, info = model.transcribe(
        file_path,
        language=lang,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt or None,
    )

    subs: list[Sub] = []
    word_data: list[list[WordInfo]] = []

    for i, seg in enumerate(segments, 1):
        reg.checkpoint()

        subs.append(Sub(
            index=i,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        ))

        seg_words = [
            WordInfo(word=w.word, probability=w.probability)
            for w in (seg.words or [])
        ]
        word_data.append(seg_words)

        _progress(
            seg.end / info.duration if info.duration else 0,
            f"识别中... {i} 条",
        )

    _progress(1.0, "识别完成")
    return subs, word_data

