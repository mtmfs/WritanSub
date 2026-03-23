"""语音识别：调用 faster-whisper 将媒体转录为 list[Sub]"""

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
) -> tuple[list[Sub], list[list[WordInfo]]]:
    """
    使用 faster_whisper 将媒体文件转录为字幕列表。

    纯内存操作，不写任何文件。Review 逻辑由独立模块处理。

    Args:
        file_path: 媒体文件路径
        lang: 语言代码
        device: "cuda" 或 "cpu"
        log_callback: 日志回调 (msg: str) -> None
        progress_callback: 进度回调 (pct, msg) -> None
        condition_on_previous_text: 是否用前一句结果作为下一句上下文
        model: 预加载的 WhisperModel，为 None 时内部创建

    Returns:
        (subs, word_data):
            subs — list[Sub] 字幕列表
            word_data — list[list[WordInfo]] 每句的词级数据
    """
    _log = log_callback or (lambda msg: None)

    def _progress(pct: float, msg: str = "") -> None:
        if progress_callback:
            progress_callback(min(pct, 1.0), msg)

    own_model = model is None
    _progress(0.0, "加载模型..." if own_model else "开始识别...")
    if own_model:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type="int8")

    _progress(0.02, "识别中...")
    segments, info = model.transcribe(
        file_path,
        language=lang,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
    )

    subs: list[Sub] = []
    word_data: list[list[WordInfo]] = []

    for i, seg in enumerate(segments, 1):
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

    if own_model:
        from writansub.bridge import ResourceRegistry
        reg = ResourceRegistry.instance()
        h = reg.register_model(f"whisper:{model_size}", model, device)
        reg.unload_model(h)

    _progress(1.0, "识别完成")
    return subs, word_data


