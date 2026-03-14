"""语音识别：调用 faster-whisper 将媒体转录为 List[Sub]"""

import os
from typing import Any, Callable, List, Optional, Tuple

from writansub.types import Sub, WordInfo


def transcribe(
    file_path: str,
    lang: str = "ja",
    device: str = "cuda",
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    condition_on_previous_text: bool = True,
    model: Optional[Any] = None,
) -> Tuple[List[Sub], List[List[WordInfo]]]:
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
            subs — List[Sub] 字幕列表
            word_data — List[List[WordInfo]] 每句的词级数据
    """
    _log = log_callback or (lambda msg: None)

    def _progress(pct: float, msg: str = "") -> None:
        if progress_callback:
            progress_callback(min(pct, 1.0), msg)

    own_model = model is None
    _progress(0.0, "加载模型..." if own_model else "开始识别...")
    if own_model:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device=device, compute_type="int8")

    _progress(0.02, "识别中...")
    segments, info = model.transcribe(
        file_path,
        language=lang,
        word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
    )

    subs: List[Sub] = []
    word_data: List[List[WordInfo]] = []

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
        h = reg.register_model("whisper", model, device)
        reg.unload_model(h)

    _progress(1.0, "识别完成")
    return subs, word_data


def transcribe_to_srt(
    file_path: str,
    lang: str = "ja",
    device: str = "cuda",
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    word_conf_threshold: float = 0.50,
    condition_on_previous_text: bool = True,
    model: Optional[Any] = None,
) -> str:
    """
    兼容 wrapper：转录 + 生成 review 文件 + 写 SRT 到磁盘。

    Returns:
        生成的 SRT 文件路径 (干净版)
    """
    from writansub.subtitle.srt_io import write_srt
    from writansub.subtitle.review import generate_review, write_review_files

    _log = log_callback or (lambda msg: None)

    subs, word_data = transcribe(
        file_path,
        lang=lang,
        device=device,
        log_callback=log_callback,
        progress_callback=progress_callback,
        condition_on_previous_text=condition_on_previous_text,
        model=model,
    )

    srt_path = os.path.splitext(file_path)[0] + ".srt"
    write_srt(subs, srt_path)

    if word_conf_threshold > 0.0:
        srt_content, ass_content, low_count, total_words = generate_review(
            subs, word_data, word_conf_threshold,
        )
        if low_count > 0:
            base = os.path.splitext(file_path)[0]
            write_review_files(base, srt_content, ass_content)
            _log(f"低置信词 {low_count}/{total_words}，已生成标记版")

    return srt_path
