"""语音识别：调用 faster-whisper 将媒体转录为 SRT"""

import os
from typing import Any, Callable, List, Optional

from writansub.core.types import Sub, fmt_srt_time, fmt_ass_time, _ASS_REVIEW_HEADER

_keep_alive: list = []  # prevent CTranslate2 model destructor crash


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
    使用 faster_whisper 将媒体文件转录为 SRT 字幕。

    启用 word_timestamps 后，同时生成 _review.srt 标记版，
    将识别概率低于 word_conf_threshold 的词用【?...】包裹。

    Args:
        file_path: 媒体文件路径
        lang: 语言代码
        device: "cuda" 或 "cpu"
        log_callback: 日志回调 (msg: str) -> None
        word_conf_threshold: 词级置信阈值 (0~1), 0 表示禁用标记
        condition_on_previous_text: 是否用前一句结果作为下一句上下文
        model: 预加载的 WhisperModel，为 None 时内部创建

    Returns:
        生成的 SRT 文件路径 (干净版)
    """

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    def _progress(pct: float, msg: str = ""):
        if progress_callback:
            progress_callback(min(pct, 1.0), msg)

    srt_path = os.path.splitext(file_path)[0] + ".srt"
    review_path = os.path.splitext(file_path)[0] + "_review.srt"
    ass_path = os.path.splitext(file_path)[0] + "_review.ass"
    mark_enabled = word_conf_threshold > 0.0

    _own_model = model is None
    _progress(0.0, "加载模型..." if _own_model else "开始识别...")
    if _own_model:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device=device, compute_type="int8")

    _progress(0.02, "识别中...")
    segments, info = model.transcribe(
        file_path, language=lang, word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
    )

    low_conf_count = 0
    total_words = 0

    f_review = open(review_path, "w", encoding="utf-8") if mark_enabled else None
    f_ass = None
    if mark_enabled:
        f_ass = open(ass_path, "w", encoding="utf-8-sig")
        f_ass.write(_ASS_REVIEW_HEADER)

    try:
        with open(srt_path, "w", encoding="utf-8") as f_clean:
            for i, seg in enumerate(segments, 1):
                text_clean = seg.text.strip()
                time_line = (
                    f"{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}"
                )

                f_clean.write(f"{i}\n{time_line}\n{text_clean}\n\n")

                if mark_enabled:
                    if seg.words:
                        srt_parts: List[str] = []
                        ass_parts: List[str] = []
                        for w in seg.words:
                            total_words += 1
                            if w.probability < word_conf_threshold:
                                stripped = w.word.lstrip()
                                leading = w.word[: len(w.word) - len(stripped)]
                                srt_parts.append(f"{leading}【?{stripped}】")
                                ass_parts.append(
                                    f"{leading}{{\\c&H0000FF&}}{stripped}{{\\c}}"
                                )
                                low_conf_count += 1
                            else:
                                srt_parts.append(w.word)
                                ass_parts.append(w.word)
                        text_review = "".join(srt_parts).strip()
                        ass_review = "".join(ass_parts).strip()
                    else:
                        text_review = text_clean
                        ass_review = text_clean

                    if f_review is not None:
                        f_review.write(
                            f"{i}\n{time_line}\n{text_review}\n\n"
                        )
                    if f_ass is not None:
                        f_ass.write(
                            f"Dialogue: 0,"
                            f"{fmt_ass_time(seg.start)},"
                            f"{fmt_ass_time(seg.end)},"
                            f"Default,,0,0,0,,{ass_review}\n"
                        )

                _progress(
                    seg.end / info.duration if info.duration else 0,
                    f"识别中... {i} 条",
                )
    finally:
        if f_review is not None:
            f_review.close()
        if f_ass is not None:
            f_ass.close()

    if _own_model:
        _keep_alive.append(model)
    _progress(1.0, "识别完成")

    if mark_enabled:
        if low_conf_count > 0:
            _log(f"低置信词 {low_conf_count}/{total_words}，已生成标记版")
        else:
            for p in (review_path, ass_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return srt_path
