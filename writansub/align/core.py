import re
from dataclasses import replace
from typing import Any, Callable

from writansub.types import Sub

_ROMAJI_FILTER_RE = re.compile(r'[^a-z]')
_PUNCT_CJK_RE = re.compile(
    r'[「」『』【】（）()\[\]{}'
    r'、。，．！？!?…♪～~♡♥☆★※＊・：；""\'\'《》〈〉'
    r'─―—\-'
    r'0-9０-９\s]'
)
_PUNCT_LATIN_RE = re.compile(
    r'[「」『』【】（）()\[\]{}'
    r'、。，．！？!?…♪～~♡♥☆★※＊・：；""\'\'《》〈〉'
    r'─―—\-,.:;!?\'"'
    r'0-9０-９\s]'
)
_katsu: Any | None = None


def _get_katsu() -> Any:
    global _katsu
    if _katsu is None:
        import cutlet
        _katsu = cutlet.Cutlet()
    return _katsu


def japanese_to_romaji(text: str) -> str:
    cleaned = _PUNCT_CJK_RE.sub('', text)
    if not cleaned:
        return ""
    romaji = _get_katsu().romaji(cleaned)
    return _ROMAJI_FILTER_RE.sub('', romaji.lower())


def text_to_romaji(text: str, lang: str) -> str:
    if lang == "ja":
        return japanese_to_romaji(text)

    cleaned = _PUNCT_LATIN_RE.sub('', text)
    if not cleaned:
        return ""

    if lang == "zh":
        try:
            from pypinyin import lazy_pinyin
            romaji = ''.join(lazy_pinyin(cleaned))
        except ImportError:
            from unidecode import unidecode
            romaji = unidecode(cleaned)
    elif lang == "ko":
        try:
            from korean_romanizer.romanizer import Romanizer
            romaji = Romanizer(cleaned).romanize()
        except ImportError:
            from unidecode import unidecode
            romaji = unidecode(cleaned)
    else:
        try:
            from unidecode import unidecode
            romaji = unidecode(cleaned)
        except ImportError:
            romaji = cleaned

    return _ROMAJI_FILTER_RE.sub('', romaji.lower())


def load_audio(path: str):
    from torchaudio.pipelines import MMS_FA as bundle
    from writansub.bridge import ResourceRegistry
    waveform, _ = ResourceRegistry.instance().decode_audio(path, sample_rate=bundle.sample_rate)
    return waveform


def init_model(device: str) -> tuple:
    from torchaudio.pipelines import MMS_FA as bundle
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    return model, tokenizer, aligner


def align_segment(
    waveform_chunk,
    romaji: str,
    model,
    tokenizer,
    aligner,
    device: str,
) -> tuple[float, float, float] | None:
    """返回 (start_sec, end_sec, avg_score) 或 None。"""
    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    if not romaji:
        return None

    with torch.inference_mode():
        emission, _ = model(waveform_chunk.to(device))

    try:
        token_spans = aligner(emission[0], tokenizer(["*", romaji, "*"]))
    except RuntimeError:
        return None

    if len(token_spans) < 3 or not token_spans[1]:
        return None

    text_spans = token_spans[1]

    ratio = waveform_chunk.shape[1] / emission.shape[1] / bundle.sample_rate
    start_sec = text_spans[0].start * ratio
    end_sec = text_spans[-1].end * ratio
    avg_score = sum(s.score for s in text_spans) / len(text_spans)

    return (start_sec, end_sec, avg_score)


LANG_MAP: dict[str, str] = {
    "ja": "Japanese",
    "zh": "Chinese",
    "en": "English",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "vi": "Vietnamese",
}


def init_qwen3_model(device: str) -> Any:
    import os
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from writansub.paths import MODELS_DIR

    local_path = os.path.join(MODELS_DIR, "Qwen3-ForcedAligner-0.6B")
    model_id = local_path if os.path.isdir(local_path) else "Qwen/Qwen3-ForcedAligner-0.6B"

    model = Qwen3ForcedAligner.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device if device != "cuda" else "cuda:0",
    )
    return model



def _align_one_mms(chunk, sub, model_bundle, device, sr):
    if not sub.romaji:
        return None
    model, tokenizer, aligner = model_bundle
    return align_segment(chunk, sub.romaji, model, tokenizer, aligner, device)


def _align_one_qwen3(chunk, sub, qwen3_model, sr, qwen_lang):
    import numpy as np

    text = sub.text.strip()
    if not text:
        return None
    try:
        chunk_np = chunk.squeeze(0).numpy().astype(np.float32)
        align_results = qwen3_model.align(
            audio=(chunk_np, sr), text=text, language=qwen_lang,
        )
        segments = align_results[0] if align_results else []
        if segments:
            aligned_dur = segments[-1].end_time - segments[0].start_time
            original_dur = sub.end - sub.start
            if aligned_dur > 0 and original_dur > 0:
                score = min(aligned_dur, original_dur) / max(aligned_dur, original_dur)
            else:
                score = 0.0
            return (segments[0].start_time, segments[-1].end_time, score)
    except Exception:
        pass
    return None


def run_alignment(
    waveform,
    subs: list[Sub],
    device: str = "cuda",
    pad_sec: float = 0.5,
    progress_callback: Callable[[float, str], None] | None = None,
    model_bundle: tuple | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *,
    qwen3_model: Any | None = None,
    lang: str = "ja",
) -> list[Sub]:
    """传 model_bundle → MMS_FA，传 qwen3_model → Qwen3，都不传 → 默认 MMS_FA。"""
    use_qwen3 = qwen3_model is not None

    if use_qwen3:
        sr = 16000
        qwen_lang = LANG_MAP.get(lang, "Japanese")
        align_fn = lambda chunk, sub: _align_one_qwen3(chunk, sub, qwen3_model, sr, qwen_lang)
    else:
        from torchaudio.pipelines import MMS_FA as bundle
        sr = bundle.sample_rate
        if model_bundle is None:
            model_bundle = init_model(device)
        align_fn = lambda chunk, sub: _align_one_mms(chunk, sub, model_bundle, device, sr)

    from writansub.bridge import ResourceRegistry

    total_duration = waveform.shape[1] / sr
    reg = ResourceRegistry.instance()
    _log = log_callback or (lambda msg: None)

    results = []
    success = 0
    fail = 0
    total_subs = len(subs)

    for i, sub in enumerate(subs):
        reg.checkpoint()  # 暂停 / 取消

        if progress_callback:
            progress_callback(i / total_subs, f"对齐中... {i+1}/{total_subs}")

        win_start = max(0.0, sub.start - pad_sec)
        win_end = min(total_duration, sub.end + pad_sec)
        start_sample = int(win_start * sr)
        end_sample = int(win_end * sr)
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < 400:
            results.append(sub)
            fail += 1
            continue

        result = align_fn(chunk, sub)

        if result is not None:
            aligned_start, aligned_end, avg_score = result
            results.append(Sub(
                index=sub.index,
                start=aligned_start + win_start,
                end=aligned_end + win_start,
                text=sub.text,
                romaji=sub.romaji,
                score=avg_score,
            ))
            success += 1
        else:
            results.append(sub)
            fail += 1

    if progress_callback:
        progress_callback(1.0, f"对齐完成 ({success}成功/{fail}跳过)")
    _log(f"对齐完成: {success} 成功, {fail} 失败/跳过")
    return results


def run_qwen3_alignment(
    waveform,
    subs: list[Sub],
    device: str = "cuda",
    pad_sec: float = 0.5,
    progress_callback: Callable[[float, str], None] | None = None,
    model: Any | None = None,
    lang: str = "ja",
    log_callback: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Sub]:
    """run_alignment 的 Qwen3 便捷入口。"""
    if model is None:
        model = init_qwen3_model(device)
    return run_alignment(
        waveform, subs, device=device, pad_sec=pad_sec,
        progress_callback=progress_callback,
        log_callback=log_callback, cancelled=cancelled,
        qwen3_model=model, lang=lang,
    )


def post_process(
    subs: list[Sub],
    extend_end: float = 0.30,
    extend_start: float = 0.00,
    gap_threshold: float = 0.50,
    min_gap: float = 0.30,
    min_duration: float = 0.30,
) -> list[Sub]:
    """
    打轴后处理:
    1. 前端向前延伸 extend_start
    2. 后端向后延伸 extend_end
    3. 相邻字幕间距处理:
       - 原始间距 >= gap_threshold → 延伸后至少保留 min_gap 空白
       - 原始间距 < gap_threshold  → 前轴延伸到后轴开头
    4. 极短字幕向前合并 (min_duration):
       时长 < min_duration 的字幕合并到前一条，设为 0 禁用
    """
    if not subs:
        return subs

    out = [replace(s) for s in subs]
    raw_starts = [s.start for s in out]
    raw_ends = [s.end for s in out]

    for s in out:
        s.start = max(0.0, s.start - extend_start)
        s.end += extend_end

    for i in range(len(out) - 1):
        curr = out[i]
        nxt = out[i + 1]
        original_gap = raw_starts[i + 1] - raw_ends[i]

        if original_gap >= gap_threshold:
            max_end = nxt.start - min_gap
            if curr.end > max_end:
                curr.end = max_end
        else:
            curr.end = nxt.start

        if curr.end < curr.start:
            curr.end = curr.start + 0.01

    if min_duration > 0:
        merged: list[Sub] = []
        for sub in out:
            if (sub.end - sub.start) < min_duration and merged:
                prev = merged[-1]
                prev.text = prev.text + sub.text
                prev.end = max(prev.end, sub.end)
            else:
                merged.append(sub)
        for i, s in enumerate(merged, 1):
            s.index = i
        out = merged

    return out
