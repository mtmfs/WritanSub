from __future__ import annotations

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


def load_audio(path: str) -> torch.Tensor:
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
    waveform_chunk: torch.Tensor,
    romaji: str,
    model: Any,
    tokenizer: Any,
    aligner: Any,
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



def _align_one_mms(
    chunk: torch.Tensor,
    sub: Sub,
    model_bundle: tuple[Any, Any, Any],
    device: str,
    sr: int,
) -> tuple[float, float, float] | None:
    if not sub.romaji:
        return None
    model, tokenizer, aligner = model_bundle
    return align_segment(chunk, sub.romaji, model, tokenizer, aligner, device)


def _align_one_qwen3(
    chunk: torch.Tensor,
    sub: Sub,
    qwen3_model: Any,
    sr: int,
    qwen_lang: str,
) -> tuple[float, float, float] | None:
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
    waveform: torch.Tensor,
    subs: list[Sub],
    device: str = "cuda",
    pad_sec: float = 0.5,
    progress_callback: Callable[[float, str], None] | None = None,
    model_bundle: tuple[Any, Any, Any] | None = None,
    log_callback: Callable[[str], None] | None = None,
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
        reg.checkpoint()

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
            # replace 而非重建 Sub：low_words/speaker 等携带字段必须原样保留
            results.append(replace(
                sub,
                start=aligned_start + win_start,
                end=aligned_end + win_start,
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
    waveform: torch.Tensor,
    subs: list[Sub],
    device: str = "cuda",
    pad_sec: float = 0.5,
    progress_callback: Callable[[float, str], None] | None = None,
    model: Any | None = None,
    lang: str = "ja",
    log_callback: Callable[[str], None] | None = None,
) -> list[Sub]:
    """run_alignment 的 Qwen3 便捷入口。"""
    if model is None:
        model = init_qwen3_model(device)
    return run_alignment(
        waveform, subs, device=device, pad_sec=pad_sec,
        progress_callback=progress_callback,
        log_callback=log_callback,
        qwen3_model=model, lang=lang,
    )


def _combine_overlap_group(group: list[Sub]) -> Sub | None:
    """把一组跨说话人重叠 cue 压成一条：时间取并集，按先开口顺序
    每个说话人一行 "- 台词"（同说话人内部顺序直拼），score 取 min 保守标注。"""
    by_spk: dict[int, list[Sub]] = {}
    for s in group:
        by_spk.setdefault(s.speaker, []).append(s)

    parts = []
    for _spk, items in sorted(by_spk.items(), key=lambda kv: min(x.start for x in kv[1])):
        text = "".join(x.text for x in items).strip()
        # 去重：TIGER 分离串扰常使两轨听写出同一句，相同文本只留一份
        if text and text not in parts:
            parts.append(text)
    if not parts:
        return None

    text = parts[0] if len(parts) == 1 else "\n".join(f"- {p}" for p in parts)
    return Sub(
        index=group[0].index,
        start=min(s.start for s in group),
        end=max(s.end for s in group),
        text=text,
        score=min(s.score for s in group),
        low_words=[w for s in group for w in s.low_words],
        speaker=0,
    )


def _merge_speaker_overlaps(out: list[Sub]) -> list[Sub]:
    """overlap_mode="merge"：跨说话人时间重叠的 cue 传递性成组后压成单条。
    组内不足两个说话人（含同说话人自身重叠）原样保留。输入须已按 start 排序。"""
    result: list[Sub] = []
    i = 0
    n = len(out)
    while i < n:
        group = [out[i]]
        group_end = out[i].end
        j = i + 1
        while j < n and out[j].start < group_end:
            group.append(out[j])
            group_end = max(group_end, out[j].end)
            j += 1
        speakers = {s.speaker for s in group if s.speaker}
        if len(group) == 1 or len(speakers) < 2:
            result.extend(group)
        else:
            merged = _combine_overlap_group(group)
            if merged is not None:
                result.append(merged)
        i = j
    return result


def post_process(
    subs: list[Sub],
    extend_end: float = 0.30,
    extend_start: float = 0.00,
    gap_threshold: float = 0.50,
    min_gap: float = 0.30,
    min_duration: float = 0.30,
    overlap_mode: str = "merge",
) -> list[Sub]:
    """
    打轴后处理:
    1. separate 模式跨说话人重叠: merge=压成一句 "- A / - B"（默认），keep=保留双条重叠
    2. 前端向前延伸 extend_start，后端向后延伸 extend_end
    3. 相邻字幕间距处理:
       - 原始间距 >= gap_threshold → 延伸后至少保留 min_gap 空白
       - 原始间距 < gap_threshold  → 前轴延伸到后轴开头
       - 异说话人的真实重叠不压平（keep 模式的存在意义）
    4. 极短字幕向前合并 (min_duration):
       时长 < min_duration 且与前一条原始间距 <= gap_threshold 才合并，设为 0 禁用
    """
    if not subs:
        return subs

    # 显式拷贝 low_words：replace 浅拷贝共享列表引用，原地 += 会污染调用方数据
    out = [replace(s, low_words=list(s.low_words)) for s in subs]

    # separate 模式先按时间稳定排序（对齐可能轻微乱序）；无 speaker 时不排，遗留行为零变化
    if any(s.speaker for s in out):
        out.sort(key=lambda s: s.start)
        if overlap_mode == "merge":
            out = _merge_speaker_overlaps(out)

    # 快照必须在重叠合并之后，否则 pairwise 循环索引错位
    raw_starts = [s.start for s in out]
    raw_ends = [s.end for s in out]

    for s in out:
        s.start = max(0.0, s.start - extend_start)
        s.end += extend_end

    for i in range(len(out) - 1):
        curr = out[i]
        nxt = out[i + 1]

        # keep 模式：异说话人的真实重叠不做贴合/压平
        # （同说话人 <= extend_end 的轻微重叠接受，不值得引入分轨钳制的复杂度）
        if (curr.speaker and nxt.speaker and curr.speaker != nxt.speaker
                and raw_starts[i + 1] < raw_ends[i]):
            continue

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
        last_raw_end: float | None = None
        for i, sub in enumerate(out):
            cross_speaker = bool(
                merged and sub.speaker and merged[-1].speaker
                and sub.speaker != merged[-1].speaker
            )
            near_prev = (
                last_raw_end is not None
                and (raw_starts[i] - last_raw_end) <= gap_threshold  # T34: 用原始间距判相邻
            )
            if (sub.end - sub.start) < min_duration and merged and near_prev and not cross_speaker:
                prev = merged[-1]
                prev.text = prev.text + sub.text
                prev.low_words = prev.low_words + sub.low_words  # 重绑，勿用 +=
                prev.end = max(prev.end, sub.end)
                last_raw_end = max(last_raw_end, raw_ends[i])
            else:
                merged.append(sub)
                last_raw_end = raw_ends[i]
        for i, s in enumerate(merged, 1):
            s.index = i
        out = merged

    return out
