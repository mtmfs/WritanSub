"""强制对齐：罗马音转换、音频加载、MMS_FA 对齐、后处理"""

import re
from typing import Any, Callable, List, Optional, Tuple

from writansub.core.types import Sub

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
_katsu: Optional[Any] = None


def _get_katsu() -> Any:
    global _katsu
    if _katsu is None:
        import cutlet
        _katsu = cutlet.Cutlet()
    return _katsu


def japanese_to_romaji(text: str) -> str:
    """日语文本 → 小写罗马音，只保留 a-z。"""
    cleaned = _PUNCT_CJK_RE.sub('', text)
    if not cleaned:
        return ""
    romaji = _get_katsu().romaji(cleaned)
    return _ROMAJI_FILTER_RE.sub('', romaji.lower())


def text_to_romaji(text: str, lang: str) -> str:
    """多语言文本 → 小写罗马音 (a-z)，按语言分派转换方法。"""
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
    """加载音频，重采样到 16kHz，返回单声道 [1, T] Tensor。
    使用 ffmpeg 解码，支持 MP4/MKV/MP3 等任意格式。"""
    import shutil
    import numpy as np
    import torch
    from torchaudio.pipelines import MMS_FA as bundle

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()

    cmd = [
        ffmpeg, "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(bundle.sample_rate),
        "-loglevel", "error", "-",
    ]
    from writansub.registry import ResourceRegistry
    proc = ResourceRegistry.instance().run_subprocess(cmd, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

    data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform


def init_model(device: str) -> Tuple:
    """初始化 MMS_FA 模型、tokenizer、aligner。"""
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
) -> Optional[Tuple[float, float, float]]:
    """
    对单个音频片段做 forced alignment。
    返回: (start_sec, end_sec, avg_score) 相对于 chunk 起点，或 None。
    """
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


def run_alignment(
    waveform,
    subs: List[Sub],
    device: str = "cuda",
    pad_sec: float = 0.5,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    model_bundle: Optional[Tuple] = None,
) -> List[Sub]:
    """
    对所有字幕段执行 forced alignment。

    策略: 按原 SRT 时间戳截取音频窗口（前后各扩展 pad_sec），
    在窗口内用 MMS_FA 精确定位语音起止点。

    Args:
        model_bundle: 预加载的 (model, tokenizer, aligner)，为 None 时内部创建
    """
    from torchaudio.pipelines import MMS_FA as bundle

    sr = bundle.sample_rate
    total_duration = waveform.shape[1] / sr

    if model_bundle is not None:
        model, tokenizer, aligner = model_bundle
    else:
        model, tokenizer, aligner = init_model(device)

    results = []
    success = 0
    fail = 0
    total_subs = len(subs)

    for i, sub in enumerate(subs):
        if progress_callback:
            progress_callback(i / total_subs, f"对齐中... {i+1}/{total_subs}")

        if not sub.romaji:
            results.append(sub)
            fail += 1
            continue

        win_start = max(0.0, sub.start - pad_sec)
        win_end = min(total_duration, sub.end + pad_sec)
        start_sample = int(win_start * sr)
        end_sample = int(win_end * sr)
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < 400:
            results.append(sub)
            fail += 1
            continue

        result = align_segment(chunk, sub.romaji, model, tokenizer, aligner, device)

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
    print(f"对齐完成: {success} 成功, {fail} 失败/跳过")
    return results


def post_process(
    subs: List[Sub],
    extend_end: float = 0.30,
    extend_start: float = 0.00,
    gap_threshold: float = 0.50,
    min_gap: float = 0.30,
    min_duration: float = 0.30,
) -> List[Sub]:
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

    out = [Sub(s.index, s.start, s.end, s.text, s.romaji, s.score) for s in subs]
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
        merged: List[Sub] = []
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
