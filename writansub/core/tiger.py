"""TIGER 音频分离：降噪 (DnR) + 说话人分轨 (Speech) + VAD 重叠检测"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as T

from writansub.registry import ResourceRegistry


@dataclass
class TimeSpan:
    """时间区间（秒）"""
    start: float
    end: float


# ── 音频 I/O ──────────────────────────────────────────────────

def load_audio_for_tiger(path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """加载音频为 [1, T] float32 tensor，重采样到 target_sr。
    使用 ffmpeg 解码，支持 mp3/mp4/mkv 等任意格式。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()

    cmd = [
        ffmpeg, "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(target_sr),
        "-loglevel", "error", "-",
    ]
    reg = ResourceRegistry.instance()
    proc = reg.run_subprocess(cmd, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

    data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    return waveform, target_sr


def save_wav(waveform: torch.Tensor, path: str, sr: int):
    """保存 waveform [C, T] 或 [T] 为 16-bit WAV。使用标准库 wave 模块。"""
    import wave
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.clamp(-1.0, 1.0).cpu()
    nch = waveform.shape[0]
    # interleave channels: [C, T] -> [T, C] -> flat
    pcm = (waveform.T.contiguous() * 32767).to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


# ── TIGER 模型调用 ────────────────────────────────────────────

def _load_dnr_model(device: str, cache_dir: str = "cache") -> Any:
    """加载 TIGER-DnR 模型（降噪三轨分离）。"""
    from writansub.vendor.tiger import TIGERDNR
    os.makedirs(cache_dir, exist_ok=True)
    model = TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir=cache_dir)
    model.to(device).eval()
    return model


def _load_speech_model(device: str, cache_dir: str = "cache") -> Any:
    """加载 TIGER-Speech 模型（说话人分轨）。"""
    from writansub.vendor.tiger import TIGER
    os.makedirs(cache_dir, exist_ok=True)
    model = TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir=cache_dir)
    model.to(device).eval()
    return model


def separate_dnr(
    waveform: torch.Tensor,
    sr: int,
    device: str = "cpu",
    cache_dir: str = "cache",
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DnR 三轨分离：对话 / 音效 / 音乐。

    直接调用 wav_chunk_inference 而非 model.forward，
    以便在 3 个子模型之间报告进度。

    Args:
        waveform: [1, T] 原始音频
        sr: 采样率
        device: 推理设备
        cache_dir: 模型缓存目录

    Returns:
        (dialog, effects, music) 各 [1, T] @ 44100Hz
    """
    _log = log_callback or (lambda msg: None)
    _progress = progress_callback or (lambda p, m: None)
    reg = ResourceRegistry.instance()

    dnr_sr = 44100
    if sr != dnr_sr:
        waveform = T.Resample(sr, dnr_sr)(waveform)

    def _factory():
        _log("加载 TIGER-DnR 模型...")
        return _load_dnr_model(device, cache_dir)

    h = reg.acquire_model("tiger_dnr", device, _factory)
    model = reg.get_model(h)
    _log("TIGER-DnR 模型就绪")

    mixture = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    track_names = ["对话", "音效", "音乐"]
    # TIGERDNR 内部: dialog=[2], effect=[1], music=[0]
    track_indices = [2, 1, 0]
    sub_models = [model.dialog, model.effect, model.music]

    results = []
    try:
        for i, (sub_model, name, idx) in enumerate(
            zip(sub_models, track_names, track_indices)
        ):
            _log(f"分离 {name} ({i+1}/3) ...")
            _progress(i / 3.0, f"分离 {name}...")
            track = model.wav_chunk_inference(sub_model, mixture)[idx]
            results.append(track.cpu())
        dialog, effects, music = results
    finally:
        reg.release_model(h)

    return dialog, effects, music


def _chunk_inference(
    model: Any,
    mixture: torch.Tensor,
    sr: int,
    n_tracks: int = 2,
    target_length: float = 12.0,
    hop_length: float = 4.0,
) -> torch.Tensor:
    """通用分 chunk 推理，避免长音频 OOM。

    逻辑与 TIGERDNR.wav_chunk_inference 一致，但独立于模型实例。

    Args:
        model: TIGER 模型
        mixture: [1, nch, T] 输入张量（已在目标设备上）
        sr: 采样率
        n_tracks: 输出轨道数
        target_length: 每 chunk 目标秒数
        hop_length: chunk 滑动步长秒数

    Returns:
        [n_tracks, nch, T] 分离结果
    """
    device = mixture.device
    batch_length = mixture.shape[-1]

    session = int(sr * target_length)
    target = int(sr * target_length)
    ignore = (session - target) // 2
    hop = int(sr * hop_length)
    tr_ratio = target_length / hop_length

    batch_mixture_pad = mixture
    if ignore > 0:
        pad = torch.zeros(mixture.shape[0], mixture.shape[1], ignore,
                          dtype=mixture.dtype, device=device)
        batch_mixture_pad = torch.cat([pad, batch_mixture_pad, pad], -1)
    if target - hop > 0:
        pad = torch.zeros(mixture.shape[0], mixture.shape[1], target - hop,
                          dtype=mixture.dtype, device=device)
        batch_mixture_pad = torch.cat([pad, batch_mixture_pad, pad], -1)

    skip_idx = ignore + target - hop
    zero_pad = torch.zeros(mixture.shape[0], mixture.shape[1], session,
                           dtype=mixture.dtype, device=device)
    num_session = (batch_mixture_pad.shape[-1] - session) // hop + 2
    all_target = torch.zeros(
        mixture.shape[0], n_tracks, mixture.shape[1], batch_mixture_pad.shape[2],
        device=device,
    )

    all_input = []
    all_segment_length = []
    for i in range(num_session):
        chunk = batch_mixture_pad[:, :, i * hop:i * hop + session]
        seg_len = chunk.shape[-1]
        if seg_len < session:
            chunk = torch.cat([chunk, zero_pad[:, :, :session - seg_len]], -1)
        all_input.append(chunk)
        all_segment_length.append(seg_len)

    all_input = torch.cat(all_input, 0)
    for i in range(all_input.shape[0]):
        this_input = all_input[i:i + 1]
        with torch.no_grad():
            est = model(this_input).unsqueeze(2)  # [1, n_tracks, 1, T]
        seg = est[0, :, :, :all_segment_length[i]][:, :, ignore:ignore + target].unsqueeze(0)
        all_target[:, :, :, ignore + i * hop:ignore + i * hop + target] += seg

    all_target = all_target[:, :, :, skip_idx:skip_idx + batch_length].contiguous() / tr_ratio
    return all_target.squeeze(0)  # [n_tracks, nch, T]


def separate_speakers(
    dialog_wav: torch.Tensor,
    dialog_sr: int,
    device: str = "cpu",
    cache_dir: str = "cache",
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """说话人分轨：从对话轨分离出 spk1 / spk2。

    使用分 chunk 推理避免长音频 OOM。

    Args:
        dialog_wav: [1, T] 对话轨 (DnR 输出)
        dialog_sr: 对话轨采样率 (通常 44100)
        device: 推理设备
        cache_dir: 模型缓存目录

    Returns:
        (spk1, spk2) 各 [1, T] @ 16000Hz
    """
    _log = log_callback or (lambda msg: None)
    reg = ResourceRegistry.instance()

    speech_sr = 16000
    wav = T.Resample(dialog_sr, speech_sr)(dialog_wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    def _factory():
        _log("加载 TIGER-Speech 模型...")
        return _load_speech_model(device, cache_dir)

    h = reg.acquire_model("tiger_speech", device, _factory)
    model = reg.get_model(h)
    _log("TIGER-Speech 模型就绪")

    try:
        _log("分离说话人 ...")
        separated = _chunk_inference(
            model, wav.unsqueeze(0).to(device),
            sr=speech_sr, n_tracks=2,
        )
        separated = separated.cpu()  # [2, 1, T] → squeeze later
    finally:
        reg.release_model(h)

    return separated[0:1].squeeze(1), separated[1:2].squeeze(1)


# ── VAD 重叠检测 ──────────────────────────────────────────────

def _run_silero_vad(
    waveform: torch.Tensor,
    sr: int = 16000,
    threshold: float = 0.5,
) -> List[TimeSpan]:
    """使用 Silero VAD 检测有声时间段。

    Args:
        waveform: [1, T] @ sr Hz
        sr: 采样率（Silero 要求 16000）

    Returns:
        有声区间列表
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    wav = waveform.squeeze(0)  # [T]
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)

    timestamps = get_speech_timestamps(wav, model, threshold=threshold)
    spans = []
    for ts in timestamps:
        spans.append(TimeSpan(
            start=ts["start"] / 16000.0,
            end=ts["end"] / 16000.0,
        ))
    return spans


def _intersect_spans(
    spans_a: List[TimeSpan],
    spans_b: List[TimeSpan],
) -> List[TimeSpan]:
    """计算两组时间区间的交集（两人同时说话的区间）。"""
    result = []
    i, j = 0, 0
    while i < len(spans_a) and j < len(spans_b):
        start = max(spans_a[i].start, spans_b[j].start)
        end = min(spans_a[i].end, spans_b[j].end)
        if start < end:
            result.append(TimeSpan(start=start, end=end))
        if spans_a[i].end < spans_b[j].end:
            i += 1
        else:
            j += 1
    return result


def detect_overlaps(
    spk1_wav: torch.Tensor,
    spk2_wav: torch.Tensor,
    sr: int = 16000,
    vad_threshold: float = 0.5,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[TimeSpan], float]:
    """检测两条说话人轨道的重叠时间段。

    Args:
        spk1_wav: [1, T] 说话人1
        spk2_wav: [1, T] 说话人2
        sr: 采样率
        vad_threshold: VAD 阈值

    Returns:
        (overlaps, overlap_ratio):
            overlaps — 重叠区间列表
            overlap_ratio — 重叠时长占总有声时长的比例
    """
    _log = log_callback or (lambda msg: None)

    _log("VAD 检测说话人 1 ...")
    vad_1 = _run_silero_vad(spk1_wav, sr, vad_threshold)
    _log("VAD 检测说话人 2 ...")
    vad_2 = _run_silero_vad(spk2_wav, sr, vad_threshold)

    overlaps = _intersect_spans(vad_1, vad_2)

    total_speech = sum(s.end - s.start for s in vad_1) + sum(s.end - s.start for s in vad_2)
    overlap_duration = sum(s.end - s.start for s in overlaps)
    overlap_ratio = overlap_duration / total_speech if total_speech > 0 else 0.0

    _log(f"重叠区间 {len(overlaps)} 段, 重叠比例 {overlap_ratio:.1%}")
    return overlaps, overlap_ratio


# ── 批处理接口（pipeline 按模型分批调用）────────────────────────

def run_dnr_batch(
    media_files: List[str],
    device: str = "cpu",
    cache_dir: str = "cache",
    save_intermediate: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """对多个文件批量执行 DnR 降噪，模型只加载一次。

    Returns:
        {media_path: {"dialog_wav", "dialog_sr", "effects_wav", "music_wav"}}
    """
    _log = log_callback or (lambda msg: None)
    results = {}
    total = len(media_files)

    try:
        for idx, media in enumerate(media_files):
            base_pct = idx / total
            file_pct = 1.0 / total

            def _dnr_progress(pct, msg, _base=base_pct, _fp=file_pct):
                if progress_callback:
                    progress_callback(_base + pct * _fp, msg)

            _dnr_progress(0.0, "加载音频...")
            waveform, sr = load_audio_for_tiger(media, target_sr=44100)
            _log(f"音频加载完成: {waveform.shape[1] / sr:.1f}s")

            dialog, effects, music = separate_dnr(
                waveform, sr, device=device, cache_dir=cache_dir,
                log_callback=_log, progress_callback=_dnr_progress,
            )
            _log("DnR 分离完成")

            if save_intermediate:
                out_dir = os.path.dirname(media)
                bname = os.path.splitext(os.path.basename(media))[0]
                for name, track in [("dialog", dialog), ("effects", effects), ("music", music)]:
                    save_wav(track, os.path.join(out_dir, f"{bname}_{name}.wav"), 44100)

            results[media] = {
                "dialog_wav": dialog,
                "dialog_sr": 44100,
            }
    finally:
        # 无论成功还是异常都释放 DnR 模型
        ResourceRegistry.instance().flush_models()

    return results


def run_speech_batch(
    dnr_results: dict,
    device: str = "cpu",
    cache_dir: str = "cache",
    save_intermediate: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    """对多个文件批量执行说话人分轨 + VAD 重叠检测，模型只加载一次。

    结果直接写入 dnr_results 的每个 dict 中（追加 spk/overlap 字段）。
    """
    _log = log_callback or (lambda msg: None)
    media_list = list(dnr_results.keys())
    total = len(media_list)

    try:
        for idx, media in enumerate(media_list):
            data = dnr_results[media]
            dialog = data["dialog_wav"]
            base_pct = idx / total
            file_pct = 1.0 / total

            def _progress(pct, msg, _base=base_pct, _fp=file_pct):
                if progress_callback:
                    progress_callback(_base + pct * _fp, msg)

            _progress(0.0, "说话人分轨...")
            spk1, spk2 = separate_speakers(
                dialog, 44100, device=device, cache_dir=cache_dir, log_callback=_log,
            )
            _log("说话人分离完成")

            if save_intermediate:
                out_dir = os.path.dirname(media)
                bname = os.path.splitext(os.path.basename(media))[0]
                save_wav(spk1, os.path.join(out_dir, f"{bname}_spk1.wav"), 16000)
                save_wav(spk2, os.path.join(out_dir, f"{bname}_spk2.wav"), 16000)

            _progress(0.7, "VAD 重叠检测...")
            overlaps, overlap_ratio = detect_overlaps(
                spk1, spk2, sr=16000, log_callback=_log,
            )

            data.update({
                "spk1_wav": spk1,
                "spk2_wav": spk2,
                "spk_sr": 16000,
                "overlap_regions": overlaps,
                "overlap_ratio": overlap_ratio,
            })
            _progress(1.0, "完成")
    finally:
        # 无论成功还是异常都释放 Speech 模型
        ResourceRegistry.instance().flush_models()
