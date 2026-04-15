"""TIGER 语音分离逻辑 (DnR) + 说话人分离 (Speech) + VAD 重叠检测"""

import os
import wave
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torchaudio.transforms as T

from writansub.bridge import ResourceRegistry


@dataclass
class TimeSpan:
    start: float
    end: float


def save_wav(waveform: torch.Tensor, path: str, sr: int) -> None:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.clamp(-1.0, 1.0).cpu()
    nch = waveform.shape[0]
    pcm = (waveform.T.contiguous() * 32767).to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def _load_dnr_model(device: str, cache_dir: str = "") -> Any:
    from writansub.vendor.tiger import TIGERDNR
    from writansub.paths import CACHE_DIR
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir=cache_dir).to(device).eval()


def _load_speech_model(device: str, cache_dir: str = "") -> Any:
    from writansub.vendor.tiger import TIGER
    from writansub.paths import CACHE_DIR
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir=cache_dir).to(device).eval()


def separate_dnr_demucs(
    waveform: torch.Tensor,
    sr: int,
    device: str = "cpu",
    model_name: str = "htdemucs_ft",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _log = log_callback or (lambda msg: None)
    _progress = progress_callback or (lambda pct, msg: None)

    reg = ResourceRegistry.instance()

    def _factory() -> Any:
        _log(f"加载 Demucs 模型 ({model_name})...")
        from demucs.pretrained import get_model
        model = get_model(model_name)
        model.to(device)
        return model

    h = reg.acquire_model(f"demucs:{model_name}", device, _factory)
    model = reg.get_model(h)

    # Demucs 需要 stereo [2, T] float32
    wav = waveform.float()
    if wav.shape[0] == 1:
        wav = wav.expand(2, -1).clone()

    try:
        _progress(0.1, "正在分离 (Demucs)...")
        _log("正在进行音源分离 (Demucs)...")

        from demucs.separate import apply_model
        # apply_model expects [batch, channels, time]
        sources = apply_model(model, wav.unsqueeze(0).to(device))
        # sources: [1, n_sources, channels, time]
        sources = sources.squeeze(0).cpu()

        # 按 model.sources 顺序取轨道
        src_idx = {name: i for i, name in enumerate(model.sources)}
        _progress(0.9, "合并音轨...")

        # vocals → dialog (mono)
        dialog = sources[src_idx["vocals"]].mean(dim=0, keepdim=True)
        # drums+bass+other → music (mono)
        music = sum(sources[src_idx[s]] for s in ("drums", "bass", "other"))
        music = music.mean(dim=0, keepdim=True)
        # Demucs 没有独立 effects，用空 tensor 占位
        effects = torch.zeros_like(dialog)
    finally:
        reg.release_model(h)

    return dialog, effects, music


def separate_dnr(
    waveform: torch.Tensor,
    sr: int,
    device: str = "cpu",
    cache_dir: str = "",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _log = log_callback or (lambda msg: None)
    _progress = progress_callback or (lambda pct, msg: None)

    reg = ResourceRegistry.instance()

    dnr_sr = 44100
    if sr != dnr_sr:
        waveform = T.Resample(sr, dnr_sr)(waveform)

    def _factory() -> Any:
        _log("加载 TIGER-DnR 模型...")
        return _load_dnr_model(device, cache_dir)

    h = reg.acquire_model("tiger_dnr", device, _factory)
    model = reg.get_model(h)

    # 每条轨道：(子模型, 显示名称, wav_chunk_inference 输出索引)
    tracks = [
        (model.dialog, "人声", 2),
        (model.effect, "音效", 1),
        (model.music,  "伴奏", 0),
    ]

    mixture = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    results = []
    try:
        for i, (sub_model, name, idx) in enumerate(tracks):
            reg.checkpoint()
            _log(f"正在分离 {name} ({i + 1}/3) ...")
            _progress(i / 3.0, f"正在分离 {name}...")
            track = model.wav_chunk_inference(sub_model, mixture)[idx]
            results.append(track.cpu())
    finally:
        reg.release_model(h)

    dialog, effects, music = results
    return dialog, effects, music


def _chunk_inference(
    model: Any,
    mixture: torch.Tensor,
    sr: int,
    n_tracks: int = 2,
    chunk_length: float = 12.0,
    hop_length: float = 4.0,
) -> torch.Tensor:
    """分块推理，防止长音频导致的显存溢出 (OOM)"""
    device = mixture.device
    batch_length = mixture.shape[-1]

    chunk_size = int(sr * chunk_length)
    hop = int(sr * hop_length)
    tr_ratio = chunk_length / hop_length

    # 在两端填充 hop 长度，确保首尾块有足够的上下文
    edge_pad = torch.zeros(
        mixture.shape[0], mixture.shape[1], chunk_size - hop,
        dtype=mixture.dtype, device=device,
    )
    padded = torch.cat([edge_pad, mixture, edge_pad], dim=-1)

    skip_idx = chunk_size - hop
    zero_pad = torch.zeros(
        mixture.shape[0], mixture.shape[1], chunk_size,
        dtype=mixture.dtype, device=device,
    )
    num_chunks = (padded.shape[-1] - chunk_size) // hop + 1

    accumulator = torch.zeros(
        mixture.shape[0], n_tracks, mixture.shape[1], padded.shape[-1],
        device=device,
    )

    reg = ResourceRegistry.instance()

    for i in range(num_chunks):
        reg.checkpoint()
        chunk = padded[:, :, i * hop:i * hop + chunk_size]
        curr_len = chunk.shape[-1]
        if curr_len < chunk_size:
            chunk = torch.cat([chunk, zero_pad[:, :, :chunk_size - curr_len]], dim=-1)

        with torch.no_grad():
            est = model(chunk).unsqueeze(2)  # [1, n_tracks, 1, T]

        seg = est[0, :, :, :curr_len][:, :, :chunk_size].unsqueeze(0)
        accumulator[:, :, :, i * hop:i * hop + chunk_size] += seg

    output = accumulator[:, :, :, skip_idx:skip_idx + batch_length].contiguous() / tr_ratio
    return output.squeeze(0)


def separate_speakers(
    dialog_wav: torch.Tensor,
    dialog_sr: int,
    device: str = "cpu",
    cache_dir: str = "",
    log_callback: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()

    speech_sr = 16000
    wav = T.Resample(dialog_sr, speech_sr)(dialog_wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    def _factory() -> Any:
        _log("加载 TIGER-Speech 模型...")
        return _load_speech_model(device, cache_dir)

    h = reg.acquire_model("tiger_speech", device, _factory)
    model = reg.get_model(h)

    try:
        _log("正在进行说话人分离...")
        separated = _chunk_inference(
            model, wav.unsqueeze(0).to(device),
            sr=speech_sr, n_tracks=2,
        )
        separated = separated.cpu()
    finally:
        reg.release_model(h)

    spk1 = separated[0].unsqueeze(0)
    spk2 = separated[1].unsqueeze(0)
    return spk1, spk2


def separate_speakers_tfgridnet(
    dialog_wav: torch.Tensor,
    dialog_sr: int,
    device: str = "cpu",
    log_callback: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用 TF-GridNet (ESPnet) 进行说话人分离。

    输出重采样到 16kHz 以保持与 TIGER-Speech 兼容。
    """
    _log = log_callback or (lambda msg: None)

    model_sr = 8000
    out_sr = 16000

    wav = T.Resample(dialog_sr, model_sr)(dialog_wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    _log("加载 TF-GridNet 模型...")
    import os
    from writansub.paths import MODELS_DIR

    reg = ResourceRegistry.instance()

    def _factory() -> Any:
        from espnet2.bin.enh_inference import SeparateSpeech

        local_base = os.path.join(
            MODELS_DIR, "tfgridnet",
            "models--espnet--yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw",
        )
        snapshot_dir = os.path.join(local_base, "snapshots")
        if os.path.isdir(snapshot_dir):
            snaps = os.listdir(snapshot_dir)
            if snaps:
                snap = os.path.join(snapshot_dir, snaps[0])
                exp_dir = os.path.join(snap, "exp", "enh_train_enh_tfgridnet_waspaa2023_raw")
                return SeparateSpeech(
                    train_config=os.path.join(exp_dir, "config.yaml"),
                    model_file=os.path.join(exp_dir, "25epoch.pth"),
                    normalize_output_wav=True,
                    device=device,
                )

        # 回退到远程下载
        return SeparateSpeech.from_pretrained(
            model_tag="espnet/yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw",
            normalize_output_wav=True,
            device=device,
        )

    h = reg.acquire_model("tfgridnet", device, _factory)
    model = reg.get_model(h)

    try:
        _log("正在进行说话人分离 (TF-GridNet)...")
        mono_np = wav.squeeze(0).numpy().astype(np.float32)
        # spatialized 模型需要双声道输入，复制单声道为立体声
        stereo_np = np.stack([mono_np, mono_np], axis=0)  # [2, T]
        outputs = model(stereo_np[np.newaxis, :], fs=model_sr)  # [1, 2, T]

        spk1 = torch.from_numpy(np.atleast_1d(outputs[0])).float()
        spk2 = torch.from_numpy(np.atleast_1d(outputs[1])).float()
        if spk1.dim() == 1:
            spk1 = spk1.unsqueeze(0)  # [1, T]
        elif spk1.dim() > 1:
            spk1 = spk1.mean(dim=0, keepdim=True)
        if spk2.dim() == 1:
            spk2 = spk2.unsqueeze(0)
        elif spk2.dim() > 1:
            spk2 = spk2.mean(dim=0, keepdim=True)

        # 重采样到 16kHz（与 TIGER-Speech 输出一致）
        spk1 = T.Resample(model_sr, out_sr)(spk1)
        spk2 = T.Resample(model_sr, out_sr)(spk2)
    finally:
        reg.release_model(h)

    return spk1, spk2


_silero_cache: tuple[Any, Any] | None = None


def _get_silero_vad() -> tuple[Any, Any]:
    global _silero_cache
    if _silero_cache is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _silero_cache = (model, utils[0])
    return _silero_cache


def _run_silero_vad(waveform: torch.Tensor, sr: int = 16000, threshold: float = 0.5) -> list[TimeSpan]:
    model, get_speech_timestamps = _get_silero_vad()

    wav = waveform.squeeze(0)
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)

    timestamps = get_speech_timestamps(wav, model, threshold=threshold)
    return [TimeSpan(start=ts["start"] / 16000.0, end=ts["end"] / 16000.0) for ts in timestamps]


def _intersect_spans(spans_a: list[TimeSpan], spans_b: list[TimeSpan]) -> list[TimeSpan]:
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
    log_callback: Callable[[str], None] | None = None,
) -> tuple[list[TimeSpan], float]:
    _log = log_callback or (lambda msg: None)

    _log("正在对说话人 1 进行 VAD 检测...")
    vad_1 = _run_silero_vad(spk1_wav, sr, vad_threshold)
    _log("正在对说话人 2 进行 VAD 检测...")
    vad_2 = _run_silero_vad(spk2_wav, sr, vad_threshold)

    overlaps = _intersect_spans(vad_1, vad_2)

    total_speech = sum(s.end - s.start for s in vad_1) + sum(s.end - s.start for s in vad_2)
    overlap_duration = sum(s.end - s.start for s in overlaps)
    overlap_ratio = overlap_duration / total_speech if total_speech > 0 else 0.0

    _log(f"检测到 {len(overlaps)} 段重叠音频, 重叠比例: {overlap_ratio:.1%}")
    return overlaps, overlap_ratio


def _make_file_progress(idx: int, total: int, cb: Callable[[float, str], None] | None):
    if not cb:
        return lambda pct, msg: None
    base, scale = idx / total, 1.0 / total
    return lambda pct, msg: cb(base + pct * scale, msg)


def run_dnr_batch(
    media_files: list[str],
    device: str = "cpu",
    cache_dir: str = "",
    save_intermediate: bool = False,
    mss_model: str = "tiger-dnr",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()
    results = {}
    total = len(media_files)

    for idx, media in enumerate(media_files):
        reg.checkpoint()
        file_info = f"[{idx + 1}/{total}]"
        _file_progress = _make_file_progress(idx, total, progress_callback)

        _file_progress(0.0, f"{file_info} 正在加载音频...")
        waveform, sr = ResourceRegistry.instance().decode_audio(media, sample_rate=44100)

        _sub_progress = lambda pct, msg, _fp=_file_progress, _fi=file_info: _fp(pct, f"{_fi} {msg}")

        if mss_model.startswith("htdemucs"):
            dialog, effects, music = separate_dnr_demucs(
                waveform, sr, device=device, model_name=mss_model,
                log_callback=_log, progress_callback=_sub_progress,
            )
        else:
            dialog, effects, music = separate_dnr(
                waveform, sr, device=device, cache_dir=cache_dir,
                log_callback=_log, progress_callback=_sub_progress,
            )

        if save_intermediate:
            out_dir = os.path.dirname(media)
            bname = os.path.splitext(os.path.basename(media))[0]
            for name, track in [("dialog", dialog), ("effects", effects), ("music", music)]:
                save_wav(track, os.path.join(out_dir, f"{bname}_{name}.wav"), 44100)

        results[media] = {"dialog_wav": dialog, "dialog_sr": 44100}

    return results


def run_speech_batch(
    dnr_results: dict,
    device: str = "cpu",
    cache_dir: str = "",
    save_intermediate: bool = False,
    ss_model: str = "tiger-speech",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()
    media_list = list(dnr_results.keys())
    total = len(media_list)

    for idx, media in enumerate(media_list):
        reg.checkpoint()
        data = dnr_results[media]
        _file_progress = _make_file_progress(idx, total, progress_callback)

        _file_progress(0.0, "正在分离说话人...")
        if ss_model == "tfgridnet-wsj0-2mix":
            spk1, spk2 = separate_speakers_tfgridnet(
                data["dialog_wav"], 44100, device=device, log_callback=_log,
            )
        else:
            spk1, spk2 = separate_speakers(
                data["dialog_wav"], 44100, device=device, cache_dir=cache_dir, log_callback=_log,
            )

        if save_intermediate:
            out_dir = os.path.dirname(media)
            bname = os.path.splitext(os.path.basename(media))[0]
            save_wav(spk1, os.path.join(out_dir, f"{bname}_spk1.wav"), 16000)
            save_wav(spk2, os.path.join(out_dir, f"{bname}_spk2.wav"), 16000)

        _file_progress(0.7, "正在进行 VAD 检测...")
        overlaps, overlap_ratio = detect_overlaps(spk1, spk2, sr=16000, log_callback=_log)

        data.update({
            "spk1_wav": spk1,
            "spk2_wav": spk2,
            "spk_sr": 16000,
            "overlap_regions": overlaps,
            "overlap_ratio": overlap_ratio,
        })
        _file_progress(1.0, "完成")
