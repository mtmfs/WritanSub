import hashlib
import os
import shutil
import wave
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torchaudio.transforms as T

from writansub.bridge import ResourceRegistry
from writansub.preprocess.cache import CacheSession, media_fingerprint
from writansub.preprocess.identity import DNR_REPO, SPEECH_REPO, local_model_identity, resolve_model_identity


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


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    """save_wav 的对称读取（stdlib wave，16-bit PCM）。返回 ([nch, T] float32, sr)。

    torchaudio 2.10 的 load 需要 torchcodec 后端，本项目不引入；wav 直读足够。
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32767.0
    return data.view(-1, nch).T.contiguous(), sr


def _media_key(media: str) -> str:
    """媒体绝对路径短哈希：溢出文件名的稳定 key，同名不同目录不冲突。"""
    return hashlib.sha1(os.path.abspath(media).encode("utf-8")).hexdigest()[:12]


def spill_wav(waveform: torch.Tensor, sr: int, spill_dir: str, media: str, kind: str) -> str:
    """T17 临时音轨写入；T41 通过 CacheSession 校验身份并发布完整缓存条目。"""
    path = os.path.join(spill_dir, f"{_media_key(media)}_{kind}.wav")
    save_wav(waveform, path, sr)
    return path


def _hf_model_cached(cache_dir: str, repo_id: str) -> bool:
    dir_name = "models--" + repo_id.replace("/", "--")
    snap_dir = os.path.join(cache_dir, dir_name, "snapshots")
    if not os.path.isdir(snap_dir):
        return False
    return any(os.scandir(snap_dir))


def _from_pretrained_cached(cls: Any, repo_id: str, cache_dir: str, device: str, revision: str | None = None) -> Any:
    """缓存命中时先离线加载；半截缓存加载失败则降级联网重试，避免把用户锁死在离线模式。

    .to(device) 保持在 try 之外：显存不足/驱动错误应原样抛出，而不是被误诊为缓存损坏。
    """
    model = None
    kwargs = {"revision": revision} if revision else {}
    if _hf_model_cached(cache_dir, repo_id):
        try:
            model = cls.from_pretrained(
                repo_id, cache_dir=cache_dir, local_files_only=True, **kwargs,
            )
        except Exception as e:
            from writansub.logger import log_line
            log_line(f"[model] {repo_id} 本地缓存加载失败 ({e!r})，转为联网重试")
    if model is None:
        model = cls.from_pretrained(repo_id, cache_dir=cache_dir, **kwargs)
    return model.to(device).eval()


def _load_dnr_model(device: str, cache_dir: str = "", model_identity: dict | None = None) -> Any:
    from writansub.vendor.tiger import TIGERDNR
    from writansub.paths import CACHE_DIR
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    identity = model_identity or resolve_model_identity(DNR_REPO, cache_dir)
    model = _from_pretrained_cached(TIGERDNR, DNR_REPO, cache_dir, device, identity["revision"])
    model._writansub_identity = identity
    return model


def _load_speech_model(device: str, cache_dir: str = "", model_identity: dict | None = None) -> Any:
    from writansub.vendor.tiger import TIGER
    from writansub.paths import CACHE_DIR
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    identity = model_identity or resolve_model_identity(SPEECH_REPO, cache_dir)
    model = _from_pretrained_cached(TIGER, SPEECH_REPO, cache_dir, device, identity["revision"])
    model._writansub_identity = identity
    return model


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
    full_tracks: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    *,
    model_identity: dict | None = None,
    tracks: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """TIGER-DnR 分离。full_tracks=False 时只跑人声子模型（流水线只消费人声，
    跳过音效/伴奏两遍全长推理约 3 倍提速），effects/music 返回 None。"""
    _log = log_callback or (lambda msg: None)
    _progress = progress_callback or (lambda pct, msg: None)

    reg = ResourceRegistry.instance()
    identity = model_identity or resolve_model_identity(DNR_REPO, cache_dir)

    dnr_sr = 44100
    if sr != dnr_sr:
        waveform = T.Resample(sr, dnr_sr)(waveform)

    def _factory() -> Any:
        _log("加载 TIGER-DnR 模型...")
        return _load_dnr_model(device, cache_dir, identity)

    key = f"tiger_dnr:{identity['revision']}"
    h = reg.acquire_model(key, device, _factory)
    model = reg.get_model(h)
    if model_identity is not None:
        model_identity.update(getattr(model, "_writansub_identity", None) or identity or {})

    # 每条轨道：(子模型, 显示名称, wav_chunk_inference 输出索引)
    names = tracks if tracks is not None else (("dialog", "effects", "music") if full_tracks else ("dialog",))
    definitions = {"dialog": (model.dialog, "人声", 2),
                   "effects": (model.effect, "音效", 1), "music": (model.music, "伴奏", 0)}

    mixture = waveform.unsqueeze(0).to(device)  # [1, 1, T]
    results = {}
    try:
        for i, track_name in enumerate(names):
            sub_model, name, idx = definitions[track_name]
            reg.checkpoint()
            _log(f"正在分离 {name} ({i + 1}/{len(names)}) ...")
            _progress(i / len(names), f"正在分离 {name}...")
            track = model.wav_chunk_inference(sub_model, mixture)[idx]
            results[track_name] = track.cpu()
    finally:
        reg.release_model(h)

    return results.get("dialog"), results.get("effects"), results.get("music")


def _chunk_inference(
    model: Any,
    mixture: torch.Tensor,
    sr: int,
    n_tracks: int = 2,
    chunk_length: float = 12.0,
    hop_length: float = 4.0,
) -> torch.Tensor:
    device = mixture.device
    batch_length = mixture.shape[-1]

    chunk_size = int(sr * chunk_length)
    hop = int(sr * hop_length)
    tr_ratio = chunk_length / hop_length

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
    num_chunks = (padded.shape[-1] - chunk_size) // hop + 2

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
    *,
    model_identity: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()
    identity = model_identity or resolve_model_identity(SPEECH_REPO, cache_dir)

    speech_sr = 16000
    wav = T.Resample(dialog_sr, speech_sr)(dialog_wav)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    def _factory() -> Any:
        _log("加载 TIGER-Speech 模型...")
        return _load_speech_model(device, cache_dir, identity)

    key = f"tiger_speech:{identity['revision']}"
    h = reg.acquire_model(key, device, _factory)
    model = reg.get_model(h)
    if model_identity is not None:
        model_identity.update(getattr(model, "_writansub_identity", None) or identity or {})

    try:
        _log("正在进行说话人分离...")
        separated = _chunk_inference(
            model, wav.unsqueeze(0).to(device),
            sr=speech_sr, n_tracks=2,
        )
        separated = separated.cpu()
    finally:
        reg.release_model(h)

    spk1 = separated[0]
    spk2 = separated[1]
    return spk1, spk2


_silero_cache: tuple[Any, Any] | None = None


def _get_silero_vad() -> tuple[Any, Any]:
    global _silero_cache
    if _silero_cache is None:
        # T13: silero-vad pip 包模型内置随包分发, 加载零联网;
        # 旧 torch.hub.load 从 GitHub 在线拉取, 国内开 VAD 必败
        from silero_vad import load_silero_vad, get_speech_timestamps
        _silero_cache = (load_silero_vad(), get_speech_timestamps)
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
    spill_dir: str,
    device: str = "cpu",
    cache_dir: str = "",
    save_intermediate: bool = False,
    mss_model: str = "tiger-dnr",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cache_session: CacheSession | None = None,
) -> dict:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()
    results = {}
    total = len(media_files)
    use_cache = cache_session is not None and cache_session.enabled and not mss_model.startswith("htdemucs")

    for idx, media in enumerate(media_files):
        reg.checkpoint()
        file_info = f"[{idx + 1}/{total}]"
        _file_progress = _make_file_progress(idx, total, progress_callback)

        _sub_progress = lambda pct, msg, _fp=_file_progress, _fi=file_info: _fp(pct, f"{_fi} {msg}")
        source = media_fingerprint(media) if use_cache else None
        model_identity = (local_model_identity(DNR_REPO, cache_dir) or {}) if use_cache else None
        identity = cache_session.identity(media, "dnr", model_identity, device) if use_cache and model_identity else None
        if identity is not None:
            identity["source"] = source
        hit = cache_session.lookup(identity) if use_cache else None
        effects = music = dialog = None
        if hit is None or save_intermediate:
            _file_progress(0.0, f"{file_info} 正在加载音频...")
            waveform, sr = reg.decode_audio(media, sample_rate=44100)
            if mss_model.startswith("htdemucs"):
                dialog, effects, music = separate_dnr_demucs(
                    waveform, sr, device=device, model_name=mss_model,
                    log_callback=_log, progress_callback=_sub_progress,
                )
            else:
                extra = {"model_identity": model_identity} if use_cache else {}
                if hit:
                    extra["tracks"] = ("effects", "music")
                dialog, effects, music = separate_dnr(
                    waveform, sr, device=device, cache_dir=cache_dir,
                    full_tracks=save_intermediate,
                    log_callback=_log, progress_callback=_sub_progress, **extra,
                )
            del waveform
        if hit:
            dialog_path = hit["dialog"]
        else:
            dialog_path = spill_wav(dialog, 44100, spill_dir, media, "dialog")
            if use_cache and model_identity:
                identity = cache_session.identity(media, "dnr", model_identity, device)
                if identity is not None:
                    identity["source"] = source  # Fingerprint from before decoding/inference.
                published = cache_session.publish(identity, {"dialog": dialog_path})
                if published["dialog"] != dialog_path:
                    os.unlink(dialog_path)
                dialog_path = published["dialog"]
        reg.checkpoint()
        if save_intermediate:
            out_dir = os.path.dirname(media)
            bname = os.path.splitext(os.path.basename(media))[0]
            shutil.copyfile(dialog_path, os.path.join(out_dir, f"{bname}_dialog.wav"))
            for name, track in [("effects", effects), ("music", music)]:
                save_wav(track, os.path.join(out_dir, f"{bname}_{name}.wav"), 44100)

        # T17: 波形即分即落盘，results 只存路径——批处理 N 个文件不再全量驻留内存
        results[media] = {
            "dialog_path": dialog_path,
            "dialog_sr": 44100,
        }
        if identity is not None:
            results[media]["cache_identity"] = identity
        if mss_model.startswith("htdemucs"):
            results[media]["dnr_backend"] = "demucs"
        del dialog, effects, music  # 防上一文件的波形活到下一文件分离完成
        _file_progress(1.0, f"{file_info} 人声就绪")

    return results


def run_speech_batch(
    dnr_results: dict,
    spill_dir: str,
    device: str = "cpu",
    cache_dir: str = "",
    save_intermediate: bool = False,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cache_session: CacheSession | None = None,
) -> None:
    _log = log_callback or (lambda msg: None)

    reg = ResourceRegistry.instance()
    media_list = list(dnr_results.keys())
    total = len(media_list)

    for idx, media in enumerate(media_list):
        reg.checkpoint()
        data = dnr_results[media]
        _file_progress = _make_file_progress(idx, total, progress_callback)
        legacy_vad = data.get("dnr_backend") == "demucs"

        upstream = data.get("cache_identity")
        use_cache = cache_session is not None and cache_session.enabled and upstream is not None
        model_identity = (local_model_identity(SPEECH_REPO, cache_dir) or {}) if use_cache else None
        identity = cache_session.identity(media, "speech", model_identity, device, upstream) if use_cache and model_identity else None
        if identity is not None:
            identity["source"] = upstream["source"]
        hit = cache_session.lookup(identity) if use_cache else None
        if hit is None:
            _file_progress(0.0, "正在分离说话人...")
            dialog, dialog_sr = load_wav(data["dialog_path"])
            extra = {"model_identity": model_identity} if use_cache else {}
            spk1, spk2 = separate_speakers(
                dialog, dialog_sr, device=device, cache_dir=cache_dir, log_callback=_log, **extra,
            )
            del dialog
            hit = {"spk1": spill_wav(spk1, 16000, spill_dir, media, "spk1"),
                   "spk2": spill_wav(spk2, 16000, spill_dir, media, "spk2")}
            if not legacy_vad:
                del spk1, spk2
            if use_cache and model_identity:
                identity = cache_session.identity(media, "speech", model_identity, device, upstream)
                if identity is not None:
                    identity["source"] = upstream["source"]
                published = cache_session.publish(identity, hit)
                for name, old_path in hit.items():
                    if old_path != published[name]:
                        os.unlink(old_path)
                hit = published
        reg.checkpoint()
        if save_intermediate:
            out_dir = os.path.dirname(media)
            bname = os.path.splitext(os.path.basename(media))[0]
            for name, path in hit.items():
                shutil.copyfile(path, os.path.join(out_dir, f"{bname}_{name}.wav"))

        _file_progress(0.7, "正在进行 VAD 检测...")
        # TIGER cold/warm paths use identical PCM16 samples; Demucs keeps its old input precision.
        if not legacy_vad:
            spk1 = load_wav(hit["spk1"])[0]
            spk2 = load_wav(hit["spk2"])[0]
        overlaps, overlap_ratio = detect_overlaps(spk1, spk2, sr=16000, log_callback=_log)

        data.update({
            "spk1_path": hit["spk1"],
            "spk2_path": hit["spk2"],
            "spk_sr": 16000,
            "overlap_regions": overlaps,
            "overlap_ratio": overlap_ratio,
        })
        del spk1, spk2
        _file_progress(1.0, "完成")
