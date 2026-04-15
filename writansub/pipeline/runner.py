import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from writansub.bridge import ResourceRegistry, CancelledError
from writansub.config import PP_DEFAULTS, TRANSLATE_DEFAULTS
from writansub.subtitle.srt_io import parse_srt, write_srt, populate_romaji, merge_bilingual
from writansub.subtitle.review import generate_review, write_review_files, mark_low_align_in_review
from writansub.transcribe.core import transcribe
from writansub.translate.core import translate_subs
from writansub.align.core import load_audio, run_alignment, post_process, init_model


@dataclass
class PipelineConfig:
    media_files: list[str] = field(default_factory=list)
    lang: str = "ja"
    device: str = "cuda"
    whisper_model: str = "large-v3"
    align_model: str = "mms_fa"
    condition_on_prev: bool = True
    vad_filter: bool = False
    initial_prompt: str | None = None
    tiger_mode: str | None = None       # None | "denoise" | "separate"
    mss_model: str = "tiger-dnr"
    ss_model: str = "tiger-speech"
    save_intermediate: bool = False
    keep_whisper_srt: bool = False
    keep_aligned_srt: bool = False
    generate_review: bool = False
    translate: bool = False
    api_base: str = TRANSLATE_DEFAULTS["api_base"]
    api_key: str = TRANSLATE_DEFAULTS["api_key"]
    llm_model: str = TRANSLATE_DEFAULTS["model"]
    target_lang: str = TRANSLATE_DEFAULTS["target_lang"]
    batch_size: int = TRANSLATE_DEFAULTS["batch_size"]
    ref_srt: str | None = None
    use_ref_sub: bool = False
    ref_sub_track: int | None = None        # 内嵌字幕轨索引（None = 按语言自动匹配）
    ref_direct: bool = False                # True 时跳过对齐，直接用参考时间轴
    extend_end: float = PP_DEFAULTS["extend_end"]
    extend_start: float = PP_DEFAULTS["extend_start"]
    gap_threshold: float = PP_DEFAULTS["gap_threshold"]
    min_gap: float = PP_DEFAULTS["min_gap"]
    word_conf_threshold: float = PP_DEFAULTS["word_conf_threshold"]
    align_conf_threshold: float = PP_DEFAULTS["align_conf_threshold"]
    min_duration: float = PP_DEFAULTS["min_duration"]
    pad_sec: float = PP_DEFAULTS["pad_sec"]


def run_pipeline(
    cfg: PipelineConfig,
    log: Callable[[str], None],
    progress: Callable[[float, str], None],
) -> None:
    """TIGER → Whisper → 对齐 → 翻译。"""
    import torch

    reg = ResourceRegistry.instance()
    _cancelled = lambda: reg.cancelled

    pp_keys = {"extend_end", "extend_start", "gap_threshold", "min_gap", "min_duration"}
    pp = {k: getattr(cfg, k) for k in pp_keys}

    has_ref = bool(cfg.ref_srt) or cfg.use_ref_sub
    skip_align = cfg.ref_direct and has_ref
    num_phases = (1 if skip_align else 2) + (1 if cfg.translate else 0) + (1 if cfg.tiger_mode else 0)
    phase_offset = 1 if cfg.tiger_mode else 0
    total = len(cfg.media_files)

    sub_results: dict[str, Any] = {}
    word_results: dict[str, Any] = {}
    tiger_results: dict[str, Any] = {}

    # Phase: TIGER 增强
    if cfg.tiger_mode and not _cancelled():
        tiger_results = _run_tiger_phase(
            cfg, num_phases, log, progress, _cancelled,
        )

    # Phase: Whisper 转录
    w_phase = 1 + phase_offset
    log(f">>> Phase {w_phase}/{num_phases}: Whisper 转录")

    def _w_factory():
        from faster_whisper import WhisperModel
        return WhisperModel(cfg.whisper_model, device=cfg.device, compute_type="int8")

    wh = reg.acquire_model(f"whisper:{cfg.whisper_model}", cfg.device, _w_factory)
    whisper_model = reg.get_model(wh)

    try:
        for idx, media in enumerate(cfg.media_files, 1):
            if _cancelled():
                break

            def _w_p(pct, msg, _idx=idx):
                progress(
                    (phase_offset * total + (_idx - 1) + pct) / (total * num_phases),
                    f"[Whisper {_idx}/{total}] {msg}",
                )

            subs, word_data = _transcribe_single(
                media, tiger_results.get(media), cfg, whisper_model, _w_p, log,
            )
            sub_results[media] = subs
            word_results[media] = word_data

            base = os.path.splitext(media)[0]
            if cfg.generate_review and cfg.word_conf_threshold > 0:
                srt_c, ass_c, low_c, tot_w = generate_review(subs, word_data, cfg.word_conf_threshold)
                if low_c > 0:
                    write_review_files(base, srt_c, ass_c)
            if cfg.keep_whisper_srt:
                write_srt(subs, base + ".srt")
    finally:
        reg.release_model(wh)

    # Phase: 参考字幕映射（可选）
    if sub_results and not _cancelled() and has_ref:
        from writansub.subtitle.ref_align import map_whisper_to_ref
        from writansub.subtitle.extract import probe_subtitle_tracks, select_track, extract_subtitle

        external_ref: list | None = None
        if cfg.ref_srt:
            external_ref = parse_srt(cfg.ref_srt, lang=cfg.lang)
            log(f"使用外部参考字幕: {cfg.ref_srt} ({len(external_ref)} 条)")

        for media in cfg.media_files:
            if media not in sub_results:
                continue
            try:
                if external_ref is not None:
                    ref_subs = external_ref
                else:
                    tracks = probe_subtitle_tracks(media)
                    if not tracks:
                        log(f"未找到内嵌字幕轨: {os.path.basename(media)}")
                        continue
                    track_idx = cfg.ref_sub_track if cfg.ref_sub_track is not None else select_track(tracks, cfg.lang)
                    if track_idx is None:
                        log(f"未匹配到字幕轨: {os.path.basename(media)}")
                        continue
                    ref_subs = extract_subtitle(media, track_idx)
                    log(f"提取内嵌字幕轨 #{track_idx}: {len(ref_subs)} 条")

                mapped = map_whisper_to_ref(sub_results[media], ref_subs)
                log(f"参考字幕映射完成: {len(mapped)} 条 (原 Whisper {len(sub_results[media])} 条)")
                sub_results[media] = mapped
            except CancelledError:
                raise
            except Exception as e:
                log(f"参考字幕处理失败，跳过: {e}")

    a_phase = 2 + phase_offset
    aligned_results: dict[str, Any] = {}
    if sub_results and not _cancelled() and not skip_align:
        use_qwen3 = (cfg.align_model == "qwen3-fa-0.6b")
        label = "Qwen3 对齐" if use_qwen3 else "MMS 对齐"
        log(f">>> Phase {a_phase}/{num_phases}: {label}")

        if use_qwen3:
            from writansub.align.core import init_qwen3_model, run_qwen3_alignment
            mh = reg.acquire_model("qwen3_fa", cfg.device, lambda: init_qwen3_model(cfg.device))
            qwen3_model = reg.get_model(mh)
        else:
            mh = reg.acquire_model("mms_fa", cfg.device, lambda: init_model(cfg.device))
            mms_bundle = reg.get_model(mh)

        try:
            for idx, media in enumerate(cfg.media_files, 1):
                if _cancelled() or media not in sub_results:
                    continue

                def _a_p(pct, msg, _idx=idx):
                    progress(
                        ((1 + phase_offset) * total + (_idx - 1) + pct) / (total * num_phases),
                        f"[对齐 {_idx}/{total}] {msg}",
                    )

                tiger_data = tiger_results.get(media)
                if tiger_data and "dialog_wav" in tiger_data:
                    import torchaudio.transforms as T
                    from torchaudio.pipelines import MMS_FA as _mms_bundle
                    target_sr = 16000 if use_qwen3 else _mms_bundle.sample_rate
                    src_sr = tiger_data["dialog_sr"]
                    waveform = tiger_data["dialog_wav"]
                    if src_sr != target_sr:
                        waveform = T.Resample(src_sr, target_sr)(waveform)
                else:
                    waveform = load_audio(media)

                if use_qwen3:
                    aligned = run_qwen3_alignment(
                        waveform, sub_results[media],
                        device=cfg.device, pad_sec=cfg.pad_sec,
                        progress_callback=_a_p,
                        model=qwen3_model, lang=cfg.lang,
                        log_callback=log,
                        cancelled=_cancelled,
                    )
                else:
                    populate_romaji(sub_results[media], cfg.lang)
                    aligned = run_alignment(
                        waveform, sub_results[media],
                        device=cfg.device, pad_sec=cfg.pad_sec,
                        progress_callback=_a_p, model_bundle=mms_bundle,
                        log_callback=log,
                        cancelled=_cancelled,
                    )

                final = post_process(aligned, **pp)
                aligned_results[media] = final

                base = os.path.splitext(media)[0]
                if cfg.generate_review and cfg.align_conf_threshold > 0:
                    low_a = {s.index for s in final if s.score < cfg.align_conf_threshold}
                    if low_a:
                        mark_low_align_in_review(base, low_a)
                if cfg.keep_aligned_srt:
                    write_srt(final, base + "_aligned.srt")
        finally:
            reg.release_model(mh)

    # ref_direct: 跳过对齐，直接后处理
    if skip_align and sub_results and not _cancelled():
        log("参考字幕直接模式: 跳过强制对齐")
        for media, subs in sub_results.items():
            aligned_results[media] = post_process(subs, **pp)

    if cfg.translate and aligned_results and not _cancelled():
        log(f">>> Phase {num_phases}/{num_phases}: AI 翻译")
        for idx, media in enumerate(cfg.media_files, 1):
            if _cancelled() or media not in aligned_results:
                continue

            def _t_p(pct, msg, _idx=idx):
                progress(
                    ((num_phases - 1) * total + (_idx - 1) + pct) / (total * num_phases),
                    f"[翻译 {_idx}/{total}] {msg}",
                )

            translate_subs(
                aligned_results[media],
                target_lang=cfg.target_lang,
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                model=cfg.llm_model,
                batch_size=cfg.batch_size,
                log_callback=log, progress_callback=_t_p,
                cancelled=_cancelled,
            )
            write_srt(
                merge_bilingual(aligned_results[media]),
                os.path.splitext(media)[0] + ".srt",
            )
    elif aligned_results and not _cancelled():
        for media, subs in aligned_results.items():
            write_srt(subs, os.path.splitext(media)[0] + ".srt")

    if not _cancelled():
        progress(1.0, "任务完成")
        log(f"全部完成! 已处理 {total} 个文件")



def _run_tiger_phase(
    cfg: PipelineConfig,
    num_phases: int,
    log: Callable[[str], None],
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> dict:
    log(f">>> Phase 1/{num_phases}: TIGER 增强")
    from writansub.preprocess.core import run_dnr_batch, run_speech_batch

    do_speech = (cfg.tiger_mode == "separate")
    dnr_weight = 0.5 if do_speech else 1.0

    def _dnr_p(pct, msg):
        progress(pct * dnr_weight / num_phases, f"[TIGER] {msg}")

    tiger_results = run_dnr_batch(
        cfg.media_files, device=cfg.device, save_intermediate=cfg.save_intermediate,
        mss_model=cfg.mss_model,
        log_callback=log, progress_callback=_dnr_p,
    )

    if do_speech and tiger_results and not cancelled():
        def _spk_p(pct, msg):
            progress((0.5 + pct * 0.5) / num_phases, f"[TIGER] {msg}")

        run_speech_batch(
            tiger_results, device=cfg.device, save_intermediate=cfg.save_intermediate,
            ss_model=cfg.ss_model,
            log_callback=log, progress_callback=_spk_p,
        )

    return tiger_results


def _transcribe_single(
    media: str,
    tiger_data: dict | None,
    cfg: PipelineConfig,
    whisper_model: Any,
    progress_callback: Callable[[float, str], None],
    log: Callable[[str], None],
) -> tuple[list, list]:
    from writansub.preprocess.core import save_wav

    whisper_input = media
    tmp_dialog = None

    if tiger_data and "dialog_wav" in tiger_data:
        tmp_dialog = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_dialog.close()
        save_wav(tiger_data["dialog_wav"], tmp_dialog.name, tiger_data["dialog_sr"])
        whisper_input = tmp_dialog.name

    try:
        overlap_r = tiger_data.get("overlap_regions") if tiger_data else None
        separated = None
        if tiger_data and "spk1_wav" in tiger_data:
            separated = (tiger_data["spk1_wav"], tiger_data["spk2_wav"])

        if overlap_r and separated:
            return _whisper_with_overlap(
                whisper_input, overlap_r, separated,
                tiger_data.get("spk_sr", 16000),
                cfg, whisper_model, progress_callback, log,
            )

        return transcribe(
            whisper_input, lang=cfg.lang, device=cfg.device,
            log_callback=log,
            progress_callback=progress_callback,
            condition_on_previous_text=cfg.condition_on_prev,
            model=whisper_model,
            cancelled=lambda: ResourceRegistry.instance().cancelled,
            vad_filter=cfg.vad_filter,
            initial_prompt=cfg.initial_prompt,
        )
    finally:
        if tmp_dialog:
            try:
                os.unlink(tmp_dialog.name)
            except OSError:
                pass


def _whisper_with_overlap(
    media: str,
    overlap_regions: list,
    separated_tracks: tuple,
    spk_sr: int,
    cfg: PipelineConfig,
    whisper_model: Any,
    progress_callback: Callable[[float, str], None],
    log: Callable[[str], None],
) -> tuple[list, list]:
    from writansub.preprocess.core import save_wav

    reg = ResourceRegistry.instance()
    _cancelled = lambda: reg.cancelled

    full_subs, full_word_data = transcribe(
        media, lang=cfg.lang, device=cfg.device,
        log_callback=log,
        progress_callback=progress_callback,
        condition_on_previous_text=cfg.condition_on_prev,
        model=whisper_model,
        cancelled=_cancelled,
        initial_prompt=cfg.initial_prompt,
    )
    if not overlap_regions:
        return full_subs, full_word_data

    overlap_subs = []
    spk1_wav, spk2_wav = separated_tracks

    for region in overlap_regions:
        if _cancelled():
            break
        start = int(region.start * spk_sr)
        end = int(region.end * spk_sr)

        for spk_wav in [spk1_wav, spk2_wav]:
            chunk = spk_wav[:, start:end]
            if chunk.shape[1] < 1600:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                save_wav(chunk, tmp_path, spk_sr)
                local_subs, _ = transcribe(
                    tmp_path, lang=cfg.lang, device=cfg.device,
                    condition_on_previous_text=False, model=whisper_model,
                    initial_prompt=cfg.initial_prompt,
                )
                for s in local_subs:
                    s.start += region.start
                    s.end += region.start
                overlap_subs.extend(local_subs)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    commented = set()
    for i, sub in enumerate(full_subs):
        mid = (sub.start + sub.end) / 2
        if any(r.start <= mid <= r.end for r in overlap_regions):
            commented.add(i)

    active = [s for i, s in enumerate(full_subs) if i not in commented] + overlap_subs
    active.sort(key=lambda s: s.start)
    for i, s in enumerate(active, 1):
        s.index = i

    return active, [[] for _ in active]
