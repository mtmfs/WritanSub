import argparse
import json
import os
import signal
import sys

from writansub import __version__
from writansub.config import PP_DEFAULTS, TRANSLATE_DEFAULTS


def _ensure_utf8() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")



def _progress_bar(pct: float, msg: str) -> None:
    width = 30
    filled = int(width * min(pct, 1.0))
    bar = "#" * filled + "-" * (width - filled)
    line = f"\r  [{bar}] {pct:5.1%}  {msg}"
    sys.stderr.write(f"{line:<80}")
    sys.stderr.flush()
    if pct >= 1.0:
        sys.stderr.write("\n")


def _log(msg: str) -> None:
    sys.stderr.write(f"\r{'':<80}\r")
    sys.stderr.write(f"  {msg}\n")
    sys.stderr.flush()


def _setup_cancel_handler() -> None:
    from writansub.bridge import ResourceRegistry

    def _handler(sig, frame):
        reg = ResourceRegistry.instance()
        if reg.cancelled:
            sys.stderr.write("\n强制退出\n")
            sys.exit(1)
        reg.cancelled = True
        reg.resume()
        sys.stderr.write("\n取消请求已发送，将在当前段结束后停止\n")

    signal.signal(signal.SIGINT, _handler)



def _add_device_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="推理设备 (默认: cuda)")


def _add_lang_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lang", default="ja",
                   help="识别语言 (默认: ja)")


def _add_pp_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("后处理参数")
    g.add_argument("--extend-end", type=float, default=None,
                   help=f"向后延伸秒数 (默认: {PP_DEFAULTS['extend_end']})")
    g.add_argument("--extend-start", type=float, default=None,
                   help=f"向前延伸秒数 (默认: {PP_DEFAULTS['extend_start']})")
    g.add_argument("--gap-threshold", type=float, default=None,
                   help=f"间距阈值秒数 (默认: {PP_DEFAULTS['gap_threshold']})")
    g.add_argument("--min-gap", type=float, default=None,
                   help=f"最小间距秒数 (默认: {PP_DEFAULTS['min_gap']})")
    g.add_argument("--min-duration", type=float, default=None,
                   help=f"最小时长秒数 (默认: {PP_DEFAULTS['min_duration']})")
    g.add_argument("--pad-sec", type=float, default=None,
                   help=f"对齐窗口余量秒数 (默认: {PP_DEFAULTS['pad_sec']})")
    g.add_argument("--word-conf-threshold", type=float, default=None,
                   help=f"识别置信阈值 (默认: {PP_DEFAULTS['word_conf_threshold']})")
    g.add_argument("--align-conf-threshold", type=float, default=None,
                   help=f"对齐置信阈值 (默认: {PP_DEFAULTS['align_conf_threshold']})")


def _add_translate_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("翻译参数")
    g.add_argument("--api-base", default=None,
                   help=f"API 地址 (默认: {TRANSLATE_DEFAULTS['api_base']})")
    g.add_argument("--api-key", default=None,
                   help="API Key")
    g.add_argument("--llm-model", default=None,
                   help=f"LLM 模型名 (默认: {TRANSLATE_DEFAULTS['model']})")
    g.add_argument("--target-lang", default=None,
                   help=f"目标语言 (默认: {TRANSLATE_DEFAULTS['target_lang']})")
    g.add_argument("--batch-size", type=int, default=None,
                   help=f"每批翻译条数 (默认: {TRANSLATE_DEFAULTS['batch_size']})")


def _load_config_file(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pp(args: argparse.Namespace, file_cfg: dict) -> dict[str, float]:
    result = {}
    for key, default in PP_DEFAULTS.items():
        cli_val = getattr(args, key, None) if hasattr(args, key) else None
        file_val = file_cfg.get(key)
        if cli_val is not None:
            result[key] = cli_val
        elif file_val is not None:
            result[key] = float(file_val)
        else:
            result[key] = default
    return result


def _resolve_translate(args: argparse.Namespace, file_cfg: dict) -> dict:
    mapping = {
        "api_base": "api_base",
        "api_key": "api_key",
        "llm_model": "model",
        "target_lang": "target_lang",
        "batch_size": "batch_size",
    }
    result = {}
    for cli_attr, cfg_key in mapping.items():
        cli_val = getattr(args, cli_attr, None)
        file_val = file_cfg.get(cfg_key)
        default = TRANSLATE_DEFAULTS[cfg_key]
        if cli_val is not None:
            result[cfg_key] = cli_val
        elif file_val is not None:
            result[cfg_key] = file_val
        else:
            result[cfg_key] = default
    return result



def cmd_pipeline(args: argparse.Namespace) -> None:
    from writansub.bridge import ResourceRegistry, CancelledError
    from writansub.pipeline.runner import PipelineConfig, run_pipeline

    file_cfg = _load_config_file(args.config)
    pp = _resolve_pp(args, file_cfg)
    tr = _resolve_translate(args, file_cfg) if args.translate else {}

    tiger_mode = None
    if args.separate:
        tiger_mode = "separate"
    elif args.denoise:
        tiger_mode = "denoise"

    cfg = PipelineConfig(
        media_files=args.files,
        lang=args.lang,
        device=args.device,
        whisper_model=args.whisper_model,
        align_model=args.align_model,
        condition_on_prev=not args.no_cond_prev,
        vad_filter=args.vad,
        initial_prompt=args.initial_prompt,
        tiger_mode=tiger_mode,
        mss_model=args.mss_model,
        ss_model=args.ss_model,
        save_intermediate=args.save_intermediate,
        ref_srt=args.ref_srt,
        use_ref_sub=args.ref_sub_track is not None,
        ref_sub_track=args.ref_sub_track,
        ref_direct=args.ref_direct,
        keep_whisper_srt=args.keep_whisper_srt,
        keep_aligned_srt=args.keep_aligned_srt,
        generate_review=args.review,
        translate=args.translate,
        api_base=tr.get("api_base", TRANSLATE_DEFAULTS["api_base"]),
        api_key=tr.get("api_key", TRANSLATE_DEFAULTS["api_key"]),
        llm_model=tr.get("model", TRANSLATE_DEFAULTS["model"]),
        target_lang=tr.get("target_lang", TRANSLATE_DEFAULTS["target_lang"]),
        batch_size=tr.get("batch_size", TRANSLATE_DEFAULTS["batch_size"]),
        **pp,
    )

    _setup_cancel_handler()
    ResourceRegistry.instance().reset_controls()

    try:
        run_pipeline(cfg, log=_log, progress=_progress_bar)
    except CancelledError:
        _log("任务已取消")
        sys.exit(1)


def cmd_preprocess(args: argparse.Namespace) -> None:
    from writansub.bridge import ResourceRegistry, CancelledError
    from writansub.preprocess.core import run_dnr_batch, run_speech_batch

    do_separate = args.separate
    if do_separate:
        args.denoise = True  # 分轨需要先降噪

    if not args.denoise and not do_separate:
        _log("错误: 请至少指定 --denoise 或 --separate")
        sys.exit(1)

    _setup_cancel_handler()
    ResourceRegistry.instance().reset_controls()

    try:
        total_phases = 2 if do_separate else 1

        _log(f"── 阶段 1/{total_phases}: DnR 降噪 ──")

        def _dnr_p(pct, msg):
            _progress_bar(pct / total_phases, f"[DnR] {msg}")

        tiger_results = run_dnr_batch(
            args.files, device=args.device,
            save_intermediate=True,
            mss_model=args.mss_model,
            log_callback=_log, progress_callback=_dnr_p,
        )

        if do_separate and tiger_results:
            _log(f"── 阶段 2/{total_phases}: 说话人分轨 ──")

            def _spk_p(pct, msg):
                _progress_bar((1 + pct) / total_phases, f"[Speech] {msg}")

            run_speech_batch(
                tiger_results, device=args.device,
                save_intermediate=True,
                ss_model=args.ss_model,
                log_callback=_log, progress_callback=_spk_p,
            )

        _progress_bar(1.0, "完成")
        _log("预处理完成")
    except CancelledError:
        _log("处理已取消")
        sys.exit(1)


def cmd_transcribe(args: argparse.Namespace) -> None:
    from writansub.bridge import ResourceRegistry, CancelledError
    from writansub.transcribe.core import transcribe as do_transcribe
    from writansub.subtitle.srt_io import write_srt
    from writansub.subtitle.review import generate_review, write_review_files

    files = args.files
    if args.output and len(files) > 1:
        _log("错误: -o/--output 只能用于单文件；多文件时输出自动命名为 <media>.srt")
        sys.exit(1)

    wc = args.word_conf_threshold if args.word_conf_threshold is not None else PP_DEFAULTS["word_conf_threshold"]

    _setup_cancel_handler()
    reg = ResourceRegistry.instance()
    reg.reset_controls()

    def _w_factory():
        from faster_whisper import WhisperModel
        return WhisperModel(args.whisper_model, device=args.device, compute_type="int8")

    wh = reg.acquire_model(f"whisper:{args.whisper_model}", args.device, _w_factory)
    whisper_model = reg.get_model(wh)

    try:
        for idx, media in enumerate(files, 1):
            if len(files) > 1:
                _log(f"── [{idx}/{len(files)}] {os.path.basename(media)} ──")
            output = args.output or os.path.splitext(media)[0] + ".srt"

            subs, word_data = do_transcribe(
                media, lang=args.lang, device=args.device,
                log_callback=_log,
                progress_callback=_progress_bar,
                condition_on_previous_text=not args.no_cond_prev,
                model=whisper_model,
                vad_filter=args.vad,
                initial_prompt=args.initial_prompt,
            )

            if wc > 0.0:
                srt_c, ass_c, low_c, tot_w = generate_review(subs, word_data, wc)
                if low_c > 0:
                    base = os.path.splitext(media)[0]
                    write_review_files(base, srt_c, ass_c)
                    _log(f"低置信词 {low_c}/{tot_w}，已生成标记版")

            write_srt(subs, output)
            _log(f"完成! 输出: {output}")
    except CancelledError:
        _log("识别已取消")
        sys.exit(1)
    finally:
        reg.release_model(wh)


def cmd_align(args: argparse.Namespace) -> None:
    from writansub.bridge import ResourceRegistry, CancelledError
    from writansub.subtitle.srt_io import parse_srt, write_srt
    from writansub.align.core import (
        load_audio, run_alignment, post_process, init_model,
        init_qwen3_model, run_qwen3_alignment,
    )

    audios = args.audio
    srts = args.srt
    if len(audios) != len(srts):
        _log(f"错误: --audio 和 --srt 数量不一致 ({len(audios)} vs {len(srts)})")
        sys.exit(1)
    if args.output and len(audios) > 1:
        _log("错误: -o/--output 只能用于单对；多对时输出自动命名为 <srt>_aligned.srt")
        sys.exit(1)

    file_cfg = _load_config_file(args.config)
    pp = _resolve_pp(args, file_cfg)
    pad_sec = pp.pop("pad_sec")
    pp.pop("word_conf_threshold", None)
    pp.pop("align_conf_threshold", None)

    _setup_cancel_handler()
    reg = ResourceRegistry.instance()
    reg.reset_controls()

    model_handle = None
    try:
        import torch
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            _log("CUDA 不可用，回退到 CPU")
            device = "cpu"

        # 模型只加载一次，跨所有文件复用
        _log("加载对齐模型...")
        if args.align_model == "qwen3-fa-0.6b":
            model_handle = reg.acquire_model("qwen3_fa", device, lambda: init_qwen3_model(device))
            qwen3_model = reg.get_model(model_handle)
            model_bundle = None
        else:
            model_handle = reg.acquire_model("mms_fa", device, lambda: init_model(device))
            model_bundle = reg.get_model(model_handle)
            qwen3_model = None

        for idx, (audio, srt) in enumerate(zip(audios, srts), 1):
            if len(audios) > 1:
                _log(f"── [{idx}/{len(audios)}] {os.path.basename(audio)} ──")
            output = args.output or os.path.splitext(srt)[0] + "_aligned.srt"

            _progress_bar(0.0, "加载音频...")
            waveform = load_audio(audio)

            _progress_bar(0.05, "解析字幕...")
            subs = parse_srt(srt, lang=args.lang)
            _log(f"字幕条数: {len(subs)}")

            if args.align_model == "qwen3-fa-0.6b":
                aligned = run_qwen3_alignment(
                    waveform, subs, device=device, pad_sec=pad_sec,
                    progress_callback=lambda p, m: _progress_bar(0.1 + p * 0.85, m),
                    model=qwen3_model, lang=args.lang,
                    log_callback=_log,
                )
            else:
                aligned = run_alignment(
                    waveform, subs, device=device, pad_sec=pad_sec,
                    progress_callback=lambda p, m: _progress_bar(0.1 + p * 0.85, m),
                    model_bundle=model_bundle,
                    log_callback=_log,
                )

            _progress_bar(0.95, "后处理...")
            final = post_process(aligned, **pp)

            write_srt(final, output)
            _progress_bar(1.0, "对齐完成")
            _log(f"完成! 输出: {output}")
    except CancelledError:
        _log("对齐已取消")
        sys.exit(1)
    finally:
        if model_handle is not None:
            reg.release_model(model_handle)


def cmd_translate(args: argparse.Namespace) -> None:
    from writansub.bridge import ResourceRegistry, CancelledError
    from writansub.subtitle.srt_io import parse_srt, write_srt, merge_bilingual
    from writansub.translate.core import translate_subs

    srt = args.file
    output = args.output or os.path.splitext(srt)[0] + "_translated.srt"

    file_cfg = _load_config_file(args.config)
    tr = _resolve_translate(args, file_cfg)

    _setup_cancel_handler()
    ResourceRegistry.instance().reset_controls()

    try:
        subs = parse_srt(srt)
        translate_subs(
            subs,
            target_lang=tr["target_lang"],
            api_base=tr["api_base"],
            api_key=tr["api_key"],
            model=tr["model"],
            batch_size=tr["batch_size"],
            log_callback=_log,
            progress_callback=_progress_bar,
        )

        if args.bilingual:
            output_subs = merge_bilingual(subs)
        else:
            output_subs = [replace(s, text=s.translated or s.text) for s in subs]
        write_srt(output_subs, output)
        _progress_bar(1.0, "翻译完成")
        _log(f"完成! 输出: {output}")
    except CancelledError:
        _log("翻译已取消")
        sys.exit(1)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="writansub-cli",
        description="WritanSub — AI 字幕处理工具 (CLI)",
    )
    parser.add_argument("--version", action="version", version=f"WritanSub {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── pipeline ──
    p_pipe = sub.add_parser("pipeline", help="完整流水线 (预处理→转录→对齐→翻译)")
    p_pipe.add_argument("files", nargs="+", help="媒体文件路径")
    p_pipe.add_argument("--config", help="JSON 配置文件（覆盖默认参数）")
    _add_lang_arg(p_pipe)
    _add_device_arg(p_pipe)
    p_pipe.add_argument("--whisper-model", default="large-v3", help="Whisper 模型 (默认: large-v3)")
    p_pipe.add_argument("--align-model", default="mms_fa",
                        choices=["mms_fa", "qwen3-fa-0.6b"], help="对齐模型 (默认: mms_fa)")
    p_pipe.add_argument("--no-cond-prev", action="store_true", help="禁用前文调优")
    p_pipe.add_argument("--vad", action="store_true", help="启用 VAD 跳过静音段")
    p_pipe.add_argument("--initial-prompt", default=None,
                        help="Whisper 初始提示 (人名/术语词典，≤224 token)")
    g_tiger = p_pipe.add_argument_group("预处理")
    g_tiger.add_argument("--denoise", action="store_true", help="启用 DnR 降噪")
    g_tiger.add_argument("--separate", action="store_true", help="启用说话人分离 (含降噪)")
    g_tiger.add_argument("--mss-model", default="tiger-dnr", help="降噪模型 (默认: tiger-dnr)")
    g_tiger.add_argument("--ss-model", default="tiger-speech", help="分轨模型 (默认: tiger-speech)")
    g_tiger.add_argument("--save-intermediate", action="store_true", help="保存中间音轨")
    g_ref = p_pipe.add_argument_group("参考字幕")
    g_ref.add_argument("--ref-srt", default=None, help="外部参考 SRT 文件路径 (时间轴参考)")
    g_ref.add_argument("--ref-sub-track", type=int, default=None,
                       help="内嵌字幕轨索引 (默认: 按语言自动匹配)")
    g_ref.add_argument("--ref-direct", action="store_true",
                       help="直接使用参考时间轴，跳过强制对齐")
    g_out = p_pipe.add_argument_group("输出")
    g_out.add_argument("--keep-whisper-srt", action="store_true", help="保留 Whisper 原始 SRT")
    g_out.add_argument("--keep-aligned-srt", action="store_true", help="保留对齐 SRT")
    g_out.add_argument("--review", action="store_true", help="生成 Review 标记文件")
    g_out.add_argument("--translate", action="store_true", help="启用 AI 翻译")
    _add_pp_args(p_pipe)
    _add_translate_args(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    # ── preprocess ──
    p_pre = sub.add_parser("preprocess", help="音频预处理 (降噪 / 说话人分轨)")
    p_pre.add_argument("files", nargs="+", help="媒体文件路径")
    _add_device_arg(p_pre)
    p_pre.add_argument("--denoise", action="store_true", help="启用 DnR 降噪")
    p_pre.add_argument("--separate", action="store_true", help="启用说话人分离 (含降噪)")
    p_pre.add_argument("--mss-model", default="tiger-dnr", help="降噪模型 (默认: tiger-dnr)")
    p_pre.add_argument("--ss-model", default="tiger-speech", help="分轨模型 (默认: tiger-speech)")
    p_pre.set_defaults(func=cmd_preprocess)

    # ── transcribe ──
    p_tr = sub.add_parser("transcribe", help="语音识别 (支持多文件批处理)")
    p_tr.add_argument("files", nargs="+", help="媒体文件路径 (可多个)")
    p_tr.add_argument("-o", "--output", help="输出 SRT 路径 (仅单文件可用；多文件时自动命名为 <media>.srt)")
    _add_lang_arg(p_tr)
    _add_device_arg(p_tr)
    p_tr.add_argument("--whisper-model", default="large-v3", help="Whisper 模型 (默认: large-v3)")
    p_tr.add_argument("--no-cond-prev", action="store_true", help="禁用前文调优")
    p_tr.add_argument("--vad", action="store_true", help="启用 VAD 跳过静音段")
    p_tr.add_argument("--initial-prompt", default=None,
                      help="Whisper 初始提示 (人名/术语词典，≤224 token)")
    p_tr.add_argument("--word-conf-threshold", type=float, default=None,
                      help=f"识别置信阈值 (默认: {PP_DEFAULTS['word_conf_threshold']})")
    p_tr.set_defaults(func=cmd_transcribe)

    # ── align ──
    p_al = sub.add_parser("align", help="强制打轴 (支持多对批处理)")
    p_al.add_argument("--audio", required=True, nargs="+", help="音频文件路径 (可多个)")
    p_al.add_argument("--srt", required=True, nargs="+", help="输入 SRT 字幕路径 (可多个，与 --audio 一一对应)")
    p_al.add_argument("-o", "--output", help="输出 SRT 路径 (仅单对可用；多对时自动命名为 <srt>_aligned.srt)")
    p_al.add_argument("--config", help="JSON 配置文件")
    _add_lang_arg(p_al)
    _add_device_arg(p_al)
    p_al.add_argument("--align-model", default="mms_fa",
                      choices=["mms_fa", "qwen3-fa-0.6b"], help="对齐模型 (默认: mms_fa)")
    _add_pp_args(p_al)
    p_al.set_defaults(func=cmd_align)

    # ── translate ──
    p_tl = sub.add_parser("translate", help="AI 翻译")
    p_tl.add_argument("file", help="输入 SRT 字幕路径")
    p_tl.add_argument("-o", "--output", help="输出 SRT 路径 (默认: _translated.srt)")
    p_tl.add_argument("--config", help="JSON 配置文件")
    p_tl.add_argument("--bilingual", action="store_true", help="双语输出 (原文+译文)")
    _add_translate_args(p_tl)
    p_tl.set_defaults(func=cmd_translate)

    return parser


def main() -> None:
    _ensure_utf8()
    from writansub.network import setup_hf_mirror
    setup_hf_mirror()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.stderr.write("\n中断\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"\n错误: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
