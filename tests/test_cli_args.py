"""CLI 参数：批次 2/3 新增旗标的默认值与解析（T04/T10/T28）。"""
import pytest

from writansub.cli import build_parser


def test_pipeline_new_flags_defaults():
    a = build_parser().parse_args(["pipeline", "x.mp4"])
    assert a.compute_type == "int8"
    assert a.overlap_mode == "merge"
    assert a.bilingual is True


def test_pipeline_new_flags_override():
    a = build_parser().parse_args([
        "pipeline", "x.mp4",
        "--compute-type", "float16",
        "--overlap-mode", "keep",
        "--no-bilingual",
    ])
    assert a.compute_type == "float16"
    assert a.overlap_mode == "keep"
    assert a.bilingual is False


def test_transcribe_compute_type():
    a = build_parser().parse_args(["transcribe", "x.mp4", "--compute-type", "int8_float16"])
    assert a.compute_type == "int8_float16"


def test_pipeline_config_defaults():
    from writansub.pipeline.runner import PipelineConfig
    from writansub.transcribe.whisper import WhisperRecognizer
    cfg = PipelineConfig()
    assert isinstance(cfg.asr, WhisperRecognizer)
    assert cfg.asr.cache_key() == "whisper:large-v3:int8"
    assert cfg.asr is not PipelineConfig().asr
    assert not hasattr(cfg, "compute_type") and cfg.keep_asr_srt is False
    assert cfg.overlap_mode == "merge"
    assert cfg.bilingual is True


def test_cmd_pipeline_builds_whisper_recognizer(monkeypatch):
    import writansub.cli as cli
    import writansub.pipeline.runner as runner
    captured = {}
    monkeypatch.setattr(runner, "run_pipeline", lambda cfg, log, progress: captured.setdefault("cfg", cfg))
    monkeypatch.setattr(cli, "_setup_cancel_handler", lambda: None)
    args = build_parser().parse_args([
        "pipeline", "x.mp4", "--compute-type", "float16", "--vad",
        "--no-cond-prev", "--initial-prompt", "名前", "--keep-whisper-srt",
    ])
    cli.cmd_pipeline(args)
    cfg = captured["cfg"]
    assert cfg.asr.cache_key() == "whisper:large-v3:float16" and cfg.asr.tag() == "whisper-large-v3"
    o = cfg.asr.options
    assert (o.vad_filter, o.condition_on_previous_text, o.initial_prompt) == (True, False, "名前")
    assert cfg.keep_asr_srt is True


def test_ss_model_flag_removed():
    """N3：--ss-model 死参数已拆，传入应报未知参数而非静默忽略。"""
    import pytest
    for cmd in ("pipeline", "preprocess"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([cmd, "x.mp4", "--ss-model", "tiger-speech"])


def test_pipeline_config_has_no_ss_model():
    from writansub.pipeline.runner import PipelineConfig
    assert not hasattr(PipelineConfig(), "ss_model")


def test_api_key_abbreviation_matches_logger_assumption():
    """T44：logger.redact_argv 假定 --api-k 起是 --api-key 的唯一前缀；将来新增 --api-k* 旗标此处先报警。"""
    a = build_parser().parse_args(["translate", "x.srt", "--api-k", "sk-1"])
    assert a.api_key == "sk-1"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["translate", "x.srt", "--api-", "sk-1"])
