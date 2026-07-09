"""CLI 参数：批次 2/3 新增旗标的默认值与解析（T04/T10/T28）。"""
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
    cfg = PipelineConfig()
    assert cfg.compute_type == "int8"
    assert cfg.overlap_mode == "merge"
    assert cfg.bilingual is True
