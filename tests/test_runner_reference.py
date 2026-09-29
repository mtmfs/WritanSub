"""真实 runner 的参考轴回退：模型推理替身，字幕解析、后处理和写盘走实际代码。"""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import writansub.align.core as alignment
import writansub.pipeline.runner as runner
import writansub.subtitle.extract as extract
from writansub.bridge import CancelledError
from writansub.subtitle.srt_io import parse_srt, write_srt
from writansub.types import Sub


@pytest.fixture
def pipeline_case(monkeypatch, registry, tmp_path):
    state = SimpleNamespace(models=[], released=[], aligned=[], logs=[], progress=[], reviews=[])

    def acquire(name, device, factory):
        state.models.append(name)
        return name

    monkeypatch.setattr(registry, "acquire_model", acquire)
    monkeypatch.setattr(registry, "get_model", lambda handle: object())
    monkeypatch.setattr(registry, "release_model", state.released.append)
    monkeypatch.setattr(runner, "load_audio", lambda path: torch.zeros(1, 16000 * 30))
    monkeypatch.setattr(runner, "populate_romaji", lambda subs, lang: None)

    def align(waveform, subs, **kwargs):
        state.aligned.append(copy.deepcopy(subs))
        kwargs["progress_callback"](.5, "模型处理中")
        kwargs["progress_callback"](1, "模型完成")
        return [replace(s, start=s.start + .1, end=s.end + .1, score=0) for s in subs]

    monkeypatch.setattr(runner, "run_alignment", align)
    monkeypatch.setattr(alignment, "run_qwen3_alignment", align)
    review = runner.generate_review_final

    def generate_review(subs, threshold):
        state.reviews.append((copy.deepcopy(subs), threshold))
        return review(subs, threshold)

    monkeypatch.setattr(runner, "generate_review_final", generate_review)

    def run(sources, references, **options):
        """sources={文件名:字幕}, references={文件名:参考字幕/异常/None(无轨)}。"""
        state.media = {name: str(tmp_path / f"{name}.mkv") for name in sources}

        def transcribe(media, tiger_data, cfg, model, progress, log):
            progress(1, "识别完成")
            subs = copy.deepcopy(sources[Path(media).stem])
            return subs, [[] for _ in subs]

        def probe(media):
            ref = references[Path(media).stem]
            if ref is None:
                return []
            return [{"index": 0, "language": "jpn", "title": "", "codec": "srt"}]

        def extract_ref(media, index):
            ref = references[Path(media).stem]
            if isinstance(ref, Exception):
                raise ref
            return copy.deepcopy(ref)

        monkeypatch.setattr(runner, "_transcribe_single", transcribe)
        monkeypatch.setattr(extract, "probe_subtitle_tracks", probe)
        monkeypatch.setattr(extract, "extract_subtitle", extract_ref)
        config = dict(media_files=list(state.media.values()), device="cpu", use_ref_sub=True,
                      ref_direct=True, generate_review=True, extend_end=0, gap_threshold=0,
                      min_gap=0, min_duration=0)
        config.update(options)
        runner.run_pipeline(runner.PipelineConfig(**config), state.logs.append,
                            lambda pct, msg: state.progress.append((pct, msg)))
        state.outputs = {name: parse_srt(str(tmp_path / f"{name}.srt")) for name in sources}
        return state

    state.run = run
    return state


@pytest.mark.parametrize("align_model", ["mms_fa", "qwen3-fa-0.6b"])
def test_complete_reference_does_not_load_aligner(pipeline_case, align_model):
    result = pipeline_case.run(
        {"good": [Sub(1, 1, 2, "甲"), Sub(2, 3, 4, "乙")]},
        {"good": [Sub(1, 0, 5, "参考原文")]}, align_model=align_model,
    )
    assert len(result.models) == 1 and result.models[0].startswith("whisper:")
    assert not result.aligned
    assert [(s.start, s.end, s.text) for s in result.outputs["good"]] == [(0, 5, "甲 乙")]
    assert result.reviews[0][1] == 0
    assert any("跳过强制对齐" in msg for msg in result.logs)


@pytest.mark.parametrize("reference", [None, [], RuntimeError("提取失败"),
                                      [Sub(1, 20, 21, "不匹配")],
                                      [Sub(1, 0, 3, "只匹配第一句")]])
@pytest.mark.parametrize("align_model,handle", [("mms_fa", "mms_fa"),
                                             ("qwen3-fa-0.6b", "qwen3_fa")])
def test_bad_reference_aligns_original_text_and_times(pipeline_case, reference, align_model, handle):
    source = [Sub(8, 1, 2, "甲"), Sub(9, 5, 6, "乙")]
    result = pipeline_case.run({"bad": source}, {"bad": reference}, align_model=align_model)
    assert result.aligned == [source]  # 部分映射时也必须回退原始时间轴
    assert result.models.count(handle) == 1 and handle in result.released
    assert [(s.start, s.text) for s in result.outputs["bad"]] == [(1.1, "甲"), (5.1, "乙")]
    assert result.reviews[0][1] == .5
    assert any("回退强制对齐" in msg for msg in result.logs)
    review_path = Path(result.media["bad"]).with_name("bad_review.srt")
    assert "【甲】" in review_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("external_kind", ["empty", "missing"])
def test_unusable_external_srt_falls_back(pipeline_case, tmp_path, external_kind):
    ref = tmp_path / "reference.srt"
    if external_kind == "empty":
        ref.write_text("", encoding="utf-8")
    result = pipeline_case.run({"bad": [Sub(1, 1, 2, "原文")]}, {}, ref_srt=str(ref))
    assert len(result.aligned) == 1
    assert result.outputs["bad"][0].text == "原文"
    assert any("回退强制对齐" in msg for msg in result.logs)


def test_valid_external_srt_uses_reference_axis(pipeline_case, tmp_path):
    ref = tmp_path / "reference.srt"
    write_srt([Sub(1, 0, 3, "参考")], str(ref))
    result = pipeline_case.run({"good": [Sub(1, 1, 2, "原文")]}, {}, ref_srt=str(ref))
    assert not result.aligned
    assert result.outputs["good"][0].start == 0


def test_mixed_batch_falls_back_per_file_with_monotonic_progress(pipeline_case):
    source = [Sub(1, 1, 2, "原文")]
    refs = [Sub(1, 0, 3, "参考")]
    result = pipeline_case.run(
        {"good1": source, "bad1": source, "good2": source, "bad2": source},
        {"good1": refs, "bad1": [], "good2": refs, "bad2": None},
    )
    assert len(result.aligned) == 2 and result.models.count("mms_fa") == 1
    assert [result.outputs[name][0].start for name in result.media] == [0, 1.1, 0, 1.1]
    assert [threshold for _, threshold in result.reviews] == [0, .5, 0, .5]
    values = [pct for pct, _ in result.progress]
    assert values == sorted(values) and values[-1] == 1
    assert all(0 <= p <= 1 for p in values)


def test_empty_recognition_never_borrows_reference_text_or_loads_aligner(pipeline_case):
    result = pipeline_case.run({"empty": []}, {"empty": [Sub(1, 0, 3, "外国語")]})
    assert result.outputs["empty"] == []
    assert not result.aligned and len(result.models) == 1


def test_partial_nondirect_mapping_keeps_unmatched_cues(pipeline_case):
    result = pipeline_case.run(
        {"partial": [Sub(1, 1, 2, "甲"), Sub(2, 5, 6, "乙")]},
        {"partial": [Sub(1, 0, 3, "参考")]}, ref_direct=False,
    )
    assert [(s.start, s.end, s.text) for s in result.aligned[0]] == [
        (0, 3, "甲"), (5, 6, "乙")]


def test_reference_cancellation_is_not_treated_as_fallback(pipeline_case):
    with pytest.raises(CancelledError):
        pipeline_case.run({"cancel": [Sub(1, 1, 2, "原文")]},
                          {"cancel": CancelledError("取消")})
    assert not pipeline_case.aligned
    assert not Path(pipeline_case.media["cancel"]).with_suffix(".srt").exists()


def test_alignment_failure_releases_model_and_leaves_existing_output(pipeline_case, monkeypatch, tmp_path):
    output = tmp_path / "bad.srt"
    output.write_text("existing result", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("alignment failed")

    monkeypatch.setattr(runner, "run_alignment", fail)
    with pytest.raises(RuntimeError, match="alignment failed"):
        pipeline_case.run({"bad": [Sub(1, 1, 2, "原文")]}, {"bad": None})
    assert "mms_fa" in pipeline_case.released
    assert output.read_text(encoding="utf-8") == "existing result"
