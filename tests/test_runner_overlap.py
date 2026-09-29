"""_whisper_with_overlap：vad_filter 补传、speaker 标签、词级数据并行携带（T10）。"""
import torch
import pytest
import os
import wave
from types import SimpleNamespace

import writansub.pipeline.runner as runner
from writansub.pipeline.runner import PipelineConfig, _whisper_with_overlap
from writansub.preprocess.core import TimeSpan
from writansub.types import Sub, WordInfo
from writansub.bridge import CancelledError


class _FakeTranscribe:
    """记录每次调用的 kwargs；全轨调用返回预置结果，区域调用返回单条短句。"""

    def __init__(self, full_subs, full_words):
        self.calls: list[dict] = []
        self._full = (full_subs, full_words)

    def __call__(self, file_path, **kwargs):
        self.calls.append({"file": file_path, **kwargs})
        if len(self.calls) == 1:  # 首次 = 全轨
            return self._full
        n = len(self.calls)
        sub = Sub(index=1, start=0.0, end=1.0, text=f"区域{n}")
        return [sub], [[WordInfo(f"区域{n}", 0.2)]]


@pytest.fixture
def fake_env(monkeypatch, registry):
    full_subs = [
        Sub(index=1, start=0.0, end=3.0, text="开场"),
        Sub(index=2, start=10.0, end=12.0, text="与区域相交将被剔除"),
        Sub(index=3, start=20.0, end=22.0, text="结尾"),
    ]
    full_words = [[WordInfo("开场", 0.9)], [WordInfo("剔除", 0.9)], [WordInfo("结尾", 0.9)]]
    fake = _FakeTranscribe(full_subs, full_words)
    monkeypatch.setattr(runner, "transcribe", fake)
    return fake


def _run(fake, vad=True):
    cfg = PipelineConfig(lang="ja", device="cpu", vad_filter=vad)
    spk = torch.zeros(1, 16000 * 30)  # 30s 假轨
    regions = [TimeSpan(start=10.0, end=12.0)]
    return _whisper_with_overlap(
        "dummy.wav", regions, (spk, spk), 16000, cfg,
        whisper_model=object(), progress_callback=lambda p, m: None, log=lambda m: None,
    )


def test_full_track_receives_vad_filter(fake_env):
    _run(fake_env, vad=True)
    assert fake_env.calls[0]["vad_filter"] is True
    # 区域短块故意不开 VAD
    assert all("vad_filter" not in c for c in fake_env.calls[1:])


def test_speaker_tags_and_intersecting_cue_dropped(fake_env):
    subs, _ = _run(fake_env)
    texts = [s.text for s in subs]
    assert "与区域相交将被剔除" not in texts          # 相交全轨 cue 被剔除
    assert {s.speaker for s in subs if s.text.startswith("区域")} == {1, 2}
    assert all(s.speaker == 0 for s in subs if not s.text.startswith("区域"))
    assert [s.index for s in subs] == list(range(1, len(subs) + 1))  # 重编号连续


def test_word_data_parallel_and_nonempty(fake_env):
    subs, word_data = _run(fake_env)
    assert len(subs) == len(word_data)
    # 区域 cue 的词级数据不再被丢弃（separate 模式 review 复活的结构保证）
    for s, w in zip(subs, word_data):
        if s.text.startswith("区域"):
            assert w and w[0].probability == 0.2
        else:
            assert w  # 全轨 cue 的词级数据也成对保留


def test_region_times_rebased(fake_env):
    subs, _ = _run(fake_env)
    region_subs = [s for s in subs if s.text.startswith("区域")]
    assert all(s.start >= 10.0 for s in region_subs)  # 时间已回加区域偏移


def _answer(text, start=0, end=1):
    return [Sub(1, start, end, text)], [[WordInfo(text, .2)]]


@pytest.fixture
def overlap_case(monkeypatch, registry):
    state = SimpleNamespace(calls=[], logs=[])

    def run(full, regions, replies, samples=16000 * 20, sr=16000):
        answers = iter(replies)
        full_words = [[WordInfo(s.text, .9)] for s in full]

        def transcribe(path, **kwargs):
            if path == "full.wav":
                return full, full_words
            with wave.open(path, "rb") as wav:
                state.calls.append((path, wav.getnframes(), wav.getframerate()))
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            if callable(answer):
                return answer()
            return answer

        monkeypatch.setattr(runner, "transcribe", transcribe)
        lengths = samples if isinstance(samples, tuple) else (samples, samples)
        tracks = tuple(torch.zeros(1, n) for n in lengths)
        return _whisper_with_overlap(
            "full.wav", regions, tracks, sr, PipelineConfig(device="cpu"),
            object(), lambda *args: None, state.logs.append,
        )

    state.run = run
    yield state
    assert all(not os.path.exists(path) for path, _, _ in state.calls)


def test_partial_overlap_retranscribes_whole_sentence_once(overlap_case):
    full = [Sub(8, 0, 10, "整句"), Sub(9, 12, 14, "后文")]
    subs, words = overlap_case.run(
        full, [TimeSpan(8, 9), TimeSpan(4, 5)],
        [_answer("完整甲", 0, 10), _answer("插话乙", 4, 5)],
    )
    assert [n for _, n, _ in overlap_case.calls] == [160000, 160000]
    assert [(s.index, s.text, s.speaker) for s in subs] == [
        (1, "完整甲", 1), (2, "插话乙", 2), (3, "后文", 0)]
    assert [w[0].word for w in words] == [s.text for s in subs]
    assert [(s.index, s.start, s.end) for s in full] == [(8, 0, 10), (9, 12, 14)]


def test_expansion_merges_transitive_intersections_but_not_touching_endpoints():
    full = [Sub(1, 0, 3, "甲"), Sub(2, 2, 6, "乙"), Sub(3, 5, 8, "丙")]
    assert runner._overlap_windows(full, [TimeSpan(1, 2), TimeSpan(8, 9)]) == [
        (0, 8, {0, 1, 2}), (8, 9, set())]


@pytest.mark.parametrize("answers", [([([], []), ([], [])]),
                                    ([_answer("  "), _answer("")])])
def test_empty_tracks_keep_original_and_words(overlap_case, answers):
    original = Sub(8, 0, 10, "原句", low_words=["原"])
    subs, words = overlap_case.run([original], [TimeSpan(4, 5)], answers)
    assert subs[0].text == "原句" and subs[0].low_words == ["原"]
    assert words == [[WordInfo("原句", .9)]]
    assert subs[0] is not original and original.index == 8
    assert any("保留原字幕" in msg for msg in overlap_case.logs)


@pytest.mark.parametrize("active_track", [0, 1])
def test_one_nonempty_track_is_valid_replacement(overlap_case, active_track):
    answers = [([], []), ([], [])]
    answers[active_track] = _answer("替换", 0, 10)
    subs, words = overlap_case.run([Sub(1, 0, 10, "原句")], [TimeSpan(4, 5)], answers)
    assert len(overlap_case.calls) == 2
    assert [(s.text, s.speaker) for s in subs] == [("替换", active_track + 1)]
    assert words == [[WordInfo("替换", .2)]]


@pytest.mark.parametrize("samples", [(160000, 159999), (159999, 160000)])
def test_incomplete_audio_keeps_whole_original(overlap_case, samples):
    subs, _ = overlap_case.run([Sub(1, 0, 10, "原句")], [TimeSpan(4, 5)], [], samples=samples)
    assert subs[0].text == "原句"
    assert overlap_case.calls == []
    assert any("不足" in msg for msg in overlap_case.logs)


def test_short_window_keeps_original(overlap_case):
    subs, words = overlap_case.run([Sub(1, 1, 1.05, "短句")], [TimeSpan(1.01, 1.04)], [])
    assert subs[0].text == "短句" and words[0][0].word == "短句"
    assert overlap_case.calls == []
    assert any("过短" in msg for msg in overlap_case.logs)


def test_rebase_uses_actual_sample_start(overlap_case):
    start = 1.00004  # floor 到第 16000 个样本，实际切片从 1.0s 开始
    subs, _ = overlap_case.run(
        [Sub(1, start, 2, "原句")], [TimeSpan(1.2, 1.3)],
        [_answer("甲"), ([], [])],
    )
    assert subs[0].start == 1.0 and subs[0].end == 2.0
    assert overlap_case.calls[0][1] == 16000


def test_region_without_original_can_add_speech(overlap_case):
    subs, _ = overlap_case.run([], [TimeSpan(2, 3)], [_answer("新句"), ([], [])])
    assert [(s.start, s.end, s.text) for s in subs] == [(2, 3, "新句")]


@pytest.mark.parametrize("exc", [CancelledError("cancel"), RuntimeError("inference failed")])
def test_second_track_failure_propagates_without_mutating_original(overlap_case, exc):
    original = Sub(8, 0, 10, "原句")
    with pytest.raises(type(exc)):
        overlap_case.run([original], [TimeSpan(4, 5)], [_answer("甲", 0, 10), exc])
    assert len(overlap_case.calls) == 2
    assert (original.index, original.text, original.start, original.end) == (8, "原句", 0, 10)


def test_cancel_after_last_track_does_not_return_partial_result(overlap_case, registry):
    def cancel_after_reply():
        registry.cancelled = True
        return _answer("乙", 0, 10)
    with pytest.raises(CancelledError):
        overlap_case.run([Sub(1, 0, 10, "原句")], [TimeSpan(4, 5)],
                         [_answer("甲", 0, 10), cancel_after_reply])
