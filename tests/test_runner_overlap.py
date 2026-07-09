"""_whisper_with_overlap：vad_filter 补传、speaker 标签、词级数据并行携带（T10）。"""
import torch
import pytest

import writansub.pipeline.runner as runner
from writansub.pipeline.runner import PipelineConfig, _whisper_with_overlap
from writansub.preprocess.core import TimeSpan
from writansub.types import Sub, WordInfo


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
