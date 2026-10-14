import types as _types

import pytest
import torch

from writansub.bridge import CancelledError
from writansub.transcribe.base import to_subs
from writansub.transcribe.qwen import QwenOptions, QwenRecognizer
from writansub.transcribe.segment import attach_punctuation, split_tokens
from writansub.types import WordInfo


def _tok(text, start, end):
    return (text, start, end)


def test_attach_punctuation_restores_marks_between_tokens():
    text = "こんにちは、世界。Hello world."
    tokens = [_tok("こんにちは", 0, 1), _tok("世界", 1, 2), _tok("Hello", 2, 3), _tok("world", 3, 4)]
    out = attach_punctuation(text, tokens)
    assert [t for t, _, _ in out] == ["こんにちは、", "世界。", "Hello", "world."]
    assert [(s, e) for _, s, e in out] == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_attach_punctuation_keeps_tokens_when_text_mismatches():
    tokens = [_tok("甲", 0, 1), _tok("乙", 1, 2)]
    assert attach_punctuation("甲丙。", tokens) == tokens


def test_split_on_sentence_end_and_gap():
    tokens = [_tok("今日は。", 0.0, 0.5), _tok("晴れ", 0.6, 1.0), _tok("です", 1.0, 1.4),
              _tok("明日", 3.0, 3.4), _tok("雨", 3.4, 3.8)]
    segs = split_tokens(tokens, max_gap=0.8, max_chars=40)
    assert [(s.text, s.start, s.end) for s in segs] == [
        ("今日は。", 0.0, 0.5), ("晴れです", 0.6, 1.4), ("明日雨", 3.0, 3.8)]
    assert segs[1].words == [WordInfo("晴れ", None, 0.6, 1.0), WordInfo("です", None, 1.0, 1.4)]
    assert all(s.time_kind == "estimate" for s in segs)


def test_split_long_group_cuts_at_last_clause_mark():
    tokens = [_tok("ああ、", 0, 1), _tok("いい", 1, 2), _tok("うう、", 2, 3), _tok("ええ", 3, 4), _tok("おお", 4, 5)]
    segs = split_tokens(tokens, max_gap=10, max_chars=8)
    assert [s.text for s in segs] == ["ああ、いいうう、", "ええおお"]


def test_split_long_prefers_pause_over_clause_mark():
    tokens = [_tok("ああ、", 0.0, 1.0), _tok("いい", 1.0, 2.0), _tok("うう", 2.6, 3.0), _tok("ええ", 3.0, 4.0), _tok("おお", 4.0, 5.0)]
    segs = split_tokens(tokens, max_gap=10, max_chars=8)
    assert [s.text for s in segs] == ["ああ、いい", "ううええおお"]


def test_split_long_never_cuts_before_particle():
    tokens = [_tok("今日", 0, 1), _tok("は", 1, 2), _tok("いい", 2, 3), _tok("天気", 3, 4), _tok("です", 4, 5), _tok("ね", 5, 6)]
    segs = split_tokens(tokens, max_gap=10, max_chars=5)
    assert [s.text for s in segs] == ["今日はいい", "天気ですね"]


def test_split_long_balances_halves_without_other_signals():
    tokens = [_tok("甲乙", i, i + 1) for i in range(6)]
    segs = split_tokens(tokens, max_gap=10, max_chars=6)
    assert [s.text for s in segs] == ["甲乙甲乙甲乙", "甲乙甲乙甲乙"]


def test_split_long_recurses_until_under_limit():
    tokens = [_tok("甲乙丙", i, i + 1) for i in range(8)]
    segs = split_tokens(tokens, max_gap=10, max_chars=7)
    assert all(len(s.text) <= 6 for s in segs) and "".join(s.text for s in segs) == "甲乙丙" * 8


def test_split_joins_ascii_words_with_spaces():
    tokens = [_tok("Hello", 0, 1), _tok("world.", 1, 2), _tok("日本", 2, 3), _tok("go", 3, 4)]
    segs = split_tokens(tokens, max_gap=10, max_chars=40)
    assert [s.text for s in segs] == ["Hello world.日本go"]


class FakeQwen:
    def __init__(self, text, items):
        self.text = text
        self.items = items
        self.kwargs = None
        self.calls = 0

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        self.calls += 1
        self.kwargs = {"audio": audio, "context": context, "language": language,
                       "return_time_stamps": return_time_stamps}
        stamps = [_types.SimpleNamespace(text=t, start_time=s, end_time=e) for t, s, e in self.items] or None
        return [_types.SimpleNamespace(language="Japanese", text=self.text, time_stamps=stamps)]


@pytest.fixture
def silent_decode(registry, monkeypatch):
    monkeypatch.setattr(registry, "decode_audio", lambda path, sample_rate=44100: (torch.zeros(1, sample_rate * 3), sample_rate))
    return registry


def test_qwen_transcribe_builds_segments(silent_decode):
    model = FakeQwen("今日は。晴れです", [("今日は", 0.0, 0.5), ("晴れ", 0.6, 1.0), ("です", 1.0, 1.4)])
    updates = []
    t = QwenRecognizer().transcribe(model, "dummy.wav", "ja", lambda p, m: updates.append((p, m)))
    assert t.backend == "qwen3" and t.duration == 3.0
    assert all(s.time_kind == "aligned" for s in t.segments)
    assert model.kwargs["language"] == "Japanese" and model.kwargs["return_time_stamps"] is True
    assert model.kwargs["audio"][1] == 16000 and model.kwargs["audio"][0].shape == (48000,)
    subs, words = to_subs(t)
    assert [(s.index, s.text, s.start, s.end) for s in subs] == [(1, "今日は。", 0.0, 0.5), (2, "晴れです", 0.6, 1.4)]
    assert words[1] == [WordInfo("晴れ", None, 0.6, 1.0), WordInfo("です", None, 1.0, 1.4)]
    assert updates[0] == (0.0, "开始识别...") and updates[-1] == (1.0, "识别完成")


def test_qwen_chunks_long_audio_and_offsets_timestamps(registry, monkeypatch):
    wave = torch.ones(1, 16000 * 18)
    wave[0, 16000 * 6] = 0.0
    wave[0, 16000 * 12] = 0.0
    monkeypatch.setattr(registry, "decode_audio", lambda path, sample_rate=44100: (wave, sample_rate))
    model = FakeQwen("甲。", [("甲", 0.1, 0.4)])
    updates = []
    rec = QwenRecognizer(QwenOptions(chunk_seconds=6.0))
    t = rec.transcribe(model, "dummy.wav", "ja", lambda p, m: updates.append((p, m)))
    assert model.calls == 3
    assert [s.start for s in t.segments] == pytest.approx([0.1, 6.1, 12.1])
    assert [m for _, m in updates if m.startswith("识别中")] == ["识别中... 1/3", "识别中... 2/3", "识别中... 3/3"]


def test_qwen_clamps_overlong_word_durations(silent_decode):
    model = FakeQwen("甲。乙。", [("甲", 0.1, 20.0), ("乙", 20.5, 21.0)])
    t = QwenRecognizer().transcribe(model, "dummy.wav", "ja")
    assert [(s.start, s.end) for s in t.segments] == [(0.1, 3.1), (20.5, 21.0)]
    assert t.segments[0].words == [WordInfo("甲。", None, 0.1, 3.1)]
    loose = QwenRecognizer(QwenOptions(max_word_seconds=30.0)).transcribe(model, "dummy.wav", "ja")
    assert loose.segments[0].end == 20.0


def test_qwen_without_timestamps_falls_back_to_window(silent_decode):
    model = FakeQwen("全文", [])
    t = QwenRecognizer().transcribe(model, "dummy.wav", "ja")
    assert len(t.segments) == 1
    assert (t.segments[0].text, t.segments[0].start, t.segments[0].end, t.segments[0].time_kind) == ("全文", 0.0, 3.0, "window")


def test_qwen_empty_text_gives_empty_transcript(silent_decode):
    t = QwenRecognizer().transcribe(FakeQwen("", []), "dummy.wav", "ja")
    assert t.segments == []


def test_qwen_cancel_raises(silent_decode):
    silent_decode.cancelled = True
    with pytest.raises(CancelledError):
        QwenRecognizer().transcribe(FakeQwen("x", []), "dummy.wav", "ja")


def test_qwen_name_tag_cache_key():
    rec = QwenRecognizer()
    assert rec.name == "qwen3"
    assert rec.tag() == "qwen3-asr-1.7b"
    assert rec.cache_key() == "qwen3:Qwen/Qwen3-ASR-1.7B:Qwen/Qwen3-ForcedAligner-0.6B"
    assert QwenRecognizer(QwenOptions(model="Qwen/Qwen3-ASR-0.6B")).tag() == "qwen3-asr-0.6b"


def test_qwen_tag_from_local_path():
    assert QwenRecognizer(QwenOptions(model="G:\\models\\Qwen3-ASR-1.7B\\")).tag() == "qwen3-asr-1.7b"
    assert QwenRecognizer(QwenOptions(model="/srv/Qwen3-ASR-0.6B")).tag() == "qwen3-asr-0.6b"
    assert QwenRecognizer(QwenOptions(model=".")).tag() == "local"
