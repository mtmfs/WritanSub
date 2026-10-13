import types as _types

import pytest

from writansub.bridge import CancelledError
from writansub.transcribe.base import to_subs
from writansub.transcribe.whisper import WhisperOptions, WhisperRecognizer
from writansub.types import Transcript, WordInfo


def _seg(start, end, text, words=None):
    return _types.SimpleNamespace(start=start, end=end, text=text, words=words)


def _word(word, probability, start=None, end=None):
    return _types.SimpleNamespace(word=word, probability=probability, start=start, end=end)


class FakeModel:
    def __init__(self, segs, duration=10.0):
        self._segs = segs
        self._duration = duration
        self.kwargs = None

    def transcribe(self, file_path, **kwargs):
        self.kwargs = kwargs
        return iter(self._segs), _types.SimpleNamespace(duration=self._duration)


def test_segments_to_subs(registry):
    model = FakeModel([
        _seg(0.0, 2.0, "  こんにちは  ", [_word("こんにちは", 0.9, 0.0, 2.0)]),
        _seg(2.5, 4.0, "世界", [_word("世", 0.8), _word("界", 0.3)]),
        _seg(4.5, 6.0, "テスト", None),
    ])
    t = WhisperRecognizer().transcribe(model, "dummy.wav", "ja")
    assert isinstance(t, Transcript) and t.backend == "whisper" and t.duration == 10.0
    subs, word_data = to_subs(t)
    assert [s.index for s in subs] == [1, 2, 3]
    assert subs[0].text == "こんにちは"
    assert word_data[0] == [WordInfo("こんにちは", 0.9, 0.0, 2.0)]
    assert word_data[1] == [WordInfo("世", 0.8), WordInfo("界", 0.3)]
    assert word_data[2] == []


def test_options_passthrough(registry):
    model = FakeModel([_seg(0.0, 1.0, "a")])
    rec = WhisperRecognizer(WhisperOptions(
        condition_on_previous_text=False, vad_filter=True, initial_prompt=""))
    rec.transcribe(model, "dummy.wav", "ja")
    assert model.kwargs["language"] == "ja"
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["initial_prompt"] is None
    assert model.kwargs["word_timestamps"] is True


def test_short_clip_overrides_context_options(registry):
    model = FakeModel([_seg(0.0, 1.0, "a")])
    rec = WhisperRecognizer(WhisperOptions(
        condition_on_previous_text=True, vad_filter=True, initial_prompt="人名"))
    rec.transcribe(model, "dummy.wav", "ja", short_clip=True)
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["initial_prompt"] == "人名"
    assert model.kwargs["language"] == "ja"


def test_progress_reaches_done(registry):
    updates = []
    model = FakeModel([_seg(0.0, 5.0, "a"), _seg(5.0, 10.0, "b")])
    WhisperRecognizer().transcribe(model, "dummy.wav", "ja", lambda p, m: updates.append((p, m)))
    assert updates[0] == (0.0, "开始识别...")
    assert updates[-1] == (1.0, "识别完成")
    assert all(0.0 <= p <= 1.0 for p, _ in updates)


def test_cancel_raises(registry):
    registry.cancelled = True
    model = FakeModel([_seg(0.0, 1.0, "a")])
    with pytest.raises(CancelledError):
        WhisperRecognizer().transcribe(model, "dummy.wav", "ja")


def test_name_tag_cache_key():
    rec = WhisperRecognizer(WhisperOptions(model="medium", compute_type="float16"))
    assert rec.name == "whisper"
    assert rec.tag() == "whisper-medium"
    assert rec.cache_key() == "whisper:medium:float16"
    assert WhisperRecognizer().cache_key() == "whisper:large-v3:int8"
