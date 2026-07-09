"""transcribe()：T40 回退后与基线一致的主路径（外部传入模型）。

假模型走真实的分段→Sub 转换、词级数据收集、进度回调、取消检查。
"""
import types as _types

import pytest

from writansub.transcribe.core import transcribe
from writansub.bridge import CancelledError
from writansub.types import WordInfo


def _seg(start, end, text, words=None):
    return _types.SimpleNamespace(start=start, end=end, text=text, words=words)


def _word(word, probability):
    return _types.SimpleNamespace(word=word, probability=probability)


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
        _seg(0.0, 2.0, "  こんにちは  ", [_word("こんにちは", 0.9)]),
        _seg(2.5, 4.0, "世界", [_word("世", 0.8), _word("界", 0.3)]),
        _seg(4.5, 6.0, "テスト", None),  # words 可能为 None
    ])
    subs, word_data = transcribe("dummy.wav", model=model)
    assert [s.index for s in subs] == [1, 2, 3]
    assert subs[0].text == "こんにちは"  # 前后空白剥掉
    assert word_data[1] == [WordInfo("世", 0.8), WordInfo("界", 0.3)]
    assert word_data[2] == []


def test_kwargs_passthrough(registry):
    model = FakeModel([_seg(0.0, 1.0, "a")])
    transcribe("dummy.wav", lang="ja", model=model,
               condition_on_previous_text=False, vad_filter=True,
               initial_prompt="")
    assert model.kwargs["language"] == "ja"
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["initial_prompt"] is None  # 空串归一为 None
    assert model.kwargs["word_timestamps"] is True


def test_progress_reaches_done(registry):
    updates = []
    model = FakeModel([_seg(0.0, 5.0, "a"), _seg(5.0, 10.0, "b")])
    transcribe("dummy.wav", model=model,
               progress_callback=lambda p, m: updates.append((p, m)))
    assert updates[-1] == (1.0, "识别完成")
    assert all(0.0 <= p <= 1.0 for p, _ in updates)


def test_cancel_raises(registry):
    registry.cancelled = True
    model = FakeModel([_seg(0.0, 1.0, "a")])
    with pytest.raises(CancelledError):
        transcribe("dummy.wav", model=model)
