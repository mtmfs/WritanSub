"""ref_align：参考映射携带 low_words/speaker（T06/T10 配套）。"""
from writansub.subtitle.ref_align import map_whisper_to_ref
from writansub.types import Sub


def _sub(i, start, end, text, speaker=0, low_words=None):
    return Sub(index=i, start=start, end=end, text=text, speaker=speaker,
               low_words=low_words or [])


REF = [_sub(1, 0.0, 5.0, "ref1"), _sub(2, 6.0, 10.0, "ref2"), _sub(3, 11.0, 15.0, "ref3")]


def test_join_drop_renumber_regression():
    whisper = [_sub(1, 0.5, 2.0, "甲"), _sub(2, 2.5, 4.5, "乙"), _sub(3, 12.0, 14.0, "丙")]
    out = map_whisper_to_ref(whisper, REF)
    # ref2 无归属被丢弃，其余重编号
    assert [(s.index, s.text) for s in out] == [(1, "甲 乙"), (2, "丙")]
    assert out[0].start == 0.0 and out[0].end == 5.0


def test_low_words_concat_in_order():
    whisper = [_sub(1, 0.5, 2.0, "甲", low_words=["か"]),
               _sub(2, 2.5, 4.5, "乙", low_words=["お", "つ"])]
    out = map_whisper_to_ref(whisper, REF)
    assert out[0].low_words == ["か", "お", "つ"]


def test_speaker_uniform_kept_mixed_zeroed():
    uniform = [_sub(1, 0.5, 2.0, "甲", speaker=2), _sub(2, 2.5, 4.5, "乙", speaker=2)]
    assert map_whisper_to_ref(uniform, REF)[0].speaker == 2
    mixed = [_sub(1, 0.5, 2.0, "甲", speaker=1), _sub(2, 2.5, 4.5, "乙", speaker=2)]
    assert map_whisper_to_ref(mixed, REF)[0].speaker == 0
