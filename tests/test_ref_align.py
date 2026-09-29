"""ref_align：参考映射携带 low_words/speaker（T06/T10 配套）。"""
from writansub.subtitle.ref_align import map_whisper_to_ref
from writansub.types import Sub
from writansub.subtitle.ref_align import _map_whisper_to_ref


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


def test_empty_whisper_never_uses_reference_text():
    assert map_whisper_to_ref([], REF) == []
    assert _map_whisper_to_ref([], REF)[1] == 0


def test_empty_reference_preserves_whisper_without_aliasing():
    whisper = [_sub(8, 3, 4, "原文", low_words=["原"])]
    out, matched = _map_whisper_to_ref(whisper, [])
    assert matched == 0
    assert [(s.index, s.start, s.end, s.text) for s in out] == [(1, 3, 4, "原文")]
    out[0].low_words.append("新")
    assert whisper[0].index == 8
    assert whisper[0].low_words == ["原"]


def test_partial_mapping_preserves_unmatched_and_reports_source_count():
    whisper = [_sub(3, 20, 21, "丙", speaker=2, low_words=["丙"]),
               _sub(1, 1, 2, "甲"), _sub(2, 3, 4, "乙")]
    out, matched = _map_whisper_to_ref(whisper, REF)
    assert matched == 2  # 两条原字幕合成一条，不得用输出条数判断匹配是否完整
    assert [(s.index, s.start, s.end, s.text) for s in out] == [
        (1, 0, 5, "甲 乙"), (2, 20, 21, "丙")]
    assert out[1].speaker == 2 and out[1].low_words == ["丙"]
    assert [(s.index, s.text) for s in whisper] == [(3, "丙"), (1, "甲"), (2, "乙")]


def test_no_overlap_preserves_all_source_text_in_time_order():
    whisper = [_sub(9, 30, 31, "后"), _sub(8, 20, 21, "前")]
    out, matched = _map_whisper_to_ref(whisper, REF)
    assert matched == 0
    assert [(s.index, s.start, s.text) for s in out] == [(1, 20, "前"), (2, 30, "后")]


def test_reference_merge_keeps_lowest_score():
    whisper = [_sub(1, 1, 2, "甲"), _sub(2, 3, 4, "乙")]
    whisper[0].score = 0.9
    whisper[1].score = 0.1
    out, matched = _map_whisper_to_ref(whisper, REF)
    assert matched == 2
    assert out[0].score == 0.1
