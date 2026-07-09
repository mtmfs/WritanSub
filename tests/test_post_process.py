"""post_process：遗留行为零回归 + 重叠双模式（T10）+ 短字幕合并守卫（T34）。"""
import pytest

from writansub.align.core import post_process
from writansub.types import Sub


def _sub(i, start, end, text="x", speaker=0, low_words=None, score=0.5):
    return Sub(index=i, start=start, end=end, text=text, speaker=speaker,
               low_words=low_words or [], score=score)


ZERO = dict(extend_end=0.0, extend_start=0.0, gap_threshold=0.0,
            min_gap=0.0, min_duration=0.0)


# ---- 遗留行为（无 speaker）回归 ----

def test_legacy_extend_and_gap():
    subs = [_sub(1, 1.0, 2.0), _sub(2, 2.2, 3.0)]
    out = post_process(subs, extend_end=0.3, extend_start=0.0,
                       gap_threshold=0.5, min_gap=0.3, min_duration=0.0)
    # 原始间距 0.2 < 0.5 → 前轴贴到后轴开头
    assert out[0].end == out[1].start
    assert out[1].end == 3.3


def test_legacy_min_gap_reserved():
    subs = [_sub(1, 1.0, 2.0), _sub(2, 3.0, 4.0)]
    out = post_process(subs, extend_end=1.5, extend_start=0.0,
                       gap_threshold=0.5, min_gap=0.3, min_duration=0.0)
    # 原始间距 1.0 >= 0.5 → 延伸后保留 min_gap
    assert out[0].end == pytest.approx(3.0 - 0.3)


def test_legacy_short_merge_and_renumber():
    subs = [_sub(1, 1.0, 2.0, "前"), _sub(2, 2.05, 2.15, "短"), _sub(3, 3.0, 4.0, "后")]
    out = post_process(subs, **{**ZERO, "min_duration": 0.3, "gap_threshold": 0.5})
    assert [s.text for s in out] == ["前短", "后"]
    assert [s.index for s in out] == [1, 2]


def test_input_not_mutated():
    subs = [_sub(1, 1.0, 2.0, "a", low_words=["a"]),
            _sub(2, 2.05, 2.15, "b", low_words=["b"])]
    post_process(subs, **{**ZERO, "min_duration": 0.3, "gap_threshold": 0.5})
    # 守卫浅拷贝坑：调用方的 low_words 列表不得被污染
    assert subs[0].low_words == ["a"]
    assert subs[0].text == "a" and subs[0].end == 2.0


# ---- T34：短字幕只合并时间相邻的 ----

def test_isolated_short_cue_survives():
    subs = [_sub(1, 1.0, 2.0, "前"), _sub(2, 7.0, 7.1, "えっ")]
    out = post_process(subs, **{**ZERO, "min_duration": 0.3, "gap_threshold": 0.5})
    # 隔 5 秒的孤立微 cue 不再被拽进前一句
    assert [s.text for s in out] == ["前", "えっ"]


def test_short_merge_carries_low_words():
    subs = [_sub(1, 1.0, 2.0, "前", low_words=["前"]),
            _sub(2, 2.05, 2.15, "短", low_words=["短"])]
    out = post_process(subs, **{**ZERO, "min_duration": 0.3, "gap_threshold": 0.5})
    assert out[0].low_words == ["前", "短"]


# ---- T10 merge 模式 ----

def test_merge_two_speakers():
    subs = [_sub(1, 36.0, 42.0, "A的台词", speaker=1, low_words=["A"], score=0.9),
            _sub(2, 40.0, 46.0, "B的台词", speaker=2, score=0.3)]
    out = post_process(subs, **ZERO, overlap_mode="merge")
    assert len(out) == 1
    m = out[0]
    assert m.text == "- A的台词\n- B的台词"
    assert m.start == 36.0 and m.end == 46.0
    assert m.speaker == 0
    assert m.low_words == ["A"]
    assert m.score == 0.3  # 取 min，保守标注


def test_merge_transitive_group_of_three():
    subs = [_sub(1, 10.0, 13.0, "甲一", speaker=1),
            _sub(2, 12.0, 15.0, "乙", speaker=2),
            _sub(3, 14.5, 16.0, "甲二", speaker=1)]
    out = post_process(subs, **ZERO, overlap_mode="merge")
    assert len(out) == 1
    # 同说话人顺序直拼，不交叉
    assert out[0].text == "- 甲一甲二\n- 乙"


def test_merge_single_speaker_region_stays_plain():
    subs = [_sub(1, 1.0, 2.0, "全轨", speaker=0), _sub(2, 5.0, 6.0, "独白", speaker=1)]
    out = post_process(subs, **ZERO, overlap_mode="merge")
    assert [s.text for s in out] == ["全轨", "独白"]  # 无跨说话人重叠，无 "- " 前缀


def test_merge_identical_texts_deduped():
    """分离串扰：两轨听写出同一句时退化为单行平文本，不出现重复双行。"""
    subs = [_sub(1, 8.0, 11.0, "同一句", speaker=1),
            _sub(2, 8.5, 11.0, "同一句", speaker=2)]
    out = post_process(subs, **ZERO, overlap_mode="merge")
    assert len(out) == 1
    assert out[0].text == "同一句"


def test_merge_empty_side_degrades_to_plain():
    subs = [_sub(1, 1.0, 3.0, "  ", speaker=1), _sub(2, 2.0, 4.0, "有词", speaker=2)]
    out = post_process(subs, **ZERO, overlap_mode="merge")
    assert len(out) == 1
    assert out[0].text == "有词"


# ---- T10 keep 模式 ----

def test_keep_preserves_cross_speaker_overlap():
    subs = [_sub(1, 36.0, 42.0, "A", speaker=1), _sub(2, 40.0, 46.0, "B", speaker=2)]
    out = post_process(subs, extend_end=0.3, extend_start=0.0,
                       gap_threshold=0.5, min_gap=0.3, min_duration=0.0,
                       overlap_mode="keep")
    assert len(out) == 2
    # 前条不得被压平到后条开头（真实重叠存活）
    assert out[0].end == pytest.approx(42.3)
    assert out[1].start == 40.0


def test_keep_min_merge_never_crosses_speakers():
    subs = [_sub(1, 1.0, 2.0, "A", speaker=1), _sub(2, 2.05, 2.15, "B短", speaker=2)]
    out = post_process(subs, **{**ZERO, "min_duration": 0.3, "gap_threshold": 0.5},
                       overlap_mode="keep")
    assert [s.text for s in out] == ["A", "B短"]
