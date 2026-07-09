"""review：attach_low_words 携带 + generate_review_final 单点生成（T06）。"""
from writansub.subtitle.review import attach_low_words, generate_review_final
from writansub.types import Sub, WordInfo


def _sub(i, text, score=1.0, low_words=None):
    return Sub(index=i, start=float(i), end=float(i) + 1, text=text,
               score=score, low_words=low_words or [])


def test_attach_counts_and_threshold():
    subs = [_sub(1, "こんにちは世界")]
    words = [[WordInfo("こんにちは", 0.9), WordInfo("世界", 0.2)]]
    low, total = attach_low_words(subs, words, 0.5)
    assert (low, total) == (1, 2)
    assert subs[0].low_words == ["世界"]


def test_attach_skips_empty_token_but_counts_it():
    subs = [_sub(1, "テスト")]
    words = [[WordInfo("  ", 0.1), WordInfo("テスト", 0.1)]]
    low, total = attach_low_words(subs, words, 0.5)
    assert (low, total) == (2, 2)
    assert subs[0].low_words == ["テスト"]  # 空白词不入列表


def test_attach_empty_word_data_noop():
    subs = [_sub(1, "テスト")]
    low, total = attach_low_words(subs, [[]], 0.5)
    assert (low, total) == (0, 0)
    assert subs[0].low_words == []


def test_final_marks_word_srt_and_ass():
    subs = [_sub(1, "日経平均が上昇", low_words=["日経"])]
    srt, ass, n_words, n_lines = generate_review_final(subs, align_threshold=0.0)
    assert "【?日経】平均が上昇" in srt
    assert "{\\c&H0000FF&}日経{\\c}" in ass
    assert (n_words, n_lines) == (1, 0)


def test_final_repeated_word_marks_successive_occurrences():
    subs = [_sub(1, "はいはい", low_words=["はい", "はい"])]
    srt, _, _, _ = generate_review_final(subs, 0.0)
    assert "【?はい】【?はい】" in srt


def test_final_line_mark_includes_score_zero():
    # score=0（对齐失败）是最可疑的行，阈值>0 时必须标
    subs = [_sub(1, "失败行", score=0.0), _sub(2, "正常行", score=0.9)]
    srt, ass, _, n_lines = generate_review_final(subs, align_threshold=0.5)
    assert "【失败行】" in srt and "【失败行】" in ass
    assert "【正常行】" not in srt
    assert n_lines == 1


def test_final_threshold_zero_disables_line_marks():
    subs = [_sub(1, "任何行", score=0.0)]
    srt, _, _, n_lines = generate_review_final(subs, align_threshold=0.0)
    assert "【任何行】" not in srt
    assert n_lines == 0


def test_final_ass_escapes_newline():
    subs = [_sub(1, "- 甲\n- 乙")]
    _, ass, _, _ = generate_review_final(subs, 0.0)
    assert "- 甲\\N- 乙" in ass
    # Dialogue 行不得被裸换行撕开
    dialogue = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogue) == 1


def test_final_block_number_follows_sub_index():
    subs = [_sub(7, "第七条", low_words=["第七条"])]
    srt, _, _, _ = generate_review_final(subs, 0.0)
    assert srt.startswith("7\n")


def test_final_clean_subs_zero_counts():
    subs = [_sub(1, "干净", score=0.9)]
    _, _, n_words, n_lines = generate_review_final(subs, 0.5)
    assert (n_words, n_lines) == (0, 0)
