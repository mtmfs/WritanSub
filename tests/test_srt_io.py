"""T18：parse_srt 编码回退链。

覆盖：探测库在场时四种编码全对；探测库缺失时 Shift-JIS 不得被 GBK
静默解成乱码（回退顺序 shift_jis 先于 gbk）；全败时报含文件名的明确错误；
默认 lang=None 不算罗马音（T27 不回归）。
"""
import pytest

from writansub.subtitle.srt_io import parse_srt, write_srt
from writansub.types import Sub

JA_LINE1 = "日経平均株価は3万円を回復しました。"
JA_LINE2 = "米連邦準備制度理事会が利上げを発表。"
ZH_LINE1 = "沪深三百指数今天大幅上涨。"

SRT_JA = (
    f"1\n00:00:01,000 --> 00:00:03,000\n{JA_LINE1}\n\n"
    f"2\n00:00:04,000 --> 00:00:06,000\n{JA_LINE2}\n"
)
SRT_ZH = f"1\n00:00:01,000 --> 00:00:03,000\n{ZH_LINE1}\n"


def _write(tmp_path, name, text, enc):
    p = tmp_path / name
    p.write_bytes(text.encode(enc))
    return str(p)


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "shift_jis"])
def test_japanese_all_encodings(tmp_path, enc):
    path = _write(tmp_path, f"ja_{enc}.srt", SRT_JA, enc)
    subs = parse_srt(path)
    assert [s.text for s in subs] == [JA_LINE1, JA_LINE2]
    assert subs[0].start == 1.0 and subs[0].end == 3.0


def test_gbk_chinese(tmp_path):
    path = _write(tmp_path, "zh.srt", SRT_ZH, "gbk")
    assert parse_srt(path)[0].text == ZH_LINE1


def test_shift_jis_without_detector(tmp_path, no_charset_normalizer):
    """探测库缺失时 Shift-JIS 必须仍正确解码，而非被 GBK 抢先解成乱码。"""
    path = _write(tmp_path, "ja_sjis.srt", SRT_JA, "shift_jis")
    assert parse_srt(path)[0].text == JA_LINE1


def test_gbk_without_detector(tmp_path, no_charset_normalizer):
    """本样本的 GBK 字节流会让 shift_jis 解码失败并落到 gbk，仍正确。"""
    path = _write(tmp_path, "zh_gbk.srt", SRT_ZH, "gbk")
    assert parse_srt(path)[0].text == ZH_LINE1


def test_undecodable_raises_with_filename(tmp_path, no_charset_normalizer):
    p = tmp_path / "bad.srt"
    # \x81\x00 在 utf-8 / shift_jis / gbk 下均非法
    p.write_bytes(b"1\n00:00:01,000 --> 00:00:03,000\n\x81\x00\n")
    with pytest.raises(ValueError, match="bad.srt"):
        parse_srt(str(p))


def test_default_no_romaji(tmp_path):
    """T27：默认不传 lang 时不得计算罗马音（翻译/ref 路径不加载 cutlet）。"""
    path = _write(tmp_path, "ja.srt", SRT_JA, "utf-8")
    assert all(s.romaji == "" for s in parse_srt(path))


def test_write_read_roundtrip(tmp_path):
    subs = [Sub(index=1, start=1.0, end=3.5, text=JA_LINE1)]
    out = str(tmp_path / "out.srt")
    write_srt(subs, out)
    back = parse_srt(out)
    assert back[0].text == JA_LINE1
    assert back[0].start == 1.0 and back[0].end == 3.5


# ---- T04 命名助手 ----

def test_stage_path():
    from writansub.subtitle.srt_io import stage_path
    assert stage_path("D:/v/ep01", "original", "whisper-large-v3") == \
        "D:/v/ep01_original_whisper-large-v3.srt"
    assert stage_path("ep01", "aligned", "mms_fa") == "ep01_aligned_mms_fa.srt"
    assert stage_path("ep01", "aligned", "qwen3-fa-0.6b") == "ep01_aligned_qwen3-fa-0.6b.srt"


def test_lang_code_mapping():
    from writansub.subtitle.srt_io import lang_code
    assert lang_code("简体中文") == "chs"
    assert lang_code("繁體中文") == "cht"
    assert lang_code("繁体中文") == "cht"
    assert lang_code("English") == "en"
    assert lang_code("日本語") == "ja"
    assert lang_code(" 简体中文 ") == "chs"  # 容忍首尾空白


def test_lang_code_fallback_sanitized():
    from writansub.subtitle.srt_io import lang_code
    assert lang_code("français") == "français"          # 未命中原样保留
    assert lang_code("Pirate Speak") == "Pirate-Speak"  # 空白转连字符
    assert lang_code('a/b\\c:d*e?f"g<h>i|j') == "a-b-c-d-e-f-g-h-i-j"
    assert lang_code("   ") == "translated"             # 全空回退兜底
