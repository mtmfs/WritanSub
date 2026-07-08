import dataclasses
from typing import Any

from writansub.types import Sub, fmt_srt_time


def _subs_from_pysrt(items: Any, lang: str | None) -> list[Sub]:
    if lang:
        from writansub.align.core import text_to_romaji
    result = []
    for s in items:
        text = s.text.replace('\n', ' ').strip()
        result.append(Sub(
            index=s.index,
            start=s.start.ordinal / 1000.0,
            end=s.end.ordinal / 1000.0,
            text=text,
            romaji=text_to_romaji(text, lang) if lang else "",
        ))
    return result


def _candidate_encodings(path: str) -> list[str]:
    """按优先级返回候选编码：utf-8-sig（兼容 BOM/纯 utf-8）→ 探测结果 → 常见东亚编码。"""
    candidates = ["utf-8-sig"]
    try:
        from charset_normalizer import from_path
        best = from_path(path).best()
        if best and best.encoding and best.encoding not in candidates:
            candidates.append(best.encoding)
    except Exception:
        pass
    for enc in ("gbk", "shift_jis"):
        if enc not in candidates:
            candidates.append(enc)
    return candidates


def parse_srt(path: str, lang: str | None = None) -> list[Sub]:
    import pysrt

    last_err: Exception | None = None
    for enc in _candidate_encodings(path):
        try:
            return _subs_from_pysrt(pysrt.open(path, encoding=enc), lang)
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
    raise ValueError(f"无法解码字幕文件 {path}: 不是 utf-8/gbk/shift_jis 等已知编码") from last_err


def parse_srt_string(text: str, lang: str | None = None) -> list[Sub]:
    import pysrt
    return _subs_from_pysrt(pysrt.from_string(text), lang)


def write_srt(subs: list[Sub], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for sub in subs:
            f.write(f"{sub.index}\n")
            f.write(f"{fmt_srt_time(sub.start)} --> {fmt_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


def populate_romaji(subs: list[Sub], lang: str) -> None:
    from writansub.align.core import text_to_romaji

    for sub in subs:
        if not sub.romaji:
            sub.romaji = text_to_romaji(sub.text, lang)


def merge_bilingual(subs: list[Sub]) -> list[Sub]:
    """text + translated → "原文\\n译文"，返回新列表。"""
    result = []
    for sub in subs:
        merged_text = f"{sub.text}\n{sub.translated}" if sub.translated else sub.text
        result.append(dataclasses.replace(sub, text=merged_text))
    return result
