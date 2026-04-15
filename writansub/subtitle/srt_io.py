import dataclasses

from writansub.types import Sub, fmt_srt_time


def _subs_from_pysrt(items, lang: str | None) -> list[Sub]:
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


def parse_srt(path: str, lang: str = "ja") -> list[Sub]:
    import pysrt
    return _subs_from_pysrt(pysrt.open(path, encoding='utf-8'), lang)


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
