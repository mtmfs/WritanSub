"""SRT 读写：parse_srt、write_srt、populate_romaji、merge_bilingual"""

from typing import List

from writansub.core.types import Sub, fmt_srt_time


def parse_srt(path: str, lang: str = "ja") -> List[Sub]:
    """解析 SRT 文件"""
    import pysrt
    from writansub.core.alignment import text_to_romaji

    subs = pysrt.open(path, encoding='utf-8')
    result = []
    for s in subs:
        text = s.text.replace('\n', ' ').strip()
        result.append(Sub(
            index=s.index,
            start=s.start.ordinal / 1000.0,
            end=s.end.ordinal / 1000.0,
            text=text,
            romaji=text_to_romaji(text, lang),
        ))
    return result


def write_srt(subs: List[Sub], path: str):
    """写入 SRT 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for sub in subs:
            f.write(f"{sub.index}\n")
            f.write(f"{fmt_srt_time(sub.start)} --> {fmt_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


def populate_romaji(subs: List[Sub], lang: str) -> None:
    """为 Sub 列表填充 romaji 字段（原地修改）。"""
    from writansub.core.alignment import text_to_romaji

    for sub in subs:
        if not sub.romaji:
            sub.romaji = text_to_romaji(sub.text, lang)


def merge_bilingual(subs: List[Sub]) -> List[Sub]:
    """合并 text + translated 为双语字幕，返回新 Sub 列表。

    每条字幕的 text 变为 "原文\\n译文" 格式。
    """
    result = []
    for sub in subs:
        merged_text = sub.text
        if sub.translated:
            merged_text = f"{sub.text}\n{sub.translated}"
        result.append(Sub(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            text=merged_text,
            romaji=sub.romaji,
            score=sub.score,
            translated=sub.translated,
        ))
    return result
