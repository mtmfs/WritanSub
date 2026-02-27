"""SRT 读写：parse_srt、write_srt、review 标记"""

import os
import re
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


def mark_low_align_in_review(base: str, low_indices: set):
    """在已有的 _review.srt / _review.ass 中给低分句子加【】标记。"""
    review_srt = f"{base}_review.srt"
    review_ass = f"{base}_review.ass"

    if os.path.isfile(review_srt):
        try:
            with open(review_srt, "r", encoding="utf-8") as f:
                content = f.read()
            blocks = re.split(r"\n\n+", content.strip())
            new_blocks = []
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 3:
                    try:
                        idx = int(lines[0].strip())
                    except ValueError:
                        new_blocks.append(block)
                        continue
                    if idx in low_indices:
                        text = "\n".join(lines[2:])
                        if not text.startswith("【"):
                            text = f"【{text}】"
                        lines = lines[:2] + [text]
                        block = "\n".join(lines)
                new_blocks.append(block)
            with open(review_srt, "w", encoding="utf-8") as f:
                f.write("\n\n".join(new_blocks) + "\n\n")
        except OSError:
            pass

    if os.path.isfile(review_ass):
        try:
            with open(review_ass, "r", encoding="utf-8") as f:
                lines = f.readlines()
            dialogue_idx = 0
            new_lines = []
            for line in lines:
                if line.startswith("Dialogue:"):
                    dialogue_idx += 1
                    if dialogue_idx in low_indices:
                        pos = line.rfind(",,")
                        if pos >= 0:
                            prefix = line[:pos + 2]
                            text = line[pos + 2:].rstrip("\n")
                            if not text.startswith("【"):
                                text = f"【{text}】"
                            line = f"{prefix}{text}\n"
                new_lines.append(line)
            with open(review_ass, "w", encoding="utf-8-sig") as f:
                f.writelines(new_lines)
        except OSError:
            pass
