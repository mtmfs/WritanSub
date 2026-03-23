"""Review 模块：从转录/对齐数据生成 review 标记文件"""

import os
import re

from writansub.types import Sub, WordInfo, fmt_srt_time, fmt_ass_time

_ASS_REVIEW_HEADER = """\
[Script Info]
Title: AItrans Review
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_review(
    subs: list[Sub],
    word_data: list[list[WordInfo]],
    threshold: float,
) -> tuple[str, str, int, int]:
    """生成 review SRT + ASS 内容字符串。

    Args:
        subs: 字幕列表
        word_data: 每句的词列表（与 subs 一一对应）
        threshold: 词级置信阈值

    Returns:
        (srt_content, ass_content, low_count, total_words)
    """
    srt_lines = []
    ass_lines = [_ASS_REVIEW_HEADER]
    low_count = 0
    total_words = 0

    for sub, words in zip(subs, word_data):
        time_line = f"{fmt_srt_time(sub.start)} --> {fmt_srt_time(sub.end)}"

        if words:
            srt_parts: list[str] = []
            ass_parts: list[str] = []
            for w in words:
                total_words += 1
                if w.probability < threshold:
                    stripped = w.word.lstrip()
                    leading = w.word[: len(w.word) - len(stripped)]
                    srt_parts.append(f"{leading}【?{stripped}】")
                    ass_parts.append(f"{leading}{{\\c&H0000FF&}}{stripped}{{\\c}}")
                    low_count += 1
                else:
                    srt_parts.append(w.word)
                    ass_parts.append(w.word)
            text_review = "".join(srt_parts).strip()
            ass_review = "".join(ass_parts).strip()
        else:
            text_review = sub.text
            ass_review = sub.text

        srt_lines.append(f"{sub.index}\n{time_line}\n{text_review}\n")
        ass_lines.append(
            f"Dialogue: 0,{fmt_ass_time(sub.start)},{fmt_ass_time(sub.end)},Default,,0,0,0,,{ass_review}"
        )

    srt_content = "\n".join(srt_lines)
    ass_content = "\n".join(ass_lines) + "\n"
    return srt_content, ass_content, low_count, total_words


def write_review_files(base_path: str, srt_content: str, ass_content: str) -> None:
    """将 review 内容写到磁盘。

    Args:
        base_path: 不含扩展名的基础路径
        srt_content: review SRT 文本
        ass_content: review ASS 文本
    """
    with open(f"{base_path}_review.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)
    with open(f"{base_path}_review.ass", "w", encoding="utf-8-sig") as f:
        f.write(ass_content)


def mark_low_align_in_review(base: str, low_indices: set[int]) -> None:
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
                parts = block.split("\n")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0].strip())
                    except ValueError:
                        new_blocks.append(block)
                        continue
                    if idx in low_indices:
                        text = "\n".join(parts[2:])
                        if not text.startswith("【"):
                            text = f"【{text}】"
                        block = "\n".join(parts[:2] + [text])
                new_blocks.append(block)
            with open(review_srt, "w", encoding="utf-8") as f:
                f.write("\n\n".join(new_blocks) + "\n\n")
        except OSError:
            pass

    if os.path.isfile(review_ass):
        try:
            with open(review_ass, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
            dialogue_idx = 0
            new_lines = []
            for line in raw_lines:
                if line.startswith("Dialogue:"):
                    dialogue_idx += 1
                    if dialogue_idx in low_indices:
                        pos = line.rfind(",,")
                        if pos >= 0:
                            prefix = line[: pos + 2]
                            text = line[pos + 2 :].rstrip("\n")
                            if not text.startswith("【"):
                                text = f"【{text}】"
                            line = f"{prefix}{text}\n"
                new_lines.append(line)
            with open(review_ass, "w", encoding="utf-8-sig") as f:
                f.writelines(new_lines)
        except OSError:
            pass
