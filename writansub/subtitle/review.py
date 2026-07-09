from typing import Callable

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
    """返回 (srt_content, ass_content, low_count, total_words)。"""
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
    with open(f"{base_path}_review.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)
    with open(f"{base_path}_review.ass", "w", encoding="utf-8-sig") as f:
        f.write(ass_content)


def attach_low_words(
    subs: list[Sub],
    word_data: list[list[WordInfo]],
    threshold: float,
) -> tuple[int, int]:
    """把低置信词记到 sub.low_words 上随本体携带（T06 根修的前半）。

    之后的重编号/合并/参考映射只需拼接该列表，review 在流程末尾单点生成。
    返回 (low_count, total_words) 供日志。
    """
    low = 0
    total = 0
    for sub, words in zip(subs, word_data):
        if not words:
            continue
        picked: list[str] = []
        for w in words:
            total += 1
            if w.probability < threshold:
                low += 1
                stripped = w.word.strip()
                if stripped:
                    picked.append(stripped)
        if picked:
            sub.low_words = sub.low_words + picked
    return low, total


def _mark_text(text: str, low_words: list[str], wrap: Callable[[str], str]) -> str:
    """按出现顺序包裹低置信词；递进偏移支持重复词标注连续出现处；找不到则跳过。"""
    out = text
    pos = 0
    for w in low_words:
        i = out.find(w, pos)
        if i < 0:
            continue
        marked = wrap(w)
        out = out[:i] + marked + out[i + len(w):]
        pos = i + len(marked)
    return out


def generate_review_final(
    subs: list[Sub],
    align_threshold: float,
) -> tuple[str, str, int, int]:
    """全流程索引稳定后单点生成 review 内容（T06 根修的后半，取代已删除的
    mark_low_align_in_review 回补写盘模式）。

    - 词级：sub.low_words 逐词标注（SRT 【?词】 / ASS 红色）
    - 行级：align_threshold > 0 且 score < 阈值时整行【】（含 score=0 的对齐失败行；
      skip_align/ref_direct 场景调用方传 0 关闭行级标注）
    - ASS 文本中换行转义为 \\N（merge 模式的多说话人 cue 为多行文本）
    返回 (srt_content, ass_content, marked_words, marked_lines)。
    """
    srt_lines = []
    ass_lines = [_ASS_REVIEW_HEADER]
    marked_words = 0
    marked_lines = 0

    for sub in subs:
        time_line = f"{fmt_srt_time(sub.start)} --> {fmt_srt_time(sub.end)}"
        srt_text = _mark_text(sub.text, sub.low_words, lambda w: f"【?{w}】")
        ass_text = _mark_text(sub.text, sub.low_words, lambda w: f"{{\\c&H0000FF&}}{w}{{\\c}}")
        marked_words += len(sub.low_words)

        if align_threshold > 0 and sub.score < align_threshold:
            if not srt_text.startswith("【"):
                srt_text = f"【{srt_text}】"
                ass_text = f"【{ass_text}】"
            marked_lines += 1

        ass_text = ass_text.replace("\n", "\\N")
        srt_lines.append(f"{sub.index}\n{time_line}\n{srt_text}\n")
        ass_lines.append(
            f"Dialogue: 0,{fmt_ass_time(sub.start)},{fmt_ass_time(sub.end)},Default,,0,0,0,,{ass_text}"
        )

    return "\n".join(srt_lines), "\n".join(ass_lines) + "\n", marked_words, marked_lines
