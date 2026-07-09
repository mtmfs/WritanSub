from writansub.types import Sub


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    o = min(a_end, b_end) - max(a_start, b_start)
    return max(o, 0.0)


def map_whisper_to_ref(
    whisper_subs: list[Sub],
    ref_subs: list[Sub],
) -> list[Sub]:
    """将 Whisper 文本按时间重叠映射到参考字幕的时间轴上。

    算法（段落级别）：
    1. 对每条 Whisper sub，找重叠最大的 ref sub，建立归属
    2. 对每条 ref sub，收集归属于它的 Whisper 文本并拼接
    3. 没有匹配到 Whisper 文本的 ref 条目跳过
    """
    if not whisper_subs or not ref_subs:
        return list(ref_subs) if ref_subs else []

    # 双指针依赖时间序；外部来源（ffmpeg/pysrt）通常已排序但不强保证
    whisper_subs = sorted(whisper_subs, key=lambda s: s.start)
    ref_subs = sorted(ref_subs, key=lambda s: s.start)

    # assigned[ref_index] = [whisper subs in order]（存 Sub 本体以携带 low_words/speaker）
    assigned: dict[int, list[Sub]] = {}
    search_start = 0

    for w_sub in whisper_subs:
        while search_start < len(ref_subs) and ref_subs[search_start].end < w_sub.start:
            search_start += 1

        best_idx = -1
        best_overlap = 0.0
        for r_idx in range(search_start, len(ref_subs)):
            r_sub = ref_subs[r_idx]
            if r_sub.start > w_sub.end:
                break
            ov = _overlap(w_sub.start, w_sub.end, r_sub.start, r_sub.end)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = r_idx

        if best_idx >= 0 and best_overlap > 0:
            assigned.setdefault(best_idx, []).append(w_sub)

    result = []
    idx = 1
    for r_idx, r_sub in enumerate(ref_subs):
        group = assigned.get(r_idx)
        if not group:
            continue
        merged_text = " ".join(s.text for s in group)
        speakers = {s.speaker for s in group if s.speaker}
        result.append(Sub(
            index=idx,
            start=r_sub.start,
            end=r_sub.end,
            text=merged_text,
            low_words=[w for s in group for w in s.low_words],
            speaker=speakers.pop() if len(speakers) == 1 else 0,
        ))
        idx += 1

    return result
