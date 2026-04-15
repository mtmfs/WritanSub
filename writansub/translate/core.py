import re
from typing import Callable

from writansub.types import Sub


def translate_subs(
    subs: list[Sub],
    target_lang: str,
    api_base: str,
    api_key: str,
    model: str,
    batch_size: int = 20,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Sub]:
    """翻译结果写入每条 Sub 的 translated 字段，原地修改。"""
    from openai import OpenAI
    from writansub.bridge import ResourceRegistry

    _log = log_callback or (lambda msg: None)
    reg = ResourceRegistry.instance()

    client = OpenAI(base_url=api_base, api_key=api_key)
    total = len(subs)

    _log(f"共 {total} 条字幕")

    translated: dict[int, str] = {}
    fail_count = 0

    for batch_start in range(0, total, batch_size):
        reg.checkpoint()

        batch_end = min(batch_start + batch_size, total)
        batch = subs[batch_start:batch_end]

        lines = [
            f"{s.index}: {s.text.replace(chr(10), ' ').strip()}" for s in batch
        ]
        prompt = "\n".join(lines)

        if progress_callback:
            progress_callback(batch_start / total, f"翻译中... {batch_end}/{total}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"你是专业字幕翻译。将以下字幕翻译为{target_lang}。\n"
                            "规则:\n"
                            '1. 仅输出翻译结果，保持相同编号格式 "编号: 译文"\n'
                            "2. 译文简洁自然，适合字幕显示\n"
                            "3. 不要添加解释或注释"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            result = response.choices[0].message.content.strip()
            # 解析响应：支持多行翻译（编号行开头 → 新条目，否则追加到上一条）
            current_idx = None
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"(\d+)[:.：]\s*(.+)", line)
                if match:
                    current_idx = int(match.group(1))
                    translated[current_idx] = match.group(2).strip()
                elif current_idx is not None:
                    translated[current_idx] += " " + line
        except Exception as e:
            fail_count += 1
            _log(f"  批次 {batch_start+1}-{batch_end} 翻译出错: {e}")

    for s in subs:
        if s.index in translated:
            s.translated = translated[s.index]

    if progress_callback:
        progress_callback(1.0, "翻译完成")
    msg = f"翻译完成 {len(translated)}/{total} 条"
    if fail_count > 0:
        msg += f" ({fail_count} 个批次失败)"
    _log(msg)
    return subs

