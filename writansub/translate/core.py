"""AI 翻译：调用 OpenAI 兼容 API 翻译字幕"""

import os
import re
from typing import Callable, Dict, List, Optional

from writansub.types import Sub, fmt_srt_time


def translate_subs(
    subs: List[Sub],
    target_lang: str,
    api_base: str,
    api_key: str,
    model: str,
    batch_size: int = 20,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[Sub]:
    """
    使用 OpenAI 兼容 API 翻译字幕列表（内存操作）。

    翻译结果写入每条 Sub 的 translated 字段。

    Args:
        subs: 输入字幕列表
        target_lang: 目标语言 (如 "简体中文", "English")
        api_base: API 地址
        api_key: API Key
        model: 模型名
        batch_size: 每批翻译条数
        log_callback: 日志回调
        progress_callback: 进度回调
        cancelled: 取消检查回调

    Returns:
        同一 subs 列表（translated 字段已填充）
    """
    from openai import OpenAI

    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    def _cancelled() -> bool:
        return cancelled() if cancelled else False

    client = OpenAI(base_url=api_base, api_key=api_key)
    total = len(subs)

    _log(f"共 {total} 条字幕")

    translated: Dict[int, str] = {}

    for batch_start in range(0, total, batch_size):
        if _cancelled():
            break

        batch_end = min(batch_start + batch_size, total)
        batch = subs[batch_start:batch_end]

        lines = [
            f"{s.index}: {s.text.replace('\n', ' ').strip()}" for s in batch
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
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"(\d+)[:.：]\s*(.+)", line)
                if match:
                    idx = int(match.group(1))
                    text = match.group(2).strip()
                    translated[idx] = text
        except Exception as e:
            _log(f"  批次翻译出错: {e}")

    for s in subs:
        if s.index in translated:
            s.translated = translated[s.index]

    if progress_callback:
        progress_callback(1.0, "翻译完成")
    _log(f"翻译完成 {len(translated)}/{total} 条")
    return subs


def translate_srt(
    srt_path: str,
    target_lang: str,
    api_base: str,
    api_key: str,
    model: str,
    batch_size: int = 20,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    """
    兼容 wrapper：从 SRT 文件读取、翻译、写回磁盘。

    Returns:
        翻译后的 SRT 文件路径
    """
    import pysrt

    srt_subs = pysrt.open(srt_path, encoding="utf-8")
    subs = [
        Sub(
            index=s.index,
            start=s.start.ordinal / 1000.0,
            end=s.end.ordinal / 1000.0,
            text=s.text.replace("\n", " ").strip(),
        )
        for s in srt_subs
    ]

    translate_subs(
        subs,
        target_lang=target_lang,
        api_base=api_base,
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancelled=cancelled,
    )

    output_path = os.path.splitext(srt_path)[0] + "_translated.srt"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in subs:
            text = s.translated if s.translated else s.text
            f.write(f"{s.index}\n")
            f.write(f"{fmt_srt_time(s.start)} --> {fmt_srt_time(s.end)}\n")
            f.write(f"{text}\n\n")

    return output_path
