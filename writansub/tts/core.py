"""[废案模块] 语音合成：多模型分派（MMS_FA 对齐 / Style-Bert-VITS2 TTS）
未接入主程序，配套 GUI (tts.py) 同为废案。
"""

import json
import os
from typing import Any, Callable

import numpy as np

from writansub.types import Sub


# ── 模型元信息（不加载权重）──────────────────────────────


def load_model_meta(model_name: str, model_dir: str = "") -> dict[str, Any]:
    """读取模型元信息，供 GUI 填充下拉框。

    Returns:
        dict with keys depending on model:
        - mms_fa: {}  (无额外元信息)
        - sbv2-jp-extra: {speakers: list[str], styles: list[str], sample_rate: int}
    """
    if model_name == "mms_fa":
        return {}

    if model_name == "sbv2-jp-extra":
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "speakers": list(cfg.get("spk2id", {"default": 0}).keys()),
            "styles": list(cfg.get("style2id", {"Neutral": 0}).keys()),
            "sample_rate": cfg.get("data", {}).get("sampling_rate", 44100),
        }

    raise ValueError(f"未知模型: {model_name}")


# ── 模型加载 ──────────────────────────────────────────


def init_model(model_name: str, device: str, model_dir: str = "") -> Any:
    if model_name == "mms_fa":
        from writansub.align.core import init_model as _init_mms
        return _init_mms(device)

    if model_name == "sbv2-jp-extra":
        return _init_sbv2(model_dir, device)

    raise ValueError(f"未知模型: {model_name}")


def _init_sbv2(model_dir: str, device: str):
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
    from pathlib import Path

    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    safetensors = None
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            safetensors = os.path.join(model_dir, name)
            break
    if safetensors is None:
        raise FileNotFoundError(f"未找到 .safetensors 文件: {model_dir}")

    model = TTSModel(
        model_path=Path(safetensors),
        config_path=Path(os.path.join(model_dir, "config.json")),
        style_vec_path=Path(os.path.join(model_dir, "style_vectors.npy")),
        device=device,
    )
    return model


# ── MMS_FA 对齐 ──────────────────────────────────────


def run_mms_fa(
    audio_path: str,
    subs: list[Sub],
    device: str,
    model_bundle: tuple,
    pp_params: dict[str, float],
    lang: str = "ja",
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[Sub]:
    from writansub.align.core import load_audio, run_alignment, post_process

    _log = log_callback or (lambda msg: None)

    if progress_callback:
        progress_callback(0.0, "加载音频...")
    waveform = load_audio(audio_path)

    _log(f"字幕条数: {len(subs)}")

    aligned = run_alignment(
        waveform, subs, device=device,
        progress_callback=lambda p, m: progress_callback(p * 0.95, m) if progress_callback else None,
        model_bundle=model_bundle,
    )

    if progress_callback:
        progress_callback(0.95, "后处理...")
    pp_params.pop("align_conf_threshold", None)
    final = post_process(aligned, **pp_params)

    if progress_callback:
        progress_callback(1.0, "对齐完成")
    return final


# ── SBV2 合成 ──────────────────────────────────────


def run_sbv2(
    subs: list[Sub],
    model: Any,
    speaker: str = "",
    style: str = "Neutral",
    speed: float = 1.0,
    use_translated: bool = False,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, int]:
    """SBV2 路径：SRT → 时间对齐的 WAV 数组。

    Returns:
        (audio_int16, sample_rate)
    """
    from style_bert_vits2.constants import Languages

    _log = log_callback or (lambda msg: None)

    sr = model.hyper_parameters.data.sampling_rate
    spk2id = model.hyper_parameters.data.spk2id if hasattr(model.hyper_parameters.data, "spk2id") else {}
    speaker_id = spk2id.get(speaker, 0) if speaker else 0

    if not subs:
        return np.zeros(0, dtype=np.int16), sr

    total_seconds = max(s.end for s in subs)
    total_samples = int(total_seconds * sr) + sr
    output = np.zeros(total_samples, dtype=np.int16)

    _log(f"合成 {len(subs)} 条字幕，总时长 {total_seconds:.1f}s")

    for i, sub in enumerate(subs):
        text = (sub.translated if use_translated and sub.translated else sub.text).strip()
        if not text:
            continue

        if progress_callback:
            progress_callback(i / len(subs), f"合成中... {i + 1}/{len(subs)}")

        try:
            seg_sr, segment = model.infer(
                text=text,
                language=Languages.JP,
                speaker_id=speaker_id,
                style=style,
                length=speed,
            )
        except Exception as e:
            _log(f"  [{i + 1}] 跳过: {e}")
            continue

        start_sample = int(sub.start * sr)
        end_sample = start_sample + len(segment)

        if end_sample > len(output):
            output = np.concatenate([output, np.zeros(end_sample - len(output), dtype=np.int16)])

        seg_len = min(len(segment), len(output) - start_sample)
        output[start_sample:start_sample + seg_len] = segment[:seg_len]

    if progress_callback:
        progress_callback(1.0, "合成完成")
    _log("合成完成")
    return output, sr
