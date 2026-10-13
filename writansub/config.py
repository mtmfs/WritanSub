from typing import Any

from writansub.translate.core import DEFAULT_BATCH_SIZE

PP_DEFAULTS: dict[str, float] = {
    "extend_end": 0.30,
    "extend_start": 0.00,
    "gap_threshold": 0.50,
    "min_gap": 0.30,
    "word_conf_threshold": 0.50,
    "align_conf_threshold": 0.50,
    "min_duration": 0.30,
    "pad_sec": 0.50,
}

TRANSLATE_DEFAULTS: dict[str, Any] = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "no-key",
    "model": "deepseek-chat",
    "target_lang": "简体中文",
    "batch_size": DEFAULT_BATCH_SIZE,
}
