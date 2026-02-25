"""配置中心：读写 PP 和翻译两套 JSON 配置文件"""

import json
import os
from typing import Any, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 后处理参数 ──────────────────────────────────────────

_PP_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "aitrans_pp.json")

PP_DEFAULTS: Dict[str, float] = {
    "extend_end": 0.30,
    "extend_start": 0.00,
    "gap_threshold": 0.50,
    "min_gap": 0.30,
    "word_conf_threshold": 0.50,
    "align_conf_threshold": 0.50,
    "min_duration": 0.30,
}

PARAM_DEFS: Dict[str, Dict[str, Any]] = {
    "extend_end": {
        "label": "向后延伸(s)", "from": 0.0, "to": 5.0, "inc": 0.01, "tip": None,
    },
    "extend_start": {
        "label": "向前延伸(s)", "from": 0.0, "to": 5.0, "inc": 0.01, "tip": None,
    },
    "gap_threshold": {
        "label": "间距阈值(s)", "from": 0.0, "to": 5.0, "inc": 0.01,
        "tip": "原始字幕间距 ≥ 此值时，延伸后仍保留\"最小间距\";\n"
               "< 此值时，前一条字幕延伸到下一条开头(无间距)",
    },
    "min_gap": {
        "label": "最小间距(s)", "from": 0.0, "to": 5.0, "inc": 0.01,
        "tip": "当原始间距 ≥ 阈值时，延伸后相邻字幕之间\n至少保留的空白时间",
    },
    "word_conf_threshold": {
        "label": "识别置信阈值", "from": 0.0, "to": 1.0, "inc": 0.05,
        "tip": "Whisper 识别概率低于此值的词将在 _review.srt 中\n"
               "标记为【?词】，方便人工校对\n设为 0 可禁用标记",
    },
    "align_conf_threshold": {
        "label": "对齐置信阈值", "from": 0.0, "to": 1.0, "inc": 0.05,
        "tip": "MMS_FA 对齐分数低于此值的句子将在 _review.srt 中\n"
               "用【】包裹整句，方便人工校对\n设为 0 可禁用标记",
    },
    "min_duration": {
        "label": "最小时长(s)", "from": 0.0, "to": 5.0, "inc": 0.01,
        "tip": "字幕时长不足此值时自动向前合并到上一条\n设为 0 可禁用",
    },
}


def load_pp_config() -> Dict[str, float]:
    """从配置文件加载后处理参数，文件不存在则返回默认值"""
    try:
        with open(_PP_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {k: float(cfg.get(k, v)) for k, v in PP_DEFAULTS.items()}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return dict(PP_DEFAULTS)


def save_pp_config(values: Dict[str, float]):
    """保存后处理参数到配置文件"""
    try:
        with open(_PP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)
    except OSError:
        pass


# ── 翻译参数 ──────────────────────────────────────────

_TRANSLATE_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "aitrans_translate.json")

TRANSLATE_DEFAULTS: Dict[str, Any] = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "no-key",
    "model": "deepseek-chat",
    "target_lang": "简体中文",
    "batch_size": 20,
}


def load_translate_config() -> Dict[str, Any]:
    """加载翻译配置"""
    try:
        with open(_TRANSLATE_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        result = dict(TRANSLATE_DEFAULTS)
        result.update({k: cfg[k] for k in TRANSLATE_DEFAULTS if k in cfg})
        return result
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(TRANSLATE_DEFAULTS)


def save_translate_config(values: Dict[str, Any]):
    """保存翻译配置"""
    try:
        with open(_TRANSLATE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
