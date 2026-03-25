"""配置中心：读写 PP 和翻译两套 JSON 配置文件"""

import json
import os
from typing import Any

from writansub.paths import PP_CONFIG_PATH, TRANSLATE_CONFIG_PATH, GUI_STATE_PATH

# ── 后处理参数 ──────────────────────────────────────────

PP_DEFAULTS: dict[str, float] = {
    "extend_end": 0.30,
    "extend_start": 0.00,
    "gap_threshold": 0.50,
    "min_gap": 0.30,
    "word_conf_threshold": 0.50,
    "align_conf_threshold": 0.50,
    "min_duration": 0.30,
}

PARAM_DEFS: dict[str, dict[str, Any]] = {
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


def _load_json(path: str) -> dict[str, Any]:
    """读取 JSON 文件，失败时返回空字典"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_json(path: str, data: dict[str, Any]) -> None:
    """将数据写入 JSON 文件，写入失败时静默忽略"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def load_pp_config() -> dict[str, float]:
    """从配置文件加载后处理参数，文件不存在则返回默认值"""
    raw = _load_json(PP_CONFIG_PATH)
    try:
        return {k: float(raw.get(k, v)) for k, v in PP_DEFAULTS.items()}
    except ValueError:
        return dict(PP_DEFAULTS)


def save_pp_config(values: dict[str, float]) -> None:
    """保存后处理参数到配置文件"""
    _save_json(PP_CONFIG_PATH, values)


# ── 翻译参数 ──────────────────────────────────────────

TRANSLATE_DEFAULTS: dict[str, Any] = {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "no-key",
    "model": "deepseek-chat",
    "target_lang": "简体中文",
    "batch_size": 20,
}


def load_translate_config() -> dict[str, Any]:
    """加载翻译配置"""
    raw = _load_json(TRANSLATE_CONFIG_PATH)
    return TRANSLATE_DEFAULTS | {k: raw[k] for k in TRANSLATE_DEFAULTS if k in raw}


def save_translate_config(values: dict[str, Any]) -> None:
    """保存翻译配置"""
    _save_json(TRANSLATE_CONFIG_PATH, values)


# ── GUI 状态持久化 ──────────────────────────────────────

def load_gui_state() -> dict[str, Any]:
    """加载 GUI 状态，文件不存在返回空字典"""
    return _load_json(GUI_STATE_PATH)


def save_gui_state(state: dict[str, Any]) -> None:
    """保存 GUI 状态到 gui_state.json"""
    _save_json(GUI_STATE_PATH, state)
