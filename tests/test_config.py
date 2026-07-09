"""config：翻译配置默认合并（batch_size 常量单源）与坏文件容错（T33）。"""
import json

from writansub.config import (
    TRANSLATE_DEFAULTS, PP_DEFAULTS,
    load_translate_config, save_translate_config, load_pp_config,
)
from writansub.translate.core import DEFAULT_BATCH_SIZE


def test_batch_size_single_source():
    assert TRANSLATE_DEFAULTS["batch_size"] is DEFAULT_BATCH_SIZE


def test_missing_file_yields_defaults(isolated_translate_config):
    cfg = load_translate_config()
    assert cfg == TRANSLATE_DEFAULTS
    assert cfg["batch_size"] == DEFAULT_BATCH_SIZE


def test_partial_file_merged(isolated_translate_config):
    isolated_translate_config.write_text(
        json.dumps({"api_key": "sk-test"}), encoding="utf-8")
    cfg = load_translate_config()
    assert cfg["api_key"] == "sk-test"
    assert cfg["batch_size"] == DEFAULT_BATCH_SIZE  # 缺键必被默认表补齐
    assert cfg["model"] == TRANSLATE_DEFAULTS["model"]


def test_corrupt_file_yields_defaults(isolated_translate_config):
    isolated_translate_config.write_text("{not json", encoding="utf-8")
    assert load_translate_config() == TRANSLATE_DEFAULTS


def test_unknown_keys_dropped(isolated_translate_config):
    isolated_translate_config.write_text(
        json.dumps({"batch_size": 50, "evil": "x"}), encoding="utf-8")
    cfg = load_translate_config()
    assert cfg["batch_size"] == 50
    assert "evil" not in cfg


def test_save_roundtrip(isolated_translate_config):
    values = dict(TRANSLATE_DEFAULTS, api_key="sk-roundtrip")
    save_translate_config(values)
    assert load_translate_config()["api_key"] == "sk-roundtrip"


def test_pp_config_null_value_falls_back(isolated_pp_config):
    """T33：JSON null/类型错误不得让加载崩溃，整体回落默认。"""
    isolated_pp_config.write_text(
        json.dumps({"pad_sec": None}), encoding="utf-8")
    assert load_pp_config() == PP_DEFAULTS


def test_pp_config_array_value_falls_back(isolated_pp_config):
    isolated_pp_config.write_text(
        json.dumps({"pad_sec": [1, 2]}), encoding="utf-8")
    assert load_pp_config() == PP_DEFAULTS
