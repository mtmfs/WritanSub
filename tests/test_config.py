"""config：翻译默认表的 batch_size 与 translate.core 常量单源。"""
from writansub.config import TRANSLATE_DEFAULTS
from writansub.translate.core import DEFAULT_BATCH_SIZE


def test_batch_size_single_source():
    assert TRANSLATE_DEFAULTS["batch_size"] is DEFAULT_BATCH_SIZE
