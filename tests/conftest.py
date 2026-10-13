"""公共 fixture：registry 复位、模拟探测库缺失。"""
import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def registry():
    """真实 ResourceRegistry 单例，测试前后都复位取消状态。"""
    from writansub.bridge import ResourceRegistry
    reg = ResourceRegistry.instance()
    reg.reset_controls()
    yield reg
    reg.reset_controls()


@pytest.fixture
def no_charset_normalizer(monkeypatch):
    """模拟 charset-normalizer 未安装：import 即抛 ImportError。"""
    real_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name == "charset_normalizer" or name.startswith("charset_normalizer."):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("charset_normalizer", None)
    monkeypatch.setattr(builtins, "__import__", block)
