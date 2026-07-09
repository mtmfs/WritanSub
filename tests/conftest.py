"""公共 fixture：隔离配置路径、registry 复位、离屏 Qt、模拟探测库缺失。

所有测试不得触碰真实用户配置（%LOCALAPPDATA%\\mtmfs\\WritanSub\\*.json）。
"""
import builtins
import os
import sys

# 离屏渲染必须在任何 Qt 导入之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def isolated_translate_config(tmp_path, monkeypatch):
    """翻译配置读写重定向到临时文件，返回该路径。"""
    import writansub.config as config
    p = tmp_path / "writansub_translate.json"
    monkeypatch.setattr(config, "TRANSLATE_CONFIG_PATH", str(p))
    return p


@pytest.fixture
def isolated_pp_config(tmp_path, monkeypatch):
    import writansub.config as config
    p = tmp_path / "writansub_pp.json"
    monkeypatch.setattr(config, "PP_CONFIG_PATH", str(p))
    return p


@pytest.fixture
def registry():
    """真实 ResourceRegistry 单例，测试前后都复位取消/暂停状态。"""
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


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
