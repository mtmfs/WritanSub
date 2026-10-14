"""T30 存储布局：默认项目目录，WRITANSUB_HOME 整体覆盖，旧的分目录变量失效，import 不建目录，溢出目录在 run/ 下。"""
import importlib
import os

import pytest

import writansub.paths as paths

SUBDIRS = {"CACHE_DIR": "cache", "MODELS_DIR": "models", "LOG_DIR": "logs",
           "CONFIG_DIR": "config", "RUN_DIR": "run"}


@pytest.fixture
def reload_paths(monkeypatch):
    """环境改好后重载 paths；结束时先撤销全部 monkeypatch（含 autouse 的属性覆盖与环境变量）再重载，
    让其他测试看到的常量回到默认。不能把还原交给 fixture 自然拆除：monkeypatch 撤销晚于本 fixture，
    reload 会在环境变量还原之前执行，模块就停在 tmp 路径上。"""
    def _reload():
        importlib.reload(paths)
        return paths
    yield _reload
    monkeypatch.undo()
    importlib.reload(paths)


def test_default_root_is_project_dir(monkeypatch, reload_paths):
    monkeypatch.delenv("WRITANSUB_HOME", raising=False)
    p = reload_paths()
    assert p.HOME_DIR == p.PROJECT_ROOT
    for attr, sub in SUBDIRS.items():
        assert getattr(p, attr) == os.path.join(p.PROJECT_ROOT, sub)


def test_home_env_moves_every_subdir(monkeypatch, tmp_path, reload_paths):
    home = tmp_path / "home"
    monkeypatch.setenv("WRITANSUB_HOME", str(home))
    p = reload_paths()
    assert p.HOME_DIR == str(home)
    for attr, sub in SUBDIRS.items():
        assert getattr(p, attr) == os.path.join(str(home), sub)
    assert not home.exists()  # import 只算路径，不 makedirs


def test_empty_env_falls_back_to_project_dir(monkeypatch, reload_paths):
    monkeypatch.setenv("WRITANSUB_HOME", "")
    p = reload_paths()
    assert p.HOME_DIR == p.PROJECT_ROOT


def test_relative_env_is_made_absolute(monkeypatch, tmp_path, reload_paths):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WRITANSUB_HOME", "data")
    assert reload_paths().HOME_DIR == os.path.join(str(tmp_path), "data")


def test_old_per_dir_env_vars_ignored(monkeypatch, tmp_path, reload_paths):
    home = tmp_path / "home"
    monkeypatch.setenv("WRITANSUB_HOME", str(home))
    monkeypatch.setenv("WRITANSUB_LOG_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("WRITANSUB_MODELS_DIR", str(tmp_path / "elsewhere"))
    p = reload_paths()
    assert p.LOG_DIR == str(home / "logs")
    assert p.MODELS_DIR == str(home / "models")


def test_new_spill_dir_lives_under_run_dir(monkeypatch, tmp_path):
    run = os.path.join(str(tmp_path), "home", "run")  # 两级都不存在：new_spill_dir 必须自己 makedirs
    monkeypatch.setattr(paths, "RUN_DIR", run)
    d = paths.new_spill_dir()
    assert os.path.isdir(d)
    assert os.path.dirname(d) == run
    assert os.path.basename(d).startswith(f"spill_{os.getpid()}_")
    assert paths.new_spill_dir() != d  # 每次调用独立目录
