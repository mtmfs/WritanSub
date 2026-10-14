"""T44 会话日志脱敏；T30 日志文件命名与按角色清理。"""
import os
import sys

import pytest

import writansub.logger as logger
from writansub.logger import redact_argv


@pytest.fixture
def fresh_logger(tmp_path, monkeypatch):
    """隔离的 logger：日志目录指向 tmp_path/logs（尚不存在，顺带验证 makedirs），单例状态复位，
    不装全局钩子（faulthandler / excepthook / root handler）、不清理旧日志。返回日志目录。"""
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr(logger, "LOG_DIR", log_dir)  # Task 4 改为 logger.paths
    monkeypatch.setattr(logger, "_session_log_path", None)
    monkeypatch.setattr(logger, "_t0", None)
    monkeypatch.setattr(logger, "_install_hooks", lambda: None)
    monkeypatch.setattr(logger, "_prune_old_logs", lambda *a, **k: None)
    return log_dir


@pytest.mark.parametrize("argv, expected", [
    (["w", "translate", "a.srt", "--api-key", "sk-1", "--batch-size", "20"],
     ["w", "translate", "a.srt", "--api-key", "***", "--batch-size", "20"]),
    (["w", "--api-key=sk-1"], ["w", "--api-key=***"]),
    (["w", "--api-k", "sk-1", "--target-lang", "English"], ["w", "--api-k", "***", "--target-lang", "English"]),
    (["w", "--api-ke=sk-1"], ["w", "--api-ke=***"]),
    (["w", "--api-base", "https://h", "--whisper-model", "large-v3", "--api-keyring", "v"],
     ["w", "--api-base", "https://h", "--whisper-model", "large-v3", "--api-keyring", "v"]),
    (["w", "--api-key"], ["w", "--api-key"]),  # 末尾无值：不越界、不报错
    ([], []),
])
def test_redact_argv(argv, expected):
    assert redact_argv(argv) == expected


def test_redact_argv_does_not_mutate_input():
    argv = ["w", "--api-key", "sk-1"]
    redact_argv(argv)
    assert argv == ["w", "--api-key", "sk-1"]


def test_session_log_header_redacts_api_key(fresh_logger, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["writansub-cli", "translate", "a.srt", "--api-key", "sk-secret"])
    path = logger.init_session_log()
    assert os.path.dirname(path) == fresh_logger
    text = open(path, encoding="utf-8").read()
    assert "sk-secret" not in text
    assert "'--api-key', '***'" in text
