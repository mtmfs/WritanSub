"""T44 会话日志脱敏；T30 日志文件命名与按角色清理。"""
import os
import re
import sys
import time
from datetime import datetime

import pytest

import writansub.logger as logger
from writansub.logger import redact_argv


@pytest.fixture
def fresh_logger(tmp_path, monkeypatch):
    """隔离的 logger：日志目录指向 tmp_path/logs（尚不存在，顺带验证 makedirs），单例状态复位，
    不装全局钩子（faulthandler / excepthook / root handler）、不清理旧日志。返回日志目录。"""
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr(logger.paths, "LOG_DIR", log_dir)
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


def test_log_file_name_has_role_and_pid():
    assert logger._log_file_name("cli", 4242, datetime(2026, 10, 15, 9, 5, 7)) == "cli_20261015_090507_4242.log"


def test_session_log_named_by_role_and_pid(fresh_logger):
    path = logger.init_session_log(role="cli")
    assert os.path.dirname(path) == fresh_logger  # 目录在调用时才建，且取自 paths.LOG_DIR
    assert re.fullmatch(rf"cli_\d{{8}}_\d{{6}}_{os.getpid()}\.log", os.path.basename(path))
    text = open(path, encoding="utf-8").read()
    assert "Role cli\n" in text
    assert f"WRITANSUB_HOME {logger.paths.HOME_DIR}" in text
    assert logger.session_log_path() == path
    assert logger.init_session_log(role="worker") == path  # 进程内单例，二次调用不换文件


def test_init_session_log_rejects_role_with_underscore(fresh_logger):
    with pytest.raises(ValueError):
        logger.init_session_log(role="cli_worker")


def _touch(log_dir: str, name: str, age_s: float) -> str:
    p = os.path.join(log_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("x")
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def test_prune_is_per_role(tmp_path, monkeypatch):
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir)
    monkeypatch.setattr(logger.paths, "LOG_DIR", log_dir)
    monkeypatch.setattr(logger, "_MAX_LOG_FILES", 2)
    old = _touch(log_dir, "cli_20260101_000000_1.log", 300)
    mid = _touch(log_dir, "cli_20260102_000000_2.log", 200)
    new = _touch(log_dir, "cli_20260103_000000_3.log", 100)
    other = _touch(log_dir, "daemon_20260101_000000_9.log", 400)    # 别的角色不动
    legacy = _touch(log_dir, "writansub_20260101_000000.log", 500)  # 旧版命名不动
    logger._prune_old_logs("cli")
    assert not os.path.exists(old)
    assert os.path.exists(mid) and os.path.exists(new)
    assert os.path.exists(other) and os.path.exists(legacy)


def test_prune_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(logger.paths, "LOG_DIR", str(tmp_path / "nope"))
    logger._prune_old_logs("cli")  # 目录不存在也不抛
