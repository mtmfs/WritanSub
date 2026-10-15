import os
from pathlib import Path
import subprocess
import sys

import pytest

from writansub.cli import _resolve_cache, build_parser


@pytest.mark.parametrize("command", ["pipeline", "preprocess"])
def test_cache_options_defaults_and_override(command):
    args = build_parser().parse_args([command, "video.mp4"])
    assert _resolve_cache(args, {}).enabled is True
    assert _resolve_cache(args, {}).max_bytes == 5_000_000_000
    config = {"preprocess_cache": False, "preprocess_cache_dir": "data", "preprocess_cache_gb": 2.5}
    assert _resolve_cache(args, config).max_bytes == 2_500_000_000
    args = build_parser().parse_args([command, "video.mp4", "--preprocess-cache", "--preprocess-cache-gb", "3"])
    options = _resolve_cache(args, config)
    assert options.enabled is True and options.max_bytes == 3_000_000_000 and options.directory == "data"
    args = build_parser().parse_args([command, "video.mp4", "--no-preprocess-cache"])
    assert not _resolve_cache(args, {}).enabled


@pytest.mark.parametrize("capacity", ["0", "-1", "nan", "inf", "1e308", "1e-15"])
def test_invalid_cache_capacity_rejected(capacity):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pipeline", "video.mp4", "--preprocess-cache-gb", capacity])


@pytest.mark.parametrize("config", [{"preprocess_cache": "false"}, {"preprocess_cache_gb": float("nan")},
                                   {"preprocess_cache_gb": True}, {"preprocess_cache_dir": ""}])
def test_invalid_json_cache_settings_rejected(config):
    with pytest.raises(ValueError):
        _resolve_cache(build_parser().parse_args(["pipeline", "video.mp4"]), config)


@pytest.mark.parametrize("command", [None, "pipeline", "preprocess", "transcribe", "align", "translate"])
@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_information_commands_create_no_storage(tmp_path, command, flag):
    home = tmp_path / "absent-home"
    argv = [sys.executable, "-m", "writansub.cli"]
    if command:
        argv.append(command)
    env = dict(os.environ, WRITANSUB_HOME=str(home), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    completed = subprocess.run(argv + [flag], env=env, capture_output=True, text=True,
                               encoding="utf-8", timeout=20, cwd=str(Path(__file__).resolve().parents[1]))
    assert completed.returncode == 0, completed.stderr
    assert not home.exists()
