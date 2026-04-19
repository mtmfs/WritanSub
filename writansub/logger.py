"""Session logger: persists run events + tracebacks to a file under LOG_DIR."""
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from writansub.paths import LOG_DIR

_MAX_LOG_FILES = 20
_session_log_path: str | None = None
_lock = threading.Lock()


def _collect_system_info() -> list[str]:
    import platform
    lines = []
    try:
        import writansub
        lines.append(f"WritanSub {writansub.__version__}")
    except Exception:
        lines.append("WritanSub (version unknown)")
    lines.append(f"Python {sys.version.split()[0]}")
    lines.append(f"Platform {platform.platform()}")

    try:
        import torch
        lines.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            lines.append(f"CUDA {torch.version.cuda} / {torch.cuda.get_device_name(0)}")
        else:
            lines.append("CUDA unavailable")
    except Exception as e:
        lines.append(f"torch: {e!r}")

    for mod in ("faster_whisper", "huggingface_hub", "demucs", "espnet2"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            lines.append(f"{mod} {ver}")
        except ImportError:
            pass

    lines.append(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}")
    lines.append(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '(unset)')}")
    return lines


def init_session_log() -> str:
    """Create a new per-session log file, write system info header, prune old files."""
    global _session_log_path
    with _lock:
        if _session_log_path:
            return _session_log_path
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"writansub_{ts}.log")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"Session start: {datetime.now().isoformat(timespec='seconds')}\n")
                for line in _collect_system_info():
                    f.write(line + "\n")
                f.write(f"argv: {sys.argv}\n")
                f.write("=" * 60 + "\n\n")
        except OSError:
            return ""
        _session_log_path = path
    _prune_old_logs()
    return path


def _prune_old_logs() -> None:
    try:
        files = sorted(
            Path(LOG_DIR).glob("writansub_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[_MAX_LOG_FILES:]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception:
        pass


def log_line(msg: str) -> None:
    """Append a single timestamped line to the session log. Safe to call before init."""
    if not _session_log_path:
        return
    stamp = datetime.now().strftime("%H:%M:%S")
    try:
        with _lock, open(_session_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {msg}\n")
    except OSError:
        pass


def log_exception(where: str, exc: BaseException) -> None:
    """Append a full traceback block for `exc`."""
    if not _session_log_path:
        return
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        with _lock, open(_session_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "!" * 60 + "\n")
            f.write(f"EXCEPTION @ {where} ({stamp}):\n")
            f.write(tb)
            f.write("!" * 60 + "\n\n")
    except OSError:
        pass


def session_log_path() -> str | None:
    return _session_log_path
