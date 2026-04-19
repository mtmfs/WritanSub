"""Session logger: persists run events + tracebacks to a file under LOG_DIR."""
import faulthandler
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from writansub.paths import LOG_DIR

_MAX_LOG_FILES = 20
_session_log_path: str | None = None
_lock = threading.Lock()
_t0: float | None = None
_fault_fh = None
_root_handler: logging.Handler | None = None


def _collect_basic_info() -> list[str]:
    import platform
    lines = []
    try:
        import writansub
        lines.append(f"WritanSub {writansub.__version__}")
    except Exception:
        lines.append("WritanSub (version unknown)")
    lines.append(f"Python {sys.version.split()[0]}")
    lines.append(f"Platform {platform.platform()}")
    lines.append(f"PID {os.getpid()}")
    lines.append(f"LOG_DIR {LOG_DIR}")
    lines.append(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}")
    lines.append(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '(unset)')}")
    return lines


def _collect_runtime_info() -> list[str]:
    lines = []
    mods = sys.modules

    if "torch" in mods:
        torch = mods["torch"]
        try:
            lines.append(f"torch {torch.__version__}")
            if torch.cuda.is_available():
                lines.append(f"CUDA {torch.version.cuda} / {torch.cuda.get_device_name(0)}")
            else:
                lines.append("CUDA unavailable")
        except Exception as e:
            lines.append(f"torch inspect failed: {e!r}")

    for name in ("faster_whisper", "huggingface_hub", "demucs", "espnet2"):
        if name in mods:
            ver = getattr(mods[name], "__version__", "?")
            lines.append(f"{name} {ver}")
    return lines


def _elapsed_str() -> str:
    if _t0 is None:
        return "+00:00"
    secs = int(time.monotonic() - _t0)
    return f"+{secs // 60:02d}:{secs % 60:02d}"


def _install_hooks() -> None:
    """Install faulthandler, threading.excepthook, sys.unraisablehook, and stdlib logging root handler."""
    global _fault_fh, _root_handler
    if not _session_log_path:
        return

    try:
        _fault_fh = open(_session_log_path, "a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_fault_fh, all_threads=True)
    except Exception as e:
        sys.stderr.write(f"[writansub-logger] faulthandler enable failed: {e!r}\n")

    def _thread_hook(args):
        try:
            log_exception(f"thread:{args.thread.name if args.thread else '?'}", args.exc_value)
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass

    def _unraisable_hook(u):
        try:
            where = f"unraisable:{u.object!r}" if u.object is not None else "unraisable"
            log_exception(where, u.exc_value)
        except Exception:
            pass

    try:
        sys.unraisablehook = _unraisable_hook
    except Exception:
        pass

    try:
        handler = logging.FileHandler(_session_log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root = logging.getLogger()
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        _root_handler = handler
    except Exception as e:
        sys.stderr.write(f"[writansub-logger] root logging handler setup failed: {e!r}\n")


def init_session_log() -> str:
    """Create a new per-session log file, write system info header, prune old files."""
    global _session_log_path, _t0
    with _lock:
        if _session_log_path:
            return _session_log_path
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(LOG_DIR, f"writansub_{ts}.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"Session start: {datetime.now().isoformat(timespec='seconds')}\n")
                for line in _collect_basic_info():
                    f.write(line + "\n")
                f.write(f"argv: {sys.argv}\n")
                f.write("=" * 60 + "\n\n")
        except OSError as e:
            sys.stderr.write(f"[writansub-logger] init failed at {LOG_DIR!r}: {e!r}\n")
            return ""
        _session_log_path = path
        _t0 = time.monotonic()
    _install_hooks()
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
    elapsed = _elapsed_str()
    try:
        with _lock, open(_session_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp} {elapsed}] {msg}\n")
    except OSError:
        pass


def log_exception(where: str, exc: BaseException) -> None:
    """Append a full traceback block for `exc`, with runtime info snapshot."""
    if not _session_log_path:
        return
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    stamp = datetime.now().isoformat(timespec="seconds")
    elapsed = _elapsed_str()
    runtime_lines = _collect_runtime_info()
    try:
        with _lock, open(_session_log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "!" * 60 + "\n")
            f.write(f"EXCEPTION @ {where} ({stamp}, elapsed={elapsed}):\n")
            if runtime_lines:
                for line in runtime_lines:
                    f.write(f"  {line}\n")
                f.write("---\n")
            f.write(tb)
            f.write("!" * 60 + "\n\n")
    except OSError:
        pass


def session_log_path() -> str | None:
    return _session_log_path
