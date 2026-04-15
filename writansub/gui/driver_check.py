import subprocess
import sys

MIN_DRIVER_MAJOR = 570

_CREATE_NO_WINDOW = 0x0800_0000


def _query_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return line or None


def _parse_major(version: str) -> int | None:
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _show_warning(title: str, text: str) -> None:
    if sys.platform != "win32":
        print(f"{title}: {text}", file=sys.stderr)
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, text, title, 0x30)
    except Exception:
        print(f"{title}: {text}", file=sys.stderr)


def check_driver(min_major: int = MIN_DRIVER_MAJOR) -> None:
    version = _query_driver_version()
    if version is None:
        _show_warning(
            "WritanSub - 未检测到 NVIDIA 驱动",
            "未能通过 nvidia-smi 读取驱动版本。\n\n"
            "WritanSub 需要 NVIDIA 显卡与驱动才能运行 CUDA 加速。\n"
            "程序将继续启动,但 GPU 相关功能可能不可用。",
        )
        return

    major = _parse_major(version)
    if major is not None and major < min_major:
        _show_warning(
            "WritanSub - 驱动版本过低",
            f"检测到 NVIDIA 驱动版本:{version}\n"
            f"本版本 WritanSub 基于 CUDA 12.8,需要驱动版本 ≥ {min_major}.x。\n\n"
            f"请升级 NVIDIA 显卡驱动后再启动程序,否则可能崩溃或无法使用 GPU。",
        )
