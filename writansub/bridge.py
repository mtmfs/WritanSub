from __future__ import annotations

import functools
import gc
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import writansub_native

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


def _gpu_mem_hint(device: str) -> str:
    """Return ' [alloc=…GB free=…/…GB]' when device is CUDA and torch is loaded, else ''."""
    if not device or not device.startswith("cuda") or "torch" not in sys.modules:
        return ""
    try:
        torch = sys.modules["torch"]
        if not torch.cuda.is_available():
            return ""
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                pass
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = torch.cuda.memory_allocated(idx)
        g = 1024 ** 3
        return f" [alloc={alloc_b/g:.2f}GB free={free_b/g:.2f}/{total_b/g:.2f}GB]"
    except Exception:
        return ""


class CancelledError(Exception):
    pass


@functools.lru_cache(maxsize=1)
def _get_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


@functools.lru_cache(maxsize=1)
def _get_ffprobe() -> str:
    """查找 ffprobe 可执行路径（系统优先，否则从 ffmpeg 同目录推导），结果缓存。"""
    path = shutil.which("ffprobe")
    if path:
        return path
    ffmpeg = _get_ffmpeg()
    dirpath = os.path.dirname(ffmpeg)
    probe_name = os.path.basename(ffmpeg).replace("ffmpeg", "ffprobe")
    return os.path.join(dirpath, probe_name) if dirpath else probe_name

class ResourceRegistry:
    _instance: "ResourceRegistry | None" = None

    @classmethod
    def instance(cls) -> "ResourceRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # 使用底层原生 Registry 类（类级 API，非实例）
        self._native = writansub_native.ResourceRegistry
        self._model_handles: dict[tuple[str, str], int] = {}
        self.cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    @property
    def paused(self) -> bool:
        return not self._pause_event.is_set()

    def reset_controls(self) -> None:
        self.cancelled = False
        self._pause_event.set()

    def checkpoint(self) -> None:
        self._pause_event.wait()
        if self.cancelled:
            raise CancelledError("任务已取消")

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        from writansub.logger import log_line
        handle = self._native.register_model(obj)
        self._model_handles[(name, device)] = handle
        log_line(f"[model] registered name={name!r} device={device!r} handle={handle}{_gpu_mem_hint(device)}")
        return handle

    def acquire_model(self, name: str, device: str, factory: Callable[[], Any]) -> int:
        from writansub.logger import log_line
        key = (name, device)
        if key in self._model_handles:
            handle = self._model_handles[key]
            # 尝试在原生层获取，如果失败（可能被 unload 了），则重新加载
            try:
                self._native.get_model(handle)
                log_line(f"[model] reuse cached name={name!r} device={device!r} handle={handle}")
                return handle
            except Exception as e:
                log_line(f"[model] cached handle={handle} invalid for {name!r}: {e!r}, reloading")
                del self._model_handles[key]

        log_line(f"[model] loading name={name!r} device={device!r} ...{_gpu_mem_hint(device)}")
        t0 = time.monotonic()
        try:
            obj = factory()
        except BaseException as e:
            elapsed = time.monotonic() - t0
            log_line(
                f"[model] LOAD FAILED name={name!r} device={device!r} "
                f"after {elapsed:.2f}s: {type(e).__name__}: {e}"
            )
            raise
        elapsed = time.monotonic() - t0
        log_line(f"[model] loaded name={name!r} device={device!r} in {elapsed:.2f}s{_gpu_mem_hint(device)}")
        return self.register_model(name, obj, device)

    def get_model(self, handle: int) -> Any:
        return self._native.get_model(handle)

    def release_model(self, handle: int) -> None:
        from writansub.logger import log_line
        self._native.release_model(handle)
        log_line(f"[model] released handle={handle}")

    def unload_model(self, handle: int) -> None:
        from writansub.logger import log_line
        key = next((k for k, v in self._model_handles.items() if v == handle), None)
        device = key[1] if key else ""
        log_line(f"[model] unloading handle={handle} key={key}{_gpu_mem_hint(device)}")
        self._native.unload_model(handle)
        self._model_handles = {k: v for k, v in self._model_handles.items() if v != handle}
        gc.collect()
        log_line(f"[model] unloaded handle={handle}{_gpu_mem_hint(device)}")

    def run_subprocess(self, cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
        handle = self._native.spawn_process(cmd)
        result: list[tuple[int, Any, Any]] = []
        error: list[BaseException] = []

        def _wait():
            try:
                result.append(self._native.wait_process(handle))
            except Exception as e:
                error.append(e)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # 超时，强制终止
            self._native.shutdown()
            raise TimeoutError(f"子进程超时 ({timeout}s): {cmd[0]}")

        if error:
            raise error[0]

        code, stdout, stderr = result[0]

        # Rust 层返回 list[int]，转换为 Python bytes
        if isinstance(stdout, list):
            stdout = bytes(stdout)
        if isinstance(stderr, list):
            stderr = bytes(stderr)

        if stderr:
            try:
                from writansub.logger import log_line
                exe = os.path.basename(cmd[0]) if cmd else "?"
                raw = stderr.decode("utf-8", errors="replace")
                limit = 4000
                if len(raw) > limit:
                    head_n, tail_n = 2000, 1000
                    elided = len(raw) - head_n - tail_n
                    raw = f"{raw[:head_n]} ... [{elided} chars elided] ... {raw[-tail_n:]}"
                text = raw.replace("\n", " | ").replace("\r", "")
                log_line(f"stderr[{exe}] ({len(stderr)}B, rc={code}): {text}")
            except Exception:
                pass

        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def decode_audio(self, path: str, sample_rate: int = 44100) -> tuple["torch.Tensor", int]:
        import numpy as np
        import torch

        cmd = [
            _get_ffmpeg(), "-i", path,
            "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
            "-loglevel", "error", "-",
        ]
        proc = self.run_subprocess(cmd, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        return waveform, sample_rate

    def shutdown(self) -> None:
        log.info("Native Shutdown Initiated...")
        self.cancelled = True
        self._pause_event.set()
        self._native.shutdown()
