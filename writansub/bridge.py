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
    candidate = os.path.join(dirpath, probe_name) if dirpath else probe_name
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            "未找到 ffprobe: imageio-ffmpeg 只自带 ffmpeg 不含 ffprobe，"
            "内嵌字幕轨探测需要系统 ffmpeg。请安装 ffmpeg(含 ffprobe)并加入 PATH"
        )
    return candidate


# Windows 下子进程不建控制台窗口(GUI/pythonw 场景防闪黑框);
# 代价是子进程收不到控制台 Ctrl+C——取消一律走显式 kill(cancelled setter)
_CREATE_NO_WINDOW = 0x0800_0000


class ResourceRegistry:
    _instance: "ResourceRegistry | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ResourceRegistry":
        # 每任务仅调用数次，非热路径：无条件加锁，不做双检快路径
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_handle = 1
        self._models: dict[int, Any] = {}
        self._model_handles: dict[tuple[str, str], int] = {}
        self._procs: set[subprocess.Popen] = set()
        self._cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    # ── 取消 / 暂停 ──

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @cancelled.setter
    def cancelled(self, value: bool) -> None:
        # 置 True 即杀活子进程：解码/提取中的取消从"等 ffmpeg 跑完"变即时
        self._cancelled = value
        if value:
            self._kill_procs()

    def _kill_procs(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    @property
    def paused(self) -> bool:
        return not self._pause_event.is_set()

    def reset_controls(self) -> None:
        self._cancelled = False
        self._pause_event.set()

    def checkpoint(self) -> None:
        self._pause_event.wait()
        if self._cancelled:
            raise CancelledError("任务已取消")

    # ── 模型注册表 ──

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        from writansub.logger import log_line
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._models[handle] = obj
            self._model_handles[(name, device)] = handle
        log_line(f"[model] registered name={name!r} device={device!r} handle={handle}{_gpu_mem_hint(device)}")
        return handle

    def acquire_model(self, name: str, device: str, factory: Callable[[], Any]) -> int:
        from writansub.logger import log_line
        key = (name, device)
        with self._lock:
            handle = self._model_handles.get(key)
            if handle is not None and handle in self._models:
                cached = handle
            else:
                cached = None
                if handle is not None:
                    del self._model_handles[key]
        if cached is not None:
            log_line(f"[model] reuse cached name={name!r} device={device!r} handle={cached}")
            return cached

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
        with self._lock:
            if handle not in self._models:
                raise KeyError(f"model handle {handle} not found")
            return self._models[handle]

    def release_model(self, handle: int) -> None:
        # 兼容 API：native 时代的 in_use 标志 Python 侧从未消费，退役后仅留日志
        from writansub.logger import log_line
        log_line(f"[model] released handle={handle}")

    def unload_model(self, handle: int) -> None:
        from writansub.logger import log_line
        with self._lock:
            key = next((k for k, v in self._model_handles.items() if v == handle), None)
            device = key[1] if key else ""
            self._models.pop(handle, None)
            self._model_handles = {k: v for k, v in self._model_handles.items() if v != handle}
        log_line(f"[model] unloading handle={handle} key={key}{_gpu_mem_hint(device)}")
        gc.collect()
        log_line(f"[model] unloaded handle={handle}{_gpu_mem_hint(device)}")

    # ── 子进程 ──

    def run_subprocess(self, cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
        creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        with self._lock:
            self._procs.add(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TimeoutError(f"子进程超时 ({timeout}s): {cmd[0]}") from None
        finally:
            with self._lock:
                self._procs.discard(proc)

        # 取消触发的 kill：返回码必非零，按取消上抛，调用方不得误报"执行失败"
        if self._cancelled and proc.returncode != 0:
            raise CancelledError("任务已取消")

        code = proc.returncode
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
        proc = self.run_subprocess(cmd, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        return waveform, sample_rate

    def shutdown(self) -> None:
        log.info("Shutdown initiated...")
        self.cancelled = True  # property：顺带 kill 所有活子进程
        self._pause_event.set()
        with self._lock:
            self._models.clear()
            self._model_handles.clear()
