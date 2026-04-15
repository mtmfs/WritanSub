import functools
import gc
import logging
import os
import shutil
import subprocess
import threading
from typing import Any, Callable

import numpy as np
import torch
import writansub_native

log = logging.getLogger(__name__)


class CancelledError(Exception):
    """任务被用户取消时抛出。"""


@functools.lru_cache(maxsize=1)
def _get_ffmpeg() -> str:
    """查找 ffmpeg 可执行路径（系统优先，imageio-ffmpeg 兜底），结果缓存。"""
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
    """Resource Registry (Rust Native Wrapper)

    接管了所有核心资源管理：
    - 进程控制 (ffmpeg)
    - 模型句柄 (引用计数)
    - 线程安全调度
    """

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

    # ── 暂停 / 取消 ─────────────────────────────────────

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
        """在后台线程的循环中调用：先等暂停恢复，再检查取消。

        Raises:
            CancelledError: 用户取消时抛出。
        """
        self._pause_event.wait()
        if self.cancelled:
            raise CancelledError("任务已取消")

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        handle = self._native.register_model(obj)
        self._model_handles[(name, device)] = handle
        log.debug(f"Native Model Registered: {handle} ({name})")
        return handle

    def acquire_model(self, name: str, device: str, factory: Callable[[], Any]) -> int:
        key = (name, device)
        if key in self._model_handles:
            handle = self._model_handles[key]
            # 尝试在原生层获取，如果失败（可能被 unload 了），则重新加载
            try:
                self._native.get_model(handle)
                log.debug(f"Reusing cached model: {name} on {device}")
                return handle
            except Exception:
                log.debug(f"Cached handle {handle} invalid, reloading...")
                del self._model_handles[key]

        obj = factory()
        return self.register_model(name, obj, device)

    def get_model(self, handle: int) -> Any:
        return self._native.get_model(handle)

    def release_model(self, handle: int) -> None:
        self._native.release_model(handle)

    def unload_model(self, handle: int) -> None:
        self._native.unload_model(handle)
        self._model_handles = {k: v for k, v in self._model_handles.items() if v != handle}
        gc.collect()

    def run_subprocess(self, cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
        """使用 Rust 接管子进程，提供更强的生命周期保证。

        超时后自动 shutdown 该进程并抛出 TimeoutError。
        """
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

        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def decode_audio(self, path: str, sample_rate: int = 44100) -> tuple[torch.Tensor, int]:
        """将任意媒体文件解码为单声道 float32 tensor [1, T]。

        使用 ffmpeg（系统优先，imageio-ffmpeg 兜底）。
        """
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
