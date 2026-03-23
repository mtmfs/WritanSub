"""由 Rust 驱动的高性能原生资源管理器"""
import gc
import logging
import subprocess
from typing import Any, Callable

import writansub_native

log = logging.getLogger(__name__)

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
        self._threads: dict[int, Any] = {}
        self._thread_counter: int = 0
        self._model_handles: dict[tuple[str, str], int] = {}

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        handle = self._native.register_model(obj)
        self._model_handles[(name, device)] = handle
        log.debug(f"Native Model Registered: {handle} ({name})")
        return handle

    def acquire_model(self, name: str, device: str, factory: Callable[[], Any]) -> int:
        """高性能模型获取：优先从原生池复用"""
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
        gc.collect()

    def run_subprocess(self, cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
        """使用 Rust 接管子进程，提供更强的生命周期保证"""
        handle = self._native.spawn_process(cmd)
        # 简化：目前直接等待。Rust 层可以扩展非阻塞监控。
        code, stdout, stderr = self._native.wait_process(handle)

        # Rust 层返回 list[int]，转换为 Python bytes
        if isinstance(stdout, list):
            stdout = bytes(stdout)
        if isinstance(stderr, list):
            stderr = bytes(stderr)

        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def register_thread(self, thread: Any) -> int:
        self._thread_counter += 1
        self._threads[self._thread_counter] = thread
        return self._thread_counter

    def unregister_thread(self, handle: int) -> None:
        self._threads.pop(handle, None)

    def shutdown(self) -> None:
        """物理级清理：强制终止所有 Rust 接管的子进程"""
        log.info("Native Shutdown Initiated...")
        self._native.shutdown()
        # 清理 Python 线程引用
        self._threads.clear()
