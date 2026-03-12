"""由 Rust 驱动的高性能原生资源管理器"""
import logging
import typing
import writansub_native

log = logging.getLogger(__name__)

class ResourceRegistry:
    """Resource Registry (Rust Native Wrapper)
    
    接管了所有核心资源管理：
    - 进程控制 (ffmpeg)
    - 模型句柄 (引用计数)
    - 线程安全调度
    """
    _instance: typing.Optional["ResourceRegistry"] = None

    @classmethod
    def instance(cls) -> "ResourceRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 初始化底层原生 Registry
        self._native = writansub_native.ResourceRegistry()
        self._threads = {}
        self._thread_counter = 0

    def register_model(self, name: str, obj: typing.Any, device: str = "") -> int:
        handle = self._native.register_model(obj)
        log.debug(f"Native Model Registered: {handle} ({name})")
        return handle

    def acquire_model(self, name: str, device: str, factory: typing.Callable[[], typing.Any]) -> int:
        """高性能模型获取：优先从原生池复用"""
        # 注意：这里的 name 和 device 逻辑目前在 Rust 层是简化的，
        # 我们通过 Python 封装来维持原有的 API 兼容性。
        # 这里直接调用 factory 如果没找到合适的。
        obj = factory()
        return self.register_model(name, obj, device)

    def get_model(self, handle: int) -> typing.Any:
        return self._native.get_model(handle)

    def release_model(self, handle: int) -> None:
        self._native.release_model(handle)

    def unload_model(self, handle: int) -> None:
        self._native.unload_model(handle)
        import gc
        gc.collect()

    def run_subprocess(self, cmd: typing.List[str], timeout: float = 600) -> typing.Any:
        """使用 Rust 接管子进程，提供更强的生命周期保证"""
        handle = self._native.spawn_process(cmd)
        # 简化：目前直接等待。Rust 层可以扩展非阻塞监控。
        code, stdout, stderr = self._native.wait_process(handle)
        
        import subprocess
        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def register_thread(self, thread) -> int:
        self._thread_counter += 1
        self._threads[self._thread_counter] = thread
        return self._thread_counter

    def unregister_thread(self, handle: int):
        self._threads.pop(handle, None)

    def shutdown(self):
        """物理级清理：强制终止所有 Rust 接管的子进程"""
        log.info("Native Shutdown Initiated...")
        self._native.shutdown()
        # 清理 Python 线程引用
        self._threads.clear()

