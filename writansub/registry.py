"""资源注册表：统一管理模型、子进程、线程的生命周期"""

import logging
import subprocess
import threading
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

__all__ = ["ResourceRegistry"]


class _ModelEntry:
    __slots__ = ("obj", "device", "name", "in_use")

    def __init__(self, obj: Any, device: str, name: str, in_use: bool = False):
        self.obj = obj
        self.device = device
        self.name = name
        self.in_use = in_use


class ResourceRegistry:
    """集中资源管理单例，线程安全。

    三组资源各有独立的锁：
    - 模型（GPU 显存）
    - 子进程（ffmpeg 等）
    - 工作线程
    """

    _instance: Optional["ResourceRegistry"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ResourceRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 模型管理
        self._model_lock = threading.Lock()
        self._models: Dict[int, _ModelEntry] = {}
        self._model_counter = 0
        # CTranslate2 退出守卫 —— 阻止最后一个 CT2 模型被析构（会崩溃）
        self._ct2_exit_guard: Any = None

        # 子进程管理
        self._proc_lock = threading.Lock()
        self._procs: Dict[int, subprocess.Popen] = {}
        self._proc_counter = 0

        # 线程管理
        self._thread_lock = threading.Lock()
        self._threads: Dict[int, threading.Thread] = {}
        self._thread_counter = 0

        # 关闭标志
        self._shutting_down = False

    # ── 模型管理 ──────────────────────────────────────────────

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        """登记模型，返回 handle。"""
        with self._model_lock:
            self._model_counter += 1
            handle = self._model_counter
            self._models[handle] = _ModelEntry(obj, device, name)
            log.debug("register_model: handle=%d name=%s device=%s", handle, name, device)
            return handle

    def get_model(self, handle: int) -> Any:
        """取回模型对象。"""
        with self._model_lock:
            entry = self._models.get(handle)
            return entry.obj if entry else None

    def unload_model(self, handle: int) -> None:
        """卸载模型，释放显存。

        对 whisper (CTranslate2) 模型：保留最后一个引用到 _ct2_exit_guard
        防止进程退出时 CT2 析构器崩溃。旧的 guard 正常释放。
        """
        with self._model_lock:
            entry = self._models.pop(handle, None)
            if entry is None:
                return
            log.debug("unload_model: handle=%d name=%s", handle, entry.name)

        # CTranslate2 (faster-whisper) 特殊处理
        if entry.name == "whisper":
            old_guard = self._ct2_exit_guard
            self._ct2_exit_guard = entry.obj
            del old_guard
        else:
            del entry

        # 尝试释放 CUDA 显存
        self._cuda_cleanup()

    @staticmethod
    def _cuda_cleanup():
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 模型池（高层 API）────────────────────────────────────────

    def acquire_model(self, name: str, device: str,
                      factory: Callable[[], Any]) -> int:
        """请求模型：池中有同 name+device 的空闲模型就复用，否则调 factory 创建。
        返回 handle。调用者用完后必须调用 release_model。"""
        with self._model_lock:
            for handle, entry in self._models.items():
                if entry.name == name and entry.device == device and not entry.in_use:
                    entry.in_use = True
                    log.debug("acquire_model: reuse handle=%d name=%s device=%s",
                              handle, name, device)
                    return handle
        # 池中没有可用的，创建新模型（factory 可能耗时，不持锁）
        obj = factory()
        with self._model_lock:
            self._model_counter += 1
            handle = self._model_counter
            self._models[handle] = _ModelEntry(obj, device, name, in_use=True)
            log.debug("acquire_model: new handle=%d name=%s device=%s",
                      handle, name, device)
            return handle

    def release_model(self, handle: int) -> None:
        """释放模型回池中（不卸载）。调用者用完即释放，
        模型留在池中供下次 acquire。"""
        with self._model_lock:
            entry = self._models.get(handle)
            if entry is not None:
                entry.in_use = False
                log.debug("release_model: handle=%d name=%s", handle, entry.name)

    def flush_models(self) -> None:
        """清空模型池，释放所有 GPU 显存。pipeline 结束时调用。"""
        with self._model_lock:
            handles = list(self._models.keys())
        for h in handles:
            self.unload_model(h)

    # ── 子进程管理 ─────────────────────────────────────────────

    def run_subprocess(self, cmd, timeout: float = 300) -> subprocess.CompletedProcess:
        """替代 subprocess.run，可在 shutdown 时被 kill。

        返回 CompletedProcess，接口与 subprocess.run(capture_output=True) 一致。
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with self._proc_lock:
            self._proc_counter += 1
            pid = self._proc_counter
            self._procs[pid] = proc

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        finally:
            with self._proc_lock:
                self._procs.pop(pid, None)

        return subprocess.CompletedProcess(
            cmd, proc.returncode, stdout, stderr,
        )

    # ── 线程管理 ───────────────────────────────────────────────

    def register_thread(self, thread: threading.Thread) -> int:
        """登记工作线程，返回 handle。"""
        with self._thread_lock:
            self._thread_counter += 1
            handle = self._thread_counter
            self._threads[handle] = thread
            log.debug("register_thread: handle=%d name=%s", handle, thread.name)
            return handle

    def unregister_thread(self, handle: int) -> None:
        """线程结束时注销。"""
        with self._thread_lock:
            self._threads.pop(handle, None)
            log.debug("unregister_thread: handle=%d", handle)

    # ── 生命周期 ───────────────────────────────────────────────

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def shutdown(self, thread_timeout: float = 5.0) -> None:
        """关窗口时调用：杀进程 → 等线程 → 卸模型。"""
        self._shutting_down = True
        log.debug("shutdown: begin")

        # 1. 杀掉所有子进程
        with self._proc_lock:
            procs = list(self._procs.values())
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass

        # 2. 等待所有线程结束
        with self._thread_lock:
            threads = list(self._threads.values())
        for t in threads:
            try:
                t.join(timeout=thread_timeout)
            except Exception:
                pass
        with self._thread_lock:
            self._threads.clear()

        # 3. 卸载所有模型
        with self._model_lock:
            handles = list(self._models.keys())
        for h in handles:
            self.unload_model(h)

        log.debug("shutdown: done")
