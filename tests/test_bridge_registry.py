"""ResourceRegistry：单例锁（简化后）与 checkpoint 取消/暂停语义。"""
import threading

import pytest

from writansub.bridge import ResourceRegistry, CancelledError


def test_singleton_identity():
    assert ResourceRegistry.instance() is ResourceRegistry.instance()


def test_concurrent_instance_single_object():
    """32 线程同时冷启动 instance()，必须只创建一个实例。"""
    original = ResourceRegistry._instance
    results: list[ResourceRegistry] = []
    barrier = threading.Barrier(32)

    def race():
        barrier.wait()
        results.append(ResourceRegistry.instance())

    try:
        ResourceRegistry._instance = None
        threads = [threading.Thread(target=race) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(r) for r in results}) == 1
    finally:
        ResourceRegistry._instance = original


def test_checkpoint_cancel_and_reset(registry):
    registry.cancelled = True
    with pytest.raises(CancelledError):
        registry.checkpoint()
    registry.reset_controls()
    registry.checkpoint()  # 复位后不再抛


def test_pause_flag(registry):
    registry.pause()
    assert registry.paused
    registry.resume()
    assert not registry.paused
    registry.checkpoint()


def test_resolve_device_cuda_fallback(monkeypatch):
    """T40：请求 cuda 但不可用时回退 cpu 并知会。"""
    import torch
    from writansub.bridge import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    msgs = []
    assert resolve_device("cuda", msgs.append) == "cpu"
    assert any("CUDA" in m for m in msgs)


def test_resolve_device_passthrough(monkeypatch):
    import torch
    from writansub.bridge import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
