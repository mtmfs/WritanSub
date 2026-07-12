"""纯 Python ResourceRegistry:子进程超时/取消即杀/模型注册表/解码回环。

全部离线:子进程用 sys.executable 起 Python 单行,解码用 imageio-ffmpeg 自带 ffmpeg。
"""
import sys
import threading
import time

import pytest

from writansub.bridge import ResourceRegistry, CancelledError


def _wait_live_proc(reg, deadline=5.0):
    """等 run_subprocess 把子进程登记进活进程表(防 kill 早于 spawn 的竞态)。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        if getattr(reg, "_procs", None):
            return True
        time.sleep(0.05)
    return False


def test_run_subprocess_captures_output(registry):
    proc = registry.run_subprocess(
        [sys.executable, "-c",
         "import sys; sys.stdout.write('out'); sys.stderr.write('err')"],
        timeout=60,
    )
    assert proc.returncode == 0
    assert proc.stdout == b"out"
    assert b"err" in proc.stderr


def test_run_subprocess_timeout_kills(registry):
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        registry.run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert time.monotonic() - t0 < 10  # 旧 native 层这里会干等 30s


def test_cancel_kills_running_process(registry):
    result = {}

    def run():
        try:
            registry.run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert _wait_live_proc(registry)
    t0 = time.monotonic()
    registry.cancelled = True  # setter 应立即 kill 活子进程
    t.join(timeout=10)
    assert not t.is_alive()
    assert time.monotonic() - t0 < 10
    assert isinstance(result["exc"], CancelledError)


def test_model_registry_roundtrip(registry):
    calls = []

    def factory():
        calls.append(1)
        return object()

    h1 = registry.acquire_model("m", "cpu", factory)
    assert registry.acquire_model("m", "cpu", factory) == h1  # 缓存复用
    assert len(calls) == 1
    assert registry.get_model(h1) is not None
    registry.release_model(h1)  # 兼容 API,无副作用
    registry.unload_model(h1)
    with pytest.raises(KeyError):
        registry.get_model(h1)
    h2 = registry.acquire_model("m", "cpu", factory)  # 失效句柄自动重建
    assert len(calls) == 2
    assert h2 != h1


def test_shutdown_clears_models_and_kills(registry):
    h = registry.register_model("x", object())
    registry.shutdown()
    with pytest.raises(KeyError):
        registry.get_model(h)
    registry.reset_controls()  # 还原给后续测试


def test_decode_audio_roundtrip(registry, tmp_path):
    import torch
    from writansub.preprocess.core import save_wav

    wav = tmp_path / "t.wav"
    save_wav(torch.zeros(1, 4410), str(wav), 44100)  # 0.1s 静音
    waveform, sr = registry.decode_audio(str(wav), sample_rate=16000)
    assert sr == 16000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert waveform.dtype == torch.float32
