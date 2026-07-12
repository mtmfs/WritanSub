"""T19：_from_pretrained_cached 的四条分流。

设备侧错误（.to 抛出）必须原样传播、不得触发"缓存坏→联网重试"误诊；
真缓存坏才记日志并联网自愈。全部用桩类，不碰真实模型。
"""
import pytest

import writansub.preprocess.core as ppc
import writansub.logger


@pytest.fixture
def logs(monkeypatch):
    captured = []
    monkeypatch.setattr(writansub.logger, "log_line", captured.append)
    return captured


@pytest.fixture
def cache_hit(monkeypatch):
    monkeypatch.setattr(ppc, "_hf_model_cached", lambda cache_dir, repo: True)


@pytest.fixture
def cache_miss(monkeypatch):
    monkeypatch.setattr(ppc, "_hf_model_cached", lambda cache_dir, repo: False)


class _GoodModel:
    def to(self, device):
        return self

    def eval(self):
        return "MODEL_OK"


def _make_cls(offline_raises=None, model_factory=_GoodModel):
    class Cls:
        calls: list[bool] = []

        @classmethod
        def from_pretrained(cls, repo_id, cache_dir=None, local_files_only=False):
            cls.calls.append(local_files_only)
            if local_files_only and offline_raises is not None:
                raise offline_raises
            return model_factory()

    return Cls


def test_offline_hit_loads_once(cache_hit, logs, tmp_path):
    cls = _make_cls()
    assert ppc._from_pretrained_cached(cls, "fake/repo", str(tmp_path), "cpu") == "MODEL_OK"
    assert cls.calls == [True]
    assert logs == []


def test_broken_cache_retries_online(cache_hit, logs, tmp_path):
    cls = _make_cls(offline_raises=OSError("half-downloaded"))
    assert ppc._from_pretrained_cached(cls, "fake/repo", str(tmp_path), "cpu") == "MODEL_OK"
    assert cls.calls == [True, False]
    assert any("联网重试" in m for m in logs)


def test_device_error_propagates_untouched(cache_hit, logs, tmp_path):
    """核心断言：OOM/驱动错误不得被吞、不得误诊为缓存损坏。"""
    class OomModel:
        def to(self, device):
            raise RuntimeError("CUDA out of memory (simulated)")

    cls = _make_cls(model_factory=OomModel)
    with pytest.raises(RuntimeError, match="out of memory"):
        ppc._from_pretrained_cached(cls, "fake/repo", str(tmp_path), "cuda")
    assert cls.calls == [True]          # 未做无谓的二次加载
    assert not any("联网重试" in m for m in logs)  # 未误诊


def test_cache_miss_goes_online_directly(cache_miss, logs, tmp_path):
    cls = _make_cls()
    assert ppc._from_pretrained_cached(cls, "fake/repo", str(tmp_path), "cpu") == "MODEL_OK"
    assert cls.calls == [False]
    assert logs == []


def test_save_load_wav_roundtrip(tmp_path):
    """T28：load_wav 与 save_wav 对称，int16 量化误差内还原。"""
    import torch
    wav = (torch.linspace(-0.9, 0.9, 1600)).unsqueeze(0)  # [1, T]
    p = str(tmp_path / "rt.wav")
    ppc.save_wav(wav, p, 16000)
    back, sr = ppc.load_wav(p)
    assert sr == 16000
    assert back.shape == wav.shape
    assert (back - wav).abs().max() < 1e-3  # 16-bit 量化误差内


def test_silero_loader_uses_pip_package(monkeypatch):
    """T13：_get_silero_vad 走 silero_vad 包而非 torch.hub 在线下载。"""
    import sys
    import types as pytypes
    from writansub.preprocess import core

    stub = pytypes.ModuleType("silero_vad")
    stub.load_silero_vad = lambda: "MODEL"
    stub.get_speech_timestamps = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "silero_vad", stub)
    monkeypatch.setattr(core, "_silero_cache", None)

    model, gst = core._get_silero_vad()
    assert model == "MODEL"
    assert gst is stub.get_speech_timestamps
    assert core._get_silero_vad() == (model, gst)  # 二次调用命中缓存


def test_silero_real_load_no_network(monkeypatch):
    """真实加载：模型随包内置，无联网必须成功；纯静音无语音段。"""
    import torch
    from writansub.preprocess import core

    monkeypatch.setattr(core, "_silero_cache", None)
    spans = core._run_silero_vad(torch.zeros(1, 16000))
    assert spans == []
