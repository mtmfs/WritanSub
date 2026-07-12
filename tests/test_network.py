"""T12：镜像探测经代理 HTTPS 实测（urlopen 打桩，全离线）。"""
import os
import urllib.error
import urllib.request

import pytest

from writansub import network


@pytest.fixture
def clean_hf_env():
    saved = {k: os.environ.pop(k) for k in ("HF_ENDPOINT", "HF_HUB_OFFLINE") if k in os.environ}
    yield
    for k in ("HF_ENDPOINT", "HF_HUB_OFFLINE"):
        os.environ.pop(k, None)
    os.environ.update(saved)


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub(monkeypatch, effect):
    def fake(req, timeout=0.0):
        if isinstance(effect, Exception):
            raise effect
        return _Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_reachable_keeps_official(clean_hf_env, monkeypatch):
    _stub(monkeypatch, None)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ


def test_unreachable_switches_mirror(clean_hf_env, monkeypatch):
    _stub(monkeypatch, urllib.error.URLError("blocked"))
    network.setup_hf_mirror()
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_http_error_counts_as_reachable(clean_hf_env, monkeypatch):
    """4xx/5xx 说明 TLS+HTTP 全程走通（旧探测的 SNI 盲区场景），视为可达。"""
    err = urllib.error.HTTPError("https://huggingface.co", 403, "Forbidden", None, None)
    _stub(monkeypatch, err)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ


def test_user_endpoint_respected(clean_hf_env, monkeypatch):
    os.environ["HF_ENDPOINT"] = "https://my-mirror.example"

    def boom(*a, **k):
        raise AssertionError("已设 HF_ENDPOINT 时不应发起探测")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    network.setup_hf_mirror()
    assert os.environ["HF_ENDPOINT"] == "https://my-mirror.example"


def test_offline_mode_skips_probe(clean_hf_env, monkeypatch):
    os.environ["HF_HUB_OFFLINE"] = "1"

    def boom(*a, **k):
        raise AssertionError("离线模式不应发起探测")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ
