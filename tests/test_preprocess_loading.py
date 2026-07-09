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
