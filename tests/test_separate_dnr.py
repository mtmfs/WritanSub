"""T24：separate_dnr 按 full_tracks 裁剪子模型推理次数。

假模型记录哪些子模型被跑过；registry 整体替换为桩，不碰真实权重。
"""
import torch
import pytest

import writansub.preprocess.core as ppc


class _FakeReg:
    def acquire_model(self, name, device, factory):
        return 1

    def get_model(self, handle):
        return _fake_model_holder["model"]

    def release_model(self, handle):
        pass

    def checkpoint(self):
        pass


class _FakeRegistry:
    @classmethod
    def instance(cls):
        return _FakeReg()


_fake_model_holder = {}


class _FakeDnrModel:
    """dialog/effect/music 三个子模型用字符串标记；wav_chunk_inference 记录调用。"""
    dialog = "SUB_DIALOG"
    effect = "SUB_EFFECT"
    music = "SUB_MUSIC"

    def __init__(self):
        self.calls: list[str] = []

    def wav_chunk_inference(self, sub_model, mixture):
        self.calls.append(sub_model)
        # [n_tracks=3, nch, T]，行值区分轨道索引
        out = torch.zeros(3, 1, mixture.shape[-1])
        for i in range(3):
            out[i] = float(i)
        return out


@pytest.fixture
def fake_env(monkeypatch):
    model = _FakeDnrModel()
    _fake_model_holder["model"] = model
    monkeypatch.setattr(ppc, "ResourceRegistry", _FakeRegistry)
    return model


def _wave():
    return torch.zeros(1, 1000)  # sr=44100 传入时跳过重采样


def test_default_full_tracks_runs_three(fake_env):
    dialog, effects, music = ppc.separate_dnr(_wave(), 44100, device="cpu")
    assert fake_env.calls == ["SUB_DIALOG", "SUB_EFFECT", "SUB_MUSIC"]
    assert dialog is not None and effects is not None and music is not None
    # 各轨取对应输出索引：dialog=2, effect=1, music=0
    assert float(dialog[0, 0]) == 2.0
    assert float(effects[0, 0]) == 1.0
    assert float(music[0, 0]) == 0.0


def test_single_track_skips_effect_music(fake_env):
    dialog, effects, music = ppc.separate_dnr(
        _wave(), 44100, device="cpu", full_tracks=False)
    assert fake_env.calls == ["SUB_DIALOG"]  # 只跑人声子模型
    assert float(dialog[0, 0]) == 2.0
    assert effects is None and music is None


def test_progress_labels_match_track_count(fake_env):
    msgs = []
    ppc.separate_dnr(_wave(), 44100, device="cpu", full_tracks=False,
                     log_callback=msgs.append)
    assert any("(1/1)" in m for m in msgs)
    assert not any("2/" in m or "3/" in m for m in msgs)
