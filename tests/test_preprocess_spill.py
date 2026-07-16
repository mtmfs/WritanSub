"""T17：run_dnr_batch / run_speech_batch 波形即分即落盘，结果字典只存路径。

registry 与分离函数整体替换为桩（循 test_separate_dnr 模式），不碰真实权重。
"""
import os

import torch
import pytest

import writansub.preprocess.core as ppc
from writansub.preprocess.core import TimeSpan, _media_key, load_wav, save_wav


class _FakeReg:
    def decode_audio(self, path, sample_rate=44100):
        return torch.zeros(1, 4410), sample_rate

    def checkpoint(self):
        pass


class _FakeRegistry:
    @classmethod
    def instance(cls):
        return _FakeReg()


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(ppc, "ResourceRegistry", _FakeRegistry)

    def _fake_separate_dnr(waveform, sr, **kwargs):
        z = torch.zeros(1, 4410)
        return z, z.clone(), z.clone()

    monkeypatch.setattr(ppc, "separate_dnr", _fake_separate_dnr)


def _media(tmp_path, name="video.mkv"):
    p = tmp_path / "media" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"\x00")
    return str(p)


def test_run_dnr_batch_spills_paths(fake_env, tmp_path):
    media = _media(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()

    results = ppc.run_dnr_batch([media], str(spill), device="cpu")

    data = results[media]
    assert "dialog_wav" not in data  # 旧键必须消失，防止漏改的读方静默走兜底
    assert data["dialog_sr"] == 44100
    assert os.path.dirname(data["dialog_path"]) == str(spill)
    wav, sr = load_wav(data["dialog_path"])
    assert wav.shape == (1, 4410) and sr == 44100


def test_save_intermediate_keeps_user_wavs(fake_env, tmp_path):
    media = _media(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()

    results = ppc.run_dnr_batch([media], str(spill), save_intermediate=True, device="cpu")

    # 面向用户的中间产物照常写在媒体旁，溢出副本独立存在
    media_dir = os.path.dirname(media)
    for name in ("video_dialog.wav", "video_effects.wav", "video_music.wav"):
        assert os.path.isfile(os.path.join(media_dir, name))
    assert os.path.isfile(results[media]["dialog_path"])


def test_run_speech_batch_reads_path_and_spills_spk(monkeypatch, tmp_path):
    monkeypatch.setattr(ppc, "ResourceRegistry", _FakeRegistry)
    monkeypatch.setattr(
        ppc, "separate_speakers",
        lambda wav, sr, **kw: (torch.zeros(1, 1600), torch.zeros(1, 1600)),
    )
    monkeypatch.setattr(
        ppc, "detect_overlaps",
        lambda s1, s2, sr=16000, **kw: ([TimeSpan(start=0.1, end=0.2)], 0.05),
    )

    media = _media(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()
    dialog_path = str(spill / "seed_dialog.wav")
    save_wav(torch.zeros(1, 4410), dialog_path, 44100)
    dnr_results = {media: {"dialog_path": dialog_path, "dialog_sr": 44100}}

    ppc.run_speech_batch(dnr_results, str(spill), device="cpu")

    data = dnr_results[media]
    assert "spk1_wav" not in data and "spk2_wav" not in data
    assert data["spk_sr"] == 16000
    assert data["overlap_regions"] and data["overlap_ratio"] == 0.05
    for key in ("spk1_path", "spk2_path"):
        wav, sr = load_wav(data[key])
        assert wav.shape == (1, 1600) and sr == 16000


def test_media_key_stable_and_distinct(tmp_path):
    a = str(tmp_path / "dir_a" / "video.mkv")
    b = str(tmp_path / "dir_b" / "video.mkv")
    assert _media_key(a) == _media_key(a)  # 稳定：同路径可跨进程命中（T41 前提）
    assert _media_key(a) != _media_key(b)  # 同名不同目录不冲突
    assert len(_media_key(a)) == 12
