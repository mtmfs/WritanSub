"""T17 runner 侧：溢出 wav 直接喂 whisper、spk 按需读回、薄壳目录随任务生灭。"""
import os

import torch
import pytest

import writansub.pipeline.runner as runner
from writansub.bridge import CancelledError
from writansub.pipeline.runner import PipelineConfig, _transcribe_single, run_pipeline
from writansub.preprocess.core import TimeSpan, save_wav
from writansub.types import Sub, WordInfo


class _RecordingTranscribe:
    """记录每次调用的输入路径；首次(全轨)返回预置结果，后续(区域)返回短句。"""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, file_path, **kwargs):
        self.calls.append(file_path)
        if len(self.calls) == 1:
            return (
                [Sub(index=1, start=0.0, end=3.0, text="全轨")],
                [[WordInfo("全轨", 0.9)]],
            )
        n = len(self.calls)
        return [Sub(index=1, start=0.0, end=1.0, text=f"区域{n}")], [[WordInfo("区", 0.2)]]


def test_transcribe_single_feeds_spill_path_directly(monkeypatch, registry, tmp_path):
    fake = _RecordingTranscribe()
    monkeypatch.setattr(runner, "transcribe", fake)
    dialog_path = str(tmp_path / "abc_dialog.wav")
    save_wav(torch.zeros(1, 4410), dialog_path, 44100)

    cfg = PipelineConfig(device="cpu")
    _transcribe_single(
        "video.mkv", {"dialog_path": dialog_path, "dialog_sr": 44100},
        cfg, object(), lambda p, m: None, lambda m: None,
    )

    assert fake.calls == [dialog_path]  # 直接喂溢出 wav，无二次临时拷贝


def test_transcribe_single_loads_spk_paths_for_overlap(monkeypatch, registry, tmp_path):
    fake = _RecordingTranscribe()
    monkeypatch.setattr(runner, "transcribe", fake)

    dialog_path = str(tmp_path / "abc_dialog.wav")
    save_wav(torch.zeros(1, 4410), dialog_path, 44100)
    spk1_path = str(tmp_path / "abc_spk1.wav")
    spk2_path = str(tmp_path / "abc_spk2.wav")
    save_wav(torch.zeros(1, 16000 * 15), spk1_path, 16000)
    save_wav(torch.zeros(1, 16000 * 15), spk2_path, 16000)

    tiger_data = {
        "dialog_path": dialog_path, "dialog_sr": 44100,
        "spk1_path": spk1_path, "spk2_path": spk2_path, "spk_sr": 16000,
        "overlap_regions": [TimeSpan(start=1.0, end=3.0)],
    }
    cfg = PipelineConfig(device="cpu")
    subs, word_data = _transcribe_single(
        "video.mkv", tiger_data, cfg, object(), lambda p, m: None, lambda m: None,
    )

    assert fake.calls[0] == dialog_path  # 全轨走溢出 wav
    assert {s.speaker for s in subs if s.text.startswith("区域")} == {1, 2}
    assert len(subs) == len(word_data)


def test_run_pipeline_cleans_spill_dir_on_cancel(monkeypatch):
    seen = {}

    def fake_impl(cfg, log, progress, spill_dir):
        seen["dir"] = spill_dir
        seen["existed"] = spill_dir is not None and os.path.isdir(spill_dir)
        raise CancelledError("任务已取消")

    monkeypatch.setattr(runner, "_run_pipeline_impl", fake_impl)
    cfg = PipelineConfig(media_files=["x.mkv"], tiger_mode="denoise", device="cpu")
    with pytest.raises(CancelledError):
        run_pipeline(cfg, lambda m: None, lambda p, m: None)

    assert seen["existed"]  # 运行中目录真实存在
    assert not os.path.isdir(seen["dir"])  # 取消后 finally 已清除


def test_run_pipeline_no_spill_dir_without_tiger(monkeypatch):
    seen = {}

    def fake_impl(cfg, log, progress, spill_dir):
        seen["dir"] = spill_dir

    monkeypatch.setattr(runner, "_run_pipeline_impl", fake_impl)
    cfg = PipelineConfig(media_files=["x.mkv"], tiger_mode=None, device="cpu")
    run_pipeline(cfg, lambda m: None, lambda p, m: None)

    assert seen["dir"] is None  # 无 TIGER 阶段不建目录，行为同旧版
