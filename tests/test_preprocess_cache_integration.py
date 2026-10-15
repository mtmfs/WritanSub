"""Real disk IO around fake engines: never load or download model weights."""
from pathlib import Path

import pytest
import torch

from writansub.bridge import CancelledError
from writansub.preprocess import core
from writansub.preprocess.cache import CacheSession, PreprocessCacheOptions
from writansub.preprocess.identity import DNR_REPO, local_model_identity


@pytest.fixture
def processing(tmp_path, monkeypatch):
    calls = {"decode": 0, "dialog": 0, "effects": 0, "music": 0, "speech": 0, "vad": []}
    class Registry:
        def checkpoint(self):
            pass
        def decode_audio(self, *args, **kwargs):
            calls["decode"] += 1
            return torch.full((1, 4410), 0.12345), 44100
    reg = Registry()
    monkeypatch.setattr(core.ResourceRegistry, "instance", classmethod(lambda cls: reg))
    def model_id(repo, *args):
        return {"repo": repo, "revision": "a" * 40}
    monkeypatch.setattr(core, "local_model_identity", model_id)
    def dnr(waveform, sr, **kwargs):
        names = kwargs.get("tracks") or (("dialog", "effects", "music") if kwargs.get("full_tracks", True) else ("dialog",))
        if kwargs.get("model_identity") == {}:
            kwargs["model_identity"].update(model_id(DNR_REPO))
        result = {}
        for name in names:
            calls[name] += 1
            result[name] = torch.full((1, 4410), 0.12345)
        return tuple(result.get(name) for name in ("dialog", "effects", "music"))
    def speech(waveform, sr, **kwargs):
        calls["speech"] += 1
        return torch.full((1, 1600), 0.32123), torch.full((1, 1600), 0.45678)
    def vad(a, b, **kwargs):
        calls["vad"].append((a.clone(), b.clone()))
        return [core.TimeSpan(0.1, 0.2)], 0.25
    monkeypatch.setattr(core, "separate_dnr", dnr)
    monkeypatch.setattr(core, "separate_dnr_demucs", dnr)
    monkeypatch.setattr(core, "separate_speakers", speech)
    monkeypatch.setattr(core, "detect_overlaps", vad)
    media = tmp_path / "video.mp4"
    media.write_bytes(b"media")
    spill = tmp_path / "spill"
    spill.mkdir()
    options = PreprocessCacheOptions(directory=str(tmp_path / "persistent"))
    def run(separate=False, save=False, model="tiger-dnr", enabled=True):
        opts = options if enabled else PreprocessCacheOptions(False, options.directory)
        with CacheSession(opts) as session:
            data = core.run_dnr_batch([str(media)], str(spill), save_intermediate=save,
                                      mss_model=model, cache_session=session)
            if separate:
                core.run_speech_batch(data, str(spill), save_intermediate=save, cache_session=session)
            return data[str(media)]
    return calls, run, media, options, reg


def test_denoise_then_separate_reuses_upstream_and_repeats_vad(processing):
    calls, run, media, options, reg = processing
    first = run()
    second = run(separate=True)
    cold = {name: Path(second[name]).read_bytes() for name in ("dialog_path", "spk1_path", "spk2_path")}
    third = run(separate=True)
    assert calls["decode"] == calls["dialog"] == calls["speech"] == 1
    assert first["dialog_path"] == second["dialog_path"] == third["dialog_path"]
    assert third["overlap_regions"] == second["overlap_regions"]
    assert len(calls["vad"]) == 2
    for a, b in zip(*calls["vad"]):
        assert torch.equal(a, b)
    assert cold == {name: Path(third[name]).read_bytes() for name in cold}


def test_export_hit_only_computes_effects_and_music(processing):
    calls, run, media, options, reg = processing
    run(separate=True)
    result = run(separate=True, save=True)
    assert calls["dialog"] == calls["speech"] == 1
    assert calls["effects"] == calls["music"] == 1
    for name in ("dialog", "effects", "music", "spk1", "spk2"):
        assert media.with_name(f"video_{name}.wav").is_file()
    assert media.with_name("video_dialog.wav").read_bytes() == Path(result["dialog_path"]).read_bytes()
    assert not list(Path(options.directory).rglob("effects.wav"))
    assert not list(Path(options.directory).rglob("music.wav"))


def test_disabled_cache_recomputes_and_leaves_no_persistent_directory(processing):
    calls, run, media, options, reg = processing
    run(enabled=False)
    run(enabled=False)
    assert calls["dialog"] == calls["decode"] == 2
    assert not Path(options.directory).exists()


def test_unwritable_cache_keeps_pipeline_audio_in_spill(processing):
    calls, run, media, options, reg = processing
    Path(options.directory).write_text("not a directory")
    result = run(separate=True)
    assert Path(result["dialog_path"]).is_file()
    assert Path(result["spk1_path"]).is_file()
    assert not Path(result["dialog_path"]).is_relative_to(Path(options.directory))
    assert calls["dialog"] == calls["speech"] == 1


def test_demucs_does_not_use_tiger_cache(processing):
    calls, run, media, options, reg = processing
    run(model="htdemucs_ft", separate=True)
    run(model="htdemucs_ft", separate=True)
    assert calls["dialog"] == calls["speech"] == 2
    assert not Path(options.directory).exists()
    assert torch.equal(calls["vad"][0][0], torch.full((1, 1600), 0.32123))


def test_source_and_model_revision_change_invalidate(processing, monkeypatch):
    calls, run, media, options, reg = processing
    run()
    media.write_bytes(b"new media")
    run()
    assert calls["dialog"] == 2
    monkeypatch.setattr(core, "local_model_identity", lambda repo, *a: {"repo": repo, "revision": "b" * 40})
    run()
    assert calls["dialog"] == 3


def test_cold_model_identity_is_captured_after_inference(processing, monkeypatch):
    calls, run, media, options, reg = processing
    monkeypatch.setattr(core, "local_model_identity", lambda *a: None)
    result = run()
    assert result["cache_identity"]["model"]["revision"] == "a" * 40
    assert Path(result["dialog_path"]).is_relative_to(Path(options.directory))


def test_source_modified_during_inference_not_cached(processing, monkeypatch):
    calls, run, media, options, reg = processing
    original = core.separate_dnr
    def changing(*args, **kwargs):
        output = original(*args, **kwargs)
        media.write_bytes(b"replaced while processing")
        return output
    monkeypatch.setattr(core, "separate_dnr", changing)
    result = run()
    assert not Path(result["dialog_path"]).is_relative_to(Path(options.directory))
    assert not list(Path(options.directory).rglob("entry.json"))


@pytest.mark.parametrize("error", [CancelledError("cancel"), RuntimeError("inference failed")])
def test_interrupted_speech_preserves_completed_dnr_only(processing, monkeypatch, error):
    calls, run, media, options, reg = processing
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(core, "separate_speakers", broken)
    with pytest.raises(type(error)):
        run(separate=True)
    assert len(list(Path(options.directory).rglob("entry.json"))) == 1
    assert not list(Path(options.directory).rglob("spk1.wav"))
    run()
    assert calls["dialog"] == 1


def test_pipeline_session_keeps_audio_through_final_phase(tmp_path):
    # Separate from GPU runner tests: exercise the real lifetime wrapper.
    from unittest.mock import patch
    from writansub.pipeline.runner import PipelineConfig, run_pipeline
    media = tmp_path / "media.mp4"
    media.write_bytes(b"media")
    seen = {}
    def pipeline(cfg, log, progress, spill, cache_session=None):
        wav = str(Path(spill) / "dialog.wav")
        core.save_wav(torch.zeros(1, 400), wav, 44100)
        identity = cache_session.identity(str(media), "dnr", {"revision": "a"}, "cpu")
        seen.update(cache_session.publish(identity, {"dialog": wav}))
        assert Path(seen["dialog"]).exists()
        # End of pipeline, after every consumer: file is still present despite 1-byte limit.
        assert cache_session.lookup(identity) == seen
    cfg = PipelineConfig(tiger_mode="denoise", preprocess_cache=PreprocessCacheOptions(max_bytes=1))
    with patch("writansub.pipeline.runner._run_pipeline_impl", pipeline):
        run_pipeline(cfg, lambda msg: None, lambda *args: None)
    assert not Path(seen["dialog"]).exists()


def test_snapshot_identity_local_only_and_requires_complete_files(tmp_path):
    root = tmp_path / "models--JusperLee--TIGER-DnR"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("a" * 40)
    snapshot = root / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert local_model_identity(DNR_REPO, str(tmp_path)) is None
    (snapshot / "model.safetensors").write_bytes(b"fake weight")
    assert local_model_identity(DNR_REPO, str(tmp_path)) == {"repo": DNR_REPO, "revision": "a" * 40}


def test_loader_passes_pinned_revision_and_device_errors_still_propagate(monkeypatch, tmp_path):
    calls = []
    class Model:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            calls.append(kwargs)
            return cls()
        def to(self, device):
            raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(core, "_hf_model_cached", lambda *a: True)
    with pytest.raises(RuntimeError, match="out of memory"):
        core._from_pretrained_cached(Model, DNR_REPO, str(tmp_path), "cuda", revision="b" * 40)
    assert calls == [{"cache_dir": str(tmp_path), "local_files_only": True, "revision": "b" * 40}]
