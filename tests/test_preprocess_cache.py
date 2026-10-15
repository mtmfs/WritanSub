"""T41 storage tests: real WAV IO, real subprocesses and kernel-owned leases."""
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from writansub.preprocess import cache
from writansub.preprocess.cache import CacheSession, PreprocessCacheOptions, media_fingerprint


def wav_file(path, sr=44100):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(b"\x01\x00" * 400)
    return str(path)


@pytest.fixture
def inputs(tmp_path):
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")
    identity = {"schema": 1, "source": media_fingerprint(str(media)), "stage": "dnr",
                "model": {"repo": "fake", "revision": "a" * 40}, "device": "cpu", "algorithm": "v1"}
    return identity, {"dialog": wav_file(tmp_path / "dialog.wav")}


def options(tmp_path, max_bytes=5_000_000_000, enabled=True):
    return PreprocessCacheOptions(enabled, str(tmp_path / "cache"), max_bytes)


def test_cross_session_reuse_and_close_preserves_complete_entry(tmp_path, inputs):
    identity, tracks = inputs
    with CacheSession(options(tmp_path)) as first:
        assert first.lookup(identity) is None
        result = first.publish(identity, tracks)
        assert result != tracks
        assert Path(result["dialog"]).read_bytes() == Path(tracks["dialog"]).read_bytes()
    with CacheSession(options(tmp_path)) as second:
        assert second.lookup(identity) == result
    assert Path(result["dialog"]).exists()
    assert not list((tmp_path / "cache" / "staging").iterdir())
    assert not list((tmp_path / "cache" / "leases").iterdir())


@pytest.mark.parametrize("change", ["source", "model", "algorithm", "device"])
def test_identity_changes_miss(tmp_path, inputs, change):
    identity, tracks = inputs
    with CacheSession(options(tmp_path)) as session:
        session.publish(identity, tracks)
        changed = dict(identity, **{change: "changed"})
        if change == "source":
            changed["source"] = dict(identity["source"], mtime_ns=0)
        assert session.lookup(changed) is None


@pytest.mark.parametrize("damage", ["json", "missing", "truncated", "wrong_header"])
def test_corrupt_entry_is_a_miss_and_can_be_rebuilt(tmp_path, inputs, damage):
    identity, tracks = inputs
    with CacheSession(options(tmp_path)) as session:
        result = session.publish(identity, tracks)
        path = Path(result["dialog"])
        if damage == "json":
            (path.parent / "entry.json").write_text("broken")
        elif damage == "missing":
            path.unlink()
        elif damage == "truncated":
            path.write_bytes(path.read_bytes()[:-2])
        else:
            wav_file(path, 16000)
        assert session.lookup(identity) is None
        rebuilt = session.publish(identity, tracks)
        assert rebuilt != result  # Never overwrite a generation leased by a reader.
        assert session.lookup(identity) == rebuilt


def test_active_batch_can_exceed_limit_then_is_cleaned(tmp_path, inputs):
    identity, tracks = inputs
    with CacheSession(options(tmp_path, max_bytes=1)) as session:
        a = session.publish(identity, tracks)
        b = session.publish(dict(identity, algorithm="v2"), tracks)
        assert Path(a["dialog"]).exists() and Path(b["dialog"]).exists()
    assert not Path(a["dialog"]).exists() and not Path(b["dialog"]).exists()


def test_lru_uses_last_access_not_creation(tmp_path, inputs):
    identity, tracks = inputs
    with CacheSession(options(tmp_path)) as session:
        older = session.publish(identity, tracks)
        newer = session.publish(dict(identity, algorithm="new"), tracks)
        session.lookup(identity)
    size = sum(p.stat().st_size for p in Path(older["dialog"]).parent.iterdir())
    with CacheSession(options(tmp_path, max_bytes=size + 10)) as session:
        assert session.lookup(identity) == older
        assert not Path(newer["dialog"]).exists()


def test_source_change_blocks_publish(tmp_path, inputs):
    identity, tracks = inputs
    Path(identity["source"]["path"]).write_bytes(b"changed")
    with CacheSession(options(tmp_path)) as session:
        assert session.publish(identity, tracks) == tracks
    assert not (tmp_path / "cache").exists()


def test_disabled_cache_does_not_touch_disk(tmp_path, inputs):
    identity, tracks = inputs
    with CacheSession(options(tmp_path, enabled=False)) as session:
        assert session.lookup(identity) is None
        assert session.publish(identity, tracks) == tracks
    assert not (tmp_path / "cache").exists()


def test_bad_cache_directory_falls_back_once(tmp_path, inputs):
    identity, tracks = inputs
    (tmp_path / "cache").write_text("not a directory")
    logs = []
    with CacheSession(options(tmp_path), logs.append) as session:
        assert session.lookup(identity) is None
        assert session.publish(identity, tracks) == tracks
    assert len(logs) == 1 and "回退" in logs[0]


def test_invalid_cache_path_falls_back_lazily(tmp_path, inputs):
    opts = PreprocessCacheOptions(directory=str(tmp_path / "cache") + "\0")
    with CacheSession(opts) as session:
        assert session.lookup(inputs[0]) is None
        assert session.failed


def test_failed_publish_is_not_visible(tmp_path, inputs, monkeypatch):
    identity, tracks = inputs
    original = os.replace
    def fail(source, destination):
        if Path(destination).parent.name == "entries":
            raise OSError("simulated disk failure")
        return original(source, destination)
    with CacheSession(options(tmp_path)) as session:
        monkeypatch.setattr(os, "replace", fail)
        assert session.publish(identity, tracks) == tracks
    with CacheSession(options(tmp_path)) as session:
        assert session.lookup(identity) is None
    assert not list((tmp_path / "cache" / "staging").iterdir())


def test_pair_is_published_together(tmp_path, inputs):
    identity, _ = inputs
    identity = dict(identity, stage="speech")
    tracks = {"spk1": wav_file(tmp_path / "a.wav", 16000), "spk2": wav_file(tmp_path / "b.wav", 16000)}
    with CacheSession(options(tmp_path)) as session:
        result = session.publish(identity, tracks)
        assert set(session.lookup(identity)) == {"spk1", "spk2"}
        assert Path(result["spk1"]).parent == Path(result["spk2"]).parent
    with CacheSession(options(tmp_path)) as session:
        assert session.publish(dict(identity, algorithm="other"), {"spk1": tracks["spk1"]}) == {"spk1": tracks["spk1"]}


_CHILD = """
import json, sys
from pathlib import Path
from writansub.preprocess.cache import CacheSession, PreprocessCacheOptions
root, identity, tracks, mode = sys.argv[1:]
with CacheSession(PreprocessCacheOptions(directory=root)) as session:
    if mode == 'staging':
        session._start()
        directory = session.root / 'staging' / session.id
        directory.mkdir()
        (directory / 'partial.wav').write_bytes(b'partial')
        result = {'stage': str(directory)}
    else:
        result = session.publish(json.loads(identity), json.loads(tracks))
    print(json.dumps(result), flush=True)
    sys.stdin.readline()
"""


def child_process(tmp_path, inputs, mode="publish"):
    identity, tracks = inputs
    return subprocess.Popen([sys.executable, "-c", _CHILD, str(tmp_path / "cache"),
                             json.dumps(identity), json.dumps(tracks), mode],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", cwd=str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("kill", [False, True])
def test_other_process_lease_survives_eviction_and_releases(tmp_path, inputs, kill):
    identity, tracks = inputs
    child = child_process(tmp_path, inputs)
    try:
        result = json.loads(child.stdout.readline())
        with CacheSession(options(tmp_path, max_bytes=1)) as parent:
            # A concurrent writer reuses the complete entry rather than overwriting it.
            assert parent.publish(identity, tracks) == result
        assert Path(result["dialog"]).exists()  # Still used by the child.
        if kill:
            child.kill()
        else:
            child.stdin.write("done\n")
            child.stdin.flush()
        child.wait(timeout=10)
        with CacheSession(options(tmp_path, max_bytes=1)) as cleanup:
            assert cleanup.lookup(identity) is None
        assert not Path(result["dialog"]).exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_strong_kill_cleans_incomplete_staging(tmp_path, inputs):
    child = child_process(tmp_path, inputs, "staging")
    try:
        result = json.loads(child.stdout.readline())
        child.kill()
        child.wait(timeout=10)
        with CacheSession(options(tmp_path)) as session:
            assert session.lookup(inputs[0]) is None
        assert not Path(result["stage"]).exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_cache_module_does_not_import_torch():
    subprocess.run([sys.executable, "-c",
                    "import sys; import writansub.preprocess.cache; assert 'torch' not in sys.modules"],
                   check=True, timeout=20)


def test_root_lock_failure_falls_back(tmp_path, inputs):
    root = tmp_path / "cache"
    root.mkdir()
    with cache.FileLock(root / "cache.lock"):
        with CacheSession(options(tmp_path)) as session:
            assert session.lookup(inputs[0]) is None
            assert session.failed


def test_eviction_never_deletes_neighboring_model_or_user_output(tmp_path, inputs):
    snapshot = tmp_path / "models--example" / "snapshots" / "model.bin"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"weights")
    with CacheSession(options(tmp_path, max_bytes=1)) as session:
        result = session.publish(*inputs)
    assert not Path(result["dialog"]).exists()
    assert snapshot.read_bytes() == b"weights"
    assert Path(inputs[1]["dialog"]).is_file()
