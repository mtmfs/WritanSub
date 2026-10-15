"""T41 disk cache. Only paths/metadata cross this boundary; no tensor imports.

Entries are immutable generations. Short root locks serialize metadata changes;
OS-locked session leases protect files until their last pipeline consumer exits.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from importlib.metadata import version

from writansub import paths

_ENTRY = re.compile(r"[0-9a-f]{64}\.[0-9a-f]{32}\Z")
_SESSION = re.compile(r"[0-9a-f]{32}\Z")
_TRACKS = {"dnr": {"dialog": 44100}, "speech": {"spk1": 16000, "spk2": 16000}}


@dataclass(frozen=True)
class PreprocessCacheOptions:
    enabled: bool = True
    directory: str | None = None
    max_bytes: int = 5_000_000_000

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("preprocess_cache 必须为布尔值")
        if not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("预处理缓存容量必须大于零")
        if self.directory is not None and (not isinstance(self.directory, str) or not self.directory.strip()):
            raise ValueError("预处理缓存目录必须为非空路径")


def capacity_bytes(gb) -> int:
    if isinstance(gb, bool):
        raise ValueError("预处理缓存容量必须为有限正数")
    value = float(gb)
    if not math.isfinite(value * 1_000_000_000) or value <= 0 or value * 1_000_000_000 < 1:
        raise ValueError("预处理缓存容量必须为有限正数且至少 1 字节")
    return int(value * 1_000_000_000)


def media_fingerprint(media: str) -> dict:
    path = os.path.normcase(os.path.realpath(os.path.abspath(media)))
    stat = os.stat(path)
    return {"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def algorithm_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "vendor" / "tiger").glob("*.py"))
    files += [root / "preprocess" / "core.py", root / "preprocess" / "identity.py"]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return {"code": digest.hexdigest(), "torch": version("torch"),
            "torchaudio": version("torchaudio"), "format": "pcm16-mono-v1"}


def cache_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FileLock:
    """A kernel-owned lock: process death releases it, including on Windows."""
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def acquire(self, timeout: float = 0) -> bool:
        self.file = open(self.path, "a+b")
        if os.fstat(self.file.fileno()).st_size == 0:
            self.file.write(b"\0")
            self.file.flush()
        deadline = time.monotonic() + timeout
        while True:
            self.file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    self.file.close()
                    self.file = None
                    return False
                time.sleep(0.02)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        if not self.acquire(timeout=2):
            raise OSError("预处理缓存锁等待超时")
        return self

    def __exit__(self, *exc):
        self.close()


def _write_json(path: Path, data: dict):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wav_spec(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("缓存音轨不能为符号链接")
    size = path.stat().st_size
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        channels, width, sr = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if channels != 1 or width != 2 or frames < 1 or wav.getcomptype() != "NONE":
            raise ValueError("无效的缓存音轨格式")
        # Seek to the final frame rather than reading the full (possibly GB) WAV.
        wav.setpos(frames - 1)
        if len(wav.readframes(1)) != channels * width:
            raise ValueError("缓存音轨被截断")
    return {"size": size, "frames": frames, "channels": channels, "width": width, "sr": sr}


class CacheSession:
    def __init__(self, options: PreprocessCacheOptions, log=None):
        self.options = options
        self.log = log or (lambda msg: None)
        self.root = Path(options.directory or os.path.join(paths.CACHE_DIR, "preprocess"))
        self.id = uuid.uuid4().hex
        self.lease = None
        self.pins: set[str] = set()
        self.failed = False
        self.algorithm = None

    @property
    def enabled(self):
        return self.options.enabled and not self.failed

    def __enter__(self):
        return self  # Lazy: help, disabled cache and non-TIGER work create nothing.

    def __exit__(self, *exc):
        self.close()

    def _warn(self, exc):
        if not self.failed:
            self.log(f"[缓存] 不可用，回退任务临时文件: {exc}")
        self.failed = True

    def _lock(self):
        return FileLock(self.root / "cache.lock")

    def _start(self):
        if self.lease is not None:
            return
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock():
            for name in ("entries", "leases", "staging"):
                directory = self.root / name
                if directory.is_symlink() or directory.resolve().parent != self.root:
                    raise OSError("缓存子目录不能指向外部路径")
                directory.mkdir(exist_ok=True)
            lease = FileLock(self.root / "leases" / f"{self.id}.lock")
            if not lease.acquire():
                raise OSError("无法建立缓存会话")
            self.lease = lease
            self._save_pins()
            self._maintain()

    def identity(self, media: str, stage: str, model: dict, device: str, upstream: dict | None = None):
        if not self.enabled:
            return None
        try:
            if self.algorithm is None:
                self.algorithm = algorithm_identity()
            return {"schema": 1, "source": media_fingerprint(media), "stage": stage,
                    "model": model, "device": device, "algorithm": self.algorithm,
                    "upstream": cache_key(upstream) if upstream else None}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._warn(exc)
            return None

    def _save_pins(self):
        _write_json(self.root / "leases" / f"{self.id}.json", {"entries": sorted(self.pins)})

    def _pin(self, name):
        self.pins.add(name)
        self._save_pins()

    def _remove(self, directory: Path):
        # Only generated entry/session directories, never metadata-supplied paths.
        if directory.is_symlink() or directory.resolve().parent != directory.parent.resolve():
            raise OSError("拒绝清理缓存目录外的路径")
        if directory.parent not in (self.root / "entries", self.root / "staging"):
            raise OSError("拒绝清理非缓存目录")
        shutil.rmtree(directory)

    def _valid(self, directory, identity):
        try:
            if directory.is_symlink():
                return None
            metadata = json.loads((directory / "entry.json").read_text(encoding="utf-8"))
            if metadata["identity"] != identity:
                return None
            expected = _TRACKS[identity["stage"]]
            if set(metadata["tracks"]) != set(expected):
                return None
            for track, sr in expected.items():
                spec = _wav_spec(directory / f"{track}.wav")
                if spec != metadata["tracks"][track] or spec["sr"] != sr:
                    return None
            if not all(isinstance(metadata[k], int) for k in ("created_ns", "used_ns")):
                return None
            return metadata
        except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError):
            return None

    def _lookup(self, identity):
        key = cache_key(identity)
        for directory in sorted((self.root / "entries").glob(key + ".*"), reverse=True):
            if not _ENTRY.fullmatch(directory.name):
                continue
            metadata = self._valid(directory, identity)
            if metadata is None:
                continue
            self._pin(directory.name)
            metadata["used_ns"] = time.time_ns()
            _write_json(directory / "entry.json", metadata)
            return {track: str(directory / f"{track}.wav") for track in metadata["tracks"]}
        return None

    def lookup(self, identity):
        if not self.enabled or identity is None:
            return None
        try:
            self._start()
            with self._lock():
                result = self._lookup(identity)
            self.log(f"[缓存] {identity['stage']} {'命中' if result else '未命中'}: {identity['source']['path']}")
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._warn(exc)
            return None

    def publish(self, identity, tracks: dict[str, str]):
        if not self.enabled or identity is None:
            return tracks
        generation = None
        try:
            if media_fingerprint(identity["source"]["path"]) != identity["source"]:
                self.log("[缓存] 源文件已变化，本次结果不写入缓存")
                return tracks
            self._start()
            # This directory belongs to a live OS-locked session, including while copying.
            stage = self.root / "staging" / self.id
            stage.mkdir(exist_ok=True)
            generation = stage / uuid.uuid4().hex
            generation.mkdir()
            specs = {}
            for track, source in tracks.items():
                if track not in _TRACKS[identity["stage"]]:
                    raise ValueError("非缓存音轨")
                target = generation / f"{track}.wav"
                shutil.copyfile(source, target)
                specs[track] = _wav_spec(target)
                if specs[track]["sr"] != _TRACKS[identity["stage"]][track]:
                    raise ValueError("缓存音轨采样率错误")
            if set(specs) != set(_TRACKS[identity["stage"]]):
                raise ValueError("缓存音轨不完整")
            now = time.time_ns()
            _write_json(generation / "entry.json", {"identity": identity, "tracks": specs,
                                                     "created_ns": now, "used_ns": now})
            with self._lock():
                if media_fingerprint(identity["source"]["path"]) != identity["source"]:
                    self.log("[缓存] 源文件已变化，本次结果不发布")
                    return tracks
                result = self._lookup(identity)
                if result is None:
                    name = cache_key(identity) + "." + uuid.uuid4().hex
                    self._pin(name)  # Protect before publishing; failure cannot expose a usable unpinned result.
                    destination = self.root / "entries" / name
                    os.replace(generation, destination)
                    result = {track: str(destination / f"{track}.wav") for track in tracks}
                self._maintain()
            return result
        except (OSError, ValueError, KeyError, TypeError, wave.Error, EOFError) as exc:
            self._warn(exc)
            return tracks
        finally:
            if generation is not None and generation.exists():
                try:
                    owner = self.root / "staging" / self.id
                    if generation.is_symlink() or generation.resolve().parent != owner.resolve():
                        raise OSError("拒绝清理暂存目录外的路径")
                    shutil.rmtree(generation)
                except OSError as exc:
                    self._warn(exc)

    def _maintain(self):
        """Called with the root lock held. A lease lock, never a PID, proves liveness."""
        active = set()
        active_sessions = set()
        for path in (self.root / "leases").glob("*.lock"):
            session_id = path.stem
            if not _SESSION.fullmatch(session_id):
                continue
            probe = FileLock(path)
            alive = session_id == self.id and self.lease is not None
            if not alive:
                alive = not probe.acquire()
                probe.close()
            metadata = path.with_suffix(".json")
            if alive:
                active_sessions.add(session_id)
                # If a live lease is unreadable, abort eviction rather than risk its files.
                data = json.loads(metadata.read_text(encoding="utf-8"))
                entries = data["entries"]
                if not isinstance(entries, list) or any(not isinstance(n, str) or not _ENTRY.fullmatch(n) for n in entries):
                    raise ValueError("无效的活跃缓存租约")
                active.update(entries)
            else:
                metadata.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
        for directory in (self.root / "staging").iterdir():
            if _SESSION.fullmatch(directory.name) and directory.name not in active_sessions:
                self._remove(directory)
        candidates = []
        total = 0
        for directory in (self.root / "entries").iterdir():
            if not _ENTRY.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
                continue
            size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file() and not p.is_symlink())
            total += size
            try:
                data = json.loads((directory / "entry.json").read_text(encoding="utf-8"))
                valid = self._valid(directory, data["identity"])
                used = int(data["used_ns"]) if valid else 0
            except (OSError, ValueError, KeyError, TypeError):
                used = 0
            candidates.append((used, directory.name, size, directory))
        for used, name, size, directory in sorted(candidates):
            if name in active or (total <= self.options.max_bytes and used != 0):
                continue
            self._remove(directory)
            total -= size
            self.log(f"[缓存] 清理 {name[:12]} ({size} 字节)")
        if total > self.options.max_bytes:
            self.log(f"[缓存] 活跃音轨暂超容量: {total}/{self.options.max_bytes} 字节，使用结束后清理")

    def close(self):
        if self.lease is None:
            return
        try:
            with self._lock():
                self.lease.close()
                self.lease = None
                self._maintain()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._warn(exc)
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None
