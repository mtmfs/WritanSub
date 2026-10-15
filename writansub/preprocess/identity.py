"""Resolve an immutable local HF revision without loading a model or contacting HF."""
from pathlib import Path
import re

from writansub import paths

DNR_REPO = "JusperLee/TIGER-DnR"
SPEECH_REPO = "JusperLee/TIGER-speech"


def local_model_identity(repo: str, cache_dir: str = "") -> dict | None:
    root = Path(cache_dir or paths.CACHE_DIR) / ("models--" + repo.replace("/", "--"))
    try:
        revision = (root / "refs" / "main").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            return None
        snapshot = root / "snapshots" / revision
        if not (snapshot / "config.json").is_file():
            return None
        if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            return None
        return {"repo": repo, "revision": revision}
    except OSError:
        return None


def resolve_model_identity(repo: str, cache_dir: str = "") -> dict:
    identity = local_model_identity(repo, cache_dir)
    if identity:
        return identity
    # Only the model loader calls this network-capable path, never a cache lookup.
    from huggingface_hub import hf_hub_download
    config = Path(hf_hub_download(repo, "config.json", cache_dir=cache_dir or paths.CACHE_DIR))
    return {"repo": repo, "revision": config.parent.name}
