import os
import socket


def _can_reach(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_hf_mirror() -> None:
    """若用户未设置 HF_ENDPOINT，且 huggingface.co 不可达，则自动切换到镜像站。"""
    if os.environ.get("HF_ENDPOINT"):
        return
    if not _can_reach("huggingface.co"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
