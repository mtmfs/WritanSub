import os
import urllib.error
import urllib.request


def _can_reach(url: str, timeout: float = 5.0) -> bool:
    """经系统代理的真实 HTTPS 探测。

    urllib 默认读取系统代理（Windows 注册表/环境变量）并完整 TLS 握手：
    Clash 场景不再误判不可达（旧裸 socket 不走代理），SNI 阶段被重置也
    不再误判可达（旧探测 TCP 通即真）。收到任何 HTTP 响应（含 4xx/5xx）
    即视为可达。
    """
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def setup_hf_mirror() -> None:
    """若用户未设置 HF_ENDPOINT，且 huggingface.co 不可达，则自动切换到镜像站。"""
    if os.environ.get("HF_ENDPOINT"):
        return
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    if not _can_reach("https://huggingface.co"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
