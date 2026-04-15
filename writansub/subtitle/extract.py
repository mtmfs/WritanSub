import json

from writansub.bridge import _get_ffmpeg, _get_ffprobe, ResourceRegistry
from writansub.subtitle.srt_io import parse_srt_string
from writansub.types import Sub

# WritanSub lang code → ISO 639-2 (ffprobe 使用)
_LANG_MAP = {
    "ja": ["jpn", "ja"],
    "zh": ["chi", "zho", "zh"],
    "en": ["eng", "en"],
    "ko": ["kor", "ko"],
    "fr": ["fre", "fra", "fr"],
    "de": ["ger", "deu", "de"],
    "es": ["spa", "es"],
    "ru": ["rus", "ru"],
    "vi": ["vie", "vi"],
}


def probe_subtitle_tracks(media: str) -> list[dict]:
    """列出媒体文件中的字幕轨信息。

    返回 [{"index": 0, "language": "jpn", "codec": "ass", "title": "日本語"}, ...]
    index 是字幕流的相对索引（第几条字幕轨），用于 ffmpeg -map 0:s:<index>。
    """
    cmd = [
        _get_ffprobe(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        media,
    ]
    reg = ResourceRegistry.instance()
    proc = reg.run_subprocess(cmd, timeout=30)
    if proc.returncode != 0:
        return []

    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []

    tracks = []
    for i, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags", {})
        tracks.append({
            "index": i,
            "language": tags.get("language", ""),
            "codec": stream.get("codec_name", ""),
            "title": tags.get("title", ""),
        })
    return tracks


def select_track(tracks: list[dict], lang: str) -> int | None:
    if not tracks:
        return None

    codes = _LANG_MAP.get(lang, [])
    for track in tracks:
        track_lang = track["language"].lower()
        if any(track_lang == c or track_lang.startswith(c) for c in codes):
            return track["index"]

    # 没匹配到语言，回退第一条
    return tracks[0]["index"]


def extract_subtitle(media: str, track_index: int) -> list[Sub]:
    """用 ffmpeg 提取指定字幕轨，解析为 Sub 列表。

    仅提取时间轴，文本内容不保证准确（可能是翻译语言）。
    """
    cmd = [
        _get_ffmpeg(),
        "-i", media,
        "-map", f"0:s:{track_index}",
        "-f", "srt",
        "-loglevel", "error",
        "-",
    ]
    reg = ResourceRegistry.instance()
    proc = reg.run_subprocess(cmd, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 字幕提取失败: {proc.stderr.decode(errors='replace')}"
        )

    srt_text = proc.stdout.decode("utf-8", errors="replace")
    return [s for s in parse_srt_string(srt_text) if s.text]
