"""流水线编排：定义步骤基类、具体步骤、调度器"""

import abc
import os
from typing import Any, Callable, Dict, List, Optional

from writansub.types import Sub
from writansub.subtitle.srt_io import parse_srt, write_srt
from writansub.subtitle.review import mark_low_align_in_review
from writansub.transcribe.core import transcribe, transcribe_to_srt
from writansub.align.core import load_audio, run_alignment, post_process
from writansub.translate.core import translate_srt


class PipelineStep(abc.ABC):
    """流水线步骤基类"""

    name: str = ""              # 唯一标识
    display_name: str = ""      # 界面显示名

    @abc.abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        log_callback: Callable[[str], None],
        cancelled: Callable[[], bool],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        ...

    def get_intermediate_files(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> List[str]:
        return []


class TigerStep(PipelineStep):
    """TIGER 音频分离步骤（前处理，可选）"""
    name = "tiger"
    display_name = "TIGER 音频分离"

    def __init__(self, mode: str = "denoise", device: str = "cpu",
                 cache_dir: str = "cache", save_intermediate: bool = False):
        self.mode = mode
        self.device = device
        self.cache_dir = cache_dir
        self.save_intermediate = save_intermediate

    def execute(self, inputs, log_callback, cancelled, progress_callback=None):
        from writansub.preprocess.core import run_tiger_separation

        if cancelled():
            return {}

        media_file = inputs["media_file"]
        result = run_tiger_separation(
            media_file,
            mode=self.mode,
            device=self.device,
            cache_dir=self.cache_dir,
            save_intermediate=self.save_intermediate,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        outputs = {
            "media_file": media_file,
            "dialog_wav": result["dialog_wav"],
            "dialog_sr": result["dialog_sr"],
        }

        if self.mode == "separate":
            outputs["separated_tracks"] = (result["spk1_wav"], result["spk2_wav"])
            outputs["spk_sr"] = result["spk_sr"]
            outputs["overlap_regions"] = result["overlap_regions"]
            outputs["overlap_ratio"] = result["overlap_ratio"]

        return outputs


class WhisperStep(PipelineStep):
    """Whisper 语音识别步骤"""
    name = "whisper"
    display_name = "Whisper 语音识别"

    def __init__(self, lang: str = "ja", device: str = "cuda",
                 word_conf_threshold: float = 0.50,
                 condition_on_previous_text: bool = True):
        self.lang = lang
        self.device = device
        self.word_conf_threshold = word_conf_threshold
        self.condition_on_previous_text = condition_on_previous_text

    def execute(self, inputs, log_callback, cancelled, progress_callback=None):
        device = self.device
        if device == "cuda":
            try:
                import ctranslate2
                ctranslate2.get_supported_compute_types("cuda")
            except Exception:
                log_callback("CUDA 不可用，Whisper 回退到 CPU")
                device = "cpu"

        media_file = inputs["media_file"]
        overlap_regions = inputs.get("overlap_regions")
        separated_tracks = inputs.get("separated_tracks")

        if overlap_regions and separated_tracks:
            srt_path = self._run_with_overlap(
                media_file, overlap_regions, separated_tracks,
                inputs.get("spk_sr", 16000),
                device, log_callback, cancelled, progress_callback,
            )
        else:
            srt_path = transcribe_to_srt(
                media_file,
                lang=self.lang,
                device=device,
                log_callback=log_callback,
                progress_callback=progress_callback,
                word_conf_threshold=self.word_conf_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
            )

        return {"srt": srt_path, "media_file": media_file}

    def _run_with_overlap(
        self, media_file, overlap_regions, separated_tracks, spk_sr,
        device, log_callback, cancelled, progress_callback,
    ) -> str:
        import tempfile
        from writansub.preprocess.core import save_wav
        from writansub.subtitle.review import generate_review, write_review_files

        log_callback("完整音频 Whisper 识别...")

        full_subs, full_word_data = transcribe(
            media_file,
            lang=self.lang,
            device=device,
            log_callback=log_callback,
            progress_callback=progress_callback,
            condition_on_previous_text=self.condition_on_previous_text,
        )

        if cancelled():
            return ""

        if not overlap_regions:
            return self._write_output(media_file, full_subs, full_word_data)

        log_callback(f"处理 {len(overlap_regions)} 个重叠段...")
        spk1_wav, spk2_wav = separated_tracks

        overlap_subs_map: Dict[tuple, List[Sub]] = {}
        for region in overlap_regions:
            if cancelled():
                return ""

            start_sample = int(region.start * spk_sr)
            end_sample = int(region.end * spk_sr)

            for spk_wav in [spk1_wav, spk2_wav]:
                chunk = spk_wav[:, start_sample:end_sample]
                if chunk.shape[1] < 1600:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                try:
                    save_wav(chunk, tmp_path, spk_sr)
                    local_subs, _ = transcribe(
                        tmp_path,
                        lang=self.lang,
                        device=device,
                        condition_on_previous_text=False,
                    )
                    for s in local_subs:
                        s.start += region.start
                        s.end += region.start
                    key = (region.start, region.end)
                    overlap_subs_map.setdefault(key, []).extend(local_subs)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        for sub in full_subs:
            mid = (sub.start + sub.end) / 2
            if any(region.start <= mid <= region.end for region in overlap_regions):
                sub.commented = True

        overlap_inserts = [s for subs in overlap_subs_map.values() for s in subs]
        active_subs = [s for s in full_subs if not s.commented] + overlap_inserts
        active_subs.sort(key=lambda s: s.start)
        for i, s in enumerate(active_subs, 1):
            s.index = i

        replaced_count = sum(1 for s in full_subs if s.commented)
        log_callback(
            f"合并完成: {len(active_subs)} 条有效字幕, "
            f"{replaced_count} 条被替换"
        )

        return self._write_output(media_file, active_subs, [[] for _ in active_subs])

    def _write_output(self, media_file, subs, word_data) -> str:
        from writansub.subtitle.review import generate_review, write_review_files

        srt_path = os.path.splitext(media_file)[0] + ".srt"
        write_srt(subs, srt_path)

        if self.word_conf_threshold > 0.0:
            srt_content, ass_content, low_count, total_words = generate_review(
                subs, word_data, self.word_conf_threshold,
            )
            if low_count > 0:
                base = os.path.splitext(media_file)[0]
                write_review_files(base, srt_content, ass_content)

        return srt_path

    def get_intermediate_files(self, inputs, outputs):
        return [outputs.get("srt", "")]


class ForceAlignStep(PipelineStep):
    """MMS_FA 强制打轴步骤"""
    name = "force_align"
    display_name = "MMS_FA 强制打轴"

    def __init__(self, lang: str = "ja", device: str = "cuda",
                 extend_end: float = 0.30, extend_start: float = 0.00,
                 gap_threshold: float = 0.50, min_gap: float = 0.30,
                 min_duration: float = 0.30,
                 align_conf_threshold: float = 0.50):
        self.lang = lang
        self.device = device
        self.extend_end = extend_end
        self.extend_start = extend_start
        self.gap_threshold = gap_threshold
        self.min_gap = min_gap
        self.min_duration = min_duration
        self.align_conf_threshold = align_conf_threshold

    def execute(self, inputs, log_callback, cancelled, progress_callback=None):
        import torch

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            log_callback("CUDA 不可用，回退到 CPU")
            device = "cpu"

        media_file = inputs["media_file"]
        srt_path = inputs["srt"]

        waveform = load_audio(media_file)
        subs = parse_srt(srt_path, lang=self.lang)
        log_callback(f"字幕 {len(subs)} 条，设备: {device}")

        if cancelled():
            return {}

        aligned = run_alignment(waveform, subs, device=device,
                                progress_callback=progress_callback)

        if cancelled():
            return {}

        final = post_process(
            aligned,
            extend_end=self.extend_end,
            extend_start=self.extend_start,
            gap_threshold=self.gap_threshold,
            min_gap=self.min_gap,
            min_duration=self.min_duration,
        )

        base = srt_path.rsplit('.', 1)[0]
        output_path = f"{base}_aligned.srt"
        write_srt(final, output_path)

        if self.align_conf_threshold > 0:
            low_align = {s.index for s in final if s.score < self.align_conf_threshold}
            if low_align:
                mark_low_align_in_review(base, low_align)
                log_callback(f"低置信对齐 {len(low_align)} 句，已标记")

        return {"aligned_srt": output_path}

    def get_intermediate_files(self, inputs, outputs):
        return [outputs.get("aligned_srt", "")]


class TranslateStep(PipelineStep):
    """AI 翻译步骤"""
    name = "translate"
    display_name = "AI 翻译"

    def __init__(self, target_lang: str, api_base: str, api_key: str,
                 model: str, batch_size: int = 20):
        self.target_lang = target_lang
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

    def execute(self, inputs, log_callback, cancelled, progress_callback=None):
        srt_path = inputs.get("aligned_srt") or inputs.get("srt")
        if not srt_path:
            log_callback("未找到可翻译的 SRT 文件")
            return {}

        output = translate_srt(
            srt_path,
            target_lang=self.target_lang,
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            batch_size=self.batch_size,
            log_callback=log_callback,
            progress_callback=progress_callback,
            cancelled=cancelled,
        )
        return {"translated_srt": output}

    def get_intermediate_files(self, inputs, outputs):
        return [outputs.get("translated_srt", "")]


class PipelineOrchestrator:
    """流水线编排器，顺序执行注册的步骤"""

    def __init__(self):
        self._steps: List[PipelineStep] = []
        self._cancelled = False

    def register_step(self, step: PipelineStep):
        self._steps.append(step)

    @property
    def steps(self) -> List[PipelineStep]:
        return list(self._steps)

    def cancel(self):
        self._cancelled = True

    def run(
        self,
        initial_inputs: Dict[str, Any],
        log_callback: Callable[[str], None],
        retention_flags: Optional[Dict[str, bool]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        self._cancelled = False
        retention = retention_flags or {}
        n_steps = len(self._steps)
        all_intermediates: Dict[str, List[str]] = {}
        last_step_name = self._steps[-1].name if self._steps else ""

        current = dict(initial_inputs)

        for i, step in enumerate(self._steps, 1):
            if self._cancelled:
                log_callback("流水线已取消")
                return current

            log_callback(f"步骤 {i}/{n_steps}: {step.display_name}")

            def _step_progress(pct, msg, _i=i):
                if progress_callback:
                    progress_callback((_i - 1 + pct) / n_steps, msg)

            prev_inputs = dict(current)
            outputs = step.execute(
                current,
                log_callback,
                lambda: self._cancelled,
                progress_callback=_step_progress,
            )

            if self._cancelled:
                log_callback("流水线已取消")
                return current

            all_intermediates[step.name] = step.get_intermediate_files(prev_inputs, outputs)
            current.update(outputs)

        for step_name, files in all_intermediates.items():
            if retention.get(step_name, False):
                log_callback(f"保留 {step_name} 中间文件")
                continue
            if step_name == last_step_name:
                continue
            for f in files:
                if f and os.path.isfile(f):
                    try:
                        os.remove(f)
                        log_callback(f"已清理中间文件: {os.path.basename(f)}")
                    except OSError:
                        pass

        log_callback("流水线执行完毕")
        return current
