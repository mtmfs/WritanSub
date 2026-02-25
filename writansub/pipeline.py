"""流水线编排：定义步骤基类、三个具体步骤、调度器"""

import abc
import os
from typing import Any, Callable, Dict, List, Optional

from writansub.core.types import Sub
from writansub.core.srt_io import parse_srt, write_srt, mark_low_align_in_review
from writansub.core.whisper import transcribe_to_srt
from writansub.core.alignment import load_audio, run_alignment, post_process
from writansub.core.translate import translate_srt


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
        """
        执行步骤。

        Args:
            inputs: 上一步输出的文件路径字典
            log_callback: 日志回调
            cancelled: 返回 True 表示用户已取消
            progress_callback: 进度回调 (0.0~1.0, 状态文本)

        Returns:
            输出文件路径字典，合并入下一步的 inputs
        """
        ...

    def get_intermediate_files(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> List[str]:
        """返回此步骤产生的中间文件路径列表（可选清理）"""
        return []


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
        srt_path = transcribe_to_srt(
            media_file,
            lang=self.lang,
            device=device,
            log_callback=log_callback,
            progress_callback=progress_callback,
            word_conf_threshold=self.word_conf_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
        )
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"srt": srt_path, "media_file": media_file}

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

        media_file = inputs["media_file"]
        srt_path = inputs["srt"]

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            log_callback("CUDA 不可用，回退到 CPU")
            device = "cpu"

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

        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
            cancelled=lambda: cancelled(),
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
        """
        执行完整流水线。

        Args:
            initial_inputs: 初始输入 (如 {"media_file": "..."})
            log_callback: 日志回调
            retention_flags: {step.name: True/False} 是否保留中间文件

        Returns:
            最终输出字典
        """
        self._cancelled = False
        retention = retention_flags or {}
        all_intermediates: Dict[str, List[str]] = {}  # step_name -> files

        current = dict(initial_inputs)

        for i, step in enumerate(self._steps, 1):
            if self._cancelled:
                log_callback("流水线已取消")
                return current

            log_callback(f"步骤 {i}/{len(self._steps)}: {step.display_name}")

            n_steps = len(self._steps)

            def _step_progress(pct, msg, _i=i):
                if progress_callback:
                    overall = ((_i - 1) + pct) / n_steps
                    progress_callback(overall, msg)

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

            intermediates = step.get_intermediate_files(prev_inputs, outputs)
            all_intermediates[step.name] = intermediates

            current.update(outputs)

        for step_name, files in all_intermediates.items():
            if retention.get(step_name, False):
                log_callback(f"保留 {step_name} 中间文件")
                continue
            if step_name == self._steps[-1].name:
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
