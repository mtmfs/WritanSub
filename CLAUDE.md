# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WritanSub is an AI subtitle processing pipeline: audio preprocessing (TIGER/Demucs) → speech recognition (faster-whisper) → forced alignment (MMS_FA or Qwen3) → AI translation (OpenAI-compatible API). It provides both a PySide6 GUI and a CLI.

## Common Commands

**Install dependencies (uv recommended):**
```bash
uv sync
```

**Install dependencies (pip fallback):**
```bash
pip install -r requirements.txt
pip install -e .
```

**Launch GUI:**
```bash
python -m writansub
```

**Run CLI:**
```bash
writansub-cli pipeline <files...>
writansub-cli transcribe <files...>
writansub-cli align --audio <...> --srt <...>
writansub-cli translate <file.srt>
writansub-cli preprocess --denoise <files...>
```

**Build native Rust extension manually:**
```bash
cd native && maturin develop
```

## Architecture

### Package Layout

- `writansub/` — Main Python package.
- `native/` — Rust native extension (`writansub_native`) built with PyO3 + maturin.

### Core Abstractions

- **`types.Sub`** (`types.py:72`) — The central subtitle data structure: `index`, `start`, `end`, `text`, `romaji`, `score`, `translated`.
- **`bridge.ResourceRegistry`** (`bridge.py`) — A singleton wrapping the Rust `writansub_native.ResourceRegistry`. It manages:
  - Model lifecycle (acquire/release/unload) with per-device caching.
  - Cooperative cancellation via `checkpoint()` (raises `CancelledError`).
  - Audio decoding via ffmpeg (`decode_audio`).
  - Subprocess execution with timeout and forced shutdown.
- **`pipeline.runner.PipelineConfig`** and **`run_pipeline`** — Orchestrate the full multi-phase pipeline (TIGER → Whisper → alignment → translation), handling model reuse across files and per-file progress reporting.

### Pipeline Stages

1. **Preprocess** (`preprocess/core.py`) — Optional DnR denoising (TIGER-DnR or Demucs) and speaker separation (TIGER-Speech or TF-GridNet). Overlap detection via Silero VAD.
2. **Transcribe** (`transcribe/core.py`) — faster-whisper with word-level timestamps. If speaker separation found overlaps, the pipeline re-runs Whisper on each separated track for the overlapping regions and merges results (`runner.py:_whisper_with_overlap`).
3. **Align** (`align/core.py`) — Forced alignment using either MMS_FA (torchaudio, default) or Qwen3-ForcedAligner. Pre-populates `romaji` (or pinyin/korean-romanizer/unidecode) for MMS_FA matching. Post-processing (`post_process`) handles extend_start/extend_end, gap thresholding, and min-duration merging.
4. **Translate** (`translate/core.py`) — Calls an OpenAI-compatible chat API in batches, writing results into `sub.translated`.

### Output Conventions

All outputs are written to the **same directory as the input file**, prefixed with the input basename:
- `_aligned.srt` — aligned subtitles
- `_translated.srt` — translated subtitles (bilingual if requested)
- `_review.srt` / `_review.ass` — low-confidence markers for manual QC

### GUI State

GUI settings persist to JSON files in the project root (`writansub_pp.json`, `writansub_translate.json`, `gui_state.json`).

### CUDA / PyTorch Version

Default is CUDA 12.8 (`cu128`). For older NVIDIA drivers (<570), change all `cu128` references in `pyproject.toml` and `requirements.txt` to `cu124` before installing.

### Notes

- There are no automated tests in this repository.
- `writansub/__init__.py` version (`0.1.6.3`) is out of sync with `pyproject.toml` (`0.1.7.3`); the latter is the source of truth.
