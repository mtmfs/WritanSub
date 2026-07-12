# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [0.1.9] - 2026-07-12 批次 5

### Changed
- **镜像探测改经系统代理的 HTTPS 实测**（Clash 场景不再误判不可达，SNI 被重置不再误判可达）；探测超时 2s→5s，仅影响断网时的启动等待。
- **silero-vad 改官方 pip 包加载**，VAD 模型随包内置、零联网——国内开 VAD（separate 模式）不再必败。
- **关窗前确认**：有任务运行中时关窗弹确认框，拒绝则任务无损继续；确认后维持既有"立即终止"语义。
- transcribe 全路径（CLI/GUI/流水线）请求 cuda 但不可用时自动回退 CPU 并知会（此前仅 align 有此回退）。
- **依赖锁默认源切官方 PyPI**（清华镜像对大量 sdist/新 wheel 持续 403）；pytest/pyflakes 入锁转正，`uv sync` 不再清掉测试工具；移除 hatchling<1.31 构建钉。镜像恢复后可切回。

### Removed
- 移除无效的 `--ss-model` 参数与 GUI"分轨模型"下拉（自 tfgridnet 退役后唯一模型 tiger-speech 自动生效）。**传入 `--ss-model` 现在会报未知参数**（原为静默忽略）。

### Fixed
- 模型下拉过长名称正确中段省略，不再与右侧显存标注重叠。
- 构建脚本自适应 GNU tar / bsdtar（纯净终端不再依赖 PATH 前置 Git 的 GNU tar）。

## [0.1.8] - 2026-07-12 批次 1~4

### Removed
- **Rust 原生扩展 `writansub_native` 整层退役**，由 bridge.py 纯 Python 实现替换（对外 API 不变）：
  - GUI 在解码/字幕提取期间不再冻结（原 GIL 持锁病灶）
  - 子进程超时与取消**首次真实生效**：取消即时终止 ffmpeg（实测延迟 <0.1s），Ctrl+C 不再遗留孤儿进程
  - 1h 音频解码内存峰值大幅下降（消除字节流→int 列表转换）
  - 源码安装不再需要 Rust 工具链；Rust 源码归档 `archive/native_rust_202607/`

### Changed
- **安装口径全面 uv**：README 移除 pip 安装方法二，`requirements.txt` 降级为参考文件。
- 子进程统一以 CREATE_NO_WINDOW 启动，GUI 场景不再闪控制台黑框。
- **输出命名新规**：源语终稿恒为 `<base>.srt`（不再被翻译覆盖）；开翻译时另写 `<base>_<语言>.srt`（如 `_chs`），默认双语、`--no-bilingual` 切单语；中间/单步产物三段式 `<base>_<阶段>_<模型>.srt`。
- **review 重构**：标注随字幕本体携带、流程末尾统一生成——修复重编号后标错行、对齐低置信标注静默丢失；review 文件现在在流程结束时产出（取消时不产出，与终稿一致）。
- **separate 模式重叠双模式**：默认把跨说话人重叠压成一句（`- 甲` / `- 乙`），`--overlap-mode keep` 保留双条真实重叠（不再被压平截尾）；VAD 开关在 separate 模式下生效；该模式的词级 review 恢复可用。
- 短字幕向前合并只在与前句时间相邻（原始间距 ≤ gap-threshold）时发生，孤立短句不再被跨静音合并。
- TIGER-DnR 降噪默认只跑人声子模型，预处理约 3 倍提速（`--save-intermediate` 时仍产三轨）。
- 新增 `--compute-type {int8,int8_float16,float16}`（pipeline/transcribe），默认 int8 不变。
- GUI 后处理参数改为"编辑即存"，根除多页同名参数互相覆盖导致的全零自锁。

### Fixed
- 非 UTF-8 字幕（Shift-JIS/GBK）解析：正式声明 charset-normalizer 依赖，回退顺序防止日文字幕被静默解成乱码。
- 模型缓存加载失败与显存不足不再互相误诊。

## [0.1.7.3] - 2026-04-17

### Added
- 新增 `LICENSE`（GPL-3.0 全文）。
- 新增 `CLAUDE.md`，记录项目架构、常用命令与开发指南。
- 新增 `writansub/network.py`，自动检测 Hugging Face 连通性并在不可达时切换至 `hf-mirror.com`。
- `pyproject.toml` 补充完整打包元数据（description、readme、license、authors、keywords、classifiers、urls）。
- `pyproject.toml` 新增 `[tool.ruff]` 与 `[tool.mypy]` 配置。
- `requirements.txt` 与 `pyproject.toml` 补充缺失依赖：`qwen-asr`、`demucs`、`espnet`、`espnet-model-zoo`、`platformdirs`。

### Changed
- 统一版本号到 `0.1.7.3`（`writansub/__init__.py`、`native/Cargo.toml`、`native/pyproject.toml`）。
- 全面重写 `README.md`，增加面向零基础用户的快速开始、GUI/CLI 详细操作指南、输出文件说明、常见问题与排错。
- `WritanSub.bat` / `WritanSubCLI.bat` 重构为 `uv` → `.venv` → 系统 Python 的三层自动回退启动脚本。
- `writansub/paths.py` 改用 `platformdirs` 管理用户数据目录，避免在包目录旁写入配置文件。
- `.gitignore` 增加 IDE/编辑器产物与各类工具缓存目录。

### Fixed
- `writansub/cli.py`：移除未使用的 `from dataclasses import replace`，简化 `_resolve_pp`。
- `writansub/pipeline/runner.py`：将 `torchaudio` 相关导入移出循环；用类型化函数替换 `_cancelled` lambda。
- `writansub/align/core.py`：为关键函数补充类型注解。
- `writansub/subtitle/srt_io.py`：为 `_subs_from_pysrt` 补充参数类型。
- `writansub/gui/widgets.py`：移除无意义的空 stub `_auto_save`。
- `writansub/vendor/tiger/activations.py`：移除 `__main__` 块中的调试 `print`。
- `writansub/gui/app.py`：启动时调用 `setup_hf_mirror()`，确保国内网络环境自动走镜像。
