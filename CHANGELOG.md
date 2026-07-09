# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [Unreleased] - 2026-07-09 批次 1~3

### Changed
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
