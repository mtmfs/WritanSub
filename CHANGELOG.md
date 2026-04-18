# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

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
