# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [Unreleased]

### Changed
- **听写后端接口化**：新增 `Recognizer` 协议与 `Transcript` / `Segment` 类型，faster-whisper 收进 `transcribe/whisper.py` 适配器；`PipelineConfig` 以 `asr` 字段持有识别器，runner 与 CLI 不再直接引用 Whisper 参数、缓存键和产物命名。CLI 旗标、产物文件名、模型缓存键均不变。`WordInfo` 新增可选词时间 start/end，probability 允许为空，review 跳过无概率的词。GUI 随后整体删除，见 Removed。
- **实验性 Qwen3-ASR 后端**：`transcribe/qwen.py` 以 Qwen3-ASR-1.7B 识别、项目已用的 Qwen3-ForcedAligner-0.6B 出词时间，`transcribe/segment.py` 按句末标点与长停顿切成字幕条（标点从全文对回对齐器输出的 token）；超长段在词边界里按停顿时长、顿号、两半均衡打分选切点，不在助词或助动词前切。对齐器给出的单词时长超过 `max_word_seconds`（默认 3 s）即截断，避免配乐段里几个字的字幕条拖到二十多秒。尚未接入 CLI，仅供 `PipelineConfig(asr=QwenRecognizer())` 或脚本调用对比。
- **识别器自带对齐轴时跳过强制对齐**：`Segment.time_kind` 新增 `aligned`；某文件的识别结果全部已对齐且未经参考字幕映射时，runner 跳过 MMS/Qwen3 强制对齐阶段，review 行级阈值与参考轴直通同样置零。
- **存储布局收敛为单一根目录**：缓存、本地模型、日志、配置、运行期临时文件统一放在存储根目录的 `cache/` `models/` `logs/` `config/` `run/` 五个子目录下，根目录默认为项目文件夹，环境变量 `WRITANSUB_HOME` 整体改位置。降噪 / 分轨的临时溢出目录从系统 Temp 移到 `run/spill_*`（任务结束照旧自动删除）。会话日志改名为 `<角色>_<日期>_<时间>_<PID>.log`（现只有 `cli` 一种角色），每角色各保留最近 50 份（原统一 20 份）；两个 CLI 进程同一秒启动不再互相截断日志头。旧格式 `writansub_*.log` 不再自动清理，可手动删除。

### Removed
- **GUI 整体删除（架构批第一步）**：PySide6 界面、`python -m writansub` 入口与 `writansub` 脚本入口、`WritanSub.bat`、NVIDIA 驱动检查弹窗、任务暂停/恢复、GUI 持久化配置（后处理参数、翻译 API 设置、窗口状态）全部移除，代码归档至 `archive/gui_pyside6_202610/`。CLI 不受影响：`writansub-cli` 旗标与产物命名不变；翻译 API 设置改为每次经 `--api-key` / `--config` 传入，`%LOCALAPPDATA%\mtmfs\WritanSub\` 下的旧配置文件不再读取，可手动删除。
- 依赖移除 `PySide6`、`platformdirs`。
- 打包链（Inno 安装器、Rust 启动器、`build/scripts/*.ps1`）停止维护，归档至 `archive/build_chain_202610/`；安装方式只保留 `uv sync`。
- `transcribe/core.py` 兼容壳（仅 GUI 识别页使用）随 GUI 归档。
- 环境变量 `WRITANSUB_MODELS_DIR`、`WRITANSUB_LOG_DIR`（从未写进文档），由 `WRITANSUB_HOME` 取代。

### Fixed
- **separate 模式重叠区丢台词**：原先只重识别两人同时说话的片段，却把与之相交的整条原字幕删掉，片段以外的台词随之丢失。现在把相交的字幕和重叠区连成完整窗口再重识别；分离音轨盖不住整句、窗口过短或两条轨都没识别出文字时保留原字幕。
- **参考字幕不可用时的结果丢失**：参考字幕读不出、视频没有字幕轨或参考字幕为空时，不再清空识别结果或静默跳过对齐，改为按文件保留识别结果并回退强制对齐。`--ref-direct` 只在该文件的识别字幕全部对上参考轴时才跳过对齐；没有识别结果时也不再把参考字幕的原文当成源语输出。
- **对齐后字幕乱序**：逐句对齐可能改变相邻字幕的先后顺序，后处理此前只在 separate 模式排序，普通字幕会被压成 10 毫秒。现在所有字幕先按开始时间排序再贴合时间轴，并统一重编号。
- 短字幕向前合并时保留两句中较低的对齐分数，对齐失败的短句不再被高分前句吞掉而漏出 review。
- `writansub-cli --version` 显示 0.1.9：0.1.10 发版时漏改 `writansub/__init__.py`，已补齐。
- **`--api-key` 明文进会话日志**：日志开头记录的命令行原样写入密钥值，分享排错日志会带出；现把 `--api-key V` / `--api-key=V`（含 `--api-k` 等前缀缩写）的值改写为 `***`。
- **模型路径拼进产物文件名**：`--whisper-model` 传本地目录（如 `G:\models\x`）或 `owner/model` 仓库名时，盘符与分隔符被原样拼进 `_original_whisper-<模型>.srt`，识别跑完才报写文件失败。现文件名只取模型标识最后一段并做安全化（`Systran/faster-whisper-large-v3` → `whisper-faster-whisper-large-v3`），加载与模型缓存键仍用原路径；`stage_path` 对模型段统一兜底，Qwen3 后端同样处理。

## [0.1.10] - 2026-07-16 批次 8（内存批）

### Changed
- **TIGER 波形即分即落盘**：批处理时分离出的人声/说话人轨写入临时溢出目录、用到时按需读回，内存峰值从"全部文件波形常驻"（约 600MB/小时音频 × 文件数）降为单文件量级；溢出目录随任务结束/取消自动清除。产物与行为不变。
- **取消响应压到毫秒级**：torch 模型（TIGER/MMS/Qwen3/demucs）统一挂逐层取消钩子，降噪/对齐推理中途取消不再等完当前前向窗口（CPU 模式原需数秒~数十秒）。暂停仍为窗口粒度。

### Fixed
- GUI 转录页 whisper 缓存键与 CLI/流水线格式不一致，同一模型在一个会话里会加载两份（各占 ~1.5GB）。

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
