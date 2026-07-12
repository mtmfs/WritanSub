# WritanSub 修改报告 — 批次 0 + 批次 1

- 日期：2026-07-09
- 范围：`BUG_TRACKER.md` 路线的前两批（止血 + 简单批），共 22 项（21 项代码修复 + 1 项运维止血）
- 基线 commit：`d5ff0f4`；本批 5 个 commit，全部在 main，**未 push**
- 总量：30 文件，+378 / −1186 行（净删 808 行）；其中 writansub 源码 23 文件 +180 / −844

| Commit | 组 | 规模 |
|--------|----|------|
| `4d83bdf` | 组 C 死代码清理 (T25/T26/T31) | 16 文件 +191/−1091 |
| `927b7cb` | 组 B 一行级速修 (9 项) | 11 文件 +54/−25 |
| `ed53117` | 组 D 独立小修 (10 项，含 T01 先遣行) | 13 文件 +145/−89 |
| `5919424` | tracker 标记批次 0 完成 | 文档 |
| `2026fc5` | hatchling 构建约束（临时） | pyproject + lock |

---

## 组 C · 死代码清理

**T25 espnet 依赖族移除**
- 删 `preprocess/core.py` 的 `separate_speakers_tfgridnet`（75 行）及其唯一调用分支（GUI 下拉本就无此选项，分支不可达）；顺带清掉因此孤儿化的 `import numpy as np`
- 删 `types.py` 的 tfgridnet TODO 注释、`logger.py` 版本打印元组里的 `"espnet2"`
- `requirements.txt` / `pyproject.toml` 删去 espnet、espnet-model-zoo；`uv lock` 重解析，**净删 38 个传递依赖包**（aiohttp/lightning/hydra/nltk/sentencepiece 等）；锁文件顺带把 writansub-native 版本从 0.1.7.3 修正为 native/pyproject 实际的 0.1.7
- 效果：新装机体积大幅缩减；`.venv` 已同步卸载

**T26 TTS 废案归档**
- `gui/tabs/tts.py`（485 行，自述废案、import 不存在的符号）→ `archive/tts_deprecated/gui_tab_tts.py`
- `writansub/tts/`（core.py 164 行 + 空 `__init__.py`）→ `archive/tts_deprecated/tts/`
- 删除前已核实全仓库零引用（未注册 tab、无 bat/README/scripts 提及）

**T31 杂项死码**
- `git rm aitrans.py`（与 `python -m writansub` 入口重复的旧启动器，零引用）
- 删空包 `writansub/core/`（仅剩 `__pycache__`）、空目录 `tmpkanji_ws/`
- 根目录三个遗留配置（gui_state / writansub_pp / writansub_translate）**移入 `archive/legacy_root_configs/` 而非删除**——发现 translate.json 内存有 DeepSeek key 明文，保留备份更稳妥
- README 删除指向不存在的 CLAUDE.md 的死链
- `runner.py`：删未用的函数内 `import torch`；删只写不读的 `word_results`（2 处）
- `gui/tabs/pipeline.py`：删从不 emit 的 `enable_start` 信号（定义 + connect）
- `cancelled` 死参数：从 `run_alignment` / `run_qwen3_alignment` / `translate_subs` 三个签名删除，同步删 4 个调用点的 kwarg（`_cancelled` 本体在 runner 中另有用途，保留）
- 决定不动的两项：`_InfoDelegate`（核实为在用类，仅 elide 子逻辑无效，删除会改变模型下拉渲染，遗留后续）；`transcribe/core.py` 的 `_log`（由 T40 启用，从死变量转为活代码）

## 组 B · 一行级速修

| ID | 修改 | 位置 |
|----|------|------|
| T02 | 补 `from dataclasses import replace`，CLI translate 默认路径不再在写盘前 NameError 丢掉已付费译文 | `cli.py` 顶部 |
| T05 | `import torchaudio.transforms as T` 提出 MMS 分支，Qwen3+TIGER 组合不再 UnboundLocalError | `runner.py` 对齐阶段 |
| T33 | `load_pp_config` 兼捕 `TypeError`，配置含 null/数组不再启动即崩 | `config.py` |
| T37 | 超时占位调大：轨道探测 30→120s、字幕提取 60→600s、音频解码 600→3600s（超时机制真正生效要等批次 4 的 T03） | `extract.py`、`bridge.py` |
| T38 | CLI bat 的无条件 `pause` 改为仅出错时暂停，不再阻塞脚本化调用 | `WritanSubCLI.bat` |
| T39 | `ResourceRegistry.instance()` 加类级双检锁 | `bridge.py` |
| T40 | transcribe 模型创建前加"CUDA 不可用回退 CPU"（对齐 cli/align 页既有写法），提示走 `_log` 输出 | `transcribe/core.py` |
| T32 | 两处多选删除改为按 row 降序删，修非连续多选可能删错行 | `gui/tabs/pipeline.py`、`preprocess.py` |
| T35 | 驱动警告弹窗改 MB_YESNO 带"不再提示"，选是后标志存 gui_state，启动时先查标志 | `gui/driver_check.py` |

## 组 D · 独立小修

| ID | 修改 | 位置 |
|----|------|------|
| T07 | 翻译批次循环包 `try/finally`，回写移入 finally——中途取消/异常也保留已完成批次的译文 | `translate/core.py` |
| T11 | `PipelineConfig.api_key` 改 `field(repr=False)`，日志打 cfg 不再泄 key；translate 页 save/restore 去掉 gui_state 里的 key，关窗时清理历史残留；key 唯一存放地为 writansub_translate.json | `runner.py`、`gui/tabs/translate.py`、`gui/app.py` |
| T23 | 保存翻译配置时保留 batch_size（UI 无控件，从现有配置透传），GUI 翻译实际把 batch_size 传给 `translate_subs`（此前恒默认 20） | `gui/tabs/translate.py` |
| T14 | `_get_ffprobe` 推导路径不存在时抛明确 FileNotFoundError 并提示安装系统 ffmpeg（此前静默表现为"无字幕轨"） | `bridge.py` |
| T15 | CLI 新增 `--ref-embedded` 开关（按语言自动选轨终于可达）；`select_track` 语言未命中回退第一轨前写告警日志 | `cli.py`、`extract.py` |
| T18 | `parse_srt` 编码回退链：utf-8-sig → charset-normalizer 探测 → gbk/shift_jis，全败抛带文件名的明确错误 | `srt_io.py` |
| T27 | `parse_srt` 默认 `lang=None` 不再强制算罗马音；翻译路径不再白载 cutlet/MeCab；runner 的 ref 解析去掉 lang 实参；对齐路径保持显式传 lang | `srt_io.py`、`runner.py` |
| T19 | TIGER 模型加载重构为共享助手：缓存命中先离线加载，失败记日志并降级联网重试——半截下载自愈而非锁死离线 | `preprocess/core.py` |
| T20 | align 页从"每次 init+register+unload 冷加载"改为 `acquire_model`/`release_model` 缓存复用，键与流水线一致（跨页共享） | `gui/tabs/align.py` |
| T21 | vendor `wav_chunk_inference` 推理批循环逐迭代 `checkpoint()`，DnR 分离期间可暂停/取消；文件头加本地化改动标注 | `vendor/tiger/tiger_dnr.py` |

## 批次 0 · 止血（含 T01 先遣行）

- 确认 GUI 未运行、实际生效配置仍为全 0 后，删除 `%LOCALAPPDATA%\mtmfs\WritanSub\writansub_pp.json`
- T01 先遣行：transcribe 页 `word_conf_threshold` spinbox 补 `setValue(load_pp_config()[...])`（随 `ed53117` 入库）——没有这一行，删文件的效果会在下次关 GUI 时被 0 值部分冲掉
- 离屏 GUI 完整开→关一轮验证：**8 个参数全部回到默认（pad_sec 0.5、extend_end 0.3、双阈值 0.5），且关窗重存后不再变 0**——自锁循环确认已断
- T01 主修（closeEvent 按 tab 命名空间化）仍在批次 2

## 验证记录

1. `compileall` + 全模块导入冒烟：通过（每组提交前各跑一轮）
2. pyflakes 基线对比：修前 19 条 → 修后 16 条，只减不增；剩余全部为修前已存在项（align/core 的 torch 延迟注解误报 6 条、vendor 杂项 6 条、零星未用导入）
3. CLI 冒烟：主命令 + 5 个子命令 `--help` 全通过，`--ref-embedded` 出现在帮助中
4. 编码定向测试：GBK / UTF-8-BOM / Shift-JIS 三个迷你 SRT 解析后与原文**严格相等**；默认 romaji 为空、显式传 lang 时正常计算
5. T02 定向：`replace(...)` 列表推导实跑无 NameError；T11 定向：`repr(PipelineConfig(api_key=...))` 确认不含 key
6. 离屏 GUI 开关一轮：pp 默认值回归且稳定、gui_state 无 `translate.api_key`、translate 配置 key 与 batch_size 完好
7. `uv run` 恢复正常（见下方偏差 2），锁文件 diff 核查：31 增 979 删，全部为 espnet 族 + native 版本对齐，无无关升级

## 计划外偏差（3 处）

1. 根目录遗留配置**移入 archive 而非删除**——translate.json 含 DeepSeek key 明文，保留备份
2. 清华源当日对多个 sdist 及新发布的 hatchling 1.31.0 wheel 持续 403：`uv lock/sync` 改走本地缓存离线模式 + Clash 代理补漏完成；为保 `WritanSub.bat`（走 `uv run` 自动 sync）可用，pyproject 增加 `build-constraint-dependencies = ["hatchling<1.31"]`（`2026fc5`）——**镜像缓存恢复后可移除**
3. A1 先遣行未单独成 commit，随组 D（`ed53117`）入库

## 遗留与注意

- **review 从本日起真正开启**（阈值恢复 0.5）。T06 索引错位尚未修（批次 3），review 标注可能标错行——从隐性变显性，属预期，批次 3 消除
- T37 为占位值：T03（批次 4）修好 GIL 前超时机制不生效
- push 未执行，由用户自行跑 push.bat
- `Audit_Report_2026-07-02.md` 保持未跟踪状态，是否入库由用户决定
- 批次 2（10 项：T01 主修、T04、T08、T24、T09/T36、T12/T13、T29→T22 标定）待启动
