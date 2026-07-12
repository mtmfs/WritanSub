# WritanSub 代码审计 / Bug 报告

- 审计日期：2026-07-02
- 版本：0.1.7.3
- 范围：`writansub/` 全部 Python 源、`native/src/lib.rs`、`vendor/tiger/`、构建与配置文件
- 严重度分级：**P0 阻断/数据损坏** · **P1 功能不可用** · **P2 设计缺陷** · **P3 冗余/性能** · **P4 小问题**

---

## 摘要

| ID | 严重度 | 组件 | 一句话 |
|----|--------|------|--------|
| WS-01 | P0 | GUI/config | 后处理参数被静默清零并自我固化，实测线上配置全为 0.0 |
| WS-02 | P0 | CLI/translate | `translate` 命令缺 `replace` 导入，翻译完成后写文件前必崩，译文全丢 |
| WS-03 | P0 | native | GIL 不释放 + 全局写锁，超时/取消/关窗全部失效，进程冻结 |
| WS-04 | P1 | pipeline | 输出文件互相覆盖，`keep_whisper_srt` 无效，与 README 矛盾 |
| WS-05 | P1 | pipeline/align | qwen3 对齐 + TIGER 预处理组合触发 `NameError: T` |
| WS-06 | P1 | pipeline/review | 合并/映射后 review 低置信标注落到错误句子 |
| WS-07 | P1 | translate | 中途取消丢弃所有已付费译文 |
| WS-08 | P1 | GUI | 跨标签页共享全局控制状态，并发互相破坏 |
| WS-09 | P2 | pipeline | separate 产出的重叠字幕被后处理压平；review 失效 |
| WS-10 | P2 | 安全 | API key 明文进日志（另存于 gui_state + translate 配置共 3 处） |
| WS-11 | P2 | network | 镜像探测走裸 socket，不经系统代理，判断可能与真实下载相反 |
| WS-12 | P2 | subtitle/bridge | imageio-ffmpeg 不含 ffprobe，内嵌字幕功能无系统 ffmpeg 时必败 |
| WS-13 | P2 | CLI/ref | 无法表达"自动选轨"；选轨回退第一轨会静默丢台词 |
| WS-14 | P2 | 打包 | 纯 pip 安装路径因 `writansub_native` 断裂 |
| WS-15 | P2 | 资源 | 波形常驻内存、模型只加载不卸载，后期阶段 OOM 风险 |
| WS-16 | P2 | subtitle | 非 UTF-8 字幕直接 UnicodeDecodeError |
| WS-17 | P3 | 依赖/死码 | espnet 全家桶只服务不可达的 tfgridnet 路径 |
| WS-18 | P3 | 死码 | TTS 整条线（tabs/tts.py + tts/core.py）是废案且会崩 |
| WS-19 | P3 | 死码 | 空包 core/、空目录 tmpkanji_ws/、根目录遗留配置、失效的 CLAUDE.md 链接、多处未用参数/变量 |
| WS-20 | P3 | 性能 | TIGER-DnR 跑 3 个全长 pass，只用 1 个 → 可 3 倍提速 |
| WS-21 | P3 | 性能 | parse_srt 默认强制算罗马音，翻译/参考路径白白加载 MeCab |
| WS-22 | P3 | 性能 | 同一文件被 ffmpeg 解码 2–3 次；`compute_type` 硬编码 int8 |
| WS-23 | P4 | GUI | 多选删除用 reversed(selectedItems()) 可能删错文件 |
| WS-24 | P4 | 多项 | 见"P4 小问题清单" |

---

## P0 — 阻断 / 数据损坏

### WS-01　后处理参数被静默清零，且状态自我固化
- **组件**：`gui/tabs/transcribe.py`、`gui/app.py`、`config.py`
- **位置**：`transcribe.py:100-104`、`app.py:56-61`、`widgets.py:261`
- **现象**：实测 `%LOCALAPPDATA%\mtmfs\WritanSub\writansub_pp.json` 全部参数为 `0.0`，包括 `pad_sec=0`（对齐零余量，成功率下降）、`extend_end=0`（尾部不延伸）、`word_conf_threshold=0` 与 `align_conf_threshold=0`（review 功能整体静默关闭）。
- **根因**：
  1. `transcribe.py` 创建 `ParamSpinBox("word_conf_threshold")` 只设了 range/step/decimals，**漏了 `setValue`**，值恒为 0.0；
  2. `app.py::closeEvent` 用 `findChildren(ParamSpinBox)` 收集**所有标签页**的 spinbox 存盘，pipeline/align/transcribe 三页存在重复 key，后遍历者覆盖先遍历者，最终值取决于 Qt 子控件遍历顺序；
  3. 坏值写回后，下次启动 `build_params_grid` 从配置读回 0（`widgets.py:261`），关窗再存 0，`PP_DEFAULTS` 永无机会生效——形成自锁。
- **影响**：对齐质量与 review 标注长期劣化，且用户无感知。
- **建议**：
  - transcribe 页 spinbox 补 `setValue(cfg.get(key, PP_DEFAULTS[key]))`；
  - closeEvent 改为各 tab 通过 `save_state()` 命名空间化各自的 pp 值，或按 tab 分别存，避免 findChildren 全局收集撞 key；
  - 立即处置：删除线上 `writansub_pp.json` 让默认值回归。

### WS-02　CLI `translate` 缺少 `replace` 导入 → 完成翻译后必崩
- **组件**：`cli.py`
- **位置**：`cli.py:416`（`replace(s, text=s.translated or s.text)`）；文件内无 `from dataclasses import replace`（AST 已确认）。
- **现象**：`writansub-cli translate x.srt`（不加 `--bilingual` 的默认路径）会先跑完全部翻译（**API 已计费**），随后在写文件前抛 `NameError: name 'replace' is not defined`，**译文全部丢失**。
- **根因**：从 GUI 版移植时漏带 import（`gui/tabs/translate.py:3` 有正确导入）。
- **建议**：`cli.py` 顶部加 `from dataclasses import replace`。

### WS-03　Rust native 层：GIL 不释放 + 全局写锁，超时/取消/关窗全失效
- **组件**：`native/src/lib.rs`、`bridge.py`
- **位置**：`lib.rs:85-94`（`wait_process`）、`96-106`（`shutdown`）、`bridge.py:158-204`
- **现象与根因**：
  - `wait_process` 在持有 `REGISTRY.write()` 期间调用 `child.wait_with_output()` 阻塞，且**整个 pyo3 调用从不 `allow_threads` 释放 GIL**。ffmpeg 解码/字幕提取期间，持 GIL 的等待线程**冻结整个 Python 进程**（GUI 事件循环、进度条、暂停/取消按钮全部无响应，Windows 显示"未响应"）。
  - `bridge.py:171` 的 `t.join(timeout)` 超时机制失效：超时后需重新获取 GIL 才能回到 Python，而 GIL 正被等待线程占用。
  - 即便进入超时分支，`shutdown()` 需拿 `REGISTRY.write()`，会死锁至子进程自然退出；且 `wait_process` 开头已把 child 从 map `remove`，`shutdown` 的 kill 循环根本触及不到该子进程。
  - `closeEvent → shutdown()` 同样卡住主线程直到子进程结束。
  - 附带：`Vec<u8>` 经 pyo3 转成 Python `list[int]`（`bridge.py:184-187` 注释自证），1 小时音频 317 MB PCM → 3.17 亿元素 list，内存峰值约 2.5 GB + 转换耗时。
- **架构结论**：该模块当前提供负价值——
  - 模型注册等价于一个 dict；`acquire_model`/`in_use` 标志在 Python 侧从未调用（死机制）；`release_model` 实为空操作却在 cli/tab 中被当资源释放调用（误导）；
  - 进程管理不如 `subprocess.run`（自带 GIL 释放、timeout、kill）；
  - 还给用户带来 maturin/Rust 工具链安装负担（README Q4 教人 `maturin develop`）。
- **建议**：二选一——
  - (A) 用 `py.allow_threads` 包裹阻塞等待，保留 child kill 句柄（wait 期间不从 map 移除），修正 shutdown 锁；返回 `PyBytes` 而非 `Vec<u8>`；
  - (B) 退回纯 Python `subprocess.run(timeout=...)`，删除整个 native 构建链与依赖。推荐 (B)，除非有明确的跨调用模型持久化需求。

---

## P1 — 功能不可用

### WS-04　流水线输出互相覆盖，`keep_whisper_srt` 失效，与 README 矛盾
- **位置**：`pipeline/runner.py:126`（whisper 写 `<base>.srt`）、`274-280`（最终结果也写 `<base>.srt`）、`239-240`（`_aligned.srt` 仅 keep 时写）
- **现象**：
  - 勾"保留 Whisper SRT"仍被最终输出覆盖，**选项无效**；
  - README 承诺"`_aligned.srt` 只要跑打轴就生成"，实际仅勾 `keep_aligned_srt` 才写，最终产物落在 `<base>.srt`；
  - 流水线开翻译时**强制双语输出**（`merge_bilingual`），无 GUI 翻译页的"仅译文"选择。
- **建议**：最终结果统一写 `_aligned.srt`（或 `_translated.srt`），不占用 `<base>.srt`；whisper srt 写 `_whisper.srt`；流水线暴露双语/仅译文开关。

### WS-05　qwen3 对齐 + TIGER 预处理 → `NameError: T`
- **位置**：`runner.py:202`（`T.Resample`）；`T` 仅在 mms 分支 `179` 行导入
- **现象**：走 qwen3 分支且存在 `dialog_wav`（`dialog_sr=44100 ≠ 16000` 恒真）时抛 `NameError: name 'T' is not defined`。即 `--align-model qwen3-fa-0.6b` 配合任意 `--denoise/--separate` 完全不可用。
- **建议**：把 `import torchaudio.transforms as T` 提到两分支之外。

### WS-06　Review 低置信标注索引错位
- **位置**：`runner.py:121-124`（review 基于原始 whisper index 生成）、`align/core.py:343-354`（post_process 合并短字幕并重编号）、`runner.py:235-238`（用新 index 回标旧文件）
- **现象**：一旦发生短字幕合并或 ref 映射（都会重编号），`mark_low_align_in_review` 用新 index 去标旧 review 文件的块，低置信标注落在**错误的句子**上。
- **建议**：review 生成推迟到 index 稳定后，或以稳定标识（时间戳）而非 index 关联。

### WS-07　翻译中途取消丢弃全部已付费译文
- **位置**：`translate/core.py:30、83-85`（译文攒在局部 dict，全部批次完成后才回写）
- **现象**：`reg.checkpoint()` 在取消时抛异常，已完成批次（钱已花）的译文全丢。
- **建议**：每批完成后立即回写对应 `sub.translated`。

### WS-08　GUI 跨标签页共享全局控制状态
- **位置**：五个 tab 的 `_start/_cancel/_toggle_pause` 均操作同一个 `ResourceRegistry` 单例
- **现象**：A 页运行中，B 页点"开始"会 `reset_controls()` 清掉 A 的取消请求；任一页"取消"取消所有任务；按钮禁用只管本页，两个任务可同时抢 GPU。
- **建议**：运行中全局互斥（禁用其他页启动），或将 cancel/pause 状态按任务隔离。

---

## P2 — 设计缺陷

### WS-09　separate 的重叠字幕被压平，且 review 失效
- **位置**：`align/core.py:337-338`（`curr.end = nxt.start` 强制顺序化）、`runner.py:390-398`（`_whisper_with_overlap` 漏传 `vad_filter`）、`runner.py:445`（返回 `[[] for _ in active]` 清空 word_data）
- **现象**：重叠说话人分离辛苦产出的重叠字幕在后处理里被压平，重叠对话不再重叠；separate 模式下 word_data 被清空 → review 整体失效。
- **建议**：重叠段字幕跳过间距压平；保留 word_data 或在该模式下明确关闭 review 并提示。

### WS-10　API key 明文写入日志（共 3 处留存）
- **位置**：`gui/tabs/pipeline.py:497`（`log_line(f"Pipeline config: {cfg}")` 含 api_key）；另存于 `gui_state.json`（`translate.api_key`）与 translate 配置
- **风险**：日志是用户求助时最常随手外发的文件。
- **建议**：入日志前对 key 打码；考虑 gui_state 不落 key。

### WS-11　镜像探测走裸 socket，不经系统代理
- **位置**：`network.py:19`（`socket.create_connection(("huggingface.co", 443))`）
- **现象**：huggingface_hub 的 requests 会读 Windows 系统代理（本机 Clash），裸 socket 不会 → 代理可用而直连不通时误切 hf-mirror；反之亦可能误判。
- **建议**：探测经代理，或直接以 hf_transfer/requests 的实际结果为准。

### WS-12　ffprobe 回退推导路径不存在
- **位置**：`bridge.py:63-66`（把 ffmpeg 文件名 replace 成 ffprobe）
- **现象**：imageio-ffmpeg **不附带 ffprobe**，推导出的路径不存在；未装系统 ffmpeg 的用户"参考内嵌字幕"必败（pipeline 里被 `except Exception` 吞成一行日志）。
- **建议**：找不到 ffprobe 时明确报错并提示安装系统 ffmpeg，或改用 ffmpeg 本身探测流信息。

### WS-13　CLI 无法表达"自动选轨"；选轨回退会静默丢台词
- **位置**：`cli.py:168`（`use_ref_sub=args.ref_sub_track is not None`）、`subtitle/extract.py:57-68`（`select_track` 未匹配回退第一轨）
- **现象**：不给 `--ref-sub-track` 索引，内嵌字幕功能整体关闭，帮助文本"默认按语言自动匹配"永不可达；回退第一轨若是少量 signs 轨，ref 映射会**静默丢弃**大部分未命中窗口的台词。
- **建议**：新增独立 `--ref-embedded` 开关；选轨未命中语言时报警而非静默回退。

### WS-14　纯 pip 安装路径断裂
- **位置**：`pyproject.toml` 依赖含 `writansub_native`，其 path 映射仅对 uv 生效；`requirements.txt` 无该项
- **现象**：README 方法 B 的 `pip install -e .` 会去 PyPI 找 `writansub_native` 失败；文档未提 pip 用户需先构建 native。
- **建议**：README 补 pip 用户先 `maturin develop`；或把 native 设为可选并提供纯 Python 回退（配合 WS-03）。

### WS-15　波形常驻内存、模型只加载不卸载
- **位置**：`preprocess/core.py`（dialog/spk1/spk2 全量存 dict 传递）、`runner.py` 各阶段 `acquire_model` 后从不 unload
- **现象**：批量 N 个长视频波形累积数 GB RAM；tiger_dnr + tiger_speech + whisper + mms_fa 全程常驻 VRAM，6–8 GB 卡跑 separate + large-v3 后期易 OOM（README Q5 只能劝"别同时开"）。
- **建议**：预处理阶段结束后 unload TIGER 模型；波形按需落临时文件而非全量驻留。

### WS-16　非 UTF-8 字幕直接崩
- **位置**：`subtitle/srt_io.py`（强制 `encoding='utf-8'`）
- **现象**：GBK/Shift-JIS 存量 SRT（目标用户极常见）→ UnicodeDecodeError。
- **建议**：utf-8 失败后回退 chardet/GBK 探测。

---

## P3 — 冗余 / 死代码 / 性能

### WS-17　espnet 全家桶只服务不可达路径
- espnet + espnet-model-zoo 两个重依赖仅供 `separate_speakers_tfgridnet`，而其模型在 `types.py:49` 自述"不兼容单声道输入"，GUI 下拉无该选项。装机体积与时间的大头喂给了死路径。**建议**：移除依赖与 `separate_speakers_tfgridnet`，或明确标注为实验性并从默认依赖剔除。

### WS-18　TTS 整条线是废案且会崩
- `gui/tabs/tts.py`（485 行，文件头自述"废案…未接入"，import 的 `TTS_MODELS`/`load_tts_config` 不存在，一接入即崩）+ `tts/core.py`（164 行，`run_mms_fa` 是 align 流程重复实现），依赖 style_bert_vits2 也不在依赖表。**建议**：移入 `archive/`。

### WS-19　零散死代码
- 空包 `writansub/core/`（仅剩 `__pycache__`）；空目录 `tmpkanji_ws/`；根目录三个旧位置遗留 config；
- README 链接的 `CLAUDE.md` 不存在（CHANGELOG 声称添加后又删，链接未更新）；
- `run_alignment`/`run_qwen3_alignment`/`translate_subs` 的 `cancelled` 参数从未使用（实际靠 `reg.checkpoint()`）；
- `runner.py:58` 未用的 `import torch`；`word_results` 只写不读；`_PipelineSignals.enable_start` 从不 emit；
- `widgets._InfoDelegate` 的 elide 逻辑因 `paint` 会重新 `initStyleOption` 而实际无效。

### WS-20　TIGER-DnR 三倍浪费（最大单点提速）
- **位置**：`preprocess/core.py:143-156`，对 dialog/effect/music 三个子模型各跑一遍全长推理（vendor `wav_chunk_inference` 确认每次独立完整 pass），而流水线只消费 dialog。
- **建议**：`save_intermediate=False` 时只跑 dialog 一个 pass → DnR 阶段直接 3 倍提速。

### WS-21　parse_srt 默认强制算罗马音
- **位置**：`srt_io.py:23`（默认 `lang="ja"`）
- **现象**：翻译命令/翻译页、ref_srt 解析都白白加载 cutlet+MeCab 并逐条形态素分析，这些路径不用 romaji。
- **建议**：默认 `lang=None`，仅对齐路径显式传语言。

### WS-22　重复解码 + 硬编码 int8
- 同一文件被 ffmpeg 解码 2–3 次（tiger 44.1k、whisper 自解、无 tiger 时 align 16k）；
- `transcribe/core.py:31、cli.py:262、runner.py:97` 硬编码 `compute_type="int8"`——CUDA 上 fp16 通常更快更准，且 README 显存表（large-v3 ~10GB）是 fp16 口径，与 int8 实况（~3–4GB）不符，文档与代码需二选一对齐。

---

## P4 — 小问题清单

| 位置 | 问题 |
|------|------|
| `pipeline.py:362`、`preprocess.py:200` | 多选删除用 `reversed(selectedItems())`，Qt 不保证按行序返回 → 非连续多选可能删错文件；应取 row 后降序删 |
| `gui/tabs/translate.py` | 不读 `batch_size`（永远默认 20）；`save_translate_config` 只写 4 个 key，会把手工配置的 batch_size 从 json 抹掉 |
| `config.py:71-76` | `load_pp_config` 只捕获 ValueError；值若为 list 等，`float()` 抛 TypeError 未捕获 → GUI 启动即崩 |
| `subtitle/review.py:112` | `mark_low_align_in_review` 用 `rfind(",,")` 定位 ASS 文本，正文含 ",," 时切错位置 |
| transcribe/pipeline | 无 CUDA 不可用回退（align 有），风格不一致 |
| native + Ctrl+C | 二次 Ctrl+C 强退后，spawn 的 ffmpeg 子进程在 Windows 上成孤儿继续跑 |
| `align/core.py:346-349` | 合并短字幕时 `prev.text + sub.text` 无分隔符（拉丁语言错误），且不看时间距离硬合并 |
| `gui/driver_check.py` | 无 N 卡用户每次启动都弹驱动警告框，无"不再提示" |

---

## 建议处理优先级

1. **立即**：删除线上 `writansub_pp.json` 让默认值回归（WS-01 正影响日常产出）；修 transcribe 页 spinbox 初始化 + closeEvent 重复 key 收集。
2. **本周级**：`cli.py` 补 `replace` 导入（WS-02）；理顺 runner 输出命名不覆盖（WS-04）；提出 qwen3 分支的 `T` 导入（WS-05）；翻译逐批回写（WS-07）。
3. **重构级**：native 层去留决策（倾向退回纯 Python，WS-03）；TIGER 按需单轨推理（WS-20）；espnet/TTS 死代码清理（WS-17/18）。
4. **顺手**：README 与实际行为对齐（输出文件表、`_dnr`/`_speech` 实为 `_dialog`/`_spk1`、CLAUDE.md 链接、显存表口径）。
