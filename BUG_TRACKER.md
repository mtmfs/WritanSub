# WritanSub 缺陷统一跟踪清单

- 生成日期：2026-07-09
- 合并来源（去重后 40 项）：
  - **A** = `Audit_Report_2026-07-02.md`（本地审计，WS-01～24）
  - **B** = `Claude-代码审查与架构优化分析.md`（claude.ai 远程盲审，#1～30，2026-06-10 起，2026-07-02 导出）
- 两份报告重叠约 2/3，互相印证的项目基本可视为免复核；单源项目已尽量现场抽查。
- 字段说明：
  - **难度** = 实现复杂度（极低 / 低 / 中 / 高）
  - **风险** = 改动引入回归或破坏既有行为的概率（低 / 中 / 高）
  - **改动量** = 预估净代码行（不含测试）
- 状态标记：`[ ]` 未修 / `[x]` 已修 / `[~]` 部分修复 / `[-]` 决定不修

> 现场核实记录（2026-07-08~09）：线上 `%LOCALAPPDATA%\mtmfs\WritanSub\writansub_pp.json` 全 0 属实；`cli.py` 确无 `replace` 导入；`transcribe.py:100-104`、`app.py:56-61`、`bridge.py` 全文与两份报告描述一致；native 层仅被 `bridge.py` 一个文件引用，其余 15 个使用方只接触 Python API。

---

## P0 — 阻断 / 数据损坏（3 项）

| 状态 | ID | 问题 | 来源 | 难度 | 风险 | 改动量 |
|---|---|---|---|---|---|---|
| [x] | T01 | 后处理参数全零自锁：transcribe 页 spinbox 缺 `setValue`（`transcribe.py:100`）+ `closeEvent` 用 `findChildren` 全局收集撞 key（`app.py:56-61`）+ 坏值写盘后自锁。2026-07-09 主修：改"编辑即单键合并写盘"（widgets.bind_pp_autosave，keyboardTracking 关闭），关窗收集整块删除——未触碰的 spinbox 永不写盘，未初始化污染机制上不可能；多页同键=同配置多视图、最后编辑者赢。离屏真 GUI 四轮验证通过 | A:WS-01 + B:#16(半) | 低 | 低 | 20–40 行 |
| [x] | T02 | CLI `translate` 缺 `from dataclasses import replace`（`cli.py:416`），默认路径翻译跑完必崩、译文全丢（API 已计费）。引入点：0.1.7.3 的"清理未用导入" | A:WS-02 + B:#1 | 极低 | 无 | 1 行 |
| [x] | T03 | native 层三合一：`wait_process` 持 GIL 全程冻结解释器（GUI 卡死、超时/取消/关窗全失效）+ `shutdown` 写锁死锁 + `Vec<u8>`→`list[int]` 内存灾难（1h 音频峰值 ~2.5GB）。两份报告一致结论：整层负价值，建议纯 Python 替换（dict + `subprocess.run(timeout=...)`）。2026-07-12 修复：bridge.py 纯 Python 替换（模型注册表 dict+锁；Popen+活进程登记；`cancelled` 改 property 置 True 即杀活子进程），对外 API 不变、消费方零改动；Rust 源码归档 archive/native_rust_202607，maturin/版本校验/打包链拆除。超时与取消首次真实生效（实测取消延迟 0.016s、无孤儿 ffmpeg、解码期间解释器心跳最大间隔 0.094s）；子进程加 CREATE_NO_WINDOW | A:WS-03 + B:#6/#7/#12 | 中 | 中 | bridge.py 改写 60–100 行；删 native/ 113 行 Rust + maturin/版本校验/打包链清理 |

T01 止血步：✅ 2026-07-09 已执行（删除线上全 0 配置 + transcribe 页 spinbox 补 setValue 先遣行）。主修：✅ 同日完成（方案由"closeEvent 命名空间化"改为更深的"编辑即写"，见表内）。⚠️ T01 修复后 review 真正复活，T06 索引错位从隐性变显性，属预期，批次 3 消除。
T03 附带收益：修复即顺带消掉 T16（pip 安装断裂）、P4 的 Ctrl+C 孤儿 ffmpeg、超时不生效（T37 的前置）。风险集中在打包链（Inno/launcher/`check_versions.ps1` 都要同步改），Python 侧爆炸半径已确认仅 bridge.py 一个文件。

---

## P1 — 功能不可用（6 项）

| 状态 | ID | 问题 | 来源 | 难度 | 风险 | 改动量 |
|---|---|---|---|---|---|---|
| [x] | T04 | 输出互相覆盖：whisper SRT 与最终结果都写 `<base>.srt`，`keep_whisper_srt` 无效；流水线开翻译时强制双语（`runner.py:126/274-280`）。2026-07-09 修复：终稿双轨（源语恒写 `<base>.srt`、翻译另写 `<base>_<lang>.srt` 如 `_chs`，两个都产）；中间/单步产物三段式 `<base>_<stage>_<model>.srt`；新增 `--no-bilingual`；助手 `stage_path`/`lang_code` 在 srt_io.py；README/ws 技能已同步。GUI 输出框自动填名维持旧样式 | A:WS-04 + B:#3 | 低 | 中※ | 15–30 行 + README |
| [x] | T05 | Qwen3 + TIGER 组合必崩：`import torchaudio.transforms as T` 只在 MMS 分支内，Qwen3 路径引用 `T.Resample` 抛 UnboundLocalError；TIGER 输出 44100≠16000 使该路径必然触发（`runner.py:179/202`，B 已复现验证） | A:WS-05 + B:#2 | 极低 | 低 | 1–2 行 |
| [x] | T06 | review 索引体系错位：按原始编号生成 → ref 映射/短字幕合并重编号 → 用新编号回标旧文件，标错行；ASS 侧 `rfind(",,")` 解析脆弱；词全高置信时对齐标注静默丢失。2026-07-09 根治：低置信词落 `Sub.low_words` 随本体携带（对齐/参考映射/合并三处变换自然跟随），流程末尾 `generate_review_final` 单点生成，`mark_low_align_in_review` 删除；无标记时清陈旧 review 文件。cli/GUI transcribe 单步路径索引稳定，保留旧 API | A:WS-06+P4 + B:#4/#5 | 中高 | 中 | 60–120 行（runner.py + review.py 数据流重排） |
| [ ] | T07 | 翻译中途取消丢弃全部已付费译文：译文攒局部 dict 最后才回写（`translate/core.py:30/83-85`）。改为每批完成即回写。**注意**：真正的丢失点在三个调用方（GUI translate 页 / CLI / runner）均在 CancelledError 之后跳过 write_srt——只改 core.py 的内存回写无效，必须让调用方在取消路径也落盘。2026-07-12 对齐：与三件套 b（弃单）、T08 同判暂缓——翻译功能不常用，将来用到时一批做 | A:WS-07 | 低 | 低 | 5–10 行 |
| [ ] | T08 | 翻译完全信任 LLM 回显编号：不校验编号属于当前批次（整批重编号=常见失败模式，译文写错条并覆盖前批）；失败批次无重试只记日志 | B:#25 | 低中 | 低 | 30–50 行 |
| [-] | T09 | GUI 跨页共享全局控制状态：B 页"开始"清掉 A 页取消请求，任一页"取消"取消所有任务，两任务可同抢 GPU。最简方案 = 运行中全局互斥禁用其他页启动。2026-07-12 用户拍板不修：批 2 定稿方案（本 tab 无任务取消置灰）经核实系老版本（60a361a）既有行为、与现状重合；双任务并发跨页互杀属罕见场景且 GUI 半废弃，按"设计即现状"关单 | A:WS-08 | 中 | 中 | 30–60 行 |

※ T04 的风险在行为变更：输出文件名是对外契约，改名要同步 README 与既有使用习惯。
T07 与 T08 同在 `translate/core.py`（全文件仅 94 行），建议同批修。
2026-07-09 复核：批次 1 对 T07 的修复（try/finally 内存回写）经验证无效已回退，T40 同（死代码已回退）；两项均退回 `[ ]`，修复要点已写入各自条目。

---

## P2 — 设计缺陷（14 项）

| 状态 | ID | 问题 | 来源 | 难度 | 风险 | 改动量 |
|---|---|---|---|---|---|---|
| [x] | T10 | separate 模式三连：重叠字幕被 `curr.end=nxt.start` 压平；`_whisper_with_overlap` 漏传 `vad_filter`；word_data 算完即弃 → review 静默失效。2026-07-09 修复：cue 打 `speaker` 标签，重叠处理移入 post_process 双模式——默认 merge 压成一句（`- A` / `- B`，时间取并集，score 取 min），`--overlap-mode keep` 保留双条真实重叠（gap 循环跳过异说话人重叠对）；vad_filter 补传全轨（区域短块故意不开）；词级数据 (sub,word) 成对携带，separate 模式 review 复活 | A:WS-09 + B:#30(半) | 中 | 中 | 20–40 行 |
| [x] | T11 | API key 明文三处：pipeline 日志（`pipeline.py:497`）+ `gui_state.json` + `writansub_translate.json`。日志脱敏 + gui_state 去 key | A:WS-10 + B:#11 | 低 | 低 | 10–20 行 |
| [x] | T12 | 镜像探测不可靠，双盲区：裸 socket 不走系统代理（Clash 场景误判，A 视角）+ TCP 通但 SNI 阶段被重置误判可达（B 视角）。2026-07-12 修复：network.py 改 urllib HEAD 实测（自动走系统代理+完整 TLS 握手），HTTPError 视为可达，超时 2s→5s；5 例打桩测试 + Clash 开态实测 | A:WS-11 + B:#28a | 低中 | 中 | 10–25 行 |
| [x] | T13 | silero-vad 走 `torch.hub` 从 GitHub 下载、无任何镜像处理，国内开 VAD 必败。2026-07-12 修复：改官方 silero-vad pip 包（模型内置随包，加载零联网），消费方零改动；顺带整锁切官方 PyPI + pytest/pyflakes 入锁转正（清华镜像 403，N1 提前完成） | B:#28b | 中 | 中 | 10–30 行 |
| [x] | T14 | ffprobe 回退路径必败：imageio-ffmpeg 不含 ffprobe，推导路径不存在，"参考内嵌字幕"无系统 ffmpeg 必败且被吞异常。明确报错+提示，或改用 ffmpeg 探测 | A:WS-12 + B:#8 | 低 | 低 | 10–30 行 |
| [x] | T15 | CLI 无法表达"自动选轨"（帮助文本承诺永不可达）；`select_track` 未命中语言时静默回退第一轨，signs 轨会静默丢大量台词 | A:WS-13 + B:#30(半) | 低 | 低 | 15–30 行 |
| [x] | T16 | pip 安装路径断裂：`writansub_native` 不在 PyPI（B 实测 404），uv sources 映射对 pip 无效，README 方法二不可用。2026-07-12：病根随 T03 依赖删除消失；文档口径经对齐定为**全面 uv**——README 删 pip 方法二，requirements.txt 降级为参考文件 | A:WS-14 + B:#9 | — | — | 做 T03 方案 B 自动消失；否则文档+pyproject ~10 行 |
| [ ] | T17 | 波形常驻内存（批处理全量留 RAM，2h 文件单条 dialog ~1.27GB）+ 模型阶段间从不卸载（`release_model` 只清死标志），6–8GB 卡后期 OOM。**2026-07-09 用户判暂缓**（生产未暴露，验证成本高收益低），移出批次 3 待回头；T20 复核遗留（align 页模型常驻）继续并在此项 | A:WS-15 + B:#13 | 中 | 中高※ | 30–60 行 |
| [x] | T18 | 非 UTF-8 字幕直接 UnicodeDecodeError（GBK/Shift-JIS 存量极常见）。utf-8-sig 优先 + 编码探测回退 | A:WS-16 + B:#27 | 低 | 低 | 10–20 行 |
| [x] | T19 | `_hf_model_cached` 见 snapshots 非空即强制 `local_files_only`，半截下载把用户锁死离线且报错不指真因 | B:#29 | 低 | 低 | 5–15 行 |
| [x] | T20 | align 页每次运行冷加载模型（init+register+finally unload），与 pipeline/whisper 页的 acquire 缓存模式不一致，重复打轴极慢 | B:#14b | 低 | 低中 | 10–20 行 |
| [x] | T21 | vendor `wav_chunk_inference` 内部循环无 checkpoint，DnR 单轨分离数分钟内取消/暂停完全无响应（自研 `_chunk_inference` 反而每块都有） | B:#15b | 低 | 低 | 5–10 行 |
| [ ] | T22 | MMS（token 后验均值）与 Qwen3（时长比）分数语义不同却共用 `align_conf_threshold=0.5`，阈值对其一无标定意义 | B:#24 | 中※※ | 中 | 10–30 行 |
| [x] | T23 | translate 页 `save_state()` 带写盘副作用且 dict 缺 `batch_size`：每次自动保存抹掉手工配置；运行时也不读 batch_size（恒默认 20） | A:P4 + B:#10 | 低 | 低 | 5–15 行 |

※ T17 风险高在生命周期改动会牵动模型缓存复用逻辑，需按"预处理→识别→对齐"全流程回归。
※※ T22 难点不在代码，在 Qwen3 阈值标定，需真实素材验证。
2026-07-09 复核修正：T18 补声明 charset-normalizer 为正式依赖（此前仅经 requests 传递偶然存在，缺失时静默降级），回退顺序改为 shift_jis 先于 gbk（GBK 会把 Shift-JIS 字节流"成功"解成乱码）；T19 把 `.to(device).eval()` 移出 try，显存/驱动错误不再误诊为缓存损坏。T20 遗留窄场景（align 页首跑后模型常驻 VRAM）并入 T17 处理。

---

## P3 — 性能 / 冗余 / 死代码（8 项）

| 状态 | ID | 问题 | 来源 | 难度 | 风险 | 改动量 |
|---|---|---|---|---|---|---|
| [x] | T24 | TIGER-DnR 三倍浪费：dialog/effect/music 各跑全长推理，流水线只消费 dialog。`save_intermediate=False` 时只跑 dialog → 预处理直接 3 倍提速（最大单点优化）。2026-07-09 修复：`separate_dnr` 按 `full_tracks` 裁剪子模型循环（浪费在 Python 层三次全长推理，vendor 未动）；实测单轨产物与三轨版逐字节全等 | A:WS-20 + B:#20 | 低中 | 低中 | 10–25 行 |
| [x] | T25 | espnet + espnet-model-zoo 重依赖只服务被注释掉的 tfgridnet 分支（~70 行死码），装机体积大头 | A:WS-17 + B:#18 | 低 | 低 | 净删 ~70 行 + 2 依赖 |
| [x] | T26 | TTS 整条线废案（~650 行）：tts.py import 的 `TTS_MODELS`/`load_tts_config` 不存在，一碰即崩；`run_mms_fa` 重复对齐逻辑。移入 archive/ | A:WS-18 + B:#17 | 低 | 低 | 净删 ~650 行 |
| [x] | T27 | `parse_srt` 默认 `lang="ja"` 强制算罗马音：翻译路径、ref 解析白白加载 cutlet/MeCab 逐条形态素分析。默认改 `lang=None`，对齐调用方显式传 | A:WS-21 + B:#19 | 低 | 低中 | 10–20 行（需核查全部调用方） |
| [~] | T28 | 同一文件被 ffmpeg 解码 2–3 次；`compute_type` 三处硬编码 int8。2026-07-09 处置：**参数化半项已做**（`--compute-type {int8,int8_float16,float16}` 默认 int8，whisper 缓存键带量化档）；**消重解半项做后回退**——探查发现重复仅存在于无预处理场景，而用户工作流恒开降噪（裸跑全是音乐幻觉），该场景零收益且预解码会改变 whisper 输入前端使转录结果变化，经对齐判不做。教训：探查后范围缩水应回头重新拍板。`load_wav`（stdlib wave 读取）保留为工具函数备 T41 用 | A:WS-22 + B(思考流) | 低→中 | 低→中 | 参数化 15–30 行 |
| [ ] | T29 | 数字被删致对齐系统性偏移：`japanese_to_romaji` 送 cutlet 前删光数字，音频里数字是读出来的（"3人"→只对"人"）。**财经素材满屏数字，疑似日常影响最大的质量项**。让数字进 cutlet 转读音 | B:#23 | 中※ | 中 | 5–15 行 |
| [ ] | T30 | 存储布局割裂：launcher 设 `WRITANSUB_HOME` 无人读；`CACHE_DIR` 无环境变量覆盖而 MODELS/LOG 有；配置走 platformdirs、模型/缓存/日志走 PROJECT_ROOT；卸载器不清理数 GB 模型残留。2026-07-12 用户定夺：**留档，待将来架构优化时再改**（含卸载残留最小修）；现场核实勘误：数 GB 大头在 CACHE_DIR（TIGER HF 快照），MODELS_DIR 仅 Qwen3 探测位，whisper/Qwen3 走用户 HF 默认缓存 app 管不到 | B:#21b（部分 A:WS-19） | 中 | 中※※ | 20–50 行 + 安装器脚本 |
| [ ] | T41 | **新增需求（2026-07-09 用户提出）**：wav 内存缓存模式——同一媒体分多次命令处理时，解码后的 wav 存内存复用，生存期到进程退出或显式清空。**待对齐的前提**：CLI 分开敲的多条命令是多个进程，内存不跨进程共享——目标形态需三选一：a) GUI 同进程多页复用（registry 加波形缓存）；b) CLI 常驻/守护模式（大改）；c) 退化为落盘缓存（跨进程可用，"内存"换"磁盘"）。注意与 T17（减内存驻留）方向相反，须显式开关默认关 | 用户需求 | 中 | 中 | 待对齐后估 |
| [ ] | T42 | 取消响应毫秒级（留档可优化）：torch forward-pre-hook 逐层查取消标志，挂 acquire_model 统一装钩 ~20 行；降噪/对齐取消从"等完一个前向窗口"（数秒~数十秒）压到毫秒级。2026-07-12 对齐：暂不做，留档 | 批次4对齐 | 低 | 中 | ~20 行 |
| [~] | T31 | 死代码杂项：`aitrans.py` 旧名残留、空包 core/、空目录 tmpkanji_ws/、根目录遗留配置、CLAUDE.md 死链、`cancelled` 死参数、`runner.py` 未用 `import torch`、`_log` 组装未调用、`word_results` 只写不读、`enable_start` 从不 emit、`_InfoDelegate` elide 无效 | A:WS-19 + B:#21 | 低 | 低 | 净删为主。2026-07-09 已清（根目录遗留配置移入 archive/legacy_root_configs 而非删除，内含 api_key）；`_log` 保留并由 T40 启用；`_InfoDelegate` 系在用类非死码，遗留待后续批次（2026-07-12 批次 5 N4 已修：initStyleOption+drawControl 展开，elide 生效） |

※ T29 代码量小但必须用真实素材 A/B 验证对齐质量后再上。
※※ T30 涉及老用户目录迁移，要写迁移逻辑或接受一次性断档。
2026-07-09 复核遗留：T25 删 tfgridnet 分支后 `ss_model` 成死参数链（5 文件空转传递、GUI 单项下拉、CLI 自由文本静默忽略）。经评审判定无运行时风险（唯一触发=手动输错；GUI 下拉与 --config 均无旁路），CLI 白名单决定不做；参数链择机随后续清理批次拆除。（2026-07-12 批次 5 N3 已拆：SS_MODELS/--ss-model/字段/传参全链移除，传入 --ss-model 现在报错）

---

## P4 — 小问题（9 项）

| 状态 | ID | 问题 | 来源 | 难度 | 风险 | 改动量 |
|---|---|---|---|---|---|---|
| [x] | T32 | 多选删除 `reversed(selectedItems())` 不保证行序，非连续多选可能删错（pipeline.py:362、preprocess.py:200）。取 row 降序删 | A:WS-23 | 低 | 低 | 5–10 行 ×2 处 |
| [x] | T33 | `load_pp_config` 的 `float()` 只捕 ValueError，JSON null/数组抛 TypeError → GUI 启动即崩 | A:P4 + B:#26 | 极低 | 低 | 1–3 行 |
| [x] | T34 | 合并短字幕 `prev.text + sub.text` 无分隔符（拉丁语言产出 "helloworld"）且不看时间距离硬合并。2026-07-09 修复（随 T06 批）：合并加原始间距 ≤ gap_threshold 守卫，孤立短 cue 不再被跨静音拽进前句；分隔符半项经对齐判不做（用户场景纯 CJK） | A:P4 + B:#22 | 低 | 低 | 5–10 行 |
| [x] | T35 | 无 N 卡用户每次启动弹驱动警告，无"不再提示"；且未考虑 cu124 用户（525+ 即可）的误报 | A:P4 + B(思考流) | 低 | 低 | 10–15 行 |
| [x] | T36 | GUI 关窗不检查运行中任务，daemon 线程被掐可能留半截 SRT。2026-07-12 修复：closeEvent 前置 _confirm_quit（StateMixin.is_running 取自取消按钮可用态），拒绝退出任务无损，确认后维持强杀语义；离屏真 MainWindow 四轮验证；T03 后关窗行为复测随此完成 | B:#30 | 低 | 低 | 10–20 行 |
| [x] | T37 | 字幕提取 `timeout=60` 对大 MKV/机械盘偏紧（decode 600s 同理）。注意：T03 修好前超时本来就不生效，改值应在 T03 之后。2026-07-12 关单：T03 后超时首次真实生效，2h05m/230MB MKV 实测（SSD）——探测 0.11s/120s、提取(2500 条) 0.20s/600s、全长解码 22.2s/3600s；机械盘按 ~100MB/s 外推 20GB MKV ≈ 200s 仍在 600s 内。**三值维持不改** | B:#30 | 极低 | 低 | 1–3 行 |
| [x] | T38 | `WritanSubCLI.bat` 无条件 `pause`，妨碍脚本化调用 | B:#30 | 极低 | 低 | 1–3 行 |
| [x] | T39 | `ResourceRegistry.instance()` 单例创建无锁 | B:#30 | 极低 | 低 | 3–5 行 |
| [x] | T40 | transcribe 无 CUDA 不可用回退（align 有），风格不一致。2026-07-12 修复：bridge.resolve_device 共享助手，接入 cli/GUI transcribe 工厂与 runner 入口，cli/GUI align 两处内联检查换用。**注意**：回退必须加在三个调用方的模型工厂（`_w_factory` 建 WhisperModel 之前），不能加在 `transcribe()` 的 `model is None` 分支——全部生产路径都预建模型传入，该分支不可达（2026-07-09 首次修复因此无效已回退）。修时顺带把 cli/align 两处相同检查抽成共享 `resolve_device()` | A:P4 | 低 | 低 | 5–10 行 |

---

## 联动关系与顺序约束

批次按难度从简到繁划分；同难度的不相关问题合并同批。以下硬性联动在批内/批间消化：

| 约束 | 内容 |
|------|------|
| T01 → T06 暴露链 | 现在 review 阈值全 0，review 实际关闭，T06 的错位标注不可见。T01（批 2）修复后 review 复活，错位标注在批 3 落地前是可见的——本来就一直错，只是从隐性变显性，批 2 收尾时知会即可 |
| T03 → T37 激活链 | 超时机制当前整体失效，T37 改值无意义；T03 修好后超时"复活"，60s 提取超时会开始真杀大 MKV 任务——T37 与 T03 同批收尾重估数值 |
| T29 → T22 标定依赖 | 数字删除系统性拉低含数字句子的对齐分数，先标定阈值再修 T29 = 阈值作废重标。批 2 内部顺序：先 T29，跑真实素材 A/B，末尾做 T22 标定 |
| T28 ⚡ T17 设计冲突 | 消重复解码（传波形）增加内存驻留，与减少驻留目标相反。同批一体设计（解码结果落临时文件、按需读取） |
| T36 依赖 T09 | T36 需要"任务运行中"全局信号，正是 T09 互斥引入的——同批；T03（批 4）落地后关窗从"冻结"变"立刻杀进程"，需回头复测一次（2026-07-12 批次 5 已复测） |
| T34 折进 T06 批 | 与 T06/T10 同住 `post_process` 合并循环，单独修会被重写覆盖 |
| 同文件对 | T18+T27（srt_io.py）；T07+T08（translate/core.py 同一循环）；T11+T23（save_translate_config）；T24+T21（TIGER 推理区域，分属批 1/批 2，先后落地无冲突）；T14 与 T03 同在 bridge.py 但不同函数（`_get_ffprobe` 纯 Python，native 替换不动它），可提前修 |

---

## 总拟修复批次路线（难度升序，4 批）

每批附带该区域的最小冒烟测试（T02 这类"清理引入的回归"正是缺冒烟所致）。
> 2026-07-09 起测试基建已建立：`tests/` pytest 套件 38 例，覆盖批次 1.5 改动面（srt_io 编码链 / 模型加载分流 / 翻译核心 / 转录核心 / config 容错 / registry 并发 / GUI 删行 helper），全部离线可重复。~~pytest 暂未入锁~~（2026-07-12 批次 5：整锁切官方 PyPI，pytest/pyflakes 已入锁转正，`uv sync` 不再清掉测试工具，此段作废）。后续批次的定向测试直接往 tests/ 里加。

**批次 0 · 止血**（用户许可后 1 分钟）
删线上 `%LOCALAPPDATA%\mtmfs\WritanSub\writansub_pp.json`，默认值即刻回归。属用户机器操作，须单独征得同意。

**批次 1 · 简单批**（21 项 · 全部低难度低风险 · 净增改 ~150–280 行 + 净删 ~800 行）
- 一行级速修：T02、T05、T33、T37※、T38、T39、T40、T32、T35（※T37 只先调大数值，真正生效在批 4）
- 纯删除：T25（espnet）、T26（TTS 移 archive）、T31（杂项死码）——先删干净，缩小后续批次的干扰面
- 独立小修（每项 ≤30 行）：T07（逐批回写译文）、T11+T23（key 脱敏/batch_size）、T14（ffprobe 明确报错）、T15（CLI 选轨）、T18+T27（srt_io 编码+lang=None）、T19（半截下载）、T20（align 页缓存）、T21（vendor checkpoint）
- 验证：pyflakes 全绿；CLI 各子命令冒烟；GBK 字幕解析；翻译中途取消不丢译文

**批次 2 · 中等批**（10 项 · 中难度或验证较重 · ~200–360 行）
- 单模块修复：T01（spinbox 初始化 + closeEvent 命名空间 + 默认值收敛）、T04（输出命名，同步 README）、T08（LLM 编号校验+重试）、T24（TIGER 单轨推理，3 倍提速）
- GUI 生命周期：T09（全局互斥）+ T36（复用运行信号）
- 网络场景：T12 + T13（Clash 开/关、直连、断网实测）
- 标定（批内最后做）：T29（数字转读音）→ 真实财经素材 A/B → T22（Qwen3 阈值标定）
- 验证：各页参数改→关→开不串不丢；输出文件名核对；翻译乱序编号注入测试；A/B 对齐质量对比
- ⚠️ T01 落地后 review 复活，错位标注（T06）从隐性变显性，属预期，批 3 消除

**批次 3 · 重构批**（5 项 · 中高难度 · ~150–310 行，回归测试量大）
- review/post_process 数据流：T06 + T10 + T34（内存中标记、索引稳定后一次性生成；重叠保留；合并加分隔）
- 内存/解码一体设计：T17 + T28（阶段间卸载、波形落盘缓存、int8 参数化、消重复解码）
- 两组不共函数但都动 runner.py，批内串行
- 验证：ref 映射/短字幕合并/separate 三场景 review 人工抽查；长视频全流程 VRAM/RAM 监控与峰值对比

**批次 4 · 架构批（压轴）**（4 项 · 最复杂，含打包链）
- T03 native 纯 Python 替换（bridge.py ~60–100 行 + 删 113 行 Rust + maturin/版本校验退役）
- 取消体验三件套与 T03 同批（2026-07-09 对齐新增）：解码取消随 T03 自动痊愈（subprocess kill 即时）；翻译 API 弃单（请求线程化，取消即弃连接立即返回）；可选 torch forward-pre-hook 逐层查取消标志（降噪/对齐取消从"一个窗口前向"压到毫秒级，挂 acquire_model 统一装钩 ~20 行）。关窗强杀为用户有意保留，T36 只补确认框且排 T03 之后
- T16 随之消失（README/pyproject 收尾）；T37 超时数值此时才真正生效，实测重估
- T30 存储布局统一（platformdirs 收敛 + 卸载残留），与打包链一次改完
- 验证：构建→打包→安装全链；解码中取消/暂停/关窗；Ctrl+C 无孤儿 ffmpeg；复测批 2 的 T09/T36 关窗行为
- 风险集中在打包链（Inno/launcher/check_versions.ps1），业务代码爆炸半径仅 bridge.py

> 2026-07-12 批次 4 完成（范围经对齐缩减：T03+T16+T37 实施；翻译侧 T07/T08/弃单暂缓、T30 留档待架构优化、三件套 c 留档为 T42），见 `Batch4_Report_2026-07-12.md`。版本升至 0.1.8。

> 2026-07-12 全口径复盘与新路线（见 `Replan_Report_2026-07-12.md`）：剩余 13 项经代码逐项核验均确实未修（T09 的"置灰"系 `60a361a` 既有行为非修复，方案需重新拍板：关单或升级为按任务粒度取消）；收编无 ID 零散项 N1–N5（N5 报告入库已结 `afcb9ae`）。用户裁定只排三批：**批次 5 杂修**（T12/T13/T36/T40 + N2 构建脚本 GNU tar 自愈 / N3 拆 ss_model 死参数链 / N4 _InfoDelegate elide）、**批次 8 内存**（T17+T41 一体设计，T42 顺风车，前置 T41 形态三选一对齐）、**批次 9 架构**（T30 随打包链）。素材批（T29→T22）与翻译批（T07/T08/弃单）不排期，维持搁置。N1（pytest 入锁+删 hatchling 钉）批外挂件，等清华源恢复顺手办。

> 2026-07-12 批次 5 完成：T12/T13/T36/T40 + N2（构建脚本 tar 自探测）/N3（ss_model 拆链）/N4（_InfoDelegate elide）实施，T09 关单 `[-]`；N1 提前完成（清华镜像 403 倒逼整锁切官方 PyPI，经用户拍板）。版本升至 0.1.9，测试 103 例全绿，见 `Batch5_Report_2026-07-12.md`。剩余池：批次 8（T17+T41+T42，前置 T41 形态三选一对齐）、批次 9（T30）、不排期（T29→T22、T07/T08/弃单）。

**总量**：批 1 ~150–280 行 + 删 800；批 2 ~200–360 行；批 3 ~150–310 行；批 4 Python ~100–150 行 + 删整个 native 构建链。全部完成后代码库净缩小。
