# WritanSub 复核报告 — 批次 1 复核 + 批次 1.5 返工 + 测试基建

- 日期：2026-07-09
- 范围：对批次 0+1 的 5 个 commit（`d5ff0f4..2026fc5`）做全量多智能体盲审复核，随后按逐项对齐结论返工
- 本轮 3 个 commit，全部已推送 GitHub：

| Commit | 内容 | 规模 |
|--------|------|------|
| `ef2958b` | 批次 1.5 返工（T07/T40 回退，T18/T19 修正） | 8 文件 +75/−67 |
| `b1365ac` | 复核清理三项（T32 去重 / batch_size 常量化 / 单例锁简化） | 8 文件 +31/−19 |
| `29e20c4` | pytest 测试基建（38 例） | 9 文件 +566 |

---

## 复核方法与结果

8 个盲审 finder 并行（逐行扫描 / 删除行为审计 / 跨文件追踪 / 复用 / 简化 / 效率 / 深度 / 规范）→ 34 条原始候选 → 去重 12 条 → 逐条独立验证员裁决：

**9 CONFIRMED · 1 PLAUSIBLE · 2 REFUTED**

核心结论：批次 1 的 22 项中，2 项修复无效（T07/T40，tracker 误标 [x]）、2 项修复自带新缺陷（T18/T23）、1 项修正瑕疵（T19）。共性教训：**T07/T40 都是修完没做行为级验证**——T07 没实际取消一次看盘上有没有文件，T40 没在无 CUDA 条件下跑。

## 十项发现处置

| # | 关联 | 发现（验证员裁决） | 处置 | 落点 |
|---|------|------|------|------|
| 1 | T07 | 修复无效：finally 只回写内存，三个调用方（GUI/CLI/runner）均在 CancelledError 后跳过 write_srt，已付费译文照丢 | 回退，tracker 退 `[ ]` 并记正确修法 | `ef2958b` |
| 2 | T40 | 修复是死代码：回退加在 `model is None` 分支，全部 5 个调用点预建模型传入，生产路径不可达 | 回退，tracker 退 `[ ]` 并记"应修在模型工厂" | `ef2958b` |
| 3 | T18 | charset-normalizer 未声明为依赖（仅经 requests 传递偶然存在）+ 回退链 gbk 先于 shift_jis：探测缺失时 Shift-JIS 被 GBK 静默解成乱码（已实测复现） | 声明依赖 + 顺序对调 | `ef2958b` |
| 4 | T23 | batch_size 无类型校验，手改坏值崩整跑且被永久回存 | **判不修**：该参数本就不面向用户，正常人不手改 json（产品立场） | — |
| 5 | T19 | try 包住 `.to(device)`：CUDA OOM 被误诊"缓存坏"并触发无谓联网重试 | `.to(device).eval()` 移出 try | `ef2958b` |
| 6 | — | T25 删 tfgridnet 后 ss_model 成 5 文件死参数链，CLI 任意值静默降级 tiger-speech | **判不修**：无运行时风险（GUI/--config 无旁路），tracker 记备忘择机拆链 | 备忘 |
| 7 | T20 | 对齐模型 VRAM 常驻（PLAUSIBLE：whisper 走 CTranslate2 本就吃不到 torch 释放的显存，旧模式反有更糟的孤儿泄漏） | 并入 T17（批次 3） | tracker 注记 |
| 8 | T32 | 修复逐字粘贴进两个 tab，重演双拷贝分叉模式 | 抽共享 `remove_selected_rows` 进 widgets.py | `b1365ac` |
| 9 | — | 默认值 20 散落四处，GUI 两处 .get 兜底恒为死代码 | `DEFAULT_BATCH_SIZE` 常量单源（translate/core.py 定义，config 引用） | `b1365ac` |
| 10 | T39 | 双检锁属过度设计（instance() 每任务仅个位数调用，热循环均一次捕获） | 改无条件加锁 | `b1365ac` |

REFUTED 2 条：编码检测性能损耗（实测 3MB 文件仅 ~3ms，占比 1.3%）；transcribe 页 spinbox"重复造轮子"（手工布局系基线既有，本批只加一行）。

## 对齐决策记录（产品立场，代码看不出来的）

- GUI 半废弃，改动优先 CLI 侧
- batch_size 从未打算暴露给用户；防输错类加固（choices 白名单等）判无必要——单用户工具，输错只坑作者本人
- tracker T25 行"被注释掉的 tfgridnet 分支"表述不实：基线是活代码且 CLI 可达（"不可达"仅对 GUI 成立），删除决策本身仍正确（espnet 38 包依赖 + 模型不兼容单声道）

## 验证记录

1. 回退逐字节核对：transcribe/core.py 与基线全等（diff 为空）；translate/core.py 残差恰 1 行（cancelled 死参数删除，属保留项）
2. compileall + 全模块导入 + `cli --help` 冒烟（顺带证明 config→translate.core 新 import 链无循环）；pyflakes 无新增（transcribe 两条系基线既有，`_log` 随 T40 回退归位死变量）
3. 单元测试 38/38 通过（2.97s）：srt_io 9 / preprocess 加载 4 / translate 5 / transcribe 4 / config 8 / registry 4 / gui 4，全部离线可重复（假 openai、桩模型、临时配置路径、离屏 Qt），含 2 条现状契约（翻译取消传播——T07 真修时会失败提醒更新断言）
4. 真实链路冒烟（样本 `开发者文件/项目文件/Hathaway.mp4`，76s）：
   - 裸音轨 run：3 条"初音ミク"幻觉——BGM 压人声时 Whisper 的已知行为，非回归，反证降噪价值
   - `--denoise` run：TIGER-DnR 走重写后的 `_from_pretrained_cached`（真实权重、离线缓存命中、零误诊日志）→ 20 条/136 词，对齐 20/20，avg 0.584，内容与 4 月基准逐句吻合；基准 46 条系 `--separate` 模式产物不可比数量，且基准含 T10 重叠伪影（重复 cue、40ms 微 cue）
   - 翻译：真实 DeepSeek API 6/6，时间轴保留、格式完好，实走常量化后的 batch_size 路径

## 环境偏差与注意

- 清华源持续 403（demucs sdist、pytest 9.1.1 wheel 均中招）。pytest 经官方源 + Clash 临时装入 .venv，**未入锁**——入锁需整体重解析会撞 403，且会连累 `WritanSub.bat` 的自动 sync。副作用：下次 `uv run`/`uv sync` 会清掉 pytest，重装：`uv pip install pytest --index-url https://pypi.org/simple`（挂代理）
- 镜像恢复后两件事一起办：`uv add --group dev pytest` 转正 + 移除 pyproject 的 `hatchling<1.31` 钉

## 遗留

- 批次 2 待启动：原 10 项 + 回池的 T07/T40。T07 与 T08 同文件建议同批；T40 修时顺带把 cli/align 两处相同 CUDA 检查抽成共享 `resolve_device()`
- `Audit_Report_2026-07-02.md`、`Fix_Report_2026-07-09.md` 与本报告均未跟踪，是否入库由用户决定
