# 批次 5 报告 · 杂修批(T12/T13/T36/T40 + N2/N3/N4,T09 关单)

- 日期:2026-07-12(与批次 4 收尾、全口径复盘同日)
- 计划:`docs/superpowers/plans/2026-07-12-batch5-misc-fixes.md`(用户审批通过的审核版大纲另存 plan 文件)
- 版本:0.1.8 → **0.1.9**(pyproject + `__init__.py`,check_versions 通过;launcher 版号独立未动)

## Commit 表

| commit | 内容 |
|---|---|
| `e48e796` | T40 transcribe 补 CUDA 回退,抽共享 `resolve_device` |
| `d2cb3c7` | T12 镜像探测改经系统代理的 HTTPS 实测 |
| `0346df5` | N3 拆除 ss_model 死参数链 |
| `2b08f24` | N4 _InfoDelegate 过长名称省略生效 |
| `d5f387b` | T36 关窗前确认运行中任务 |
| `035e4d6` | N2 构建脚本 tar 口味自探测 |
| `1a48482` | 锁默认源切官方 PyPI,pytest/pyflakes 入锁转正,删 hatchling 钉(N1 提前) |
| `44fb97c` | T13 silero-vad 改 pip 包加载 |
| (本次) | 收口:版本 0.1.9、tracker、CHANGELOG、本报告 |

## 各任务要点与验证

**T40** — `bridge.resolve_device(requested, log_callback)`:cuda 请求但不可用 → 知会并回退 cpu。接入五处:cli/GUI transcribe 的模型工厂之前、runner 入口(`cfg.device` 一次收敛)、cli/GUI align 两处旧内联检查换用。首修教训(加在生产不可达分支)已写入函数 docstring。验证:2 例单测(monkeypatch `torch.cuda.is_available`)。

**T12** — network.py 整文件重写:裸 socket → `urllib.request.urlopen(HEAD)`,自动读系统代理 + 完整 TLS 握手,HTTPError(4xx/5xx)视为可达,超时 2s→5s。验证:5 例打桩测试(可达/不可达/403 可达/尊重 HF_ENDPOINT/尊重 HF_HUB_OFFLINE);本机 Clash 开态实测 huggingface.co 判可达(旧探测在此场景恰好误判)、无效域名判不可达。

**N3** — ss_model 死链全拆(6 文件):`SS_MODELS` 常量、两条 `--ss-model` CLI 参数、PipelineConfig 字段、决策日志片段、run_speech_batch 参数、两个 GUI tab 的下拉块+状态存取+线程传参。验证:2 例单测(旗标拒收 SystemExit、字段不存在);全文 grep 零残留(mss 子串误报已滤);离屏构造两 tab 冒烟。老 gui_state 残键无害(读取守卫已随删)。

**N4** — 根因确认:旧 `super().paint(opt)` 内部重新 `initStyleOption` 用 DisplayRole 覆盖改过的 `opt.text`,elide 结果被静默丢弃。修法:paint 内自走 `initStyleOption + style.drawControl(CE_ItemViewItem)`(基类等价展开)。验证:离屏截图——offscreen 平台字形缺失(豆腐块),但结构证据确凿:60 字符长名仅渲染 ~15 字形格(≈240px 上限),与右侧 info 留白清晰;对照 short-name(10 字符=10 格)确认一格一字。建议有空真 GUI 目检一眼。

**T36** — `StateMixin.is_running()`(取消按钮可用态即运行态,五 tab 一处继承)+ `app._confirm_quit` 在 `shutdown()` 之前拦截:拒绝退出任务无损,确认后维持既有强杀语义(用户有意保留的设计)。验证:4 例 helper 级单测 + 离屏真 MainWindow 四轮端到端(空闲直关/运行中 No 不关不杀/Yes 关且 shutdown/gui_state 写临时目录)。**计划偏差**:原定真 GUI 人工复测改为脚本化离屏验证(无人值守环境);tracker 欠的"T03 后关窗复测"以此销账,建议有空真 GUI 走一遍降噪中关窗。

**N2** — build.ps1/light.ps1 同款替换:`tar --version` 含 "GNU tar" 才带 `--force-local`,bsdtar 不带。验证:两脚本语法解析通过;scratchpad 双实测(GNU 带参/bsdtar 不带参各解同一 tar 成功;探测表达式对两种 tar 判定正确)。

**T13** — `_get_silero_vad` 改 `from silero_vad import load_silero_vad, get_speech_timestamps`(v6.2.1,模型内置随包、加载零联网),消费方 `_run_silero_vad` 零改动。验证:2 例单测(stub 模块验证走 pip 包 + 真实加载纯静音返回空 span);grep 确认 torch.hub 代码零残留(仅注释提及旧行为)。

**T09** — 无代码改动,tracker 关单 `[-]`:批 2 定稿方案("本 tab 无任务取消置灰")经 git 溯源系老版本(`60a361a`)既有行为、与现状重合;双任务并发跨页互杀属罕见场景且 GUI 半废弃,用户拍板"那就不改"。

## 计划外事件:清华镜像 403 倒逼锁源切换(N1 提前完成)

T13 加依赖触发重解析,清华镜像对大量 sdist(demucs/pysrt/unidic-lite/julius…)和新 wheel(silero-vad)持续 403。先按最小创面尝试单包钉官方源,但 `tool.uv.sources` 钩不住传递依赖(julius),该路不通。**经用户拍板整锁切官方 PyPI**(`1a48482`):

- 默认 index 清华 → pypi.org(经系统代理;镜像恢复后可切回,pyproject 有注记)
- 顺带把原"等镜像恢复"的 N1 提前办掉:pytest 入锁转正 + 删 `hatchling<1.31` 构建钉
- pyflakes 同批入锁(首次 sync 时被清,暴露与 pytest 同款问题)
- uv.lock 全量 URL 改写(清华 0 处、pypi.org 154 处),torch/torchaudio 仍走 pytorch-cu128 专用源不受影响

**行为影响**:日常 `uv sync`/`WritanSub.bat` 自动同步改走官方源,需 Clash 可用(本机常开);`uv sync` 从此不再清掉 pytest/pyflakes,批次 2 以来的"sync 后重装 pytest"惯例作废。

## 行为变化知会(对外可感知)

1. `--ss-model` 从静默忽略变为报"未知参数"错误。
2. GUI 关窗在有任务运行时多一道确认框(强杀语义不变;空闲关窗无感)。
3. 镜像探测超时 2s→5s,仅影响"断网启动"场景的等待时长;Clash 场景下探测结果从误判变正确,不再无谓切镜像。
4. 依赖锁走官方 PyPI(见上节)。

## 测试与状态

- 套件 **103 例全绿(~4s)**:批次 4 收尾 88 + 本批 15(resolve_device 2 / network 5 / cli_args 2 / gui_close_confirm 4 / preprocess silero 2)
- pyflakes:全仓无新增告警(基线既有项不变)
- CLI 冒烟:transcribe/preprocess/pipeline `--help` 正常,`--ss-model` 正确拒收
- tracker:T12/T13/T36/T40 → `[x]`,T09 → `[-]`,T31 遗留注记(_InfoDelegate/ss_model)已销,pytest 未入锁段落作废,路线区已记批次 5 完成
- 剩余池:批次 8(T17+T41+T42,前置 T41 形态三选一对齐)、批次 9(T30)、不排期(T29→T22、T07/T08/弃单);N1 已随本批消化
- push 未执行,由用户决定
