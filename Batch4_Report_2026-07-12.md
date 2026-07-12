# WritanSub 批次 4 报告 — native 层退役 + 全面 uv + 超时实测

- 日期:2026-07-12
- 范围:批次 4 架构批,经对齐缩减为 T03 + T16 + T37;实施计划详见 `docs/superpowers/plans/2026-07-12-batch4-architecture.md`
- 本轮 3 个 commit(含收尾),已推送:

| Commit | 内容 | 规模 |
|--------|------|------|
| `08cea47` | T03 bridge.py 纯 Python 改写 + 6 例新测试 | 2 文件 +195/−56 |
| `b106d26` | T03/T16 构建链拆除 + 安装口径全面 uv | 12 文件 +32/−197 |
| (收尾) | 版本 0.1.8 + tracker/CHANGELOG 收口 | 文档 |

---

## 对齐处置记录(2026-07-12,用户逐项拍板)

- **实施**:T03(可)、T37(可)、T16(病根随 T03 自动消失;文档口径定为删 pip 方法二全面 uv)
- **暂缓·翻译侧三项**:T07 / T08 / 取消三件套 b(弃单)——翻译功能不常用,将来用到时一批做;T07 重修要点已在 tracker 条目
- **留档**:T30 整项(含卸载残留最小修)——"留档,或许哪天架构优化时要改";三件套 c(forward-pre-hook 毫秒级取消)→ 新增 **T42** 留档
- 三件套 a(解码取消)随 T03 自动痊愈,未单列

## T03 — native 层纯 Python 替换

三宗罪复述:`wait_process` PyO3 全程持 GIL(超时/取消/GUI 三症一根)、`shutdown` 写锁死锁、`Vec<u8>`→list[int] 内存灾难(1h 音频 ~2.5GB)。红灯运行给出病灶现场演示:测试里 5 秒的轮询等待被 GIL 冻结,35 秒(= 子进程时长)后才返回。

替换设计:模型注册表 = dict + threading.Lock(Rust 侧 `acquire_model`/`in_use` 标志 Python 从未调用,不搬);`run_subprocess` = Popen + 活进程登记 + `communicate(timeout=)`(阻塞在 C 层 I/O,GIL 正常释放);**`cancelled` 从普通属性改 property,置 True 即 kill 全部活子进程**——CLI 信号处理器与 GUI 取消按钮(`reg.cancelled = True`)零改动获得即时取消;取消致 kill 上抛 CancelledError 防调用方误报;子进程加 CREATE_NO_WINDOW。对外 API 一字不变,消费方 15+ 文件零改动。

拆除面:pyproject 依赖/uv source/Rust classifier、uv.lock(离线重解析,净移除 writansub-native)、build.ps1 B4 maturin 段、light.ps1 C3 段 + install.bat 模板 5→4 步、check_versions 两条 native 规则、README、.gitignore。Rust 源码 git mv 入 `archive/native_rust_202607/`(本地归档惯例,`.git/info/exclude` 使新增文件不入库,已跟踪文件保留历史)。

**验证**:
- 88 例全绿(82 既有 + 6 新增),`test_bridge_registry.py` 既有 4 例零改动通过
- **`uv sync` 卸掉 native wheel 后测试仍全绿**——业务代码真正不依赖 native 的决定性证据
- 轻量包构建通过(11.3 MB;install.bat 确认 4 步、vendor 无 wheel、锁无 writansub 条目)
- 完整包构建通过(6.1 GB,内置冒烟 `smoke-ok`;runtime site-packages 无任何 writansub 残留)
- pyflakes 16 条与批次 1 基线持平,全为既存项

## T16 — pip 断裂(全面 uv 口径)

病根(pyproject 依赖不存在于 PyPI 的包)随 T03 删除自动消失。经对齐,README 删除 pip 安装方法二及其排障引用(共 6 处),uv 成为唯一官方安装路径;requirements.txt 文件头加"仅供参考"声明。剩余 `pip install` 字样仅为安装 uv 本身的引导命令。

## T37 — 超时数值实测关单

T03 落地后超时首次真实生效。实测素材:归档区真实数加加音频 `000004.wav`(41.8min)拼接 3 遍 + 合成 2500 条 SRT 轨 → 2h05m / 230MB MKV(G 盘,SSD):

| 场景 | 实测 | 现上限 | 余量 |
|------|------|--------|------|
| 字幕轨探测 | 0.11s | 120s | ~1000× |
| 字幕提取(2500 条) | 0.20s | 600s | ~3000× |
| 全长解码(44.1k 重采样) | 22.2s | 3600s | ~160× |

机械盘外推:提取/解码均为顺序读主导,按 ~100MB/s 计,20GB MKV ≈ 200s,仍在 600s 内。**结论:三值维持不改,关单。**

## 取消体验实测(CLI 代测,GUI 半弃用经用户确认)

驱动脚本复现 CLI Ctrl+C 处理器的真实动作(`reg.cancelled = True` + resume,即 cli.py `_handler` 全部逻辑),对 2h MKV 真实解码中途取消:

- 取消延迟 **0.016s**(旧实现:等 ffmpeg 跑完)
- 正确上抛 CancelledError(非 RuntimeError 误报)
- `tasklist` 复查**无孤儿 ffmpeg**
- 解码全程主线程心跳最大间隔 **0.094s**(旧实现:= 整段解码时长,即 GUI 冻结的根源)——GUI 不冻结的等效证据

## 行为变更知会

1. 子进程带 CREATE_NO_WINDOW:GUI 下 ffmpeg 不再闪黑框;子进程收不到控制台 Ctrl+C,取消一律走显式 kill(已被上述实测覆盖)
2. **超时首次真实生效**:大文件提取/解码此后真的会被超时杀(120/600/3600s,实测余量充足)
3. 安装口径全面 uv,pip 方法二删除

## 计划外偏差与环境注意

1. 构建脚本的 `tar --force-local` 需要 GNU tar(Git for Windows 自带)在 PATH 中优先——System32 的 bsdtar 不支持该参数。本轮以 `$env:PATH = 'C:\Program Files\Git\usr\bin;'+...` 前置解决;从纯净 cmd/PowerShell 跑构建脚本的人都会撞到,择机可在脚本内自愈(未列入本批)
2. 首次完整构建失败系包装方式(Windows PowerShell 5.1 + `*>&1` 把 uv 的 stderr 进度行升格为终止错误),pwsh 7 重跑通过,脚本本身无问题
3. ~~ISCC 本机未安装,打包冒烟跳过~~ **后续补测通过**:ISCC 6.7.1 实际装在 `%LOCALAPPDATA%\Programs\Inno Setup 6\`(用户级,首查漏检)。用独立测试 AppId 的脚本副本走完整装卸:编译 → `WritanSub-Setup-light-0.1.8.exe`(12.7 MB)→ 静默安装(payload 0.1.8、bridge 无 native、install.bat 4 步)→ 静默卸载(文件与注册表项全清)。正式 writansub.iss 零改动。**另发现** F:\WritanSub 存在 2026-04 的完整版真安装(6 GB,同 AppId),测试全程未触碰,注册表项复核完好;是否卸除由用户定
4. 原定"机械盘大 MKV"实测因本机全 SSD 改为 SSD 实测 + 机械盘外推

## 测试与状态

- 套件累计 **88 例全绿(~3.7s)**:批次 1.5 基建 38 + 各批次累计 44 + 本批 bridge 子进程/注册表 6
- tracker:T03/T16/T37 → `[x]`;T07/T30 批注暂缓/留档归因;T42 新增 `[ ]`
- 版本:0.1.7.3 → **0.1.8**(check_versions 三方对齐通过;native 两条规则已退役)
- 剩余池:批 2 遗留(T08、T09+T36、T12+T13)、翻译侧暂缓组(T07/T08/弃单)、等素材(T29→T22)、暂缓(T17+T20 遗留)、留档(T30、T42)、待对齐(T41)
- 本报告与此前四份报告同样保持未跟踪,入库与否用户定
