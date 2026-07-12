# WritanSub 批次 4 架构批实施计划(2026-07-12 对齐终稿)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 退役 writansub_native Rust 层(T03),超时/取消机制随之首次真实生效并实测重估数值(T37),安装口径全面转 uv(T16 收尾)。

**Architecture:** bridge.py 内部实现从 Rust 注册表换成纯 Python(dict + 锁 + subprocess.Popen + 活进程登记),对外 API 一字不变,业务代码零改动;打包链(maturin/版本校验/install.bat)同步拆除;README 删 pip 安装方法二,uv 成为唯一官方安装路径。

**Tech Stack:** Python 3.12 / subprocess / pytest;打包链 PowerShell + Inno Setup 6 + Rust launcher(保留)。

## 对齐处置记录(2026-07-12,用户逐项拍板)

- **本轮实施**:T03(可)、T37(可)、T16(病根随 T03 自动消失;文档口径定为**删方法二全面 uv**)
- **暂缓·翻译侧三项**:T07(取消丢译文,重修要点已在 tracker)、T08(编号校验)、取消三件套 b(弃单)——理由:翻译功能不常用;将来用到翻译时一批做
- **留档·T30 整项**(含卸载残留最小修):用户定夺"留档,或许哪天架构优化时要改"
- **留档**:取消三件套 c(torch forward-pre-hook 毫秒级取消)→ tracker 新增 T42,不做
- 取消三件套 a(解码取消)随 T03 自动痊愈,不单列

## Global Constraints

- `ResourceRegistry` 对外 API(方法名、签名、异常类型)保持不变——消费方 15+ 文件一个都不许改。
- 所有新增测试离线可重复:不触网、不碰真实用户配置(`%LOCALAPPDATA%\mtmfs\WritanSub\*.json`)、不下载模型。
- pytest 不在 uv 锁里(清华源 403 历史遗留):跑测试用 `G:\WritanSub\.venv\Scripts\python.exe -m pytest`,**不要用 `uv run pytest`**(auto-sync 会把 pytest 卸掉)。任何 `uv sync` 之后必须重装:`uv pip install pytest --index-url https://pypi.org/simple`。
- 提交信息沿用现有风格(中文、Txx 编号开头),每条结尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 工作目录 `G:\WritanSub`,直接在当前分支提交(与批次 1-3 相同流程),最后统一 push。
- 行为变更知会条目(写进 Batch4_Report):① 子进程带 CREATE_NO_WINDOW(GUI 下 ffmpeg 不再闪黑框;取消一律走显式 kill);② 超时首次真实生效——大文件提取/解码此后真的会被超时杀,数值以 T37 实测为准;③ README 安装口径改全面 uv,pip 方法二删除。
- 不碰翻译侧任何文件(translate/core.py、gui/tabs/translate.py、cli.py 的 cmd_translate、runner.py 翻译段)——T07/T08/弃单 已判暂缓。

---

## 各项:bug 是什么 · 原来怎么实现 · 怎么改

### T03 — native 层(writansub_native)整层负价值

**bug 是什么**:GUI 在解码/字幕提取期间整个界面冻结;所有子进程超时参数无效;子进程运行期间取消/暂停失灵;1h 音频解码峰值内存 ~2.5GB。

**原来怎么实现**:Rust 层 113 行(`native/src/lib.rs`)只干两件事——一个 HashMap 存模型 PyObject、一个 HashMap 存子进程 Child,Python 侧由 bridge.py 独家包装。三宗罪:

1. `wait_process`(lib.rs:85-94)是 PyO3 方法,默认**全程持 GIL**,`wait_with_output()` 阻塞到子进程退出。bridge.py:172-180 把它丢进 daemon 线程再 `t.join(timeout)`,但该线程一进 Rust 就攥死 GIL——主线程连 join 的超时判断都得不到执行机会,整个解释器冻结到 ffmpeg 跑完。超时永不触发、取消/暂停 checkpoint 无法运行、GUI 事件循环卡死,三症一根。
2. `shutdown`(lib.rs:97-106)要 REGISTRY 写锁,而 `wait_process` 从入口就持写锁到子进程结束(lib.rs:87)——bridge 的超时路径(bridge.py:184)调 shutdown 是死锁;只因超时本身永不触发,死锁一直藏在死机后面。
3. 返回类型 `Vec<u8>` 被 PyO3 转成 **list[int]**(bridge.py:192-196 的 `bytes(stdout)` 补丁即物证):1h 音频 s16le ≈ 317MB 字节流,变列表后每字节一个 Python int 对象,峰值 ~2.5GB 纯浪费。

另:Rust 侧 `acquire_model`/`in_use` 占用标志(lib.rs:33-51)Python 侧从未调用,纯死代码;剩下的 register/get/unload 就是一个带锁 dict。消费面已核实仅 bridge.py 一个文件 import,业务侧 4 个调用点(extract.py:36/91、preprocess/core.py:374、align/core.py:76)全走 bridge API。

**怎么改**:bridge.py 内部纯 Python 替换,对外 API 一字不变。模型注册表 = dict + threading.Lock;`run_subprocess` = `subprocess.Popen` + 活进程登记表 + `communicate(timeout=)`——阻塞发生在 C 层 I/O,GIL 正常释放,GUI 不再冻结,超时真实生效;`cancelled` 从普通属性改 property,置 True 即 kill 登记中的全部子进程——CLI 信号处理器和 GUI 各取消按钮(`reg.cancelled = True`)零改动获得即时取消,Ctrl+C 不再留孤儿 ffmpeg。Windows 下子进程加 CREATE_NO_WINDOW(pythonw 场景防闪黑框,取消一律走显式 kill 不依赖控制台信号)。

### T16 — pip 安装路径断裂(文档口径:全面 uv)

**bug 是什么**:README 安装方法二 `pip install -e .` 必败。

**原来怎么实现**:pyproject dependencies 含 `writansub_native`(pyproject.toml:45),该包不在 PyPI(远程盲审实测 404);`[tool.uv.sources]` 的 path 映射(pyproject.toml:68)只对 uv 生效,pip 径直去 PyPI 找,找不到即断。

**怎么改**:病根随 T03 删除依赖自动消失。文档口径经对齐定为**全面 uv**:README 删除 pip 安装方法二,uv 成为唯一官方安装路径,少一条要维护的承诺;requirements.txt 文件头加注释声明仅供参考、锁定以 uv 为准。

### T37 — 子进程超时数值(纯验证项)

**bug 是什么 / 原来怎么实现**:批 1 已把数值调大(轨道探测 30→120s、字幕提取 60→600s、解码 600→3600s),但 T03 修复前超时机制整体失效(见 T03 第 1 宗罪),这些数值从未真实生效过——纯占位。

**怎么改**:T03 落地后超时复活,**首次开始真杀任务**。机械盘大 MKV 实测三个场景(探测/提取/解码)的真实耗时,余量不足 3 倍就按实测调值(extract.py:36/91、bridge.py decode_audio),足够就记录实测依据关单。

---

### Task 1: bridge.py 纯 Python 改写(T03 业务侧)

**Files:**
- Modify: `writansub/bridge.py`(74 行起的 ResourceRegistry 整类重写;文件头 import;`_gpu_mem_hint`/`CancelledError`/`_get_ffmpeg`/`_get_ffprobe` 四个不动)
- Test: `tests/test_bridge_subprocess.py`(新建)
- Test(既有,须保持全绿): `tests/test_bridge_registry.py`

**Interfaces:**
- Consumes: 无(本任务是底座)
- Produces: `ResourceRegistry` 语义强化——`reg.cancelled = True` 现在会立即 kill 所有登记中的子进程;`run_subprocess` 超时真实生效(TimeoutError);取消导致的 kill 上抛 `CancelledError` 而非 RuntimeError。API 签名全部不变。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_bridge_subprocess.py`:

```python
"""纯 Python ResourceRegistry:子进程超时/取消即杀/模型注册表/解码回环。

全部离线:子进程用 sys.executable 起 Python 单行,解码用 imageio-ffmpeg 自带 ffmpeg。
"""
import subprocess
import sys
import threading
import time

import pytest

from writansub.bridge import ResourceRegistry, CancelledError


def _wait_live_proc(reg, deadline=5.0):
    """等 run_subprocess 把子进程登记进活进程表(防 kill 早于 spawn 的竞态)。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline:
        if getattr(reg, "_procs", None):
            return True
        time.sleep(0.05)
    return False


def test_run_subprocess_captures_output(registry):
    proc = registry.run_subprocess(
        [sys.executable, "-c",
         "import sys; sys.stdout.write('out'); sys.stderr.write('err')"],
        timeout=60,
    )
    assert proc.returncode == 0
    assert proc.stdout == b"out"
    assert b"err" in proc.stderr


def test_run_subprocess_timeout_kills(registry):
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        registry.run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert time.monotonic() - t0 < 10  # 旧 native 层这里会干等 30s


def test_cancel_kills_running_process(registry):
    result = {}

    def run():
        try:
            registry.run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)
        except BaseException as e:
            result["exc"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert _wait_live_proc(registry)
    t0 = time.monotonic()
    registry.cancelled = True  # setter 应立即 kill 活子进程
    t.join(timeout=10)
    assert not t.is_alive()
    assert time.monotonic() - t0 < 10
    assert isinstance(result["exc"], CancelledError)


def test_model_registry_roundtrip(registry):
    calls = []

    def factory():
        calls.append(1)
        return object()

    h1 = registry.acquire_model("m", "cpu", factory)
    assert registry.acquire_model("m", "cpu", factory) == h1  # 缓存复用
    assert len(calls) == 1
    assert registry.get_model(h1) is not None
    registry.release_model(h1)  # 兼容 API,无副作用
    registry.unload_model(h1)
    with pytest.raises(KeyError):
        registry.get_model(h1)
    h2 = registry.acquire_model("m", "cpu", factory)  # 失效句柄自动重建
    assert len(calls) == 2
    assert h2 != h1


def test_shutdown_clears_models_and_kills(registry):
    h = registry.register_model("x", object())
    registry.shutdown()
    with pytest.raises(KeyError):
        registry.get_model(h)
    registry.reset_controls()  # 还原给后续测试


def test_decode_audio_roundtrip(registry, tmp_path):
    import torch
    from writansub.preprocess.core import save_wav

    wav = tmp_path / "t.wav"
    save_wav(torch.zeros(1, 4410), str(wav), 44100)  # 0.1s 静音
    waveform, sr = registry.decode_audio(str(wav), sample_rate=16000)
    assert sr == 16000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert waveform.dtype == torch.float32
```

- [ ] **Step 2: 跑测试确认现状失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_bridge_subprocess.py::test_cancel_kills_running_process -v`
Expected: FAIL(旧 native 层:cancelled 只是普通属性,不 kill;线程 30s 内不退出,assert not t.is_alive() 失败)。注意红灯运行本身会等满 30s,只跑这一条即可,不要全跑。

- [ ] **Step 3: 重写 bridge.py 的 ResourceRegistry**

文件头:删 `import writansub_native`(第 14 行)。`_gpu_mem_hint`、`CancelledError`、`_get_ffmpeg`、`_get_ffprobe` 保持原样。第 74 行起整类替换为:

```python
# Windows 下子进程不建控制台窗口(GUI/pythonw 场景防闪黑框);
# 代价是子进程收不到控制台 Ctrl+C——取消一律走显式 kill(cancelled setter)
_CREATE_NO_WINDOW = 0x0800_0000


class ResourceRegistry:
    _instance: "ResourceRegistry | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ResourceRegistry":
        # 每任务仅调用数次，非热路径：无条件加锁，不做双检快路径
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_handle = 1
        self._models: dict[int, Any] = {}
        self._model_handles: dict[tuple[str, str], int] = {}
        self._procs: set[subprocess.Popen] = set()
        self._cancelled = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    # ── 取消 / 暂停 ──

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @cancelled.setter
    def cancelled(self, value: bool) -> None:
        # 置 True 即杀活子进程：解码/提取中的取消从"等 ffmpeg 跑完"变即时
        self._cancelled = value
        if value:
            self._kill_procs()

    def _kill_procs(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    @property
    def paused(self) -> bool:
        return not self._pause_event.is_set()

    def reset_controls(self) -> None:
        self._cancelled = False
        self._pause_event.set()

    def checkpoint(self) -> None:
        self._pause_event.wait()
        if self._cancelled:
            raise CancelledError("任务已取消")

    # ── 模型注册表 ──

    def register_model(self, name: str, obj: Any, device: str = "") -> int:
        from writansub.logger import log_line
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._models[handle] = obj
            self._model_handles[(name, device)] = handle
        log_line(f"[model] registered name={name!r} device={device!r} handle={handle}{_gpu_mem_hint(device)}")
        return handle

    def acquire_model(self, name: str, device: str, factory: Callable[[], Any]) -> int:
        from writansub.logger import log_line
        key = (name, device)
        with self._lock:
            handle = self._model_handles.get(key)
            if handle is not None and handle in self._models:
                cached = handle
            else:
                cached = None
                if handle is not None:
                    del self._model_handles[key]
        if cached is not None:
            log_line(f"[model] reuse cached name={name!r} device={device!r} handle={cached}")
            return cached

        log_line(f"[model] loading name={name!r} device={device!r} ...{_gpu_mem_hint(device)}")
        t0 = time.monotonic()
        try:
            obj = factory()
        except BaseException as e:
            elapsed = time.monotonic() - t0
            log_line(
                f"[model] LOAD FAILED name={name!r} device={device!r} "
                f"after {elapsed:.2f}s: {type(e).__name__}: {e}"
            )
            raise
        elapsed = time.monotonic() - t0
        log_line(f"[model] loaded name={name!r} device={device!r} in {elapsed:.2f}s{_gpu_mem_hint(device)}")
        return self.register_model(name, obj, device)

    def get_model(self, handle: int) -> Any:
        with self._lock:
            if handle not in self._models:
                raise KeyError(f"model handle {handle} not found")
            return self._models[handle]

    def release_model(self, handle: int) -> None:
        # 兼容 API：native 时代的 in_use 标志 Python 侧从未消费，退役后仅留日志
        from writansub.logger import log_line
        log_line(f"[model] released handle={handle}")

    def unload_model(self, handle: int) -> None:
        from writansub.logger import log_line
        with self._lock:
            key = next((k for k, v in self._model_handles.items() if v == handle), None)
            device = key[1] if key else ""
            self._models.pop(handle, None)
            self._model_handles = {k: v for k, v in self._model_handles.items() if v != handle}
        log_line(f"[model] unloading handle={handle} key={key}{_gpu_mem_hint(device)}")
        gc.collect()
        log_line(f"[model] unloaded handle={handle}{_gpu_mem_hint(device)}")

    # ── 子进程 ──

    def run_subprocess(self, cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
        creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        with self._lock:
            self._procs.add(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TimeoutError(f"子进程超时 ({timeout}s): {cmd[0]}") from None
        finally:
            with self._lock:
                self._procs.discard(proc)

        # 取消触发的 kill：返回码必非零，按取消上抛，调用方不得误报"执行失败"
        if self._cancelled and proc.returncode != 0:
            raise CancelledError("任务已取消")

        code = proc.returncode
        if stderr:
            try:
                from writansub.logger import log_line
                exe = os.path.basename(cmd[0]) if cmd else "?"
                raw = stderr.decode("utf-8", errors="replace")
                limit = 4000
                if len(raw) > limit:
                    head_n, tail_n = 2000, 1000
                    elided = len(raw) - head_n - tail_n
                    raw = f"{raw[:head_n]} ... [{elided} chars elided] ... {raw[-tail_n:]}"
                text = raw.replace("\n", " | ").replace("\r", "")
                log_line(f"stderr[{exe}] ({len(stderr)}B, rc={code}): {text}")
            except Exception:
                pass

        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    def decode_audio(self, path: str, sample_rate: int = 44100) -> tuple["torch.Tensor", int]:
        import numpy as np
        import torch

        cmd = [
            _get_ffmpeg(), "-i", path,
            "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
            "-loglevel", "error", "-",
        ]
        proc = self.run_subprocess(cmd, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace')}")

        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        return waveform, sample_rate

    def shutdown(self) -> None:
        log.info("Shutdown initiated...")
        self.cancelled = True  # property：顺带 kill 所有活子进程
        self._pause_event.set()
        with self._lock:
            self._models.clear()
            self._model_handles.clear()
```

注意:旧代码 `run_subprocess` 里的 `bytes(stdout)` list 转换段(原 192-196 行)随之消亡——communicate 返回的本来就是 bytes。GUI 关窗强杀(closeEvent 直调 shutdown)是用户有意保留的设计,新 shutdown 语义(kill 全部+清模型)与现状一致,不要加确认框。

- [ ] **Step 4: 全量跑测试确认通过**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests -q`
Expected: 全绿(82 例既有 + 6 例新增)。特别确认 `test_bridge_registry.py` 4 例(单例/checkpoint/暂停语义)零改动通过。

- [ ] **Step 5: 消费方冒烟**

Run: `G:\WritanSub\.venv\Scripts\python.exe -c "from writansub.bridge import ResourceRegistry; r = ResourceRegistry.instance(); import sys; p = r.run_subprocess([sys.executable, '-V'], timeout=30); print(p.returncode, p.stdout)"`
Expected: `0 b'Python 3.12...'`

- [ ] **Step 6: Commit**

```powershell
git add tests/test_bridge_subprocess.py writansub/bridge.py
git commit -m @'
T03 bridge.py 纯 Python 改写：退役 writansub_native 运行时依赖

wait_process 持 GIL 冻结解释器 / shutdown 写锁死锁 / Vec<u8>→list[int]
内存灾难三宗罪一次消除。模型注册表 dict+锁一比一替换（in_use 标志
Python 侧从未消费，不搬）；run_subprocess 改 Popen+活进程登记，
cancelled setter 即时 kill，超时/取消首次真实生效。对外 API 不变。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 2: native 退役——依赖与打包链拆除 + 全面 uv 口径(T03 打包侧 + T16)

**Files:**
- Modify: `pyproject.toml:24,45,68`
- Modify: `build/scripts/build.ps1`(删 B4 段 89-129 行、`$SkipNative` 参数、`--no-emit-package writansub_native`)
- Modify: `build/scripts/light.ps1`(删 C3 段 84-111 行、`$SkipNative` 参数、`--no-emit-package` 行、install.bat 模板 5→4 步)
- Modify: `build/scripts/check_versions.ps1`(删 native 两条规则)
- Modify: `README.md`(native/maturin/Rust 相关段落 + **删除 pip 安装方法二**)
- Modify: `requirements.txt`(文件头加"仅供参考"注释)
- Modify: `.gitignore`(删 `native/target/` 行)
- Create: `archive/native_rust_202607/`(Rust 源码归档)
- Delete: `native/`(源码 git mv 走,`.venv`/`target` 构建产物直接删)
- Modify: `uv.lock`(uv lock 重解析)

**Interfaces:**
- Consumes: Task 1(bridge.py 已不 import writansub_native)
- Produces: `uv sync` 后环境无 writansub_native 且测试全绿(决定性证据);构建脚本无 maturin 步骤;README 安装章节只剩 uv 路径。

- [ ] **Step 1: pyproject.toml 三处删除**

- 删第 45 行 `"writansub_native",`(dependencies)
- 删第 68 行 `writansub_native = { path = "./native" }`([tool.uv.sources])
- 删第 24 行 `"Programming Language :: Rust",`(classifiers)

- [ ] **Step 2: 归档 Rust 源码,删除构建产物**

```powershell
New-Item -ItemType Directory -Force G:\WritanSub\archive\native_rust_202607 | Out-Null
git mv native/src archive/native_rust_202607/src
git mv native/Cargo.toml archive/native_rust_202607/Cargo.toml
git mv native/Cargo.lock archive/native_rust_202607/Cargo.lock
git mv native/pyproject.toml archive/native_rust_202607/pyproject.toml
Remove-Item -Recurse -Force G:\WritanSub\native
```

(`native/.venv`、`native/target` 未被 git 跟踪,连目录一起物理删除。归档目录加一行说明文件:)

```powershell
Set-Content G:\WritanSub\archive\native_rust_202607\README.md "# writansub_native 归档 (2026-07 批次 4 退役)`n`nT03:wait_process 持 GIL / shutdown 死锁 / Vec<u8> 内存灾难,整层由 bridge.py 纯 Python 替换。仅存源码供考古,构建链已拆除。"
git add archive/native_rust_202607/README.md
```

- [ ] **Step 3: 重解析锁 + 同步环境**

```powershell
uv lock
uv sync
uv pip install pytest --index-url https://pypi.org/simple
uv pip list | Select-String -Pattern 'writansub|pytest'
```

Expected: 列表无 writansub-native,有 pytest。若清华源仍 403 导致 uv lock 失败,走系统代理(Clash 127.0.0.1:7897)重试。

- [ ] **Step 4: 全量测试证明环境无 native 也全绿**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests -q`
Expected: 全绿。这是"业务代码真正不依赖 native"的最终证据。

- [ ] **Step 5: build.ps1 拆除**

- 删参数块中 `[switch]$SkipNative,`(19 行)
- 删整个 B4 段(89-129 行,从 "---------- B4. Native extension wheel ----------" 注释到该 if 块收尾 `}`)
- B3 段 uv export 参数删 `--no-emit-package writansub_native \``(71 行)
- 文件头注释(2-15 行)如提到 native 则同步删改

- [ ] **Step 6: light.ps1 拆除**

- 删参数块 `[switch]$SkipNative,`(25 行)
- 删整个 C3 段(84-111 行)
- C4 段 uv export 删 `--no-emit-package writansub_native \``(120 行)
- 文件头注释:删第 13 行 wheel 一行,第 15 行 "5-stage installer (...) " 改为 "4-stage installer (extract Python -> patch _pth -> ensure uv -> uv pip install)"
- install.bat 模板(147-266 行 here-string 内):
  - 头部清单 `[1/5]`~`[5/5]` 五行改四行,删 `echo   [5/5] 安装 native 扩展`
  - 各阶段标签 `[1/5]`→`[1/4]`、`[2/5]`→`[2/4]`、`[3/5]`→`[3/4]`、`[4/5]`→`[4/4]`
  - 整段删除 `:: -------- [5/5] native 扩展 --------` 到其 for 循环结束(245-258 行)

- [ ] **Step 7: check_versions.ps1 拆除**

- 删规则注释第 7-8 行(native 两条)
- 删 51-52 行(`$vNativeCg`/`$vNativePy` 两个 Parse)
- 删 58-60 行(native 互等检查 if 块)
- 删 63-65 行(native major.minor 检查 if 块)
- 70 行输出改为 `"    [versions] pyproject=$vPyproject init=$vInit launcher=$vLauncher"`

Run: `powershell -NoProfile -File G:\WritanSub\build\scripts\check_versions.ps1`
Expected: 输出 `0.1.7.3`,退出码 0。

- [ ] **Step 8: README 全面 uv + requirements.txt 注释 + .gitignore**

- `.gitignore` 删 `native/target/` 行
- README.md:
  - **删除安装方法二(pip)整节**,安装章节只留 uv 路径;如有"方法一/方法二"编号措辞,顺句改写
  - 全文搜 `native`、`maturin`、`Rust`,逐处处理:599 行排障项("如果报错和 writansub_native 有关…maturin develop")整条删除;架构/目录结构介绍如列有 `native/`,删除该行
- `requirements.txt` 文件头(`--extra-index-url` 行之前)加一行:

```
# 仅供参考——官方安装与依赖锁定以 uv (pyproject.toml / uv.lock) 为准
```

Run: `Select-String -Path G:\WritanSub\README.md -Pattern 'writansub_native|maturin|pip install' -SimpleMatch`
Expected: 无 writansub_native/maturin 匹配;`pip install` 仅允许出现在与安装路径无关的上下文(如 pytest 说明),安装章节零匹配。

- [ ] **Step 9: Commit**

```powershell
git add -A
git commit -m @'
T03/T16 native 构建链拆除 + 安装口径全面 uv

Rust 源码归档 archive/native_rust_202607/，maturin 步骤与版本校验规则退役，
install.bat 5 步减为 4 步。README 删 pip 安装方法二（T16 病根已随依赖
删除消失，口径经对齐定为 uv 唯一），requirements.txt 降级为参考文件。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 3: 全链构建验证 + T37 超时实测(人机配合)

**Files:**
- 无源码改动预期;T37 若实测需调值则 Modify: `writansub/subtitle/extract.py:36,91`、`writansub/bridge.py`(decode_audio timeout)

**Interfaces:**
- Consumes: Task 1-2 全部落地
- Produces: 构建产物可安装可运行;取消/超时行为实测记录(写入 Batch4_Report)

> 本任务多步需要用户配合(安装、大文件实测),执行时逐步征询,不可跳过实测直接判通过。

- [ ] **Step 1: 单测全量 + pyflakes 终扫**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests -q` → 全绿。
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pyflakes writansub`(pyflakes 不在环境则 `uv pip install pyflakes --index-url https://pypi.org/simple`)→ 与批次 1 基线(16 条既存项)比只减不增。

- [ ] **Step 2: 轻量包全链构建**

```powershell
powershell -NoProfile -File G:\WritanSub\build\scripts\light.ps1 -Clean
```

Expected: 无 native 相关步骤出现;`build/dist/WritanSub-light/` 产出且 vendor/ 无 .whl;install.bat 内容为 4 步。

- [ ] **Step 3: 完整包构建 + 内置冒烟**

```powershell
powershell -NoProfile -File G:\WritanSub\build\scripts\build.ps1 -Clean
```

Expected: B4 native 段不存在,结尾 `Smoke test passed`。(耗时较长,walltime 10 分钟级;失败优先检查 uv export 参数删改是否干净。)

- [ ] **Step 4: Inno 打包冒烟(可选,征询用户)**

```powershell
$ver = (& powershell -NoProfile -File G:\WritanSub\build\scripts\check_versions.ps1 | Select-Object -Last 1).Trim()
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=$ver /DLIGHT G:\WritanSub\build\installer\writansub.iss
```

(版本号取 check_versions 实际输出;ISCC 路径以实机为准。writansub.iss 本批未改,此步只验证 dist 树变化后打包仍通。)

- [ ] **Step 5: 取消体验实测(需用户素材)**

用真实长媒体(用户日常财经视频)在源码环境跑:

1. `WritanSubCLI.bat pipeline <大文件> --denoise` 起跑,解码阶段 Ctrl+C → 应 1s 内退出;`tasklist | findstr ffmpeg` → **无孤儿进程**。
2. GUI 起同任务,解码阶段拖动窗口/点按钮 → **界面不冻结**;点"取消" → 即时停。

- [ ] **Step 6: T37 超时数值重估**

用机械盘大 MKV(用户提供)实测:字幕轨探测(现 120s)、字幕提取(现 600s)、解码(现 3600s)的真实耗时。若余量 <3 倍,按实测调 `extract.py:36/91` 与 `bridge.py decode_audio` 的数值并补一条 commit;若足够,tracker T37 记"数值维持,实测依据"后置 `[x]`。

- [ ] **Step 7: Commit(如 Step 6 调了值)**

```powershell
git add writansub/subtitle/extract.py writansub/bridge.py
git commit -m @'
T37 子进程超时按实测重估（T03 后超时首次真实生效）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
```

---

### Task 4: 收尾——版本号、tracker、CHANGELOG、报告、push

**Files:**
- Modify: `pyproject.toml:7`(0.1.7.3 → 0.1.8)、`writansub/__init__.py`(同步)
- Modify: `BUG_TRACKER.md`(T03/T16 置 `[x]`,T37 按实测结果,T07/T30 批注暂缓,新增 T42 行)
- Modify: `CHANGELOG.md`(0.1.8 条目)
- Create: `Batch4_Report_2026-07-12.md`

**Interfaces:**
- Consumes: 全部前序任务的实际结果(报告要写实测数据,不许写预期值)
- Produces: 版本 0.1.8;tracker 收口

- [ ] **Step 1: 版本号 + check_versions 验证**

pyproject.toml `version = "0.1.8"`;`writansub/__init__.py` → `__version__ = "0.1.8"`。
Run: `powershell -NoProfile -File G:\WritanSub\build\scripts\check_versions.ps1`
Expected: 输出 `0.1.8`(launcher 0.1.x major.minor 匹配,通过)。

- [ ] **Step 2: BUG_TRACKER.md 收口**

- T03 → `[x]`,批注:`2026-07-12 修复:bridge.py 纯 Python 替换(dict+锁 / Popen+活进程登记+cancelled setter 即杀),Rust 源码归档 archive/native_rust_202607,maturin/版本校验/打包链拆除。超时与取消首次真实生效;子进程加 CREATE_NO_WINDOW`
- T16 → `[x]`,批注:`2026-07-12:病根随 T03 消失;文档口径经对齐定为全面 uv,README 删 pip 方法二,requirements.txt 降级参考`
- T37 → 按 Task 3 Step 6 结果置 `[x]` 并记实测数值与依据
- T07 追加批注:`2026-07-12 对齐:与三件套 b(弃单)、T08 同判暂缓——翻译功能不常用,将来用到时一批做;重修要点见本条目`
- T30 追加批注并保持 `[ ]`:`2026-07-12 用户定夺:留档,待将来架构优化时再改(含卸载残留最小修);现场核实勘误:数 GB 大头在 CACHE_DIR(TIGER HF 快照),MODELS_DIR 仅 Qwen3 探测位,whisper/Qwen3 走用户 HF 默认缓存 app 管不到`
- P3 表新增一行:`| [ ] | T42 | 取消响应毫秒级(留档可优化):torch forward-pre-hook 逐层查取消标志,挂 acquire_model 统一装钩 ~20 行;降噪/对齐取消从"等完一个前向窗口"压到毫秒级。2026-07-12 对齐:暂不做,留档 | 批次4对齐 | 低 | 中 | ~20 行 |`
- 「总拟修复批次路线」批次 4 段落末尾追加:`> 2026-07-12 批次 4 完成(范围经对齐缩减:T03+T16+T37;翻译侧 T07/弃单与 T30 暂缓),见 Batch4_Report_2026-07-12.md。三件套 c 留档为 T42。`

- [ ] **Step 3: CHANGELOG.md 追加 0.1.8 条目**

按既有格式写:native 层退役(GUI 解码期不再冻结、超时/取消真实生效、Ctrl+C 无孤儿进程、解码内存峰值大幅下降)、子进程 CREATE_NO_WINDOW、安装口径全面 uv(pip 方法二移除)。

- [ ] **Step 4: Batch4_Report_2026-07-12.md**

沿用 Batch3_Report 结构:commit 表、对齐处置记录(含 T07/弃单/T08/T30 暂缓归因与 T42 留档)、T03 设计要点与验证数据(测试计数、构建产物、取消实测、T37 实测数值)、行为变化知会(Global Constraints 列的三条)、剩余池更新(批 2 遗留 T08/T09+T36/T12+T13、翻译侧暂缓组、T29→T22 等素材、T17+T30 暂缓、T41 待对齐、T42 留档)。

- [ ] **Step 5: 全量终验 + Commit + push**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests -q` → 全绿。

```powershell
git add -A
git commit -m @'
批次 4 收尾：版本 0.1.8，tracker/CHANGELOG/Batch4_Report 收口，T42 留档

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
'@
git push
```

---

## 联动提醒(执行者必读)

- Task 1 与 Task 2 顺序不可换:先让 bridge.py 不再 import native(环境里 wheel 还在,测试可跑),再拆依赖(`uv sync` 卸掉 wheel 后测试仍绿 = 决定性证据)。
- `uv sync`(Task 2 Step 3)会卸掉 pytest,之后必须立刻重装,否则后续所有测试步骤全挂。
- **翻译侧文件是禁区**:translate/core.py、gui/tabs/translate.py、cli.py 的 cmd_translate、runner.py 翻译段本批一律不碰(T07/T08/弃单 已判暂缓);tests/test_translate_core.py 的现状契约测试(test_cancel_propagates)保持原样,它记录的就是当前未修行为。
- GUI 关窗强杀是用户有意保留的:Task 1 的 shutdown() 语义(kill 全部+清模型)与现状一致,不要加确认框(那是批 2 遗留 T36 的事)。
- T37 实测(Task 3)依赖用户提供机械盘大 MKV,执行到该步先停下征询,不可用小文件糊弄了事。
