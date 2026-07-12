# 批次 5 · 杂修批实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地批次 5 全部 7 个代码项(T40/T12/N3/N4/T36/N2/T13)并收口 tracker/CHANGELOG,版本 0.1.8 → 0.1.9。

**Architecture:** 七个互不相关的小修,每项独立成任务、独立测试、独立提交;唯一的依赖是 T13 改 pyproject 会触发 `uv sync` 清掉 pytest,故排在代码任务最后。T09 已由用户拍板"不改",在收口任务标 `[-]`。

**Tech Stack:** Python 3.12 + uv;PySide6(离屏测试);pytest(`tests/`,当前 88 例全绿);PowerShell 7(build 脚本)。

## Global Constraints

- 跑测试一律用 `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q`;**绝不用 `uv run pytest`**(auto-sync 会卸掉 pytest)。任何 `uv sync` 之后必须重装:`uv pip install pytest --index-url https://pypi.org/simple`(需代理,Clash 在 `127.0.0.1:7897`)。
- 测试必须全离线可重复,绝不触碰真实用户配置(`%LOCALAPPDATA%\mtmfs\WritanSub\*.json`);Qt 测试依赖 conftest.py 的 `qapp` fixture(已设 `QT_QPA_PLATFORM=offscreen`)。
- **翻译侧文件是禁区**:`translate/core.py`、`gui/tabs/translate.py`、cli.py 的 `cmd_translate`、runner.py 翻译段一律不碰(T07/T08/弃单已判暂缓);`tests/test_translate_core.py` 的现状契约测试保持原样。
- 清华源对部分新 wheel 间歇 403:`uv lock` 失败时挂代理(`$env:HTTPS_PROXY='http://127.0.0.1:7897'`)加 `UV_DEFAULT_INDEX=https://pypi.org/simple` 重试。
- 每任务一个 commit,中文消息沿用仓库风格(`fix:`/`chore:`/`build:`/`docs:` 前缀 + 任务 ID),末尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 不 push;push 由用户决定。

## File Structure

| 文件 | 动作 | 任务 |
|---|---|---|
| `writansub/bridge.py` | 新增 `resolve_device()` | 1 |
| `writansub/cli.py` | cmd_transcribe 用 resolve_device;cmd_align 换用;删两处 `--ss-model` | 1, 3 |
| `writansub/pipeline/runner.py` | run 入口 resolve_device;删 ss_model 字段/日志/传参 | 1, 3 |
| `writansub/gui/tabs/transcribe.py` | `_run_whisper` 用 resolve_device | 1 |
| `writansub/gui/tabs/align.py` | 内联 CUDA 检查换 resolve_device | 1 |
| `writansub/network.py` | 探测改 urllib HTTPS(走系统代理) | 2 |
| `writansub/types.py` | 删 `SS_MODELS` | 3 |
| `writansub/gui/tabs/pipeline.py` | 删分轨模型下拉及状态链 | 3 |
| `writansub/gui/tabs/preprocess.py` | 删分轨模型下拉及状态链、线程传参 | 3 |
| `writansub/preprocess/core.py` | run_speech_batch 删 ss_model 参数;silero 改 pip 包加载 | 3, 7 |
| `writansub/gui/widgets.py` | `_InfoDelegate.paint` 修 elide;StateMixin 加 `is_running()` | 4, 5 |
| `writansub/gui/app.py` | `_confirm_quit` 助手 + closeEvent 确认 | 5 |
| `build/scripts/build.ps1` / `light.ps1` | tar 口味自探测 | 6 |
| `pyproject.toml` | 加 silero-vad 依赖;版本 0.1.9 | 7, 8 |
| `tests/test_bridge_registry.py` | resolve_device 测试 | 1 |
| `tests/test_network.py` | 新建 | 2 |
| `tests/test_cli_args.py` | --ss-model 拒收测试 | 3 |
| `tests/test_gui_close_confirm.py` | 新建 | 5 |
| `tests/test_preprocess_loading.py` | silero 加载测试 | 7 |
| `BUG_TRACKER.md` / `CHANGELOG.md` / `Batch5_Report_*.md` | 收口 | 8 |

---

### Task 1: T40 — CUDA 回退共享助手 resolve_device

**Files:**
- Modify: `writansub/bridge.py`(模块级函数区,`_gpu_mem_hint` 之后)
- Modify: `writansub/cli.py`(cmd_transcribe ~:245-280;cmd_align ~:335-340)
- Modify: `writansub/pipeline/runner.py`(`[决策]` 日志行 ~:78 之前)
- Modify: `writansub/gui/tabs/transcribe.py`(`_run_whisper` ~:296-310)
- Modify: `writansub/gui/tabs/align.py`(~:289-292 内联检查)
- Test: `tests/test_bridge_registry.py`

**Interfaces:**
- Produces: `writansub.bridge.resolve_device(requested: str, log_callback: Callable[[str], None] | None = None) -> str`
- 背景:2026-07-09 首次修复因加在 `transcribe()` 的 `model is None` 分支(生产不可达)被回退;本次必须加在各调用方**建模型之前**。

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_bridge_registry.py`)

```python
def test_resolve_device_cuda_fallback(monkeypatch):
    """T40：请求 cuda 但不可用时回退 cpu 并知会。"""
    import torch
    from writansub.bridge import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    msgs = []
    assert resolve_device("cuda", msgs.append) == "cpu"
    assert any("CUDA" in m for m in msgs)


def test_resolve_device_passthrough(monkeypatch):
    import torch
    from writansub.bridge import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_bridge_registry.py -q`
Expected: FAIL,`ImportError: cannot import name 'resolve_device'`

- [ ] **Step 3: 实现**(`writansub/bridge.py`,加在 `_gpu_mem_hint` 函数之后)

```python
def resolve_device(requested: str, log_callback=None) -> str:
    """请求 cuda 但 CUDA 不可用时回退 cpu 并知会调用方;其余原样返回。

    必须在建模型(WhisperModel / init_qwen3_model 等)之前调用——
    transcribe() 内部的 model is None 分支生产路径不可达(T40 首修教训)。
    """
    if requested == "cuda":
        import torch
        if not torch.cuda.is_available():
            if log_callback:
                log_callback("CUDA 不可用，回退到 CPU")
            return "cpu"
    return requested
```

- [ ] **Step 4: 接入五个调用点**

a) `cli.py` cmd_transcribe:函数内 import 行 `from writansub.bridge import ResourceRegistry, CancelledError` 追加 `, resolve_device`;在 `reg.reset_controls()` 之后插入 `device = resolve_device(args.device, _log)`,并把 cmd_transcribe 函数体内**所有** `args.device`(工厂、`acquire_model`、`do_transcribe` 调用)替换为 `device`:

```python
    reg.reset_controls()
    device = resolve_device(args.device, _log)

    def _w_factory():
        from faster_whisper import WhisperModel
        return WhisperModel(args.whisper_model, device=device, compute_type=args.compute_type)

    wh = reg.acquire_model(
        f"whisper:{args.whisper_model}:{args.compute_type}", device, _w_factory)
```

b) `cli.py` cmd_align:把内联检查(含其专用的 `import torch`,若该函数内 torch 无其他用途则一并删,以 pyflakes 为准)

```python
        import torch
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            _log("CUDA 不可用，回退到 CPU")
            device = "cpu"
```

替换为 `device = resolve_device(args.device, _log)`(同样补 import)。

c) `runner.py`:在 `log(f"[决策] tiger_mode=...")` 行之前插入 `cfg.device = resolve_device(cfg.device, log)`(bridge 已被 runner import,按现有 import 风格补 `resolve_device`)。

d) `gui/tabs/transcribe.py` `_run_whisper`:在 `log_emit` 定义之后、`_w_factory` 之前插入 `device = resolve_device(device, log_emit)`(顶部 `from writansub.bridge import ResourceRegistry` 追加 `, resolve_device`)。

e) `gui/tabs/align.py` ~:289 的内联检查(三行 if 块)替换为 `device = resolve_device(device, log_emit)`;顶部 torch import 若因此闲置则删,以 pyflakes 为准。

- [ ] **Step 5: 跑测试与静态检查**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿(88+2)
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pyflakes writansub/` → 无新增告警
Run: `G:\WritanSub\.venv\Scripts\python.exe -m writansub.cli transcribe --help` → 正常输出

- [ ] **Step 6: Commit**

```bash
git add writansub/bridge.py writansub/cli.py writansub/pipeline/runner.py writansub/gui/tabs/transcribe.py writansub/gui/tabs/align.py tests/test_bridge_registry.py
git commit -m "fix: T40 transcribe 补 CUDA 回退, 抽共享 resolve_device"
```

---

### Task 2: T12 — 镜像探测改经代理 HTTPS 实测

**Files:**
- Modify: `writansub/network.py`(全文件重写,现仅 20 行)
- Test: `tests/test_network.py`(新建)

**Interfaces:**
- Produces: `_can_reach(url: str, timeout: float = 5.0) -> bool`;`setup_hf_mirror()` 签名不变(app.py/cli.py 调用方零改动)。
- 修复双盲区:裸 socket 不走系统代理(Clash 场景误判不可达)+ TCP 通但 SNI 被重置误判可达。urllib 默认读系统代理(Windows 注册表/环境变量)且完整 TLS 握手,两个盲区同时消除。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_network.py`)

```python
"""T12：镜像探测经代理 HTTPS 实测（urlopen 打桩，全离线）。"""
import os
import urllib.error
import urllib.request

import pytest

from writansub import network


@pytest.fixture
def clean_hf_env():
    saved = {k: os.environ.pop(k) for k in ("HF_ENDPOINT", "HF_HUB_OFFLINE") if k in os.environ}
    yield
    for k in ("HF_ENDPOINT", "HF_HUB_OFFLINE"):
        os.environ.pop(k, None)
    os.environ.update(saved)


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub(monkeypatch, effect):
    def fake(req, timeout=0.0):
        if isinstance(effect, Exception):
            raise effect
        return _Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_reachable_keeps_official(clean_hf_env, monkeypatch):
    _stub(monkeypatch, None)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ


def test_unreachable_switches_mirror(clean_hf_env, monkeypatch):
    _stub(monkeypatch, urllib.error.URLError("blocked"))
    network.setup_hf_mirror()
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_http_error_counts_as_reachable(clean_hf_env, monkeypatch):
    """4xx/5xx 说明 TLS+HTTP 全程走通（旧探测的 SNI 盲区场景），视为可达。"""
    err = urllib.error.HTTPError("https://huggingface.co", 403, "Forbidden", None, None)
    _stub(monkeypatch, err)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ


def test_user_endpoint_respected(clean_hf_env, monkeypatch):
    os.environ["HF_ENDPOINT"] = "https://my-mirror.example"
    def boom(*a, **k):
        raise AssertionError("已设 HF_ENDPOINT 时不应发起探测")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    network.setup_hf_mirror()
    assert os.environ["HF_ENDPOINT"] == "https://my-mirror.example"


def test_offline_mode_skips_probe(clean_hf_env, monkeypatch):
    os.environ["HF_HUB_OFFLINE"] = "1"
    def boom(*a, **k):
        raise AssertionError("离线模式不应发起探测")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    network.setup_hf_mirror()
    assert "HF_ENDPOINT" not in os.environ
```

- [ ] **Step 2: 跑测试确认失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_network.py -q`
Expected: `test_reachable_keeps_official` 等 FAIL(旧实现走裸 socket,打桩无效 → 真连网或误判)

- [ ] **Step 3: 重写 `writansub/network.py`**(整文件替换)

```python
import os
import urllib.error
import urllib.request


def _can_reach(url: str, timeout: float = 5.0) -> bool:
    """经系统代理的真实 HTTPS 探测。

    urllib 默认读取系统代理(Windows 注册表/环境变量)并完整 TLS 握手:
    Clash 场景不再误判不可达(旧裸 socket 不走代理),SNI 阶段被重置也
    不再误判可达(旧探测 TCP 通即真)。收到任何 HTTP 响应(含 4xx/5xx)
    即视为可达。
    """
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def setup_hf_mirror() -> None:
    """若用户未设置 HF_ENDPOINT，且 huggingface.co 不可达，则自动切换到镜像站。"""
    if os.environ.get("HF_ENDPOINT"):
        return
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    if not _can_reach("https://huggingface.co"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

超时从 2s 放宽到 5s:HTTPS 经代理比裸 TCP 慢,启动最坏多等 3s,换探测结果可信。

- [ ] **Step 4: 跑测试确认通过**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_network.py -q` → 5 passed
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿

- [ ] **Step 5: 真实场景抽查**(Clash 常开的本机,双场景)

```powershell
# 场景1 系统代理开(Clash): 应判可达、不设镜像
G:\WritanSub\.venv\Scripts\python.exe -c "from writansub.network import _can_reach; print(_can_reach('https://huggingface.co'))"
# Expected: True
# 场景2 显式断路: 应判不可达
G:\WritanSub\.venv\Scripts\python.exe -c "from writansub.network import _can_reach; print(_can_reach('https://invalid.example.invalid'))"
# Expected: False
```

- [ ] **Step 6: Commit**

```bash
git add writansub/network.py tests/test_network.py
git commit -m "fix: T12 镜像探测改经系统代理的 HTTPS 实测, 消除 Clash/SNI 双盲区"
```

---

### Task 3: N3 — 拆除 ss_model 死参数链

**Files:**
- Modify: `writansub/types.py`(删 `SS_MODELS` 块 ~:45-49)
- Modify: `writansub/cli.py`(删 :463 与 :493 两条 `--ss-model` add_argument;删 :167 与 :237 的 `ss_model=args.ss_model,`)
- Modify: `writansub/pipeline/runner.py`(删 :28 `ss_model` 字段;:78 决策日志删 `ss={cfg.ss_model} ` 片段;删 :333 `ss_model=cfg.ss_model,`)
- Modify: `writansub/preprocess/core.py`(run_speech_batch 签名删 `ss_model: str = "tiger-speech",`)
- Modify: `writansub/gui/tabs/pipeline.py`(import 删 `, SS_MODELS`;删 ss_row 整块 ~:145-151;删 :260 connect、:281 save_state 键、:318-319 restore、:456 传参)
- Modify: `writansub/gui/tabs/preprocess.py`(import 删 `, SS_MODELS`;删 label+combo 块 ~:92-96;删 :154 connect、:163 save 键、:176-177 restore、:257 局部变量、:263 线程 args 中的 `ss_model`、:268-269 `_run_tiger` 签名参数、:306 传参)
- Test: `tests/test_cli_args.py`

**Interfaces:**
- 背景:T25 删 tfgridnet 后唯一分轨模型是 tiger-speech,`run_speech_batch` 的函数体早已不消费该参数(内部 `separate_speakers(...)` 调用根本不传),整条链 5 文件空转。CLI 传任意值静默忽略 → 拆除后 argparse 对 `--ss-model` 报"未知参数",属预期行为升级(静默忽略 → 显式报错)。
- 注意:老用户 `gui_state.json` 里残留的 `pipeline.ss_model` / `tiger.ss_model` 键无害——restore_state 的 `if key in state` 守卫连同键一起删除,残键只是不再被读。

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_cli_args.py`)

```python
def test_ss_model_flag_removed():
    """N3：--ss-model 死参数已拆，传入应报未知参数而非静默忽略。"""
    import pytest
    for cmd in ("pipeline", "preprocess"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([cmd, "x.mp4", "--ss-model", "tiger-speech"])


def test_pipeline_config_has_no_ss_model():
    from writansub.pipeline.runner import PipelineConfig
    assert not hasattr(PipelineConfig(), "ss_model")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_cli_args.py -q`
Expected: 两个新测试 FAIL(旗标还在、字段还在)

- [ ] **Step 3: 按 Files 列表逐文件删除**

无新增代码,纯删。GUI 两处删的是完整 UI 块,以 pipeline.py 为例删除:

```python
        ss_row = QHBoxLayout()
        ss_row.addWidget(QLabel("分轨模型:"))
        self._ss_model_combo = GroupedComboBox()
        self._ss_model_combo.set_grouped_items(SS_MODELS)
        self._ss_model_combo.setCurrentName("tiger-speech")
        ss_row.addWidget(self._ss_model_combo)
        tiger_layout.addLayout(ss_row)
```

preprocess.py 对应块是 `tiger_layout.addWidget(QLabel("分轨模型"))` 起的四行 + combo 两行。删完后全文 grep 确认:`grep -rn "ss_model\|SS_MODELS" writansub/ --include="*.py" | grep -v archive | grep -vi mss` 应零命中(`-vi mss` 同时滤掉 `mss_model` 与 `MSS_MODELS` 的子串误报;被滤行均属 mss 链,不在本任务范围)。

- [ ] **Step 4: 跑测试与静态检查**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pyflakes writansub/` → 无新增
Run: `G:\WritanSub\.venv\Scripts\python.exe -m writansub.cli preprocess --help` → 帮助文本无 `--ss-model`

- [ ] **Step 5: 离屏 GUI 冒烟**(确认两个 tab 构造不炸)

```powershell
$env:QT_QPA_PLATFORM='offscreen'; G:\WritanSub\.venv\Scripts\python.exe -c "from PySide6.QtWidgets import QApplication; app=QApplication([]); from writansub.gui.tabs.pipeline import PipelineTab; from writansub.gui.tabs.preprocess import TigerTab; PipelineTab(); TigerTab(); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add writansub/types.py writansub/cli.py writansub/pipeline/runner.py writansub/preprocess/core.py writansub/gui/tabs/pipeline.py writansub/gui/tabs/preprocess.py tests/test_cli_args.py
git commit -m "chore: N3 拆除 ss_model 死参数链 (T25 遗留, 5 文件空转)"
```

---

### Task 4: N4 — _InfoDelegate 名称省略生效

**Files:**
- Modify: `writansub/gui/widgets.py`(`_InfoDelegate.paint` ~:162-186;顶部 QtWidgets import 行补 `QStyle, QApplication`)

**Interfaces:**
- 根因:旧实现把 elide 结果塞进 `opt.text` 再 `super().paint(painter, opt, index)`——但 QStyledItemDelegate.paint 内部会重新 `initStyleOption(opt, index)`,用 index 的 DisplayRole 覆盖 `opt.text`,elide 结果被无声丢弃。
- 修法:paint 里自己走 `initStyleOption + style.drawControl(CE_ItemViewItem)`(这正是基类 paint 的展开形式),改完 `opt.text` 后不再被覆盖。`sizeHint` 不动。

- [ ] **Step 1: 替换 `_InfoDelegate.paint`**

```python
    def paint(self, painter, option, index):
        # 手动展开基类 paint = initStyleOption + drawControl:
        # 直接 super().paint(opt) 会在内部重新 initStyleOption,
        # 用 DisplayRole 覆盖我们改过的 opt.text,elide 永远不生效(N4 根因)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        info = index.data(Qt.UserRole + 1) or ""
        info_width = opt.fontMetrics.horizontalAdvance(info) if info else 0
        reserved = info_width + self._INFO_GAP + self._RIGHT_PAD if info else 0

        # 给名字的区域：原 rect 右边扣掉 info 占位；过长中段省略
        opt.rect = opt.rect.adjusted(0, 0, -reserved, 0)
        max_name_width = min(opt.rect.width(), self._NAME_MAX_PX)
        opt.text = opt.fontMetrics.elidedText(opt.text, Qt.ElideMiddle, max_name_width)

        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        if info:
            painter.save()
            painter.setPen(option.palette.color(QPalette.ColorRole.Text))
            painter.drawText(
                option.rect.adjusted(0, 0, -self._RIGHT_PAD, 0),
                Qt.AlignRight | Qt.AlignVCenter,
                info,
            )
            painter.restore()
```

import 行改为:

```python
from PySide6.QtWidgets import (
    QTextEdit, QProgressBar, QLabel, QWidget, QVBoxLayout,
    QHBoxLayout, QDoubleSpinBox, QGridLayout, QFrame,
    QScrollArea, QComboBox, QStyledItemDelegate, QStyleOptionViewItem,
    QStyle, QApplication,
)
```

- [ ] **Step 2: 离屏截图验证**(像素级行为无法用断言稳定表达,用截图人工核验;把脚本写到 scratchpad,不入库)

```python
# scratchpad/n4_check.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from writansub.gui.widgets import GroupedComboBox

app = QApplication([])
cb = GroupedComboBox()
cb.set_grouped_items([
    ("测试", [
        ("this-is-a-very-long-model-name-that-must-elide-in-the-middle", "~1 GB"),
        ("short-name", "~2 GB"),
    ]),
])
cb.showPopup()
cb.view().grab().save(os.path.join(os.path.dirname(__file__), "n4_popup.png"))
print("saved")
```

Run 后用 Read 工具查看 `n4_popup.png`,核验两点:长名中段出现 `…` 且不与右侧 `~1 GB` 重叠;短名与分组头渲染无回归。

- [ ] **Step 3: 全量回归**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿(test_gui_widgets 覆盖 widgets 模块 import 与既有行为)
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pyflakes writansub/gui/widgets.py` → 干净

- [ ] **Step 4: Commit**

```bash
git add writansub/gui/widgets.py
git commit -m "fix: N4 _InfoDelegate 过长名称省略生效 (initStyleOption+drawControl 展开)"
```

---

### Task 5: T36 — 关窗运行中确认框(兼 T03 后关窗复测)

**Files:**
- Modify: `writansub/gui/widgets.py`(StateMixin :19-27 追加方法)
- Modify: `writansub/gui/app.py`(QtWidgets import 补 `QMessageBox`;新增模块级 `_confirm_quit`;closeEvent 顶部接入)
- Test: `tests/test_gui_close_confirm.py`(新建)

**Interfaces:**
- Produces: `StateMixin.is_running() -> bool`(五个 tab 全部继承 StateMixin,一处实现);`writansub.gui.app._confirm_quit(tabs, parent) -> bool`。
- 运行态的事实源:各 tab 的 `_set_buttons_state(running)` 维护 `_cancel_btn.setEnabled(running)`,取消按钮可用 = 本 tab 有任务,不引入新状态。
- 关窗强杀(shutdown 即 kill 全部子进程+清模型)是用户有意保留的设计,只在前面加一道确认;**确认必须在 shutdown 之前**,拒绝时任务毫发无损。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_gui_close_confirm.py`)

```python
"""T36：关窗确认——运行中弹框，拒绝则不退出（离屏 Qt，helper 级）。"""
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from writansub.gui import app as app_mod
from writansub.gui.widgets import StateMixin


class _FakeTab(StateMixin, QWidget):
    def __init__(self, running: bool):
        super().__init__()
        self._cancel_btn = QPushButton(self)
        self._cancel_btn.setEnabled(running)


def test_state_mixin_is_running(qapp):
    assert _FakeTab(True).is_running() is True
    assert _FakeTab(False).is_running() is False


def test_idle_quits_without_dialog(qapp, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("空闲时不应弹确认框")
    monkeypatch.setattr(QMessageBox, "question", boom)
    assert app_mod._confirm_quit([_FakeTab(False), _FakeTab(False)], None) is True


def test_running_declined_blocks_quit(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert app_mod._confirm_quit([_FakeTab(True), _FakeTab(False)], None) is False


def test_running_confirmed_quits(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert app_mod._confirm_quit([_FakeTab(True)], None) is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_gui_close_confirm.py -q`
Expected: FAIL,`AttributeError`(is_running / _confirm_quit 不存在)

- [ ] **Step 3: 实现**

`widgets.py` StateMixin 追加:

```python
    def is_running(self) -> bool:
        """本 tab 是否有任务在跑。取消按钮可用态即运行态(_set_buttons_state 维护)。"""
        btn = getattr(self, "_cancel_btn", None)
        return bool(btn is not None and btn.isEnabled())
```

`app.py` QtWidgets import 行改为 `from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget`;MainWindow 类之前加:

```python
def _confirm_quit(tabs, parent) -> bool:
    """有任务运行中时弹确认;返回 False 表示用户放弃退出。"""
    if not any(tab.is_running() for tab in tabs):
        return True
    resp = QMessageBox.question(
        parent, "确认退出",
        "仍有任务运行中，退出将立即终止全部任务。确定退出？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return resp == QMessageBox.StandardButton.Yes
```

closeEvent 顶部(在 `ResourceRegistry.instance().shutdown()` **之前**)插入:

```python
    def closeEvent(self, event) -> None:
        if not _confirm_quit(self._tabs, self):
            event.ignore()
            return
        from writansub.bridge import ResourceRegistry
```

- [ ] **Step 4: 跑测试确认通过**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_gui_close_confirm.py tests/ -q` → 全绿

- [ ] **Step 5: 真 GUI 人工复测**(兼 tracker 欠的"T03 后关窗从冻结变立刻杀进程"复测)

启动 `G:\WritanSub\.venv\Scripts\python.exe -m writansub.gui.app`,流水线页选 `开发者文件/项目文件/Hathaway.mp4` 开降噪跑起;TIGER 阶段中点关窗:
1. 弹确认框,选"No" → 窗口不关、任务继续跑;
2. 再关窗选"Yes" → 立即退出;
3. `tasklist | findstr ffmpeg` → 无孤儿进程。
空闲状态关窗 → 不弹框直接退出。

- [ ] **Step 6: Commit**

```bash
git add writansub/gui/widgets.py writansub/gui/app.py tests/test_gui_close_confirm.py
git commit -m "feat: T36 关窗前确认运行中任务 (强杀语义保留, 仅加确认; 兼 T03 后关窗复测)"
```

---

### Task 6: N2 — 构建脚本 tar 口味自探测

**Files:**
- Modify: `build/scripts/build.ps1`(:101 附近 `& tar --force-local -xf $tar`)
- Modify: `build/scripts/light.ps1`(:74 附近,同款)

**Interfaces:**
- 背景:GNU tar 处理含盘符路径(`G:\...`)需要 `--force-local`,否则把盘符当远程主机;System32 的 bsdtar 不认 `--force-local` 但原生支持盘符路径。批次 4 靠 PATH 前置 Git 的 GNU tar 绕过,纯净 cmd/pwsh 环境仍会炸。自探测后两种 tar 都能跑,不再依赖 PATH 顺序。

- [ ] **Step 1: 两个脚本同款替换**

把

```powershell
    & tar --force-local -xf $tar
```

替换为

```powershell
    # GNU tar 处理含盘符路径需 --force-local；System32 bsdtar 不认该参数但原生支持盘符
    $tarIsGnu = ((& tar --version 2>$null | Select-Object -First 1) -match 'GNU tar')
    if ($tarIsGnu) { & tar --force-local -xf $tar } else { & tar -xf $tar }
```

(紧随其后的 `if ($LASTEXITCODE) { throw "tar extract failed" }` 两处均保持不动。)

- [ ] **Step 2: 语法与功能验证**(不跑完整构建;在 scratchpad 做最小实测)

```powershell
# 语法解析两个脚本(有解析错误会抛)
[void][scriptblock]::Create((Get-Content -Raw G:\WritanSub\build\scripts\build.ps1)); [void][scriptblock]::Create((Get-Content -Raw G:\WritanSub\build\scripts\light.ps1)); 'parse OK'
```

```powershell
# 功能实测: 同一带盘符路径的 tar, 两种 tar 各解一遍
$wd = "<scratchpad>\n2_tar"; New-Item -ItemType Directory -Force "$wd\a","$wd\b" | Out-Null
Set-Content "$wd\hello.txt" 'hi'
& 'C:\Program Files\Git\usr\bin\tar.exe' --force-local -cf "$wd\t.tar" -C $wd hello.txt   # GNU 打包
& 'C:\Windows\System32\tar.exe' -xf "$wd\t.tar" -C "$wd\a"                                # bsdtar 无 --force-local 解包
& 'C:\Program Files\Git\usr\bin\tar.exe' --force-local -xf "$wd\t.tar" -C "$wd\b"         # GNU 带 --force-local 解包
(Test-Path "$wd\a\hello.txt") -and (Test-Path "$wd\b\hello.txt")
# Expected: True
# 探测表达式对两种 tar 的判定:
((& 'C:\Program Files\Git\usr\bin\tar.exe' --version | Select-Object -First 1) -match 'GNU tar')   # True
((& 'C:\Windows\System32\tar.exe' --version | Select-Object -First 1) -match 'GNU tar')            # False
```

- [ ] **Step 3: Commit**

```bash
git add build/scripts/build.ps1 build/scripts/light.ps1
git commit -m "build: N2 tar 口味自探测, 纯净终端不再依赖 PATH 前置 GNU tar"
```

---

### Task 7: T13 — silero-vad 改 pip 包加载

**Files:**
- Modify: `pyproject.toml`(dependencies 列表加 `"silero-vad>=5.1",`)
- Modify: `writansub/preprocess/core.py`(`_get_silero_vad` ~:284-293)
- Test: `tests/test_preprocess_loading.py`

**Interfaces:**
- 背景:`torch.hub.load("snakers4/silero-vad")` 从 GitHub 在线拉取、无镜像处理,国内开 VAD(separate 模式)必败。silero-vad 官方 pip 包把模型文件内置在包数据里,加载零联网;`load_silero_vad()` 返回同款 jit 模型,`get_speech_timestamps` 直接顶层导出,签名与 hub 版 utils[0] 一致(输入 wav+model+threshold,输出 samples 级 start/end 字典列表),`_run_silero_vad` 消费代码零改动。
- 依赖面:silero-vad 依赖 torch/torchaudio(项目已钉 2.10.0,满足)与 onnxruntime(项目环境已有);包本体含模型 ~2MB。

- [ ] **Step 1: 加依赖并同步环境**

`pyproject.toml` dependencies 列表 `"demucs",` 之后插入 `"silero-vad>=5.1",`,然后:

```powershell
cd G:\WritanSub
uv lock          # 若清华源 403: $env:HTTPS_PROXY='http://127.0.0.1:7897'; $env:UV_DEFAULT_INDEX='https://pypi.org/simple'; uv lock
uv sync
uv pip install pytest --index-url https://pypi.org/simple   # sync 清掉了 pytest, 必须重装(挂代理)
```

核对锁文件 diff 只有 silero-vad 及其少量新增(`git diff uv.lock` 里不应出现无关升级;若 UV_DEFAULT_INDEX 兜底导致既有包源整体改写,回退锁文件改用"代理+清华源"重试)。

- [ ] **Step 2: 写失败测试**(追加到 `tests/test_preprocess_loading.py`)

```python
def test_silero_loader_uses_pip_package(monkeypatch):
    """T13：_get_silero_vad 走 silero_vad 包而非 torch.hub 在线下载。"""
    import sys
    import types as pytypes
    from writansub.preprocess import core

    stub = pytypes.ModuleType("silero_vad")
    stub.load_silero_vad = lambda: "MODEL"
    stub.get_speech_timestamps = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "silero_vad", stub)
    monkeypatch.setattr(core, "_silero_cache", None)

    model, gst = core._get_silero_vad()
    assert model == "MODEL"
    assert gst is stub.get_speech_timestamps
    assert core._get_silero_vad() == (model, gst)  # 二次调用命中缓存


def test_silero_real_load_no_network(monkeypatch):
    """真实加载：模型随包内置，离线必须成功；纯静音无语音段。"""
    import torch
    from writansub.preprocess import core

    monkeypatch.setattr(core, "_silero_cache", None)
    spans = core._run_silero_vad(torch.zeros(1, 16000))
    assert spans == []
```

- [ ] **Step 3: 跑测试确认失败**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/test_preprocess_loading.py -q`
Expected: `test_silero_loader_uses_pip_package` FAIL(现实现走 torch.hub,stub 不生效)

- [ ] **Step 4: 实现**(`preprocess/core.py` 替换 `_get_silero_vad`)

```python
def _get_silero_vad() -> tuple[Any, Any]:
    global _silero_cache
    if _silero_cache is None:
        # T13: silero-vad pip 包模型内置随包分发, 加载零联网;
        # 旧 torch.hub.load 从 GitHub 在线拉取, 国内开 VAD 必败
        from silero_vad import load_silero_vad, get_speech_timestamps
        _silero_cache = (load_silero_vad(), get_speech_timestamps)
    return _silero_cache
```

- [ ] **Step 5: 跑测试确认通过 + 无残留**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿
Run: `grep -rn "torch.hub" writansub/ --include="*.py" | grep -v archive` → 零命中

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock writansub/preprocess/core.py tests/test_preprocess_loading.py
git commit -m "fix: T13 silero-vad 改 pip 包加载 (模型内置), 退役 torch.hub 在线下载"
```

---

### Task 8: 收口 — tracker/CHANGELOG/版本/报告

**Files:**
- Modify: `pyproject.toml`(:7 `version = "0.1.8"` → `"0.1.9"`)
- Modify: `BUG_TRACKER.md`、`CHANGELOG.md`
- Create: `Batch5_Report_<实际完成日期>.md`

- [ ] **Step 1: 版本对齐**

pyproject 版本改 0.1.9 后跑 `pwsh -NoProfile -File build/scripts/check_versions.ps1`,按其输出把需要对齐的位置(launcher 等)一并改到 0.1.9,直到校验通过。

- [ ] **Step 2: BUG_TRACKER 更新**

- T12/T13/T36/T40 → `[x]`,各自条目追加一句修复口径与日期(格式仿 T03/T37 条目)。
- T09 → `[-]`,批注:`2026-07-12 用户拍板不修:批 2 定稿方案("本 tab 无任务取消置灰")经核实系老版本既有行为,与现状重合;双任务并发跨页互杀属罕见场景且 GUI 半废弃,按"设计即现状"关单`。
- T31 行遗留清单更新:`_InfoDelegate` 与 `ss_model` 参数链已随批次 5 清除(N4/N3)。
- T36 所在联动表行补一句"T03 后关窗复测已随批次 5 完成"。
- 路线区批次 5 段落(Replan 批注之后)追加:`> <日期> 批次 5 完成:T12/T13/T36/T40 + N2/N3/N4 实施,T09 关单,版本 0.1.9,见 Batch5_Report。`

- [ ] **Step 3: CHANGELOG 增补 0.1.9 段**

条目:CUDA 不可用自动回退 CPU(transcribe 全路径);镜像探测走系统代理真实 HTTPS(Clash 场景修正);silero-vad 模型随包内置(VAD 不再联网下载);关窗前确认运行中任务;移除无效的 `--ss-model` 参数与 GUI 分轨模型下拉(唯一模型 tiger-speech 自动生效);构建脚本兼容 bsdtar;模型下拉长名称正确省略。

- [ ] **Step 4: 写 Batch5_Report**(沿用 Batch4_Report 结构:commit 表、各任务要点与验证数据、行为变化知会、测试计数、剩余池更新)

行为变化必须知会的三条:`--ss-model` 从静默忽略变报错;关窗多一道确认(强杀语义不变);镜像探测超时 2s→5s(仅影响断网启动等待)。
剩余池更新:批次 8(T17+T41+T42,前置 T41 三选一对齐)、批次 9(T30)、不排期(T29→T22、T07/T08/弃单)、批外挂件(N1 等清华源恢复)。

- [ ] **Step 5: 最终全量验证**

Run: `G:\WritanSub\.venv\Scripts\python.exe -m pytest tests/ -q` → 全绿(预期 ~100 例)
Run: `G:\WritanSub\.venv\Scripts\python.exe -m pyflakes writansub/` → 干净
Run: `G:\WritanSub\.venv\Scripts\python.exe -m writansub.cli --help` 及各子命令 `--help` 冒烟

- [ ] **Step 6: Commit**(报告与 tracker 是否入库按当批用户口径;默认随收口提交)

```bash
git add pyproject.toml BUG_TRACKER.md CHANGELOG.md Batch5_Report_*.md build/
git commit -m "docs: 批次 5 收尾 tracker/CHANGELOG 收口, 版本 0.1.9, T09 关单"
```

不 push,由用户决定。

---

## 联动与顺序说明

- Task 7(T13)必须最后一个代码任务:`uv sync` 会清 pytest,集中一次环境扰动。
- Task 1(T40)与 Task 3(N3)都改 cli.py/runner.py,先后串行避免冲突;其余任务互不相交。
- Task 5 的人工复测同时销掉 tracker 联动表欠的"T03 落地后关窗行为复测"。
- T09 无代码任务,只在 Task 8 落 tracker 状态。
