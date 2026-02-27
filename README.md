# WritanSub

AI 字幕处理流水线：语音识别 → 强制打轴 → AI 翻译

## 功能

- **Whisper 语音识别** — 基于 faster-whisper (large-v3)，支持日/中/英/韩等多语言
- **MMS_FA 强制打轴** — 利用 Meta MMS 模型精确对齐字幕时间轴
- **AI 翻译** — 调用 OpenAI 兼容 API 批量翻译字幕
- **一键流水线** — 批量处理多个文件，两阶段模型管理（Whisper 全跑完 → MMS_FA 全跑完），峰值显存 = max(Whisper, MMS_FA)
- **置信度标记** — 低置信词/句自动标记到 `_review.srt` 和 `_review.ass`，方便人工校对

## 环境要求

- Python 3.12+
- NVIDIA 显卡 + 驱动 ≥570（CPU 模式可用但很慢）
  - 默认 CUDA 12.8，30/40 系如驱动低于 570 需升级驱动，或将 `pyproject.toml` 中 `cu128` 全部改为 `cu124`
- [ffmpeg](https://ffmpeg.org/download.html)（需要在 PATH 中）

## 安装

### 方法一：使用 uv（推荐，速度更快）

```bash
# 安装 uv（如果没有）
pip install uv

# 克隆项目
git clone https://github.com/mtmfs/WritanSub.git
cd WritanSub

# 创建虚拟环境并安装依赖
uv venv
uv pip install -e .
```

### 方法二：使用普通 pip

如果不使用 uv，**必须**通过 `requirements.txt` 安装以确保下载正确的 CUDA 版本 PyTorch（直接 `pip install .` 可能会下载错误的 CPU 版本）：

```bash
# 克隆项目
git clone https://github.com/mtmfs/WritanSub.git
cd WritanSub

# 创建虚拟环境（可选但推荐）
python -m venv .venv
# Windows 激活: .venv\Scripts\activate
# Linux/macOS 激活: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用

双击 `WritanSub.bat` 启动 GUI，或：

```bash
python -m writansub
```

GUI 提供四个页面：

| 页面       | 功能                              |
| ---------- | --------------------------------- |
| 一键流水线 | 批量 Whisper + 打轴，支持文件列表 |
| 语音识别   | 单文件 Whisper 转录               |
| 强制打轴   | 单文件 MMS_FA 对齐                |
| AI 翻译    | 单文件字幕翻译                    |

## 项目结构

```
WritanSub/
├── writansub/
│   ├── core/              # 核心引擎（无 GUI 依赖）
│   │   ├── types.py       # 数据类型、常量
│   │   ├── srt_io.py      # SRT 读写
│   │   ├── whisper.py     # 语音识别
│   │   ├── alignment.py   # 强制对齐 + 后处理
│   │   └── translate.py   # AI 翻译
│   ├── gui/               # Tkinter 界面
│   │   ├── widgets.py     # 通用控件
│   │   ├── pipeline_tab.py
│   │   ├── whisper_tab.py
│   │   ├── alignment_tab.py
│   │   ├── translate_tab.py
│   │   └── app.py         # 主窗口
│   ├── config.py          # 配置管理
│   └── pipeline.py        # 流水线编排
├── aitrans_pp.json        # 后处理参数配置
├── pyproject.toml
└── WritanSub.bat          # Windows 启动脚本
```
