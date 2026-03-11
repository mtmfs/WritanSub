# WritanSub

AI 字幕处理流水线：音频预处理 → 语音识别 → 强制打轴 → AI 翻译

## 功能

- **TIGER 音频预处理** — DnR 降噪（去除音效和音乐）、说话人分轨 + VAD 重叠段检测
- **Whisper 语音识别** — 基于 faster-whisper (large-v3)，支持日/中/英/韩等多语言
- **MMS_FA 强制打轴** — 利用 Meta MMS 模型精确对齐字幕时间轴
- **AI 翻译** — 调用 OpenAI 兼容 API 批量翻译字幕
- **一键流水线** — 批量处理多个文件，两阶段模型管理（Whisper 全跑完 → MMS_FA 全跑完），峰值显存 = max(Whisper, MMS_FA)
- **置信度标记** — 低置信词/句自动标记到 `_review.srt` 和 `_review.ass`，方便人工校对

## 环境要求

- Python 3.12+
- NVIDIA 显卡 + 驱动 ≥570（CPU 模式可用但很慢）
  - 默认 CUDA 12.8，30/40 系如驱动低于 570 需升级驱动，或将 `pyproject.toml` 中 `cu128` 全部改为 `cu124`
## 安装

### 方法一：使用 uv（推荐，速度更快）

```bash
# 安装 uv（如果没有）
pip install uv

# 克隆项目
git clone https://github.com/mtmfs/WritanSub.git
cd WritanSub

# 一键创建虚拟环境并安装所有依赖
uv sync
```

### 方法二：使用普通 pip

**必须**通过 `requirements.txt` 安装以确保下载正确的 CUDA 版本 PyTorch（直接 `pip install .` 可能会下载 CPU 版本）：

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
pip install -e .
```

## 使用

双击 `WritanSub.bat` 启动 GUI，或：

```bash
python -m writansub
```

GUI 提供五个页面：

| 页面       | 功能                                        |
| ---------- | ------------------------------------------- |
| 一键流水线 | 批量 Whisper + 打轴 + 翻译，支持文件列表    |
| 预处理     | TIGER 降噪 / 说话人分轨，批量处理           |
| 语音识别   | 单文件 Whisper 转录                          |
| 强制打轴   | 单文件 MMS_FA 对齐                           |
| AI 翻译    | 单文件字幕翻译                               |

## 输出文件

所有输出文件生成在**输入文件所在目录**，以输入文件名为前缀：

以 `video.mp4` 为例：

| 文件 | 说明 | 生成条件 |
| --- | --- | --- |
| `video.srt` | Whisper 识别的原始字幕 | 勾选"保留 Whisper 原始 SRT"时保留，否则打轴后自动清理 |
| `video_aligned.srt` | 经 MMS_FA 强制打轴后的字幕（**最终结果**） | 始终生成 |
| `video_review.srt` | 校对标记版字幕（低置信词用`【?…】`包裹，低置信句用`【】`包裹） | 存在低置信内容时生成 |
| `video_review.ass` | 校对标记版 ASS 字幕（低置信词红色高亮，可直接拖入播放器预览） | 同上 |
| `video_translated.srt` | AI 翻译后的字幕 | 使用 AI 翻译功能时生成 |

## 模型下载

首次运行需要下载以下模型，下载后缓存到本地，后续不再重复下载：

| 模型 | 用途 | 大小 |
| --- | --- | --- |
| Systran/faster-whisper-large-v3 | 语音识别 | ~1.5 GB |
| MMS_FA (torchaudio 内置) | 强制打轴 | ~1.2 GB |
| TIGER-DnR | 降噪（对话/音效/音乐分离） | — |
| TIGER-Speech | 说话人分轨 | — |

提供三种下载方式：

### 方式一：自动下载（推荐）

直接启动程序即可，首次运行时会自动从 Hugging Face 下载模型：

```bash
python -m writansub
```

### 方式二：手动下载（原版地址）

如果自动下载失败，可从 Hugging Face 手动下载：

- Whisper: [huggingface.co/Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) — 下载全部文件放入 `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/`
- MMS_FA: 由 torchaudio 管理，缓存在 `~/.cache/torch/hub/`，参考 [PyTorch 文档](https://pytorch.org/audio/stable/pipelines.html#torchaudio.pipelines.MMS_FA)

### 方式三：国内镜像下载

设置 Hugging Face 镜像环境变量后启动程序，模型会从国内镜像自动下载：

```bash
set HF_ENDPOINT=https://hf-mirror.com
python -m writansub
```

也可以直接从镜像站手动下载 Whisper 模型：[hf-mirror.com/Systran/faster-whisper-large-v3](https://hf-mirror.com/Systran/faster-whisper-large-v3)

## 项目结构

```
WritanSub/
├── writansub/
│   ├── core/              # 核心引擎（无 GUI 依赖）
│   │   ├── types.py       # 数据类型、常量
│   │   ├── srt_io.py      # SRT 读写
│   │   ├── tiger.py       # TIGER 降噪 / 说话人分轨
│   │   ├── whisper.py     # 语音识别
│   │   ├── alignment.py   # 强制对齐 + 后处理
│   │   └── translate.py   # AI 翻译
│   ├── gui/               # PySide6 界面
│   │   ├── widgets.py     # 通用控件
│   │   ├── pipeline_tab.py
│   │   ├── tiger_tab.py   # 预处理（降噪/分轨）
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
