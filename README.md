# WritanSub

AI 字幕处理流水线：音频预处理 → 语音识别 → 强制打轴 → AI 翻译

本项目面向**完全零基础**的用户。如果你只会用电脑看视频、下载软件，按照本说明一步一步操作，也可以独立完成字幕制作。

---

## 目录

1. [我能用 WritanSub 做什么？](#我能用-writansub-做什么)
2. [快速开始（第一次使用）](#快速开始第一次使用)
3. [环境要求](#环境要求)
4. [安装](#安装)
5. [使用指南（GUI）](#使用指南gui)
6. [输出文件说明](#输出文件说明)
7. [模型下载](#模型下载)
8. [CLI 命令行参考](#cli-命令行参考)
9. [常见问题与排错](#常见问题与排错)
10. [开发](#开发)
11. [许可证](#许可证)

---

## 我能用 WritanSub 做什么？

WritanSub 把「做字幕」这件事拆成了几个自动化的步骤。你只需要准备好视频或音频文件，软件会帮你：

1. **预处理** — 如果视频里有背景音乐、音效、多人同时说话，可以先分离出干净的人声
2. **语音识别** — 把说话内容转换成文字（支持日语、中文、英语、韩语等）
3. **强制打轴** — 把文字和画面里的说话时间精确对齐，生成带时间轴的字幕
4. **AI 翻译** — 把日语字幕自动翻译成中文（或其他语言）

**典型工作流程：**

- **最简单**：把视频拖到「一键流水线」，等待处理，最后拿到中文字幕
- **更精细**：先用「预处理」去噪 → 「语音识别」出日文 → 人工校对日文 → 「AI 翻译」出中文
- **已有字幕只想对齐**：用「强制打轴」把现成的 SRT 字幕和音频精确对齐

---

## 快速开始（第一次使用）

> 目标：用最快的速度，从一个视频文件拿到对齐好的字幕。

### 第一步：检查电脑能不能跑

- 操作系统：Windows 10/11（64 位）
- 显卡：NVIDIA 独立显卡，显存 ≥6 GB 最佳（4 GB 可跑小模型）
- 硬盘：至少预留 20 GB 空间（用于安装和下载模型）
- 驱动：NVIDIA 显卡驱动建议 ≥570（不清楚就先去 [NVIDIA 官网](https://www.nvidia.cn/drivers/lookup/) 更新驱动）

### 第二步：安装软件环境

**方法一：使用 uv（推荐，更快更稳）**

1. 按 `Win + R`，输入 `cmd`，回车，打开黑框框（命令提示符）
2. 输入下面命令安装 uv：
   ```bash
   pip install uv
   ```
3. 去 GitHub 下载本项目（或找人要压缩包），解压到某个文件夹，比如 `D:\WritanSub`
4. 在 cmd 里进入这个文件夹：
   ```bash
   cd /d D:\WritanSub
   ```
5. 运行安装：
   ```bash
   uv sync
   ```
   这一步会自动下载 Python 依赖和模型库，可能要 10~30 分钟，取决于网速。

**方法二：使用普通 pip**

如果你不想装 uv，也可以：

```bash
cd /d D:\WritanSub
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第三步：启动 GUI 并跑第一个视频

1. 双击文件夹里的 `WritanSub.bat`
2. 等待窗口打开（首次启动会自动下载模型，请保持网络畅通）
3. 点击顶部标签页「一键流水线」
4. 点击「添加文件」，选择你的视频（如 `video.mp4`）
5. 保持默认设置，直接点击右下角「开始处理」
6. 等待进度条跑完
7. 去视频所在的文件夹找输出文件（见下节「输出文件说明」）

> **国内用户注意**：程序已内置 Hugging Face 镜像自动切换。如果网络不好，会自动使用 `hf-mirror.com` 下载模型，无需手动设置。

---

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10/11 64 位 | Windows 11 |
| Python | 3.12 | 3.12 |
| 显卡 | NVIDIA GTX 1060 6GB | RTX 3060 12GB 或更高 |
| 显存 | 4 GB | 8 GB+ |
| 内存 | 8 GB | 16 GB+ |
| 硬盘 | 20 GB 可用 | 50 GB 可用 |
| 驱动 | NVIDIA 驱动 ≥470 | ≥570 |

**CUDA 版本说明：**

- 默认依赖 CUDA 12.8（随 PyTorch 一起安装）
- 30 系/40 系显卡如果驱动低于 570，安装时可能会报错
- 解决办法：升级显卡驱动，或者把 `pyproject.toml` 里所有的 `cu128` 改成 `cu124`，然后重新安装

---

## 安装

### 方法 A：uv 安装（推荐）

```bash
# 1. 安装 uv
pip install uv

# 2. 进入项目目录
cd WritanSub

# 3. 创建环境并安装所有依赖（包括正确的 CUDA 版 PyTorch）
#    已默认使用清华大学 PyPI 镜像，国内下载更快
uv sync
```

**安装后怎么启动？**

```bash
# GUI 模式
uv run python -m writansub

# CLI 模式
uv run writansub-cli --help
```

### 方法 B：pip 安装

```bash
# 1. 进入项目目录
cd WritanSub

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活环境（Windows）
.venv\Scripts\activate

# 4. 安装依赖（注意：必须带 requirements.txt，否则可能装成 CPU 版 PyTorch）
pip install -r requirements.txt
pip install -e .
```

**安装后怎么启动？**

```bash
# GUI 模式
python -m writansub

# CLI 模式
writansub-cli --help
```

### Windows 快捷启动

项目根目录提供了两个批处理文件，双击即可启动：

- `WritanSub.bat` — 启动 GUI（图形界面）
- `WritanSubCLI.bat` — 启动 CLI（命令行）

这两个脚本会自动检测你的环境：优先使用 `uv` → 其次用 `.venv` → 最后 fallback 到系统 Python。

---

## 使用指南（GUI）

启动 GUI 后，顶部有 5 个标签页。下面按使用频率从高到低讲解。

### 1. 一键流水线（最常用）

这是把「预处理 → 语音识别 → 强制打轴 → AI 翻译」串起来的全自动模式。

#### 界面分区说明

- **左上：媒体文件列表** — 点击「添加文件」选择视频/音频，可批量添加多个
- **左中：模型配置**
  - **识别语言**：默认 `ja`（日语），可选 `zh`（中文）、`en`（英语）、`ko`（韩语）等
  - **听写模型**：默认 `large-v3`（最准，需约 10 GB 显存），显存不够可换 `medium` 或 `small`
  - **推理设备**：默认 `cuda`（用显卡），没有 N 卡选 `cpu`（极慢）
  - **对齐模型**：默认 `mms_fa`，轻量且效果稳定；可选 `qwen3-fa-0.6b`
  - **启用前文调优**：勾选后，前一句识别结果会影响后一句，整体更连贯；如果前一句错了，可能把错误传下去
  - **跳过静音 (VAD)**：勾选后自动跳过没有人声的片段，加快识别速度；如果已经做了 TIGER 预处理，建议关闭
  - **初始提示**：输入人名、术语（如 `瀬尾拓也 伊地知琴子`），让 Whisper 统一写法
- **右中：预处理增强**
  - **人声去噪 (DnR)**：分离出人声，去掉背景音乐和音效
  - **重叠分离 (Separate)**：分离多个说话人，并检测重叠片段；会自动先执行降噪
  - **保存中间音轨**：勾选后保留 TIGER 处理后的临时 WAV 文件
- **右上：输出保留**
  - **保留 Whisper SRT**：保留原始识别结果（未经时间轴对齐）
  - **保留对齐 SRT**：保留强制对齐后的中间结果
  - **生成 Review 文件**：勾选后，如果存在低置信度的词/句，会生成 `_review.srt` 和 `_review.ass`
- **右下：参考字幕**
  - **参考内嵌字幕**：从视频里提取已有字幕的时间轴作为参考
  - **外部 SRT**：手动指定一个参考字幕文件
  - **直接使用参考轴**：跳过强制对齐，直接用参考字幕的时间轴
- **底部：后处理参数**
  - 一般用默认即可。常见调整：
    - **识别置信阈值**（默认 0.5）：低于此值的词会在 review 文件里标红
    - **对齐窗口余量**（默认 0.5）：对齐时给每句话两边留的余量，增大可提高成功率

#### 操作步骤

1. 点击「添加文件」，选择要处理的视频
2. （可选）根据需要勾选「人声去噪」或「重叠分离」
3. （可选）如果要做 AI 翻译，勾选左下角「AI 翻译」（翻译 API 设置见「AI 翻译」页）
4. 点击「开始处理」
5. 等待进度条完成。过程中可以点「暂停」临时停下，或点「取消任务」中断

#### 什么时候用参考字幕？

如果你手头有**时间轴准确但文字是错的语言**的字幕（比如原片有英文字幕，你想做中文字幕），可以勾选「参考内嵌字幕」或指定「外部 SRT」，这样 Whisper 的文本会自动映射到参考字幕的时间轴上，省去重新打轴的步骤。

---

### 2. 预处理

如果你只想做音频降噪/分轨，不跑后面的识别，用这个页面。

#### 操作步骤

1. 点击「添加」，选择媒体文件（可多个）
2. 勾选「降噪」和/或「对话分轨」
   - 勾「对话分轨」会自动勾「降噪」（因为分轨需要先降噪）
3. 选择设备（`cuda` 或 `cpu`）
4. 点击「开始处理」

#### 输出说明

处理后的音频会生成在原视频同目录下，文件名带 `_dnr`（降噪后）或 `_speech`（分轨后）。

---

### 3. 语音识别

单文件 Whisper 转录。适合只想把音频/视频转成文字 SRT，不做后续处理。

#### 操作步骤

1. 「媒体文件」选择你的视频/音频
2. 「输出 SRT」会自动填为同目录下同名的 `.srt`，可手动改
3. 选择语言、模型、设备
4. （可选）调整「识别置信阈值」— 设为 0.5 表示概率低于 50% 的词会在 review 文件里标出来
5. （可选）在「初始提示」里输入人名/术语，提高识别准确率
6. 点击「开始识别」

#### 小贴士

- **上文关联**：默认开启。做动漫/电视剧时建议开启，角色名更稳定；做访谈/演讲时如果前几句容易错，可关闭
- **跳过静音**：长视频（>30 分钟）建议开启，能省一半时间

---

### 4. 强制打轴

如果你已经有 SRT 字幕文件，但时间轴不准（或完全没有时间轴），用这个功能把字幕和音频精确对齐。

#### 操作步骤

1. 「音频文件」选择原视频/音频
2. 「字幕文件」选择要对齐的 `.srt`
3. 「输出路径」自动为 `xxx_aligned.srt`
4. 选择语言和对齐模型（默认 `mms_fa`）
5. 点击「开始对齐」

#### 对齐模型怎么选？

- **mms_fa**：Meta 开源模型，效果稳定，支持多种语言，推荐日常使用
- **qwen3-fa-0.6b**：阿里 Qwen3 强制对齐模型，显存占用约 2 GB，在某些场景下精度更高

---

### 5. AI 翻译

把已有的 SRT 字幕文件翻译成其他语言。

#### 操作步骤

1. 「字幕文件」选择要翻译的 `.srt`
2. 「输出路径」自动为 `xxx_translated.srt`
3. 填写翻译设置：
   - **目标语言**：默认「简体中文」
   - **模型**：默认 `deepseek-chat`（DeepSeek V3）
   - **API 地址**：默认 `https://api.deepseek.com/v1`
   - **API Key**：你在 DeepSeek（或其他兼容 OpenAI 的厂商）申请的 API Key
4. （可选）勾选「双语输出」，这样每句字幕会同时保留原文和译文
5. 点击「开始翻译」

#### 支持的 API 厂商

任何提供 OpenAI 兼容接口的服务都可以，常见有：

- DeepSeek (`https://api.deepseek.com/v1`)
- 阿里云百炼 (`https://dashscope.aliyuncs.com/compatible-mode/v1`)
- 硅基流动 (`https://api.siliconflow.cn/v1`)
- 本地 Ollama (`http://localhost:11434/v1`)
- OpenAI 官方

> **注意**：API Key 只保存在本地配置文件里，不会上传到任何地方。

---

## 输出文件说明

所有输出文件都生成在**输入文件所在的同一个文件夹**里，文件名以输入文件名为前缀。

假设你的输入文件叫 `video.mp4`：

| 文件 | 说明 | 生成条件 |
|------|------|---------|
| `video.srt` | Whisper 原始识别结果（未对齐） | 勾选「保留 Whisper SRT」时保留 |
| `video_aligned.srt` | 经 MMS_FA 强制打轴后的字幕，**时间轴最准**，通常这就是你要的最终结果 | 只要跑了打轴就生成 |
| `video_review.srt` | 校对版 SRT：低置信词用 `【?词】` 包裹，低置信句用 `【】` 包裹 | 勾选「生成 Review 文件」且存在低置信内容时生成 |
| `video_review.ass` | 校对版 ASS 字幕：低置信词显示为红色高亮，可直接拖进播放器预览 | 同上 |
| `video_translated.srt` | AI 翻译后的字幕 | 使用 AI 翻译功能时生成 |
| `video_dnr.wav` | TIGER 降噪后的音频 | 预处理勾选了「降噪」且勾选了「保存中间音轨」时生成 |
| `video_speech.wav` | TIGER 说话人分轨后的音频 | 预处理勾选了「对话分轨」且勾选了「保存中间音轨」时生成 |

**我该用哪个字幕文件？**

- 只要时间轴准确的对齐字幕 → `video_aligned.srt`
- 要翻译成中文 → 基于 `video_aligned.srt` 再跑 AI 翻译，得到 `video_translated.srt`
- 要校对哪些地方机器可能识别错了 → 打开 `video_review.ass` 边看边改

---

## 模型下载

首次运行需要联网下载 AI 模型。下载一次后自动缓存到本地，后续不再重复下载。

### 各模型用途与大小

| 模型 | 用途 | 大小 | 缓存位置 |
|------|------|------|---------|
| `Systran/faster-whisper-large-v3` | 语音识别 | ~1.5 GB | `~/.cache/huggingface/hub/` |
| `MMS_FA` (torchaudio 内置) | 强制打轴 | ~1.2 GB | `~/.cache/torch/hub/` |
| `TIGER-DnR` | 降噪 | ~1 GB | `~/.cache/huggingface/hub/` |
| `TIGER-Speech` | 说话人分轨 | ~1 GB | `~/.cache/huggingface/hub/` |
| `Qwen3-ForcedAligner` | 强制打轴（可选） | ~1.2 GB | `~/.cache/huggingface/hub/` |

> `~` 表示你的用户主目录，Windows 上通常是 `C:\Users\你的用户名`。

### 下载方式

#### 方式一：自动下载（推荐）

直接启动程序即可。程序会检测网络环境：

- 如果能直接访问 `huggingface.co`，就从官方下载
- **如果访问不了，会自动切换到国内镜像 `https://hf-mirror.com`**

无需手动设置任何环境变量。

#### 方式二：手动设置镜像（国内网络不稳定时）

如果你发现自动下载很慢，可以在启动前手动设置环境变量：

**Windows（cmd）：**
```cmd
set HF_ENDPOINT=https://hf-mirror.com
python -m writansub
```

**Windows（PowerShell）：**
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python -m writansub
```

#### 方式三：离线下载

如果电脑完全不能联网，可以找一台能联网的机器先下载好模型，再复制过来：

1. 在能联网的机器上运行一次程序，让模型自动下载
2. 把整个 `C:\Users\用户名\.cache\huggingface\hub\` 文件夹复制到目标机器的对应位置
3. 同样复制 `~/.cache/torch/hub/`（MMS_FA 模型）

---

## CLI 命令行参考

如果你习惯用命令行，或者想批量处理/写脚本，可以使用 `writansub-cli`。

### 全局选项

```bash
writansub-cli --version          # 查看版本
writansub-cli <command> --help   # 查看某个命令的详细参数
```

### 1. pipeline — 完整流水线

```bash
# 最简用法：转录 + 对齐
writansub-cli pipeline video.mp4

# 完整用法：预处理 → 转录 → 对齐 → 翻译 + 生成 review
writansub-cli pipeline video.mp4 --denoise --translate --review

# 批量处理多个文件
writansub-cli pipeline a.mp4 b.mp4 c.mp4 --review

# 指定语言为中文，使用 CPU 运行
writansub-cli pipeline video.mp4 --lang zh --device cpu

# 使用小模型（显存不够时）
writansub-cli pipeline video.mp4 --whisper-model small

# 使用 Qwen3 对齐模型
writansub-cli pipeline video.mp4 --align-model qwen3-fa-0.6b

# 启用说话人分离
writansub-cli pipeline video.mp4 --separate --save-intermediate

# 使用参考字幕直接映射时间轴
writansub-cli pipeline video.mp4 --ref-srt reference.srt

# 直接使用参考轴，跳过强制对齐
writansub-cli pipeline video.mp4 --ref-srt reference.srt --ref-direct
```

**pipeline 常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lang` | 识别语言 | `ja` |
| `--device` | 推理设备 (`cuda`/`cpu`) | `cuda` |
| `--whisper-model` | Whisper 模型 | `large-v3` |
| `--align-model` | 对齐模型 (`mms_fa`/`qwen3-fa-0.6b`) | `mms_fa` |
| `--vad` | 启用 VAD 跳过静音 | 不启用 |
| `--no-cond-prev` | 禁用前文调优 | 启用 |
| `--initial-prompt` | 初始提示词 | 无 |
| `--denoise` | 启用 DnR 降噪 | 不启用 |
| `--separate` | 启用说话人分离（含降噪） | 不启用 |
| `--save-intermediate` | 保存中间音轨 | 不保存 |
| `--ref-srt` | 外部参考 SRT | 无 |
| `--ref-direct` | 直接使用参考轴 | 不启用 |
| `--review` | 生成 Review 标记文件 | 不生成 |
| `--translate` | 启用 AI 翻译 | 不启用 |
| `--keep-whisper-srt` | 保留原始识别 SRT | 不保留 |
| `--keep-aligned-srt` | 保留对齐 SRT | 不保留 |

### 2. preprocess — 音频预处理

```bash
# 仅降噪
writansub-cli preprocess video.mp4 --denoise

# 降噪 + 说话人分离
writansub-cli preprocess video.mp4 --separate

# 批量处理
writansub-cli preprocess a.mp4 b.mp4 --separate --device cuda
```

### 3. transcribe — 语音识别

```bash
# 单文件识别
writansub-cli transcribe video.mp4 -o output.srt

# 批量识别（输出自动命名为 a.srt, b.srt...）
writansub-cli transcribe a.mp4 b.mp4 --lang ja

# 启用 VAD + 初始提示
writansub-cli transcribe video.mp4 --vad --initial-prompt "瀬尾拓也 伊地知琴子"

# 调整置信阈值
writansub-cli transcribe video.mp4 --word-conf-threshold 0.6
```

### 4. align — 强制打轴

```bash
# 单文件对齐
writansub-cli align --audio video.mp4 --srt input.srt -o output_aligned.srt

# 批量对齐（audio 和 srt 一一对应）
writansub-cli align --audio a.mp4 b.mp4 --srt a.srt b.srt

# 使用 Qwen3 对齐模型
writansub-cli align --audio video.mp4 --srt input.srt --align-model qwen3-fa-0.6b
```

### 5. translate — AI 翻译

```bash
# 翻译为简体中文
writansub-cli translate subtitle.srt --target-lang 简体中文

# 双语输出
writansub-cli translate subtitle.srt --bilingual

# 指定自定义 API
writansub-cli translate subtitle.srt \
  --api-base https://api.siliconflow.cn/v1 \
  --api-key sk-xxx \
  --llm-model Qwen/Qwen2.5-72B-Instruct
```

### 6. 通过配置文件批量设置参数

你可以把常用的参数写进一个 JSON 文件，避免每次都在命令行里敲：

**`myconfig.json`：**
```json
{
  "extend_end": 0.2,
  "gap_threshold": 0.4,
  "word_conf_threshold": 0.6,
  "api_base": "https://api.deepseek.com/v1",
  "api_key": "sk-你的key",
  "model": "deepseek-chat",
  "target_lang": "简体中文",
  "batch_size": 20
}
```

**使用：**
```bash
writansub-cli pipeline video.mp4 --config myconfig.json --translate --review
```

---

## 常见问题与排错

### 一、安装阶段

#### Q1：运行 `uv sync` 时报错 `uv 不是内部或外部命令`

**原因**：uv 没有安装成功，或者安装后没有加到系统 PATH 里。

**解决**：
1. 确认 `pip install uv` 执行成功
2. 如果还是找不到，用 pip 方式安装（方法 B）
3. 或者去 [uv 官方 GitHub Releases](https://github.com/astral-sh/uv/releases) 下载 `uv-x86_64-pc-windows-msvc.zip`，把 `uv.exe` 放到 `WritanSub` 文件夹里，再运行 `uv sync`

#### Q2：安装 PyTorch 时提示 `No matching distribution found`

**原因**：Python 版本不对，或者系统架构不支持 CUDA。

**解决**：
1. 确认 Python 版本是 **3.12**（`python --version`）
2. 确认是 64 位 Windows
3. 如果显卡很老（GTX 750 Ti 等），不支持 CUDA 12，改用 CPU 版：把 `requirements.txt` 里带 `cu128` 或 `cu124` 的行删掉，换成 `torch torchvision torchaudio`

#### Q3：安装时卡住不动，或下载速度极慢

**原因**：国内访问 PyPI 或 Hugging Face 较慢。

**解决**：
1. **uv 用户**：项目已默认配置清华大学 PyPI 镜像（`pyproject.toml`），`uv sync` 会自动走国内源，无需手动设置。
2. **pip 用户**：安装时加上清华镜像参数：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. Hugging Face 模型下载由程序自动处理镜像，安装阶段不需要担心

---

### 二、启动阶段

#### Q4：双击 `WritanSub.bat` 后闪一下就没了

**原因**：通常是 Python 环境没装好，或者依赖缺失。

**解决**：
1. 不要双击，改为在 cmd 里手动运行，看具体报错：
   ```bash
   cd /d D:\WritanSub
   uv run python -m writansub
   ```
2. 如果报错 `ModuleNotFoundError: No module named 'xxx'`，说明依赖没装全，重新执行 `uv sync` 或 `pip install -r requirements.txt`
3. 如果报错和 `writansub_native` 有关，说明 Rust 扩展没编译。进入 `native` 目录运行 `maturin develop`（需要安装 Rust 工具链）

#### Q5：提示 `CUDA out of memory`（显存不足）

**原因**：显卡显存不够加载当前模型。

**解决**：
1. **换小模型**：把 Whisper 模型从 `large-v3` 换成 `medium` 或 `small`
2. **关闭预处理**：TIGER 降噪和分离也会占显存，显存 <6 GB 时不建议同时开
3. **用 CPU**：虽然慢，但不会爆显存
4. **分段处理**：把长视频切成几段分别处理

**各模型显存参考：**

| 模型 | 所需显存 |
|------|---------|
| Whisper large-v3 | ~8-10 GB |
| Whisper medium | ~5 GB |
| Whisper small | ~2 GB |
| Whisper tiny | ~1 GB |
| MMS_FA | ~2 GB |
| Qwen3-FA-0.6B | ~2 GB |
| TIGER-DnR | ~2 GB |

#### Q6：首次启动时下载模型失败，提示连接超时

**原因**：网络访问 Hugging Face 不稳定。

**解决**：
1. 程序已内置自动镜像切换，通常不需要干预。如果还是失败：
2. 手动设置镜像后重启：
   ```cmd
   set HF_ENDPOINT=https://hf-mirror.com
   uv run python -m writansub
   ```
3. 或者参考「模型下载」里的「离线下载」方式

---

### 三、运行阶段

#### Q7：语音识别结果里人名/术语总是错的

**解决**：
1. 在「初始提示」里输入正确的人名写法（空格分隔）
2. 勾选「启用前文调优」，让后面的句子继承前面的写法
3. 如果某句话本身发音模糊，机器确实会猜错，需要人工校对

#### Q8：对齐后的字幕时间轴还是有偏差

**解决**：
1. 增大「对齐窗口余量」（比如从 0.5 改到 1.0）
2. 检查输入音频是否经过 TIGER 预处理（降噪后的干净人声对齐更准）
3. 尝试换用 `qwen3-fa-0.6b` 对齐模型
4. 如果原片本身语速极快、大量叠音，任何自动对齐都会有极限，需要手动微调

#### Q9：AI 翻译结果质量不好

**解决**：
1. 换更强的模型（如 `deepseek-chat` → `deepseek-reasoner`）
2. 调整 `batch_size`：太小会导致上下文不连贯，太大可能导致模型理解偏差，建议 10~30 之间
3. 先用 Whisper 出日文 → 人工简单校对 → 再跑翻译，效果会比直接翻译错识别文本好得多
4. 某些专业术语、梗、双关语，AI 翻译必然翻车，必须人工润色

#### Q10：处理到一半想暂停/取消

**解决**：
1. GUI 里直接点「暂停」按钮，再次点击恢复
2. GUI 里点「取消任务」，或 CLI 里按 `Ctrl + C`，程序会在当前段落处理完后安全退出
3. 已经生成的输出文件会保留，不会丢失

#### Q11：输出文件在哪里？

**答**：所有输出文件都在**你选择的输入文件所在的同一个文件夹**里。例如你选了 `D:\视频\demo.mp4`，那么 `demo_aligned.srt`、`demo_review.srt` 等都会生成在 `D:\视频\` 文件夹里。

---

### 四、CLI 阶段

#### Q12：运行 `writansub-cli` 提示找不到命令

**原因**：虚拟环境没有激活。

**解决**：
1. 如果用 uv：`uv run writansub-cli ...`
2. 如果用 venv：先运行 `.venv\Scripts\activate`，再运行 `writansub-cli ...`
3. 或者直接用 Python 模块方式：`python -m writansub.cli ...`

#### Q13：批量处理时 `-o` 参数报错

**原因**：`-o/--output` 只在单文件时有效。

**解决**：批量处理时输出文件名会自动生成，不要手动指定 `-o`。

---

## 开发

本项目包含一个 Rust 原生扩展（`native/`），用于进程管理与模型资源注册。

开发前请先阅读 [`CLAUDE.md`](CLAUDE.md) 了解架构与常用命令。

主要技术栈：

- Python 3.12 + PySide6（GUI）
- faster-whisper（语音识别）
- torchaudio + MMS_FA / Qwen3（强制对齐）
- TIGER / Demucs（音频预处理）
- Rust + PyO3 + maturin（原生扩展）

---

## 项目结构

```
WritanSub/
├── writansub/
│   ├── types.py              # 公共数据类型、常量
│   ├── paths.py              # 统一路径管理
│   ├── config.py             # 配置读写
│   ├── bridge.py             # Rust FFI 桥接层
│   ├── preprocess/           # TIGER 音频预处理
│   ├── transcribe/           # Whisper 语音识别
│   ├── align/                # MMS_FA / Qwen3 强制打轴
│   ├── translate/            # AI 翻译
│   ├── subtitle/             # 字幕处理（SRT 读写、置信度审查）
│   ├── pipeline/             # 流水线编排
│   ├── gui/                  # PySide6 界面
│   └── vendor/tiger/         # 第三方 TIGER 模型代码
├── native/                   # Rust 原生扩展
├── pyproject.toml
├── requirements.txt
├── WritanSub.bat             # Windows GUI 启动脚本
├── WritanSubCLI.bat          # Windows CLI 启动脚本
├── LICENSE                   # GPL-3.0 许可证全文
└── CHANGELOG.md              # 版本更新记录
```

---

## 许可证

本项目采用 [GPL-3.0](LICENSE) 许可证发布。

这意味着你可以自由使用、修改和分发本软件，但如果你发布了基于本软件的修改版本，也必须以 GPL-3.0 协议开源。
