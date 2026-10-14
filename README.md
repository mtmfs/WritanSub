# WritanSub

AI 字幕处理流水线：音频预处理 → 语音识别 → 强制打轴 → AI 翻译

本项目是一个**本地命令行工具**，只服务作者本人与 Agent（Claude Code 的 `ws` 技能据本文档构建 `writansub-cli` 命令）。图形界面已于 2026-10 整体删除，代码归档在 `archive/gui_pyside6_202610/`；安装器与启动器停止维护，归档在 `archive/build_chain_202610/`。

---

## 目录

1. [功能概览](#功能概览)
2. [环境要求](#环境要求)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [CLI 命令行参考](#cli-命令行参考)
6. [输出文件说明](#输出文件说明)
7. [存储布局](#存储布局)
8. [模型下载](#模型下载)
9. [常见问题与排错](#常见问题与排错)
10. [开发](#开发)
11. [项目结构](#项目结构)
12. [许可证](#许可证)

---

## 功能概览

WritanSub 把「做字幕」拆成四个阶段，每个阶段是一个子命令，也可以用 `pipeline` 一条命令串起来：

1. **预处理（preprocess）** — 视频里有背景音乐、音效、多人同时说话时，先用 TIGER 分离出干净人声；可选说话人分轨
2. **语音识别（transcribe）** — faster-whisper 把说话内容转成文字（支持日语、中文、英语、韩语等）
3. **强制打轴（align）** — MMS_FA / Qwen3 把文字与音频精确对齐，生成带时间轴的字幕，再做间距、时长等后处理
4. **AI 翻译（translate）** — 经 OpenAI 兼容接口把字幕翻译成目标语言（默认简体中文）

**典型工作流：**

```bash
# 一条命令跑完：降噪 → 识别 → 打轴 → 翻译 → 生成校对文件
writansub-cli pipeline video.mp4 --lang ja --denoise --translate --review --api-key sk-xxx

# 分步：先出日文字幕，人工校对后再翻译
writansub-cli pipeline video.mp4 --lang ja --denoise
writansub-cli translate video.srt --api-key sk-xxx

# 已有字幕只想重新打轴
writansub-cli align --audio video.mp4 --srt raw.srt -o aligned.srt
```

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

uv 是唯一官方安装方式：

```bash
# 1. 安装 uv
pip install uv

# 2. 进入项目目录
cd WritanSub

# 3. 创建环境并安装所有依赖（包括正确的 CUDA 版 PyTorch）
#    依赖从官方 PyPI 下载，国内网络慢时请先开启系统代理
uv sync
```

**三种等价的启动方式：**

```bash
uv run writansub-cli --help      # 推荐
python -m writansub.cli --help   # 已激活 .venv 时
WritanSubCLI.bat --help          # Windows 脚本调用；按 uv → .venv → 系统 Python 的顺序找运行环境
```

下文示例一律写 `writansub-cli ...`，按需替换成上面任一写法。

---

## 快速开始

> 目标：从一个视频文件拿到对齐好的日文字幕。

1. 确认显卡驱动 ≥570（不清楚就先去 [NVIDIA 官网](https://www.nvidia.cn/drivers/lookup/) 更新）
2. 按上节 `uv sync` 装好环境
3. 跑第一条流水线：

   ```bash
   uv run writansub-cli pipeline video.mp4 --lang ja --denoise --review
   ```

   首次运行会自动下载模型（Whisper large-v3 约 1.5 GB、MMS_FA 约 1.2 GB、TIGER 约 1 GB），请保持网络畅通
4. 处理完成后，到 `video.mp4` 所在文件夹找 `video.srt`（终稿）和 `video_review.ass`（低置信标记，拖进播放器核对）

> **国内用户注意**：程序内置 Hugging Face 镜像自动切换，访问不了 `huggingface.co` 时自动改用 `hf-mirror.com`，无需手动设置。

---

## CLI 命令行参考

### 全局选项

```bash
writansub-cli --version          # 查看版本
writansub-cli <command> --help   # 查看某个命令的详细参数
```

**参数优先级：** 命令行参数 > `--config` JSON 文件 > 内置默认值。

**取消：** `Ctrl+C` 一次发出取消请求（降噪、对齐阶段毫秒级停下，识别阶段在当前片段结束后停下），再按一次强制退出。已写到磁盘的文件都会保留。

**进度与日志** 输出到 stderr；字幕内容只写文件，不走 stdout。

### 1. pipeline — 完整流水线

```bash
# 最简用法：转录 + 对齐
writansub-cli pipeline video.mp4

# 完整用法：预处理 → 转录 → 对齐 → 翻译 + 生成 review
writansub-cli pipeline video.mp4 --denoise --translate --review --api-key sk-xxx

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

# 参考视频内嵌字幕的时间轴（按识别语言自动选轨）
writansub-cli pipeline video.mkv --ref-embedded

# 手动指定内嵌字幕轨
writansub-cli pipeline video.mkv --ref-sub-track 0
```

**pipeline 常用参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lang` | 识别语言 | `ja` |
| `--device` | 推理设备 (`cuda`/`cpu`) | `cuda` |
| `--whisper-model` | Whisper 模型 | `large-v3` |
| `--compute-type` | Whisper 量化档 (`int8`/`int8_float16`/`float16`)；CUDA 上 `float16` 更快更准但更吃显存 | `int8` |
| `--align-model` | 对齐模型 (`mms_fa`/`qwen3-fa-0.6b`) | `mms_fa` |
| `--vad` | 启用 VAD 跳过静音 | 不启用 |
| `--no-cond-prev` | 禁用前文调优 | 启用 |
| `--initial-prompt` | 初始提示词（人名/术语，空格分隔） | 无 |
| `--denoise` | 启用 DnR 降噪 | 不启用 |
| `--separate` | 启用说话人分离（含降噪） | 不启用 |
| `--mss-model` | 降噪模型 (`tiger-dnr`/`htdemucs_ft`/`htdemucs`) | `tiger-dnr` |
| `--save-intermediate` | 保存中间音轨 | 不保存 |
| `--overlap-mode` | `--separate` 时两人同时说话的处理：`merge` 压成一句（`- 甲` / `- 乙`），`keep` 保留两条重叠字幕 | `merge` |
| `--ref-srt` | 外部参考 SRT | 无 |
| `--ref-embedded` | 参考视频内嵌字幕，按识别语言自动选轨 | 不启用 |
| `--ref-sub-track` | 手动指定内嵌字幕轨索引（覆盖自动选轨） | 无 |
| `--ref-direct` | 直接使用参考轴；参考字幕不可用或没对上时自动回退强制对齐 | 不启用 |
| `--review` | 生成 Review 标记文件 | 不生成 |
| `--translate` | 启用 AI 翻译（终稿另存 `<base>_<语言>.srt`，源语终稿不受影响） | 不启用 |
| `--no-bilingual` | 翻译终稿输出单语译文 | 默认双语 |
| `--keep-whisper-srt` | 保留原始识别 SRT（`_original_whisper-<模型>.srt`） | 不保留 |
| `--keep-aligned-srt` | 保留对齐 SRT（`_aligned_<模型>.srt`） | 不保留 |
| `--config` | JSON 配置文件（见第 7 节） | 无 |

翻译相关参数（`--api-base`、`--api-key`、`--llm-model`、`--target-lang`、`--batch-size`）与 `translate` 子命令相同，见第 5 节。后处理参数见第 6 节。

### 2. preprocess — 音频预处理

```bash
# 仅降噪
writansub-cli preprocess video.mp4 --denoise

# 降噪 + 说话人分离
writansub-cli preprocess video.mp4 --separate

# 批量处理
writansub-cli preprocess a.mp4 b.mp4 --separate --device cuda
```

必须指定 `--denoise` 或 `--separate` 至少一个，`--separate` 隐含 `--denoise`。输出在原文件同目录，文件名带 `_dnr`（降噪后）或 `_speech`（分轨后）。

### 3. transcribe — 语音识别

```bash
# 单文件识别
writansub-cli transcribe video.mp4 -o output.srt

# 批量识别（输出自动命名为 a_original_whisper-large-v3.srt ...）
writansub-cli transcribe a.mp4 b.mp4 --lang ja

# 启用 VAD + 初始提示
writansub-cli transcribe video.mp4 --vad --initial-prompt "瀬尾拓也 伊地知琴子"

# 调整置信阈值
writansub-cli transcribe video.mp4 --word-conf-threshold 0.6
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `files` | (必填) | 一个或多个媒体文件；多文件时同进程内复用模型 |
| `-o/--output` | `_original_whisper-<模型>.srt` | 输出路径（**仅单文件时可用**；多文件时自动按三段式命名） |
| `--lang` | `ja` | 识别语言 |
| `--whisper-model` | `large-v3` | Whisper 模型 |
| `--compute-type` | `int8` | Whisper 量化档 |
| `--vad` | 不启用 | VAD 跳过静音段 |
| `--no-cond-prev` | 启用 | 禁用前文调优 |
| `--initial-prompt` | 无 | Whisper 初始提示（**只影响前 30 s 窗口**，对长视频效果有限） |
| `--word-conf-threshold` | 0.5 | 识别置信阈值（低于此值标记为低置信） |

### 4. align — 强制打轴

```bash
# 单文件对齐
writansub-cli align --audio video.mp4 --srt input.srt -o output_aligned.srt

# 批量对齐（audio 和 srt 一一对应）
writansub-cli align --audio a.mp4 b.mp4 --srt a.srt b.srt

# 使用 Qwen3 对齐模型
writansub-cli align --audio video.mp4 --srt input.srt --align-model qwen3-fa-0.6b
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--audio` | (必填) | 一个或多个音频/视频文件 |
| `--srt` | (必填) | 一个或多个 SRT 路径，**与 `--audio` 按位置一一对应**，数量必须相等 |
| `-o/--output` | `_aligned_<模型>.srt` | 输出路径（**仅单对时可用**；多对时自动按三段式命名） |
| `--align-model` | `mms_fa` | 对齐模型 (`mms_fa`/`qwen3-fa-0.6b`) |
| `--config` | 无 | JSON 配置文件 |

> transcribe 与 align 的多文件批处理在同一进程内只加载一次模型，12 集任务比逐集独立调用省约 3 分钟模型加载时间。

### 5. translate — AI 翻译

```bash
# 翻译为简体中文
writansub-cli translate subtitle.srt --api-key sk-xxx

# 双语输出
writansub-cli translate subtitle.srt --api-key sk-xxx --bilingual

# 指定自定义 API
writansub-cli translate subtitle.srt \
  --api-base https://api.siliconflow.cn/v1 \
  --api-key sk-xxx \
  --llm-model Qwen/Qwen2.5-72B-Instruct
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `file` | (必填) | SRT 字幕文件 |
| `-o/--output` | `_<语言>.srt`（如 `_chs.srt`） | 输出路径 |
| `--bilingual` | 不启用 | 双语输出（原文 + 译文） |
| `--api-base` | `https://api.deepseek.com/v1` | API 地址 |
| `--api-key` | 无 | API Key |
| `--llm-model` | `deepseek-chat` | LLM 模型名 |
| `--target-lang` | `简体中文` | 目标语言 |
| `--batch-size` | 20 | 每批翻译条数 |
| `--config` | 无 | JSON 配置文件 |

任何提供 OpenAI 兼容接口的服务都可以：DeepSeek、阿里云百炼（`https://dashscope.aliyuncs.com/compatible-mode/v1`）、硅基流动（`https://api.siliconflow.cn/v1`）、本地 Ollama（`http://localhost:11434/v1`）、OpenAI 官方。

> **API 设置不再持久化。** 每次运行都要经 `--api-key` 或 `--config` 文件传入；想少敲几个参数就把它们写进配置文件（第 7 节）。

### 6. 后处理参数（pipeline / align 共用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--extend-end` | 0.30 | 字幕向后延伸秒数 |
| `--extend-start` | 0.00 | 字幕向前延伸秒数 |
| `--gap-threshold` | 0.50 | 间距阈值：原始间距大于此值时保留最小间距，小于则前一条延伸到下一条开头 |
| `--min-gap` | 0.30 | 相邻字幕最小间距 |
| `--min-duration` | 0.30 | 最小字幕时长（不足则向前合并到上一条） |
| `--pad-sec` | 0.50 | 对齐窗口两侧余量；增大可提高对齐成功率，减小可提高精度 |
| `--word-conf-threshold` | 0.50 | 识别置信阈值（低于此值的词在 review 里标 `【?词】`，0 禁用） |
| `--align-conf-threshold` | 0.50 | 对齐置信阈值（低于此值的句在 review 里整句 `【】` 包裹，0 禁用） |

两套实测预设：

- **Fansub 风格（默认，不传参数即是）** — 字幕组紧凑型，cue 紧贴说话边界：
  `--extend-start 0.00 --extend-end 0.30 --min-gap 0.30 --gap-threshold 0.50 --min-duration 0.30 --pad-sec 0.50`
- **商业风格（Crunchyroll / Netflix）** — 前后留大量余量，显示时长长：
  `--extend-start 0.30 --extend-end 0.50 --min-gap 0.12 --gap-threshold 0.50 --min-duration 0.50 --pad-sec 0.50`

全部归零（`--extend-start 0 --extend-end 0 --min-gap 0 --gap-threshold 0 --min-duration 0 --pad-sec 0`）只用于测试原始对齐质量，生成的字幕不要直接拿来观看。

### 7. 通过配置文件批量设置参数

把常用参数写进 JSON，避免每次在命令行里敲：

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

## 输出文件说明

所有输出文件都生成在**输入文件所在的同一个文件夹**里，文件名以输入文件名为前缀。

命名规则：**终稿**占最简名——源语终稿 `video.srt`，翻译终稿 `video_<语言>.srt`（如 `_chs`）；**中间产物**三段式 `video_<阶段>_<模型>.srt`。

假设输入文件叫 `video.mp4`：

| 文件 | 说明 | 生成条件 |
|------|------|---------|
| `video.srt` | **源语终稿**：对齐 + 后处理完成的字幕，时间轴最准，通常这就是最终结果 | 流水线跑完必生成，不会被翻译覆盖 |
| `video_chs.srt` | **翻译终稿**：默认双语（原文 + 译文），`--no-bilingual` 时为单语译文；后缀随目标语言变化 | `--translate` 时生成，与源语终稿并存 |
| `video_original_whisper-large-v3.srt` | Whisper 原始识别结果（未对齐），模型段随所选模型 | `--keep-whisper-srt` 时保留 |
| `video_aligned_mms_fa.srt` | 强制打轴后、未做最终后处理留档的对齐版，模型段随对齐模型 | `--keep-aligned-srt` 时保留 |
| `video_review.srt` | 校对版 SRT：低置信词用 `【?词】` 包裹，低置信句用 `【】` 包裹 | `--review` 且存在低置信内容时生成 |
| `video_review.ass` | 校对版 ASS 字幕：低置信词显示为红色高亮，可直接拖进播放器预览 | 同上 |
| `video_dialog/_effects/_music.wav` | TIGER 分离的中间音轨 | `--denoise --save-intermediate` 时生成 |
| `video_spk1/_spk2.wav` | 说话人分轨后的音频 | `--separate --save-intermediate` 时生成 |

单步命令的默认输出同规则：`transcribe` → `_original_whisper-<模型>.srt`；`align` → `_aligned_<模型>.srt`；`translate` → `_<语言>.srt`（如 `_chs.srt`）。

**该用哪个字幕文件？**

- 只要时间轴准确的日文字幕 → `video.srt`
- 要中文/双语 → 开 `--translate`，拿 `video_chs.srt`（日文轴照样在 `video.srt`）
- 要校对哪些地方机器可能识别错了 → 打开 `video_review.ass` 边看边改

---

## 存储布局

程序自己产生的文件都放在一个**存储根目录**下，默认就是项目文件夹。设置环境变量 `WRITANSUB_HOME` 可以把整个根目录搬到别处（例如另一块盘），五个子目录随之一起搬；请用绝对路径。

| 子目录 | 内容 | 能否删除 |
|------|------|---------|
| `cache/` | TIGER-DnR / TIGER-Speech 两个模型的下载快照（约 2 GB，Hugging Face 缓存格式） | 可删，下次用到重新下载 |
| `models/` | 手动放置的本地模型；当前只探测 `models/Qwen3-ForcedAligner-0.6B/` | 可删，回退为在线仓库 |
| `logs/` | 会话日志 `cli_<日期>_<时间>_<PID>.log`，每种角色各保留最近 50 份 | 可删 |
| `config/` | 预留给将来的常驻后端配置，当前为空 | — |
| `run/` | 运行期临时文件：降噪 / 分轨时的溢出 wav `spill_*/`，任务结束（含取消）自动删除 | 没有任务在跑时可整个删掉；程序被强杀后残留的 `spill_*` 可手动清 |

```powershell
$env:WRITANSUB_HOME="D:\WritanSubData"
uv run writansub-cli pipeline video.mp4 --denoise
```

```cmd
set WRITANSUB_HOME=D:\WritanSubData
uv run writansub-cli pipeline video.mp4 --denoise
```

Whisper、Qwen3、MMS_FA 的权重不归这里管，仍在各自库的默认缓存，见下一节。

---

## 模型下载

首次运行需要联网下载 AI 模型。下载一次后自动缓存到本地，后续不再重复下载。

### 各模型用途与大小

| 模型 | 用途 | 大小 | 缓存位置 |
|------|------|------|---------|
| `Systran/faster-whisper-large-v3` | 语音识别 | ~1.5 GB | `~/.cache/huggingface/hub/` |
| `MMS_FA` (torchaudio 内置) | 强制打轴 | ~1.2 GB | `~/.cache/torch/hub/` |
| `TIGER-DnR` | 降噪 | ~1 GB | 存储根目录下的 `cache/`（默认项目文件夹） |
| `TIGER-Speech` | 说话人分轨 | ~1 GB | 存储根目录下的 `cache/`（默认项目文件夹） |
| `Qwen3-ForcedAligner` | 强制打轴（可选） | ~1.2 GB | `~/.cache/huggingface/hub/` |

> `~` 表示用户主目录，Windows 上通常是 `C:\Users\你的用户名`。TIGER 的两个模型不在用户主目录，而是下载到存储根目录（默认项目文件夹，见[存储布局](#存储布局)）的 `cache/` 子文件夹。

### 下载方式

**方式一：自动下载（推荐）。** 直接运行任一命令即可。程序会检测网络环境：能直接访问 `huggingface.co` 就从官方下载，访问不了会自动切换到国内镜像 `https://hf-mirror.com`，无需手动设置环境变量。

**方式二：手动设置镜像（国内网络不稳定时）。** 在运行前设置环境变量：

```cmd
set HF_ENDPOINT=https://hf-mirror.com
uv run writansub-cli pipeline video.mp4
```

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
uv run writansub-cli pipeline video.mp4
```

**方式三：离线下载。** 电脑完全不能联网时，找一台能联网的机器先跑一次让模型下好，然后把 `C:\Users\用户名\.cache\huggingface\hub\`、`~/.cache/torch/hub/`（MMS_FA）复制到目标机器的对应位置；用到降噪或分轨时再把存储根目录下的 `cache/` 复制到目标机器的同一位置（TIGER 模型）。

---

## 常见问题与排错

### 一、安装阶段

#### Q1：运行 `uv sync` 时报错 `uv 不是内部或外部命令`

**原因**：uv 没有安装成功，或者安装后没有加到系统 PATH 里。

**解决**：
1. 确认 `pip install uv` 执行成功
2. 如果还是找不到，去 [uv 官方 GitHub Releases](https://github.com/astral-sh/uv/releases) 下载 `uv-x86_64-pc-windows-msvc.zip`，把 `uv.exe` 放到 `WritanSub` 文件夹里，再运行 `uv sync`

#### Q2：安装 PyTorch 时提示 `No matching distribution found`

**原因**：Python 版本不对，或者系统架构不支持 CUDA。

**解决**：
1. 确认 Python 版本是 **3.12**（`python --version`）
2. 确认是 64 位 Windows
3. 如果显卡很老（GTX 750 Ti 等），不支持 CUDA 12，改用 CPU 版：把 `pyproject.toml` 里 torch/torchaudio 的 `pytorch-cu128` 索引配置删掉，重新 `uv sync`

#### Q3：安装时卡住不动，或下载速度极慢

**原因**：国内访问 PyPI 或 Hugging Face 较慢。

**解决**：
1. 依赖默认从官方 PyPI 下载（清华镜像对部分包持续返回 403，已不再作为默认源）。下载慢或卡住时，请开启系统代理后重新执行 `uv sync`
2. Hugging Face 模型下载由程序自动处理镜像，安装阶段不需要担心

### 二、运行阶段

#### Q4：提示 `CUDA out of memory`（显存不足）

**原因**：显卡显存不够加载当前模型。

**解决**：
1. **换小模型**：`--whisper-model medium` 或 `small`
2. **关闭预处理**：TIGER 降噪和分离也会占显存，显存 <6 GB 时不建议同时开
3. **用 CPU**：`--device cpu`，虽然慢，但不会爆显存
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

#### Q5：首次运行时下载模型失败，提示连接超时

**原因**：网络访问 Hugging Face 不稳定。

**解决**：
1. 程序已内置自动镜像切换，通常不需要干预。如果还是失败：
2. 手动设置镜像后重跑：
   ```cmd
   set HF_ENDPOINT=https://hf-mirror.com
   uv run writansub-cli pipeline video.mp4
   ```
3. 或者参考「模型下载」里的「离线下载」方式

#### Q6：语音识别结果里人名/术语总是错的

**解决**：
1. 用 `--initial-prompt "正确写法1 正确写法2"` 给出人名、术语（空格分隔）
2. 前文调优默认开启（后面的句子继承前面的写法），不要传 `--no-cond-prev`
3. 如果某句话本身发音模糊，机器确实会猜错，需要人工校对

#### Q7：对齐后的字幕时间轴还是有偏差

**解决**：
1. 增大对齐窗口余量，如 `--pad-sec 1.0`
2. 检查输入音频是否经过 TIGER 预处理（`--denoise` 后的干净人声对齐更准）
3. 尝试 `--align-model qwen3-fa-0.6b`
4. 如果原片本身语速极快、大量叠音，任何自动对齐都会有极限，需要手动微调

#### Q8：AI 翻译结果质量不好

**解决**：
1. 换更强的模型（如 `--llm-model deepseek-reasoner`）
2. 调整 `--batch-size`：太小会导致上下文不连贯，太大可能导致模型理解偏差，建议 10~30 之间
3. 先用 Whisper 出日文 → 人工简单校对 → 再跑 `translate`，效果比直接翻译错识别文本好得多
4. 某些专业术语、梗、双关语，AI 翻译必然翻车，必须人工润色

#### Q9：处理到一半想取消

**解决**：
1. 按 `Ctrl + C`：降噪和对齐阶段会立即停下，语音识别阶段会在当前片段识别完后停下。再按一次 `Ctrl + C` 强制退出
2. 已经写到磁盘的文件都会保留。注意流水线是按阶段处理整批文件的：在识别或对齐阶段取消，这一批都还没有终稿（`video.srt`）；在翻译阶段取消，源语终稿已经写好，不会丢

#### Q10：输出文件在哪里？

**答**：所有输出文件都在**输入文件所在的同一个文件夹**里。例如输入 `D:\视频\demo.mp4`，那么 `demo.srt`、`demo_review.srt` 等都会生成在 `D:\视频\` 文件夹里。

### 三、CLI 阶段

#### Q11：运行 `writansub-cli` 提示找不到命令

**原因**：虚拟环境没有激活。

**解决**：
1. 如果用 uv：`uv run writansub-cli ...`
2. 如果用 venv：先运行 `.venv\Scripts\activate`，再运行 `writansub-cli ...`
3. 或者直接用 Python 模块方式：`python -m writansub.cli ...`

#### Q12：批量处理时 `-o` 参数报错

**原因**：`-o/--output` 只在单文件时有效。

**解决**：批量处理时输出文件名会自动生成，不要手动指定 `-o`。

---

## 开发

主要技术栈：

- Python 3.12，命令行入口 `writansub/cli.py`
- faster-whisper（语音识别；另有实验性的 Qwen3-ASR 后端，尚未接入 CLI）
- torchaudio + MMS_FA / Qwen3（强制对齐）
- TIGER / Demucs（音频预处理）
- silero-vad（说话人分轨时的重叠检测）

运行测试（`uv sync` 已包含 pytest）：

```bash
uv run pytest
```

设计与迁移存档在 `docs/`。GUI（PySide6）与打包链已退役，代码在 `archive/` 下按时间归档，不再维护。

---

## 项目结构

```
WritanSub/
├── writansub/
│   ├── cli.py                # 命令行入口（writansub-cli）
│   ├── types.py              # 公共数据类型
│   ├── paths.py              # 统一路径管理（WRITANSUB_HOME 及其子目录）
│   ├── config.py             # 后处理 / 翻译默认参数
│   ├── bridge.py             # 进程与模型资源管理（子进程/取消）
│   ├── logger.py             # 会话日志
│   ├── network.py            # Hugging Face 镜像自动切换
│   ├── preprocess/           # TIGER / Demucs 音频预处理
│   ├── transcribe/           # 语音识别（统一识别器接口 + Whisper / Qwen3-ASR 适配器）
│   ├── align/                # MMS_FA / Qwen3 强制打轴与后处理
│   ├── translate/            # AI 翻译
│   ├── subtitle/             # 字幕处理（SRT 读写、内嵌字幕提取、参考轴映射、置信度审查）
│   ├── pipeline/             # 流水线编排
│   └── vendor/tiger/         # 第三方 TIGER 模型代码
├── cache/ models/ logs/ config/ run/   # 运行时生成的存储子目录（不入库，见"存储布局"）
├── tests/                    # pytest 测试
├── docs/                     # 设计与迁移存档
├── archive/                  # 已退役代码归档（GUI、打包链、Rust 原生层、TTS）
├── pyproject.toml
├── uv.lock                   # 依赖锁（安装以它为准）
├── requirements.txt          # 仅供参考，不用于安装
├── WritanSubCLI.bat          # Windows CLI 启动脚本
├── LICENSE                   # GPL-3.0 许可证全文
└── CHANGELOG.md              # 版本更新记录
```

---

## 许可证

本项目采用 [GPL-3.0](LICENSE) 许可证发布。

这意味着你可以自由使用、修改和分发本软件，但如果你发布了基于本软件的修改版本，也必须以 GPL-3.0 协议开源。
