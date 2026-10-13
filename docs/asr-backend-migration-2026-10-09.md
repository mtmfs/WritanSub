# 听写后端通用化与 VibeVoice 接入准备存档

日期：2026-10-09。项目基线：`cf6aa913ed7b6f08809b954d0be312c42920d9fb`。

> **状态更新（2026-10-13）**
>
> - 第 5 节的步骤 1、2 已完成：`Recognizer` 协议在 `writansub/transcribe/base.py`，faster-whisper 收进 `transcribe/whisper.py`，`Transcript` / `Segment` 类型在 `writansub/types.py`，runner 与 CLI 只经识别器对象取缓存键、产物标签和转写结果。改动尚未提交，记在 CHANGELOG 的 Unreleased。
> - 实际先接入的第二后端是 Qwen3-ASR-1.7B（`transcribe/qwen.py`，实验性，未上 CLI），不是本文准备的 VibeVoice。
> - VibeVoice 仍在计划内，尚未接入；第 1、2 节准备的权重、源码和增量依赖保持原状，第 4 节的接口差异和第 5 节的步骤 3～5 仍然有效。
> - 第 3 节表格的行号对应基线 `cf6aa91`，接口化之后已不准，各行现状见表下说明。
>
> 以下正文是 2026-10-09 的原始记录，未改动。

本次范围是下载和技术准备，尚未实现新听写后端，也未改变现有字幕流水线。上次字幕逻辑修复已提交，彼时非 GUI 测试 179 项通过；本次未修改业务代码，未重跑该套测试。

## 1. 已准备的文件

选择此前讨论的 **Microsoft VibeVoice-ASR-Streaming-1.5B** 作为第二个听写引擎的接入候选。它支持日语、说话人标注和自定义热词，但不能据此认定其日语影视识别质量优于当前 Whisper；需要后续真实素材对比。此次没有下载 7B、BitNet 或额外 Qwen 识别权重。

| 内容 | 固定版本 / 验证状态 |
| --- | --- |
| 模型仓库 | `microsoft/VibeVoice-ASR-Streaming-1.5B` |
| 模型 revision | `4262d23d8a539a6530cf64fbd0b1751ef9a30853` |
| 模型文件 | 14 个，共 5,644,397,665 字节，约 5.64 GB；含 3 个 safetensors 分片、权重索引、模型/预处理配置、分词器及词表 |
| 权重校验 | 3 个分片的 SHA-256 与 Hugging Face LFS 元数据一致；所有下载文件的大小匹配；权重文件头的 1177 个张量名称与索引一致 |
| 官方源码 | `microsoft/VibeVoice`，commit `16fb2cb1217c9934a886e1948ffb06120caa2df5`，提交时间 2026-10-08 |
| 源码压缩包 | 126,626,438 字节，已解压，下载地址及 SHA-256 记入 manifest |
| 增量依赖 | 18 个适合当前 Windows / Python 3.12 的 wheel，逐个核对 PyPI SHA-256 |

本机是 RTX 5060 Laptop，显存约 8GB。模型名称中的“1.5B”不是整个模型的总参数量，官方模型页标注约 3B；完整权重约 5.6GB。8GB 显存能否支持目标片长及其他模型共存，尚未验证。

**模型缓存入口：**

```text
C:\Users\25183\.cache\huggingface\hub\models--microsoft--VibeVoice-ASR-Streaming-1.5B\snapshots\4262d23d8a539a6530cf64fbd0b1751ef9a30853
```

**实际解析到的磁盘位置：**

```text
F:\C盘迁移_20261003\Models\huggingface\hub\models--microsoft--VibeVoice-ASR-Streaming-1.5B\snapshots\4262d23d8a539a6530cf64fbd0b1751ef9a30853
```

本机 Python 运行环境中的默认 `HF_HUB_CACHE` 指向 `G:\本地部署模型\faster-whisper`，但 G 盘空间不足以宽裕地容纳此次准备，故本次显式指定了上述缓存入口，没有修改全局配置。最初落在 G 盘的本次未完成下载已移到指定缓存后续传。下次优先使用已验证的绝对模型目录，避免因默认缓存不同而重复下载。

**官方源码与准备材料：**

```text
G:\WritanSub\cache\model-prep\2026-10-09\
  VibeVoice-16fb2cb1217c9934a886e1948ffb06120caa2df5.zip
  VibeVoice-16fb2cb1217c9934a886e1948ffb06120caa2df5\
  wheels\
  staged-deps\
  existing-environment-constraints.txt
  staged-requirements.txt
  model-metadata.json
  model-download-manifest.json
  source-download-manifest.json
  dependency-download-manifest.json
  offline-smoke-result.json
  download_model.py
  download_dependencies.py
```

`cache/` 受已有 `.gitignore` 管理，权重、第三方源码和临时依赖不会进入项目 Git。此存档位于 `docs/`，可随下一次文档提交保存。用户本机路径仅是本次准备记录，不应写死到未来业务代码。

## 2. 依赖与已完成的验证

当前项目：Python 3.12.10、PyTorch / torchaudio 2.10.0+cu128、transformers 4.57.6、accelerate 1.12.0、huggingface-hub 0.36.2、qwen-asr 0.0.6。

- 固定源码版本声明 `transformers>=4.51.3,<5.0.0`，与现有 4.57.6 相容；现有 qwen-asr 要求 **transformers==4.57.6**，不应照模型页示例直接执行无约束的升级命令。
- 上游未锁定 diffusers。查询时最新版 0.41.0 要求 huggingface-hub>=1.32.0，会迫使现有依赖栈变化。本次以现有环境版本为约束进行 dry-run 求解，选出 **diffusers 0.37.1**，没有计划替换现有包。
- 所有 18 个新增依赖版本保存在 `staged-requirements.txt`。除了推理所需包，也准备了官方项目声明的 demo 依赖，如 aiortc；以后正式接入可以再拆成可选依赖。
- wheel 下载后只安装到准备目录下的 `staged-deps/`，供单独进程临时加入 `sys.path` 做验证。**没有安装进项目 `.venv`，没有修改 pyproject.toml 或 uv.lock。** 这是一份增量依赖包，不是包含现有 PyTorch 等所有包的完整离线环境。
- 在离线模式下，固定源码中的 `VibeVoiceASRForConditionalGeneration`、`VibeVoiceASRProcessor` 导入成功，processor 和 config 从本地模型目录加载成功。
- `<|text_chunk_end|>` 已由所下载分词器提供，ID 为 `151665`，无需再下载基础 Qwen 分词器。误用基础分词器可能丢失这一流式结束标记。
- 权重文件头读取和索引一致性检查通过；未把整套模型加载进显存，未进行音频推理、速度、显存峰值或识别准确率测试。
- 系统 ffmpeg 已存在：`C:\ffmpeg\ffmpeg.EXE`。

导入检查有两条警告：当前环境 requests 的依赖版本警告；checkpoint 标注 `Qwen2Tokenizer` 而官方处理器使用 `VibeVoiceASRTextTokenizerFast` 的类名警告。它们没有阻止本次离线 processor 加载，但尚未通过真实推理排除影响，不能把导入通过写成模型运行已验证。

续传及校验命令（在项目根目录运行）：

```powershell
.\.venv\Scripts\python.exe cache\model-prep\2026-10-09\download_model.py
```

增量依赖已展开；如需从已下载 wheel 重建准备目录，可使用：

```powershell
uv pip install --python .venv\Scripts\python.exe --target cache\model-prep\2026-10-09\staged-deps --no-deps --no-index --find-links cache\model-prep\2026-10-09\wheels -r cache\model-prep\2026-10-09\staged-requirements.txt
```

## 3. 当前项目的 Whisper 专用问题

| 位置 | 当前绑定 | 通用化方向 |
| --- | --- | --- |
| `writansub/transcribe/core.py:6` | `transcribe()` 直接使用 faster-whisper 参数及 `(segments, info)` 返回格式，`model: Any` 不能提供真正隔离 | 将加载、专属参数及原始结果解析放入 Whisper 适配器；其他引擎实现相同听写接口 |
| `writansub/pipeline/runner.py:112`、CLI 听写入口 | 直接构造 WhisperModel；配置、缓存键、输出名称均围绕 whisper_model | 入口只选择 backend / model / backend_options，模型实例由后端工厂和缓存层管理 |
| `writansub/types.py:60` | WordInfo 只有文字和概率；听写取得词时间却丢弃；字幕和词是两份平行列表 | 引入 Transcript，词归属于片段，保留可选的词/片段时间与来源，不再按两个列表的位置配对 |
| `writansub/align/core.py:202` | 按每句已有时间轴加 pad_sec 切音频，对纯文本结果无法直接工作 | 明确时间信息种类：无时间、粗范围、模型时间戳、强制对齐结果；在构造最终 Sub 前补齐有效时间 |
| review / Sub.score | 假设存在概率；MMS token 分数与 Qwen 时长比又共用一个 score | 未提供分数使用未知值，记录来源及含义，不伪造概率，也不默认不同后端共用阈值 |
| 重叠区域识别 / Sub.speaker | 固定两条 TIGER 分离轨重识别；speaker=1/2 同时表示音轨身份 | 区分 source_track 与 speaker_id；内置说话人标注不等价于分离重叠音频 |
| `writansub/bridge.py` | 模型缓存和任务取消/暂停均挂全局单例；release_model 不卸载模型 | 为未来服务化分离任务控制和共享缓存；8GB 下的新模型加载、串行运行和卸载策略需要明确讨论 |

各行现状（2026-10-13）：

- 前三行所述的绑定已解除。`WordInfo` 带可选的词起止时间，`probability` 允许为空；词挂在 `Segment.words` 下，不再靠两个平行列表按位置配对。
- 第四行部分解决：`Segment.time_kind` 区分 `estimate` / `window` / `aligned`，识别结果全部为 `aligned` 时 runner 跳过强制对齐。完全没有时间的纯文本结果仍无路径可走。
- 第五行部分解决：review 跳过没有概率的词；MMS 与 Qwen3 对齐分数共用阈值的问题未动（BUG_TRACKER T22）。
- 第六行未动：`Segment.speaker` 已留字段，但 `to_subs()` 不传递它，`Sub.speaker` 仍以 1/2 兼表分离音轨。
- 第七行未动：`bridge.py` 的模型缓存与取消、暂停仍是进程全局单例。

SRT 读写、参考映射、字幕合并、间距后处理、翻译大部分可以复用；上次修复的整句替换、参考失败回退和 review 信息保留应作为回归约束。

## 4. VibeVoice 接入时必须处理的差异

这些是所下载固定代码和配置中已核实的接口差异，不是模型效果推测：

1. **音频为 24kHz。** 预处理配置给出 target_sample_rate=24000、speech_tok_compress_ratio=3200、chunk_frames=22、lookahead_frames=4，即正文步长约 2.933333 秒、前瞻约 0.533333 秒。应读取模型配置，不能沿用当前分轨/对齐的 16kHz 常量。
2. **返回的是流式文本块。** 官方文件示例调用 `streaming_generate()`，得到 `(chunk_index, total_chunks, chunk_text)`。默认提示要求 speaker、content，没有提供 Whisper 式逐词 probability 或精确 start/end。音频块边界只是粗定位线索，前瞻部分也不能重复算进字幕范围。
3. **要处理文本跨块与格式问题。** 保留原始 chunk 输出并验证结构化结果解析、跨块句子连接、空输出及截断；不要把返回原文连同 speaker 标记直接写进 SRT，也不能假设每块都恰好是一条完整字幕。具体输出行为待真实推理确认。
4. **必须使用该模型自带分词器。** 固定源码已优先读取 checkpoint 的 tokenizer；不要用旧示例强制替换成基础 Qwen tokenizer。当前项目 transformers 4.57.6 下已验证的是上游仓库类的导入路线，不是模型页自动生成的 AutoModel 类名示例。
5. **显式控制 dtype 和 attention。** 官方文件示例在 CUDA 上使用 BF16，支持默认 `sdpa`，本次不安装 flash-attn。checkpoint 顶层 dtype 为 float32，未来加载时应显式指定 BF16，不能依靠文件大小估算默认加载的显存。
6. **“流式”不自动保证长片内存有界。** 所下载 `streaming_generate()` 会先积累整段的 feature_chunks，随后逐块生成并延续 KV cache。另有 `init_streaming_state()` / `streaming_generate_step()` 接口；正式接入需要检查长片的特征驻留、上下文增长和取消响应，再选择分段/逐步处理方式。不能把短音频成功外推为整集可用。
7. **不能默认与旧模型同时常驻。** 目前 release_model 只记日志；Whisper、TIGER、对齐模型再叠加该模型可能超出 8GB。旧版本曾选择保留常驻策略，这次只记录新约束，未擅自改变该设计。

## 5. 建议下一步实施顺序

1. 固定现有 Whisper 行为，提取 Recognizer 接口与 Whisper 适配器，迁移模型工厂和专属配置；公共任务层只认识统一输入与 Transcript。
2. Transcript 保留原文、片段内词、可选时间/置信度、说话人和来源信息；完成时间轴后再转换为 Sub。片段相对时间只在汇总处加一次媒体偏移。
3. 接入固定版本的 VibeVoice 作为显式选择的第二后端，默认仍使用 Whisper。先跑短素材验证加载、输出解析、时间范围和资源释放，再处理长片。
4. 时间轴路径分开处理“采用识别轴 / 参考轴 / 强制对齐”；VibeVoice 的块范围标注为粗轴，配合对齐验证。不能把没有置信度的词填成 0 或 1，也不能用假精确时间绕过校验。
5. 用相同素材比较静音、日语普通对话、背景音乐、多人插话、专有名词及长片。记录漏字、重复、说话人错误、字幕轴误差、人工修改时间、耗时和显存峰值，而不只看能否生成文本。

验收应包括：当前 Whisper 的回归测试继续通过；不同后端专属参数不串用；取消、空结果和模型异常不会被伪装成正常结果；新后端的缺失时间/置信度被正确处理；实际素材上没有系统性跨块重复、漏句或时间漂移。

本存档是后续重构的起点，不代表已经完成适配、GPU 推理或质量验证。GUI 删除及前后端分离属于后续工作。

## 官方资料与本地证据

- [模型卡与权重](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B/tree/4262d23d8a539a6530cf64fbd0b1751ef9a30853)
- [固定版本官方源码](https://github.com/microsoft/VibeVoice/tree/16fb2cb1217c9934a886e1948ffb06120caa2df5)
- [流式 ASR 使用说明](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/docs/vibevoice-asr-streaming.md)
- [文件推理示例](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/demo/vibevoice_asr_streaming_inference_from_file.py)
- [流式推理实现](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/vibevoice/modular/modeling_vibevoice_asr.py)
- [固定版本依赖声明](https://github.com/microsoft/VibeVoice/blob/16fb2cb1217c9934a886e1948ffb06120caa2df5/pyproject.toml)
- 本地校验明细：准备目录下的 `model-download-manifest.json`、`source-download-manifest.json`、`dependency-download-manifest.json`、`offline-smoke-result.json`。
