# Chunk 对齐精细标注设计（1 caption = 1 bidirectional chunk）

目标：训练时模型 predict 每个 chunk 时，文本条件恰好是**当前 chunk 自己的 caption**，
不掺历史、不漏未来。本文定义标注粒度、帧数学、数据格式、VLM 标注协议和已知坑。

## 1. 结论（多少帧标一次）

**每 33 RGB 帧标一个 caption**，从 `cut_start` 开始平铺，不重叠：

```
caption_k 覆盖帧区间 [cut_start + 33k, cut_start + 33(k+1))   （CFR 视频中的帧号）
caption 数量 = floor((cut_end - cut_start) / 33)
尾巴 < 33 帧的部分：编码器直接丢弃（num_rgb // 33），不要标注
```

**注意：语料虽然叫 "24FPS"，实际是逐 clip 恒定但全局不统一的整数 fps**（抽查以 25 为主，
另有 24/30 等；原始源甚至有 50/60）。所以一个 chunk 的时长在 ~1.1–1.4 秒之间浮动，
**一切边界必须按帧号算，绝不能按秒算**。帧对齐本身是安全的：encode 路径 `base_fps=None`
→ stride 恒为 1（`dataloader_mp4_dist.py:351`），无论 fps 编码器都取连续帧。
chunk 是模型的 bidirectional 单元：
- `latent_window_size = 9`，`(9-1)*4 + 1 = 33`（Wan VAE 4× 时间压缩）；
- 离线编码是**逐 chunk 独立过 VAE** 的（`get_multievent-latents.py:224-231`：
  `s = c*33; vae.encode(pixel[s:s+33])`），chunk 之间没有 latent 泄漏；
- DiT 在 chunk 内 9 个 latent 帧上是双向 attention，跨 chunk 只通过 history 条件。

所以 33 帧窗口是唯一与训练目标严格对齐的标注粒度。若成本受不了，可退化为
**k×33 帧一个 caption（k=2 或 3）**——边界仍落在 chunk 边界上，对齐性质不变，
只是切换粒度变粗；千万不要选 32/48/64 这种与 33 不对齐的窗口。

Wan2.2-TI2V-5B 口径：时间压缩同为 4×（只有空间变 16×），chunk 仍是 33 帧，
本设计对 14B / 5B 两条线通用。

## 2. 数据格式：不需要改训练代码

现有 schema 已经支持精确 per-chunk 绑定，训练侧零改动：

- `.pt` 里 `event_idx_per_chunk: (N_chunks,)` 是 chunk→caption 的显式查表，
  dataloader 取样到 chunk c 时直接返回 `event_prompt_embeds[event_idx_per_chunk[c]]`
  （`dataloader_history_latents_dist.py:212-217`）。
- per-chunk 标注下有两种写法：
  1. **推荐：去重写法。** `cap = 去重后的 caption 列表`，
     `event_idx_per_chunk[c] = c 对应 caption 在去重表中的下标`。
     相邻 chunk 内容没变时 caption 会几乎一样，去重能把 embed 存储省一个数量级。
  2. 朴素写法：`cap` 长度 = N_chunks，`event_idx_per_chunk = [0,1,2,...]`，
     `switch_frame_index = [33, 66, ...]`（majority-overlap 退化为恒等映射）。
- **必须加断言**：`len(event_idx_per_chunk) == num_chunks == num_rgb // 33`，
  且 `max(event_idx_per_chunk) < len(cap)`。错一位就是整段错位监督。

## 3. VLM 标注协议

1. **在最终帧流上标注**：即 CFR 重采样后（逐 clip 恒定 fps，以 25 为主，非统一 24）、按 `cut` 截取后的帧
   （和 `BucketedFeatureDataset` 解码路径同一坐标系）。在原始变帧率源视频上标、
   再换算帧号，边界一定漂。最稳的做法是直接复用编码脚本同款 decode 逻辑抽帧。
2. **每个 chunk 给 VLM 采 8–16 帧**（必含窗口首、尾帧），33 帧全喂浪费且没必要。
3. **两遍法保证实体一致性**：
   - Pass 1（整段视频）：建实体表——"a man in a red jacket"、"a white dog" 等
     规范名，每段视频固定；
   - Pass 2（逐 chunk）：给 VLM 当前窗口帧 + 实体表，要求只用规范名指代。
   否则 chunk 0 是 "a man"、chunk 3 变 "the person"，文本条件自身不一致。
4. **caption 措辞纪律**（这是 train/infer 一致性的关键）：
   - 只描述窗口内可见的内容；禁止 "then / next / begins to / is about to /
     continues to" 这类指涉窗口外的语篇词——推理时用户给的 prompt 不会带这种
     上下文，训练带了就是分布错位；
   - 每条 caption 自足（scene + subject + action），不依赖"上一条说过"；
   - chunk 0 的 caption 写得更完整（场景建立 + 外观细节），它同时兼任 i2v
     条件帧所在 chunk；后续 chunk 以动作为主。
5. 保持现有确定性约定：greedy decoding、固定 seed，同输入同权重可复现。

## 4. 坑清单

| # | 坑 | 对策 |
|---|---|---|
| 1 | **帧坐标系漂移**：原始 fps / VFR 上标注 → CFR 后边界错位 | 只在 CFR + cut 后的帧流上按帧号标（§3.1），fps 不统一，绝不按秒换算 |
| 2 | **cut 偏移**：chunk 0 从 `cut_start` 开始，不是视频第 0 帧；uttid 后缀就是 `_{cut_start}-{cut_end}` | 标注 JSON 用同一 uttid 键、帧号相对 cut_start |
| 3 | **尾巴截断**：`num_chunks = num_rgb // 33`，余帧不进训练 | 尾巴不标；caption 数与 num_chunks 强断言 |
| 4 | **embed 存储爆炸**：`encode_prompt` 全 pad 到 512×4096（`utils_base.py:619,652`）。一条 42-chunk 的 1 分钟片段若逐 chunk 存独立 embed ≈ 42×512×4096×2B ≈ 176 MB/clip | caption 去重 + `event_idx_per_chunk` 查表（§2）；必要时存实际 token 长度、加载时再 pad |
| 5 | **相邻 caption 近重复**：~1.1–1.4s 窗口里多数时候什么都没变，切换监督信号被稀释 | 预期内；去重后统计"每视频独立 caption 数"，太低（<2）的片段对 switching 训练无贡献，可降采样 |
| 6 | **VLM 单窗口看不懂动作**：1.4s 太短，动作方向可能歧义（举起 vs 放下） | 允许 VLM 看前后各 1 chunk 的帧做参考，但 prompt 里强制"只描述中间窗口" |
| 7 | **事件真实边界在 chunk 中间**：该 chunk 内容本身是混合的 | 让 VLM 如实描述该窗口（含转变），这正是模型实际看到的内容，不要强行归到某一事件 |
| 8 | **标注成本**：每分钟视频 ≈ 40–55 次 VLM 调用（随 fps 浮动）（两遍法再 +1 整段调用） | 先在小子集（如 1k clips）验证 caption 质量与训练收益，再放量；或 k×33 粗粒度起步 |
| 9 | **最短长度过滤**：只有 1 个 chunk 的片段没有 history/switch 可学 | 过滤 `< 2×33 = 66` 帧（建议 ≥ 4 chunks 以上才有像样的 history） |
| 10 | **推理侧对齐**：训练是 per-chunk 硬绑定，推理时切 prompt 也必须落在 chunk 边界（interpolation_steps=0 的硬切） | 复用 `assign_chunks_to_events` 的同一套帧数学，离线/在线 byte-identical |
| 11 | **【重要】旧 latent 有随机时间偏移，不能直接复用**：`dataloader_mp4_dist.py:421-425` 在 cut 长于 length bucket 时做 `random.randint(cut_start, cut_end - bucket_len*stride)` 随机起点，且 length bucket（21,41,…,501，步长 20）不是 33 的倍数——现有 `latents_cfr_int_30b_368x640` 每条 .pt 的实际起始帧有最多 ~19 帧的**未记录**随机偏移 | 新 caption 标定死 `start = cut_start`；配套**重新编码** latent（固定 `start_frame=cut_start`，读恰好 `n_chunks*33` 帧），不要把 per-chunk caption 挂到旧 .pt 上 |

## 5. 实现（已落地，可复用）

代码：`scripts/data_prep/chunk_captioning/`（sample_30k.py / annotate_gemini_chunks.py /
merge.py / README.md）。数据根目录：
`/mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/`。

### 5.1 帧网格抽取（与训练编码 byte 级同坐标系）

```python
FRAMES_PER_CHUNK = 33
FRAME_OFFSETS = (0, 11, 22, 32)          # 每 chunk 抽 4 帧：首/三分之一/三分之二/尾
indices = [cut_start + k*33 + off for k in range(n_chunks) for off in FRAME_OFFSETS]
frames = PyVideoReader(video_path, threads=0).get_batch(indices)  # 全按帧号，不按秒
# JPEG 质量 85，长边 resize 到 512（≈20KB/帧 ≈ 1.1k token/帧）
```

### 5.2 API 调用（Forge / TensorBlock，OpenAI 兼容）

- key：`/mnt/beegfs/xiangbo/.config/forge_api_key`（chmod 600，绝不进 git）或环境变量
  `FORGE_API_KEY`；base_url `https://api.forge.tensorblock.co/v1`。
- 模型：**`tensorblock/gemini-3.1-flash-lite`**（无裸 "3.1-flash"；lite 单请求快 ~3 倍、
  更便宜，质量验证可用，连接词违规略多——都进 `flags` 可后过滤）。备选
  `tensorblock/gemini-3-flash-preview`（前 ~2.1k clips 用的它，更守纪律）。
  `--backend google` 走原生 google-genai（key=GEMINI_API_KEY）。
- 一条视频**一次调用**：messages = system + [intro, "CHUNK 0:", 4 张 data-URL 图, "CHUNK 1:", …, 指令]，
  `response_format={"type":"json_object"}`；返回
  `{header, role, background, style, chunks:[{event,scene}]×N}`。
  `len(chunks)!=n_chunks` 或全局字段缺失 → 重试（指数退避，容 429/5xx）。
- 每 clip 实测 ~24k 输入 / ~450 输出 token；并发 48 → **~119 clips/min**。

### 5.3 Caption 组装（训练直接用）

每 chunk 的最终 caption = 全片共享 header/role/Background/style（保证 <ID_x> 一致）
+ 该 chunk 自己的 event/scene，拼成训练语料同款 HTML-tag schema。输出 JSONL 同时保留
原始件（chunk_events/chunk_scenes/全局字段）便于重组去重；`flags` 记录 soft-lint
（连接词、缺 ID 标签）。

### 5.4 跑法与运维教训

```bash
# 全量（supervisor 自愈循环；断点续跑幂等，重跑只补缺）
setsid nohup /mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/run_until_done.sh &
tail -f .../chunk_captions_gemini_30k/annotate_full.log
# 合并 + 补漏清单
python merge.py --sample .../sample_30k.jsonl --shards .../shards --out .../captions_chunks.jsonl
```

- detached 进程在 shell 节点会被**静默 SIGKILL**（无栈、无 OOM 记录）→ 必须用
  supervisor 循环（`run_until_done.sh`），靠幂等 resume 兜底。
- **改 run_until_done.sh 不会影响运行中的 supervisor**（bash 按原始解析执行循环体）
  —— 改参数后要把 supervisor+annotator 都杀掉重启。
- `pkill -f "annotate_gemini_chunks.py"` 会匹配到你自己 shell 的命令行把自己杀掉
  （诡异 exit 144）→ 用 `pkill -f "annotate_gemini_chunks[.]py"`。
- 模型偶尔把字段返回成 list（schema 之外）→ `_as_text` 递归强转 + 单条 try/except，
  坏记录只记 error 不杀 run。
- prompt 里不要写死秒数（fps 不统一，见 §1）；一切用"33 frames"表述。

## 6. 与现状的差异

现状（`build_multievent_json.py` + majority-overlap）：事件边界任意帧 → 每个 chunk
按最大重叠归属某事件，一个 caption 覆盖多个 chunk，边界 chunk 的监督是"近似的"。
本设计把近似消掉：标注粒度 = 监督粒度 = 33 帧。代价是标注量上升约一个数量级
（事件级 → chunk 级），主要缓解手段是 caption 去重与 k×33 折中。

## 7. 不同标注方式下 VAE 应该怎么 encode（总结）

先说结论：**VAE 的数学运算完全一样**（同一个 Wan2.1 VAE，同样的 33 帧一组 chunk-wise 独立
encode，`num_chunks = num_rgb // 33`，尾巴丢弃）。不一样的只有一件事：**送进 VAE 的是原视频的
哪一段帧**。

| 数据 | 取帧方式 | 起点 | 长度 | encoder |
|---|---|---|---|---|
| 单 prompt（整段一个 caption） | `BucketedFeatureDataset` 随机时间裁剪：`start = randint(cut_start, cut_end - bucket_len)`（`dataloader_mp4_dist.py:421-425`） | **随机、不落盘** | 长度桶 21/41/61/…/501，**不是 33 的倍数** | `get_short-latents.py` |
| 旧 multievent（event_shift，majority-overlap） | 同上（也走 Bucketed 随机裁剪） | 随机 | 长度桶 | `get_multievent-latents.py` |
| **chunk 对齐 caption（本次）** | 确定性取帧 `[cut_start, cut_start + n_chunks*33)` | **固定 = cut_start** | **恰好 n_chunks×33** | `scripts/data_prep/chunk_captioning/encode_chunk_latents.py` |

为什么单 prompt 可以随机裁剪：整段只有一个 caption，随便切哪一段、切多长，"这段画面 ↔ 这个
caption"都成立，随机裁剪反而是数据增广。旧 multievent 也侥幸能用：caption 边界靠 majority
overlap 分配到 chunk，偏移十几帧只是让边界 chunk 的归属"差一点"，训练信号是近似对的。

而 chunk 对齐 caption 的合同是 **caption_k 精确覆盖 `[cut_start+33k, cut_start+33(k+1))`**。
标注时 Gemini 看到的就是这些帧；训练时第 k 个 latent chunk 必须也来自这些帧，否则第 k 段
caption 对着第 k±偏移 段画面训练，边界越清晰伤害越大。所以取帧必须确定性锚在 cut_start，
长度必须是 33 的整倍数（尾巴不标、也不 encode）。

其余全部不变：分辨率 368×640、stride=1（`base_fps=None`，帧号对齐、绝不按秒）、每 33 帧一组
独立过 VAE、latent shape `(n_chunks, 16, 9, 46, 80)`。新 encoder 额外把每个 chunk 的 caption
过 UMT5 存成 `event_prompt_embeds` + `event_idx_per_chunk = [0..n-1]`，训练代码零改动。

为什么旧 latent 不能"转换"成新 latent：见 `docs/VAE_LATENT_CONVERSION_WHY_NOT.md`。
