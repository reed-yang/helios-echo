# Stage-1 历史-latent dataloader:帧数、chunk,以及一个 clip 是怎么被消费的

本文说明 `helios/dataset/dataloader_history_latents_dist.py`(`BucketedFeatureDataset`)如何把一个预编码好的
clip 变成训练样本、为什么 clip 必须 **≥121 帧**、以及模型每一步到底看到什么。文末有 **3-chunk(121 帧)** 和
**7-chunk(261 帧)** 两个完整举例。

所有 `file:line` 均指向本仓库。

---

## 1. 万物的基本单位:chunk(块)

- `latent_window_size = 9` 个 latent 帧 = `(9-1)*4 + 1 = 33` 个 RGB 帧(Wan VAE:时间 4× 下采样)。
- 离线编码器(`tools/offload_data/get_short-latents.py:207-215`)把一个 clip 切成
  **`N = floor(帧数 / 33)` 个 chunk**,每个 33-RGB-帧的块编码成 9 个 latent 帧。
- 因此 `.pt` 里 `vae_latent` 的形状是 **`(N, 16, 9, H/8, W/8)`** —— N 个 chunk × 16 个 VAE 通道 × 9 个 latent 帧
  × 空间。(实测验证:121→N=3,161→4,221→6,261→7,…)

也就是说,一个 clip **不是**作为一整段长序列训练,而是 *N 个独立的 9 帧 chunk 的堆叠*。

## 2. 一个训练样本是什么(固定大小,与 clip 长度无关)

每次 `__getitem__` 只返回**一个 (history, target) 对**,由 `prepare_stage1_latent` 构造
(dataloader_history_latents_dist.py:136-196):

- **target = 1 个 chunk = 9 个 latent 帧**(这一步要去噪的块)。
- **history = `sum(history_sizes) = 16+2+1 = 19` 个 latent 帧**(target 之前的记忆上下文)。
- **section = history + target = 19 + 9 = 28 个 latent 帧。**

把 28 个连续 latent 帧解码回 RGB = `(28-1)*4 + 1 = ` **109 个 RGB 帧** —— 这就是 "109" 的来源。
**它是固定的每步训练窗口,不是 clip 长度。** clip 越长只是*能采样的 target chunk 越多*;每个样本本身永远是这个
28-latent(≈109-RGB)窗口。

这 19 个历史帧会按"新旧"进一步拆成 **Multi-Term Memory** 三档 `history_sizes = [16, 2, 1]`(长/中/短),由独立的
Conv3d 分别 patchify(`transformer_helios.py:1065-1068`,`patch_long` stride 8、`patch_mid` 4、`patch_short` 2):
**最旧的 16 帧**池化最狠(粗粒度远期记忆),**最新的 1 帧**保留最细。dataloader 只负责给出连续的 19 帧;拆分/patchify
在模型里做。

## 3. `prepare_stage1_latent` 怎么构造这一对(机制)

```
source = vae_latent (N, 16, 9, h, w)
拉平 chunk ->                时间轴有  N*9  个 latent 帧                 # rearrange "b c t h w -> c (b t) h w"
前面补 19 个 0 latent 帧 -> continue = [19个0] ++ [N*9 个真实帧]          # 长度 19 + N*9
choice_idx ~ 均匀{0 .. N-1}                                            # 随机选哪个 chunk 当 target
start = choice_idx * 9
history = continue[start        : start + 19]                          # target 前面的 19 帧
target  = continue[start + 19   : start + 28]                          # = 真实 chunk #choice_idx(9 帧)
```

前面补 **19 个 0** 的关键后果:
- `choice_idx = 0` → history **全是 0**(模拟*生成开始*,没有过去)。
- 较小的 `choice_idx` → history **部分补 0**(模型只有一点点过去)。
- 一旦 `choice_idx ≥ 3` → history **全是真实帧**(此时 target 前面已有 ≥19 帧真实过去)。

`is_keep_x0=True`:还会返回首帧 x0;若 `choice_idx==0` 则把它置 0(生成开始时没有真实锚帧)。这是
"Easy Anti-Drifting" 的首帧处理。

## 4. 为什么 clip 必须 ≥121 帧(= ≥3 chunk)

`floor(121/33) = 3`。过滤要求 **≥3 chunk**,在两处强制执行:
- **编码前:** `scripts/data_prep/build_offload_json_cfr.py:91`(`--min-frames 121`)丢掉过短的 clip。
- **训练时(冗余保险):** `dataloader_history_latents_dist.py:103` `if num_frame < 121: continue`
  (`num_frame` 从 `.pt` 文件名 `{uttid}_{num_frame}_{H}_{W}.pt` 解析)。

为什么偏偏要 ≥3 chunk:
- `num_rollout_sections` 默认 **3**(dataloader_history_latents_dist.py:20)。在
  **rollout / DMD-teacher-forcing / GAN** 这条路径(`return_all_vae_latent=True`)里,会取一段*连续*的
  `19 + num_rollout_sections*9 = 46` 帧窗口,并**硬性断言 `total_sections ≥ num_rollout_sections`**,否则直接
  `raise ValueError("Not enough sections")`(dataloader:178-182)。所以*编码出的数据必须保证 ≥3 chunk*,这些
  Stage-3 regime 才能在上面跑。
  - 在纯 flow-matching 的 LoRA 跑里,`return_all_vae_latent` 是 `False`(它由
    `dmd_teacher_forcing or is_use_gan` 决定,train_helios.py:803-810),不走那条硬路径 —— 但 121 过滤仍然生效
    (写死在共享 dataloader 里),而且 ≥3 chunk 也保证随机 `choice_idx` 能落到 零/部分/较满历史 三类。
- **121 vs 理论下限 99:** 编码会把 clip 归到长度桶(`…141, 121, 101, 81…`,`dataloader_mp4_dist.py`)。
  101 和 121 都是 3 chunk,但 121(=3.67 chunk)比临界的 101(=3.06)更稳 —— 容忍解码/裁剪掉几帧后仍稳稳 ≥3 chunk。
  过滤保留桶 ≥121。

## 5. 完整举例 A —— **121 帧 clip(N = 3 chunk)**

`vae_latent` 形状 `(3, 16, 9, h, w)`。拉平后 = 27 个真实 latent 帧。前面补 19 个 0:

```
continue 下标:  0 ............ 18 | 19 ...... 27 | 28 ...... 36 | 37 ...... 45
内容:          <-- 19 个 0 -->   |   chunk 0    |   chunk 1    |   chunk 2
                                      (9 帧)         (9 帧)         (9 帧)
```

`choice_idx` 在 {0, 1, 2} 上均匀采样。这个 clip 能产出的三种样本:

| choice_idx | start | history = continue[start : start+19]   | target  |
|---|---|---|---|
| 0 | 0  | **19 个 0**(无过去)                    | chunk 0 |
| 1 | 9  | 10 个 0 + chunk 0(9)                    | chunk 1 |
| 2 | 18 | 1 个 0 + chunk 0(9)+ chunk 1(9)        | chunk 2 |

所以 3-chunk clip 能训练 **"生成开始"**(k=0)、**"早期生成"**(k=1)、**"有一点历史"**(k=2)。它*永远*产不出
"19 帧全真实历史"(那需要 k≥3,即 ≥4 chunk)。这是合理且有意为之的 —— 这些部分历史样本教模型冷启动行为。跨 epoch 时
随机 `choice_idx`(种子 = `base_seed + epoch*1e6 + idx`,dataloader:169)会把三种都轮一遍。

## 6. 完整举例 B —— **261 帧 clip(N = 7 chunk)**

`vae_latent` 形状 `(7, 16, 9, h, w)`。时间轴 = 63 个真实 latent 帧 + 前面补 19 个 0:

```
continue: [19个0] | c0(19..27) | c1(28..36) | c2(37..45) | c3(46..54) | c4(55..63) | c5(64..72) | c6(73..81)
```

`choice_idx` 在 {0..6} 上均匀采样。举几个:

| choice_idx | start | history(target 前 19 帧)              | target  | 历史类型 |
|---|---|---|---|---|
| 0 | 0  | 19 个 0                                | chunk 0 | 全 0(冷启动) |
| 2 | 18 | 1 个 0 + c0 + c1                       | chunk 2 | 部分 |
| 3 | 27 | c0 尾 1 帧 + c1 + c2(19 帧全真实)      | chunk 3 | **全真实** |
| 6 | 54 | c3 尾 1 帧 + c4 + c5(19 帧全真实)      | chunk 6 | **全真实、深入片段** |

所以 7-chunk clip 的样本丰富得多:既有冷启动(k=0,1,2 部分历史),又有大量"深入视频、19 帧全真实记忆"的样本
(k=3..6)。这就是长 clip 更值钱的原因 —— 同一个文件能采出更多、且更多"完整条件"的 target。每个 epoch 对每个 clip
采不同的 `choice_idx`,训练久了所有 chunk 都会被轮到。

## 7. 为什么是 `(28-1)*4+1` 而不是 `28*4`?(因果 VAE)

Wan VAE 是**因果(causal)时间 VAE**,时间方向的压缩是**非对称**的:
- **第 1 个** latent 帧只编码 **1** 个 RGB 帧(起始关键帧,1:1 映射);
- **之后每 1 个** latent 帧编码 **4** 个 RGB 帧。

所以 `L` 个 latent 帧 ↔ RGB = `1 + (L-1)*4 = (L-1)*4 + 1`。那个 `-1` 就是把唯一一个 1:1 的起始帧单独拎出来,剩下
`L-1` 帧才是 4:1。验证:1 个 chunk = 9 latent → `(9-1)*4+1 = 33` ✓;28 latent → `(28-1)*4+1 = 109` ✓。
**不能**写成 `28*4=112` —— 那等于假设所有 latent 帧都是 4:1,忽略了首帧。

## 8. "19 个历史 latent" vs "short 流是 2 帧" —— 别混淆

`history_sizes = [16, 2, 1]` 之和 = **19**,这是 dataloader 交出的连续历史 latent 帧数,**始终是 19**。
"short 是 2 帧" 说的是**另一件事** —— 喂给 `patch_short` Conv3d 的那一路**输入流**有 2 帧,因为里面额外塞了一个
**x0 锚帧**,而 **x0 不计入那 19 帧**。看 `utils_helios_base.py:647-659`:

```python
indices = arange(0, sum([1, 16, 2, 1, 9]))          # = arange(0, 29):  x0(1) + 16 + 2 + 1 + 9
indices_prefix, long, mid, idx_1x, hidden = split([1, 16, 2, 1, 9])
indices_latents_history_short = cat([indices_prefix, idx_1x])   # x0 + 最新 1 帧 = 2 帧
latents_history_long, latents_history_mid, latents_history_1x = history_latents.split([16, 2, 1])  # 历史仍是 19
```

- 历史 19 帧按"新旧"拆成 **long=16 / mid=2 / 1x=1**(最旧 16、中间 2、最新 1)。
- `patch_short` 真正吃的是 **x0(1 帧) + 最新的 1x(1 帧) = 2 帧**。这个 "2" 是 **1 个历史帧 + 1 个 x0 锚帧**,
  不是 "short 取了 2 个历史帧"。
- 总账:`x0(1) + 历史 19 + target 9 = 29` 个 latent 槽位;但真正属于这段窗口的**新** latent 帧仍是 `19+9=28`
  (→ 109 RGB)。**x0 是把 clip 自己的首帧"复用"过来当锚,不重复计入 28。**

所以 `16+2+1+9 = 28`(✓)与"short 两帧"(`x0 + 1x`)二者并不矛盾。

### 8.1 x0 锚帧:为什么要把首帧塞进 short 流(attention sink / Easy Anti-Drifting)

自回归逐 chunk 生成时,误差会沿时间累积导致**漂移(drifting)**:越往后画面越偏离开头、越糊。Helios 的对策是把 clip 的
**第一帧 latent(x0)** 永远固定地放在 short 流的**位置 0**,当成一个不动的 **attention sink / 锚点**:无论生成到第几个
chunk,模型注意力里总有一个"原点参照",把后续帧拉回到与开头一致的全局色调/主体/构图上。

- `is_keep_x0=True` 时 x0 由 dataloader 单独取出(clip 真正的首个 latent 帧)。
- **冷启动特例:** `choice_idx==0`(target 就是第 0 个 chunk、还没有任何过去)时,x0 被**置 0** —— 此刻"生成尚未开始",
  不存在真实锚帧,给 0 让模型学会从零起步。其余 `choice_idx>0` 时 x0 是真实首帧。
- `is_random_drop` 的 t2v 分支(`utils_helios_base.py:662-671`)会把 x0 连同整段历史一起清零,模拟纯文生视频(无图无视频条件)。

## 9. 历史是怎么被"压缩"的:多档 patchify 的 token 账(以 368×640 为例)

第 2 节说的 19、9 都是 **latent 帧数**;真正进 attention 的是 patchify 之后的 **token**。三档历史各用一个独立 Conv3d
(`transformer_helios.py:1066-1068`),stride 决定压缩率;target 用主 `patch_embedding`(stride `(1,2,2)`)。
368×640 → latent 空间 **46×80**(mid/long 的空间会被 `pad_for_3d_conv` 把 46→48 再除):

| 流 | 输入 latent 帧 | Conv3d stride (t,h,w) | token = ⌈t/st⌉·⌈h/sh⌉·⌈w/sw⌉ | 说明 |
|---|---|---|---|---|
| **long(远期)** | 16 | (4, 8, 8) | (16/4)·(48/8)·(80/8) = 4·6·10 = **240** | 压得最狠:16 帧→240 token |
| **mid(中期)** | 2 | (2, 4, 4) | (2/2)·(48/4)·(80/4) = 1·12·20 = **240** | 中等 |
| **short(近期)** | 2(x0+最新) | (1, 2, 2) | (2/1)·(46/2)·(80/2) = 2·23·40 = **1840** | 最细,几乎不压时间 |
| **target(待生成)** | 9 | (1, 2, 2) | 9·23·40 = **8280** | 主 patch_embedding |

直观对比:**最旧的 16 帧**只占 240 token,而**最近的 2 帧**占 1840 token —— 单帧信息密度相差约 **61×**。这正是
"Multi-Term Memory" 的核心:**远的记粗、近的记细**,在固定预算下既保留长程上下文、又保住近期细节。dataloader 只负责交出
连续的 19 帧(+ x0);拆 [16,2,1]、三档 patchify 全在模型里完成。

## 10. 完整举例 C —— **161 帧 clip(N = 4 chunk),含 x0 与三档压缩的全流程**

`floor(161/33) = 4` → `vae_latent` 形状 `(4, 16, 9, 46, 80)`。拉平 = 36 个真实 latent 帧,前面补 19 个 0 →
`continue` 长度 `19+36 = 55`:

```
continue 下标: 0 ........ 18 | 19..27 | 28..36 | 37..45 | 46..54
内容:          <- 19 个 0 -> | chunk0 | chunk1 | chunk2 | chunk3
                               (9 帧)   (9 帧)   (9 帧)   (9 帧)
x0 = chunk0 的首个 latent 帧 = continue 下标 19
```

`choice_idx` 在 {0,1,2,3} 上均匀采样,4 种样本:

| choice_idx | start | history = continue[start:start+19] | target | x0 | 历史类型 |
|---|---|---|---|---|---|
| 0 | 0  | 19 个 0                              | chunk0 | **置 0**(冷启动无锚) | 全 0 |
| 1 | 9  | 10 个 0 + chunk0(9)                 | chunk1 | 真实首帧 | 部分 |
| 2 | 18 | 1 个 0 + chunk0(9) + chunk1(9)      | chunk2 | 真实首帧 | 部分 |
| 3 | 27 | chunk0 尾 1 帧 + chunk1 + chunk2     | chunk3 | 真实首帧 | **19 帧全真实** |

**以 `choice_idx=3` 为例,走完整条流水线**(target = chunk3,history 19 帧全真实):

1. **dataloader 交出**(`prepare_stage1_latent`):history = `continue[27:46]` = `[chunk0[8], chunk1(28..36), chunk2(37..45)]`
   共 19 帧;target = `continue[46:55]` = chunk3(9 帧);x0 = chunk0 的首帧(真实,非 0)。
2. **拆三档**(`prepare_stage1_clean_input_from_latents`,`split([16,2,1])`):
   - long = 这 19 帧里**最旧的 16 帧** → patch_long(4,8,8) → **240 token**
   - mid = 接下来 **2 帧** → patch_mid(2,4,4) → **240 token**
   - 1x = **最新 1 帧** → 与 x0 拼成 short 流(2 帧)→ patch_short(1,2,2) → **1840 token**
3. **target** chunk3 的 9 帧 → 主 patch_embedding → **8280 token**,加噪后由模型预测 flow。
4. **注意力序列** ≈ `[short 1840] + [mid 240] + [long 240] + [target 8280]`(各自带 RoPE 位置);x0 固定在 short 流位置 0
   当 attention sink,把 chunk3 的生成锚回开头。
5. **flow-matching 损失**只在 target 的 8280 token 上算(history/x0 是干净条件,不加噪、不回传到"预测"目标)。

> 对照:`choice_idx=0` 时 history 全 0、x0 也置 0 → 这一条流水线退化成"纯冷启动文生视频第一帧",教模型**从零开 chunk**。
> 4-chunk clip 一次性覆盖了 冷启动(k=0)、两种部分历史(k=1,2)、和首个"19 帧全真实记忆"(k=3),比 3-chunk 更全。

## 11. 一句话总结

一个 clip 被编码成 `N = floor(帧数/33)` 个 9-latent chunk。每步训练随机挑**一个** chunk 当 9 帧 target,并取它前面
**19** 个 latent 帧(开头补零)当 history,再复用 clip 首帧 x0 当锚 → 一个**固定 28-latent(≈109-RGB)窗口**
(因果 VAE:`(28-1)*4+1`)。19 帧历史按 [16,2,1] 拆成 long/mid/short 三档,远期压到 240 token、近期保留 1840 token
(远粗近细);x0 固定在 short 流位置 0 解决 attention-sink/漂移。≥121 帧规则保证 `N ≥ 3` chunk,这是 rollout/DMD/GAN
路径的硬约束(`num_rollout_sections=3`),也给随机 target 采样器提供了有用的历史状态分布。更长的 clip(如 261→7 chunk)
只是产出更多、且更多"完整条件"的样本。
