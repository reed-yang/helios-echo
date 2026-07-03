# Stage-1 前向、注意力结构与完整训练 pipeline(公式 + 伪代码)

本文承接 [`STAGE1_DATALOADER_FRAMES_zh.md`](./STAGE1_DATALOADER_FRAMES_zh.md)(讲 chunk 几何、19+9、x0、多档压缩),
进一步讲清楚 **token 进入 transformer 之后的注意力是怎么连的**,以及**从离线编码到 flow-matching loss 的完整 pipeline**。
全程用同一个例子串联:**161 帧 clip → 4 chunk,368×640,`choice_idx=3`,batch=1**。

所有 `file:line` 均指向本仓库(`helios/modules/transformer_helios.py`、`helios/utils/utils_helios_base.py`、
`helios/dataset/dataloader_history_latents_dist.py`)。

---

## 1. 常量与记号

```
latent_window_size W = 9          # 一个 chunk 的 latent 帧
history_sizes = [16, 2, 1]        # long / mid / 1x,  Σ = 19
VAE: 时间 4× / 空间 8× (causal)    # RGB = (L-1)*4 + 1
分辨率 368×640 → latent 46×80, C=16
σ ∈ (0,1) 流匹配噪声水平;  ε ~ N(0, I);  D = transformer 隐藏维
```

---

## 2. 注意力结构:`restrict_self_attn`(非对称"历史只读")

核心在 `HeliosAttnProcessor2_0`(`transformer_helios.py:201-444`)。进入每个 block 的 token 序列分成**两组**
(不是三个对等 chunk):

| 组 | 内容 | token 数(368×640 例) |
|---|---|---|
| **history** | long16 + mid2 + short(x0+1) **三档 patchify 后拼成一条**历史序列 | 240+240+1840 = **2320** |
| **target** | 待去噪的那个 chunk(9 latent 帧) | **8280** |

long/mid/short 三档 patchify 后**直接 concat 成一条 history 序列**(代码里只剩 `history_seq_len` 一个整体),靠各自的
RoPE 位置编码区分远近。每个 block 把 q/k/v 切成 `history` 段和 `target` 段(`:302-304`),做**两次独立的 attention**:

```
(a) 历史自注意力 (:365-379)
    h_out = Attn(q_history, k_history, v_history)
    → 历史 token 只在【历史内部】互相 attend,看不到 target
    → 与当前噪声/timestep 无关 ⇒ 首步算完即缓存 (:383-388),后续去噪步复用

(b) target 注意力 (:372-410)
    k' = cat(k_history, k_target)        # :372
    v' = cat(v_history, v_target)        # :373
    t_out = Attn(q_target, k', v')       # :410  —— 无 causal mask
    → target 既【内部 9 帧全双向 3D】,又能读到【全部历史】
```

非对称连接图:

```
            ┌──────── history (long|mid|short) ────────┐   ┌──── target chunk ────┐
history Q:  └─→ 只看 history(双向)                      ┘   (看不到 target)
target  Q:  ──→ 看 history ────────────────────────────────→ 且内部全双向 ←──────
```

**所以"3 个 chunk 一起做 bidirectional self-attention 吗?"的准确答案:**

- **target 那 9 帧(8280 token)之间:完整双向 3D self-attention**(空间+时间全连,**没有**逐帧 causal mask;
  `attn_varlen_func` 是普通全注意力,无 `is_causal`)。
- **target → history:能读**(单向,把 history 当额外 KV)。
- **history → target:读不到**(历史是只读的干净条件)。
- **history 三档之间:也双向**,但自成一体。

即**不是"三块平等全双向"**,而是"**历史是自包含双向块、target 双向且额外读历史、历史对 target 失明**"。

### 2.1 为什么这样设计(不是随意的)

"history 看不到 target"是**整个实时自回归的命门**:

1. **history 的 K/V 与噪声/timestep 无关** → 4/20/50 步去噪里**第一步算完 history KV 就缓存,后续步直接复用**
   (`self.kv_cache`,`:383-388`;`use_cache` 分支 `:329-331`)。这是 19.5 FPS 的来源。
2. **自回归在 chunk 级、不在 token 级**:逐 chunk 生成时,上一 chunk 解码后变成下一 chunk 的 history KV;chunk
   **内部**则一次性全双向 diffusion 去噪(画质优于 token 级 causal)。
3. `is_amplify_history`(`:394-408`)可给 history 的 key 乘 `scale_key`,**放大/抑制历史影响力**(防漂移旋钮)。
4. `restrict_lora`(`:306-309`)给 history 的 q/k/v 单独挂 LoRA,使"历史路"与"生成路"学不同投影。

每个 block 里还有一路**对文本的 cross-attention**(`enable_cross`,prompt 的 UMT5 embedding)—— 与上面的时空
self-attention 是两件事。

---

## 3. 完整 pipeline(公式 + 伪代码),逐步走例子

### 3.0 离线编码(一次性,`get_short-latents.py`)

```
clip(161 RGB帧, 368×640)
  N = floor(161/33) = 4
  vae_latent = VAE.encode → (N=4, C=16, 9, 46, 80)      # 4 个 chunk
  存盘:  {uttid}_161_368_640.pt
```

### 3.1 dataloader 取一个样本(`prepare_stage1_latent`, dataloader:136-196)

```
flat = rearrange(vae_latent, "n c t h w -> c (n t) h w")       # 时间轴 36 latent 帧
continue = cat([ zeros(C,19,46,80), flat ], dim=1)             # 前补 19 个 0 → 长 55
choice_idx = 3                       # 本例取 chunk3
s = choice_idx * 9 = 27
history = continue[:, 27:46]         # (16,2,1) 全真实:chunk0[8] ++ chunk1 ++ chunk2 → (C,19,46,80)
target  = continue[:, 46:55]         # = chunk3                                        → (C, 9,46,80)
x0      = flat[:, 0:1]               # clip 首帧(真实;choice_idx≠0 故不置 0)           → (C, 1,46,80)
```
> 若 `choice_idx==0`:`history` 全 0 且 `x0←0`(冷启动)。

### 3.2 拆三档 + x0 锚(`prepare_stage1_clean_input_from_latents`:619-659)

```
long, mid, h1x = history.split([16,2,1], dim=1)
short_stream   = cat([x0, h1x], dim=1)        # x0 + 最新1帧 = 2 帧   ← “short 是 2 帧”
```

### 3.3 加噪 —— 只加在 target 上(`prepare_stage1_noise_input`:734-856)

```
u  ~ density(weighting_scheme);  idx = floor(u·1000)
σ  = sigmas[idx];   timestep = σ·1000                  # 标量(每样本一个)  :763,:787
ε  ~ randn_like(target)
# (可选) corrupt_history: long/mid/short 各自加轻噪/降采样增广 (:816)  ← history,不是 target
noisy_target = (1-σ)·target + σ·ε                      # (C,9,46,80)   :855
v_target     = ε - target                              # 流匹配速度标签  :856
```
**关键:history(long/mid/short)与 x0 保持干净**,只有 target 被加噪。

### 3.4 patchify → token(三个 Conv3d + 主 patch_embedding,transformer:1066-1068)

mid/long 空间 46 先 `pad_for_3d_conv` → 48,再除 stride:

```
short = patch_short(short_stream) stride(1,2,2) → ( 2,23,40) = 1840 token  (D 维)
mid   = patch_mid (mid)           stride(2,4,4) → ( 1,12,20) =  240 token   (46→48)
long  = patch_long(long)          stride(4,8,8) → ( 4, 6,10) =  240 token   (46→48)
hist  = cat([long, mid, short])   → history_seq_len = 2320 token
tgt   = patch_embedding(noisy_target) stride(1,2,2) → (9,23,40) = 8280 token (= query)
seq   = cat([hist, tgt])          → 10600 token   (+ 各自 RoPE 位置)
```

### 3.5 transformer L 层(`restrict_self_attn`,见 §2)

```
for block in blocks:                    # L 层
    q,k,v = proj(seq);  RoPE(q,k)
    q_h,q_t = split(q,[2320,8280]); 同理 k_h/k_t, v_h/v_t

    h_out = Attn(q_h, k_h, v_h)                       # (a) 历史自注意, 首步缓存
    k' = cat([k_h,k_t]); v' = cat([v_h,v_t])          # (b) target 读历史+自注意
    (可选 amplify: k_h *= scale_key)                  # :394-408
    t_out = Attn(q_t, k', v')                         # 无 causal

    seq = cat([h_out, t_out])
    seq = seq + CrossAttn(seq, text_emb)              # prompt(UMT5)
    seq = seq + FFN(seq)
```

### 3.6 输出 + 损失(`proj_out` → unpatchify → `_flow_loss`:20-91)

```
v_pred = unpatchify(proj_out(t_out))                  # 仅 8280 个 target token → (C,9,46,80)
loss   = mean_over_target( w(σ) · (v_pred - v_target)² )      # :86-88,  v_target = ε - target
# history / x0 不进 loss(干净条件,不作预测目标)
```

---

## 4. 形状流总览(本例)

```
(4,16,9,46,80)         vae_latent
   │ flatten+pad19 + choice_idx=3
   ├── history (16,2,1)=19帧 ──split──> long16  mid2  1x1
   │                                      │     │    └┐
   │                                      │     │   x0(1) ┘→ short_stream(2)
   ├── x0(1帧)  ─────────────────────────┘     │
   └── target  chunk3 (9帧) ──(1-σ)x+σε──> noisy_target,  v_target=ε-x
                                            │
   patchify:   long→240   mid→240   short→1840  │  target→8280
                └──── hist 2320 ────┘            │
                          cat ──────────────> seq 10600
                          │  × L blocks (history自注意[缓存] ‖ target读历史+自注意 ‖ cross-attn文本 ‖ FFN)
                          ▼
                  v_pred(8280) → unpatchify (16,9,46,80)
                  loss = w(σ)·‖v_pred − (ε−x_target)‖²
```

## 5. 一句话总结

一个 chunk 当 target、前 19 帧当 history、首帧当 x0 锚;只给 target 加流匹配噪声 `(1-σ)x+σε`、标签是速度 `ε-x`;
history 三档分别压成 240/240/1840 token 当**干净且可缓存**的只读条件;每个 block 里 **history 自包含双向、target 内部
全双向并额外读历史、history 对 target 失明**,再叠文本 cross-attention 与 FFN;最后只在 target 上回归速度。
推理时换成逐 chunk:上一 chunk 输出 → 下一 chunk 的 history KV,σ 沿 `{…}` 多步去噪,history KV 缓存复用 → 流式实时。
