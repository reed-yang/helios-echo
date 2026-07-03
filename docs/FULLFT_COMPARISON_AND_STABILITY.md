# fine-tune 配方对比（LoRA / 全参数 / 混合 SFT）与稳定性指南

本文对比在 **cfr_int / captions_30b_32f** 语料（~154k–177k clips，均从 `Helios-Base/transformer_init` 续训、共享同一份 latents）上做的三种配方，
并给出下次训练如何更稳的清单。配套文档：`docs/FULLSFT_ANALYSIS.md`（全参数 FT 的深度失败分析）。

三个配方是同一实验的三条并行/迭代分支：

- **LoRA r128（基线）**：`stage1_lora_cfr.yaml` —— 最保守、最稳，画质基本不掉，作为对照组。
- **Round-1 全参数 FT**：`stage1_fullft_cfr.yaml` —— 最激进，`is_full_finetune: true` 解冻全部 14.31B，**失败**。
- **Round-2 混合 SFT**：`stage1_sft_cfr_v2.yaml` —— 针对 Round-1 失败做的"收敛版"重训，介于两者之间。

> 术语澄清：严格意义上只有 **Round-1 是"全参数" fine-tune**。**Round-2 是"混合 SFT"**——用 LoRA(r256) + 全量训练
> patch_embedding / 多尺度记忆 patch，但**冻结 norm/AdaLN 与 base 线性权重**；**LoRA r128** 则连 patch_embedding 也走 LoRA。

第四条分支见文末 **Round-3**（full base-linear + LoRA patch_embedding），是吸收前三轮教训后的新配方。

---

## 0. 先搞清楚：DiT 里各部分是什么、"base 线性权重 / patch_embedding" 指哪

下文反复出现"base 线性权重""patch_embedding""norm/AdaLN"这些词，它们是 Helios DiT 的不同部件。
数据流骨架如下（全部在 `helios/modules/transformer_helios.py`）：

```
噪声 latent [B,16,T,H,W]
      │
      ▼
① patch_embedding  (Conv3d, line 1023)   ← "入口分词器"，把 latent 切成 token
      │            + patch_short/mid/long (Conv3d, 1066-1068) 把 history 也切成 token
      ▼
② ×N 个 TransformerBlock (line 728)  ← DiT 的"主体/骨干"
      │   ├─ norm1 (LayerNorm) + scale_shift_table  ← AdaLN 调制
      │   ├─ attn1 self-attn:  to_q/to_k/to_v/to_out  ← 全是 nn.Linear (488-493)
      │   ├─ norm2 + attn2 cross-attn (看文本)        ← 又一组 Linear
      │   └─ norm3 + ffn (FeedForward = 2 个 Linear)  ← Linear (768)
      ▼
③ time_embedder / proj_out                ← 时间嵌入 + 输出投影
      ▼
   预测的 velocity / noise
```

**"我们理解的 DiT" ≈ base 线性权重吗？** 大体是，但不完全等同：

- 整个模型 `HeliosTransformer3DModel` **才是** DiT。文档里的 **"base 线性权重"** 特指 ② 里那些 `nn.Linear`
  ——attention 的 `to_q/to_k/to_v/to_out` + FFN 的两个 Linear + 各种 projection。它们是 DiT 的**计算骨干**，
  占 14.3B 参数的**绝大多数**。所以"base 线性权重 ≈ DiT 主体"这个直觉是对的。
- **LoRA 正是只在这些 Linear 上加低秩增量**（`lora_layers: "all-linear"`），base 权重本身冻结——这就是 LoRA 稳的原因。
- 但 DiT 里还有**不是 Linear** 的部分，它们**不算**在"base 线性权重"里：
  - **patch_embedding**（Conv3d，①）——卷积，不是 Linear；
  - **norm / AdaLN**（`scale_shift_table` 参数 + LayerNorm/RMSNorm，line 160/739/771）——归一化与调制，不是 Linear；
  - time_embedder 的一部分。

  → **全参数 FT 崩的关键**：它连 **norm/AdaLN + base 全秩权重**一起动了；LoRA 只给 Linear 加低秩 delta、norm/AdaLN 全冻结。这就是第 2/3 节里"危险层"的由来。

**patch_embedding 在哪、和 patch_short/mid/long 什么区别？**

- `patch_embedding`（`transformer_helios.py:1023`，`nn.Conv3d`）= DiT 的**最前端输入分词器**，把噪声 latent
  `[B,16,T,H,W]` 用 3D 卷积切块投影成 token 序列（类比 ViT 的 patch embed）。它是 Conv3d、不是 Linear，所以三种配方要用
  `is_train_full/lora_patch_embedding` 这套 flag 单独控制：LoRA r128 走 LoRA，全参 FT / 混合 SFT 都是全量训练。
- ⚠️ 别和 **`patch_short/mid/long`（line 1066-1068）** 搞混——那是**另外三个 Conv3d**，专门把 **history（多尺度记忆）**
  切成 token 的，不是给噪声输入用的。三种配方对这组**都全量训练**，所以它不是画质差异的来源。

**一句话**：`patch_embedding` = 噪声输入的 Conv3d 入口；"base 线性权重" = 中间 TransformerBlock 里的 Linear（DiT 主体）；
norm/AdaLN 是第三类、既不是 patch 也不是 Linear，却是全参 FT 崩掉的主因之一。

---

## 1. 结论速览

| | **LoRA r128（基线）** | **Round-1：全参数 FT** | **Round-2：混合 SFT** |
|---|---|---|---|
| config | `stage1_lora_cfr.yaml` | `stage1_fullft_cfr.yaml` | `stage1_sft_cfr_v2.yaml` |
| launcher | `train_stage1_lora_cfr.sbatch` | `train_stage1_fullft_cfr.sbatch` | `train_stage1_sft_v2.sbatch` |
| 结果 | ✅ **稳**：跑到 step 9000（DDP，画质保住） | ❌ **失败**：~step 2400 明显形变+抖动，loss 平坦 ~0.072，手动停 | ✅ **跑完** 6000 步（`checkpoint-6000-final`） |
| 运行方式 | DDP（无 DeepSpeed），16 GPU | DeepSpeed ZeRO-2，32 GPU | DeepSpeed ZeRO-2，32 GPU |

**一句话**：LoRA r128 只在冻结的 base 上加低秩增量，天然稳但改动上限低；Round-1 用「constant 1e-5 × 全部 14B ×
batch 32 × 无 EMA × 强 history 腐蚀」把预训练时空先验推飘了；Round-2 把这五个旋钮**全部往 LoRA 那一侧调**
（更低 lr、更大 batch、开 EMA、缩小可训练范围、减弱增广），于是既稳定跑完、又比纯 LoRA 多动了一点结构层。

---

## 2. 逐项差异

### 2.1 可训练范围（最关键）

| 模块 | LoRA r128 | Round-1 全参数 FT | Round-2 混合 SFT |
|---|---|---|---|
| base 线性权重（attn/FFN/proj） | ⛔ 冻结，加 **LoRA r128 增量** | ✅ **全秩训练** | ⛔ 冻结，加 **LoRA r256 增量** |
| **norm / AdaLN**（timestep/condition 调制） | ⛔ **冻结** | ✅ **训练** | ⛔ **冻结** |
| time-embedding base | ⛔ 冻结（仅 LoRA Linear 动） | ✅ 训练 | ⛔ 冻结（仅 LoRA Linear 动） |
| `patch_embedding`（输入 Conv3d） | 🟡 **LoRA**（`is_train_lora_patch_embedding: true`） | ✅ 全量训练 | ✅ **全量**（`is_train_full_patch_embedding: true`） |
| `patch_short/mid/long`（多尺度记忆） | ✅ 全量训练 | ✅ 训练 | ✅ 全量训练 |
| 可训练参数量 | 数亿级 LoRA 增量 | **14.31 B**（ZeRO-2 每卡分片 447M） | **≈ 39.7 M** |

> `FULLSFT_ANALYSIS.md` 的核心定位：真正把画质推坏的是**移动了全秩 base 权重 + norm/AdaLN**——这两者只有全参数 FT 在动。
> 三个配方都全量训多尺度记忆 patch，它并不是罪魁；差别在于对 base 线性权重是"低秩增量"(LoRA)还是"全秩移动"(全参)，以及 norm/AdaLN 是否解冻。
> Round-2 相对 LoRA 唯一多动的是把 `patch_embedding` 从 LoRA 升级为全量训练。

### 2.2 优化 / 稳定性超参

| 超参 | LoRA r128 | Round-1 全参 | Round-2 SFT | 说明 |
|---|---|---|---|---|
| learning_rate | **3e-5** const_w_warmup | **1e-5** constant | **2e-6** const_w_warmup | 全秩 14B 上 1e-5 远比 LoRA 的 3e-5 猛；Round-2 一路降到 2e-6 |
| warmup | 500 | 200 | 500 | |
| **EMA** | ❌ off | ❌ off | ✅ **on**（0.999, start250） | 全参/SFT 画质高度依赖 EMA 平滑；LoRA 因 base 冻结可不开 |
| **global batch** | **32**（bs2×16GPU） | **32**（bs1×32GPU） | **128**（bs2×32GPU×accum2） | 大 batch 抹平梯度尖刺（Round-1 有到 4.95 的 spike） |
| lr_scheduler | constant_with_warmup | **constant** | constant_with_warmup | Round-1 无 warmup 兜底更易尖刺 |
| max_train_steps | 11000（跑到 9000） | 4500（~2400 停） | 6000（跑完） | |
| random_drop v2v/t2v | 0.4 / 0.4 | 0.4 / 0.4 | **0.25 / 0.25** | Round-2 减弱 Easy-Anti-Drifting 增广 |
| corrupt_history prob | 0.9 | 0.9 | **0.6** | Round-2 减弱 history 腐蚀，别在全参移动时过度压时序 |
| max_grad_norm | 1.0 | 1.0 | 1.0 | 三者都裁剪 |
| seed / 数据 / 分辨率 | 44 / 共享 latents / 384×640 | 同左 | 同左（apples-to-apples） | |

> 注意 LoRA r128 的增广（drop 0.4 / corrupt 0.9）和 Round-1 一样激进，却没出问题——说明**强增广本身不是致命项，
> 只有在"全秩移动 base + norm/AdaLN"时才被放大成形变/抖动**。这也印证根因在可训练范围而非增广。

### 2.3 保存 / 工程差异

- LoRA r128：纯 **DDP**（无 DeepSpeed），只存 LoRA adapter + 全量的 memory patch（extra components）；ckpt world-size 锁 16。工程上最简单，不受 ZeRO-2 那些坑影响。
- Round-1：DeepSpeed **ZeRO-2** 分片存 model+optim；ckpt world-size 锁 32；产出 consolidated `transformer/`（28.6GB）走**训练后最终 save + 离线 `zero_to_fp32.py`**（save hook 内不做重活）。
- Round-2：全量训练的 `patch_embedding` / memory patch 存进 `transformer_partial.pth`（`utils_base.save/load_extra_components`），LoRA 走标准 adapter 保存。
- 两者都设 `skip_dataloader_dcp: true`：本集群 `dcp.save` 的 `gather_object` NCCL collective 在**多机**会崩（Error 2）；跳过后 resume 只丢 dataloader 迭代位置（对 SGD 无所谓），model+optim 仍由 `accelerator.save_state` 完整存取。

---

## 3. Round-1 为什么失败（根因）

摘自 `FULLSFT_ANALYSIS.md`，五个因素叠加导致 **灾难性漂移/遗忘**，而非 bug：

1. **lr 1e-5 作用在全部 14B**（含 patchify、多尺度记忆、norm/AdaLN、time-embed）——全参续训一般用 **1e-6 ~ 5e-6 + 衰减**；
2. **无 EMA**——直接采样原始权重，抖动被放大；
3. **batch 只有 32**——梯度方差大，出现真实梯度尖刺（max 4.95）；
4. **constant LR 无衰减**——全程满强度推；
5. **强 history 腐蚀 + 离分布的短 caption**——被迫拟合新格式，进一步压时序一致性。

现象签名：**loss 全程平坦(~0.072) 但样本明显变差** ⇒ 权重被推动却没有可测收益 = 纯漂移/遗忘。
→ 时序动态漂移=抖动；空间/结构先验漂移=形变。

---

## 4. 下次训练稳定性清单（按优先级）

优先级从高到低——前 3 条基本能消掉大部分漂移，且 Round-2 已验证有效：

1. **压低 lr**：全参 FT 用 `1e-6 ~ 3e-6`，**并加 cosine/linear 衰减**（Round-2 用了 2e-6 但仍是 constant——可进一步上衰减）。这是单条最有效的。
2. **开 EMA 并从 EMA 权重采样/评测**：`use_ema: true`, `ema_decay 0.999~0.9999`。全参画质对此高度敏感（Distilled 本身就用 EMA）。
   注意 Round-2 `use_ema_validation: false`——若做内建 validation 想看真实质量，应打开或在外部 eval 时加载 EMA 权重。
3. **放大有效 batch 到 ~128–512**：靠 `gradient_accumulation_steps`。Round-2 的 128 已明显压住了尖刺；纯全参建议更大。
4. **缩小可训练范围到"LoRA 会冻结的那些层之外别乱动"**：
   - **冻结 norm / AdaLN**（`train_norm_layers: false` 且不要 `requires_grad_(True)` 全解冻）——这是最伤画质的一类层；
   - 谨慎对待 **time-embed base**（Round-2 冻结）；
   - 多尺度记忆 patch（`patch_short/mid/long`）可以训——两轮都训且安全；
   - 如果全秩 base 权重的移动仍太猛，**直接回退到 LoRA/混合 SFT 配方**（Round-2 已证明能保住画质），全参在这里可能并无必要。
5. **减弱增广**：`random_drop_* ≈ 0.25`、`corrupt_mode_prob_history ≈ 0.6`（Round-2 值），别在全参移动时过度压时序。
6. **caption 分布**：考虑把过于简短/离分布的 `<header>/<event>/<role>` caption 改写得更接近预训练风格，减少全参必须吸收的分布偏移。

### 工程/集群稳定性（别踩的坑）

- **`skip_dataloader_dcp: true` 必开**（多机 `dcp.save` NCCL 崩溃）。
- **ZeRO-2 epoch-rollover 死锁**：全参 DeepSpeed 在 **epoch-1→2 交界后的第一个 checkpoint** 会挂（torchdata StatefulDataLoader 的 set_epoch/reshuffle 交互，未解决）——所以 Round-1 才 cap 在 <1 epoch。要跑多 epoch 需先修此问题，或改用 Round-2 的 LoRA/DDP 路径（不受影响）。
- **ZeRO-2 None-grad 崩溃**：t2v microbatch（`is_random_drop`）下 history/记忆参数拿不到梯度 → 在 `_flow_loss` 里加 `loss += 0.0*Σ p.sum()` 的 zero touch term（仅 `is_full_finetune` 门控）。
- **DeepSpeed WarmupLR 的 lr**：`warmup_max_lr: "auto"` **不会**接住 client-optimizer 的 lr（会退化成默认 0.001 bug）——必须在 `zero2_sched*.json` 里**硬编码**与 config 一致的 lr。
- 每次 crash-resume 会开新的 wandb run（旧的显示 stopped，即"断了"），属正常。
- 别在同一 `output_dir` 上跑两个 Claude/训练 session——会互相覆盖 checkpoint 造成假 crash。

---

## 5. 相关文件索引

- config：`scripts/training/configs/stage1_lora_cfr.yaml`、`stage1_fullft_cfr.yaml`、`stage1_sft_cfr_v2.yaml`
- launcher：`scripts/training/train_stage1_lora_cfr.sbatch`、`train_stage1_fullft_cfr.sbatch`、`train_stage1_sft_v2.sbatch`
- accelerate/ds：`scripts/accelerate_configs/multi_node_zero2_sched{,_2e6}.yaml` + `zero2_sched{,_2e6}.json`
- 代码：`is_full_finetune` 分支在 `train_helios.py`（解冻/跳过 add_adapter、touch term、save/load）；
  extra-components 存取在 `helios/utils/utils_base.py`；flow loss touch term 在 `helios/utils/utils_helios_base.py`
- 深度分析：`docs/FULLSFT_ANALYSIS.md`

---

## 6. Round-3：full base-linear + LoRA patch_embedding（新配方，2026-07-01 开跑）

吸收前三轮的结论后的折中配方：**base 线性骨干全量 fine-tune**（表达力最强的部分放开），但把两个最"伤画质"的部件保持保守——
**patch_embedding 只用 LoRA r128、norm/AdaLN 冻结、memory patch 全量**。等于"Round-1 的骨干放开 + Round-2 的结构层保守"。

- config `stage1_fullbase_cfr368.yaml`；launcher `train_stage1_fullbase_cfr368.sbatch`（**c-node05,06 = 16 GPU**，ZeRO-2）。
- **起点 = LoRA-17000（368×640）**，用 `tools/merge_lora_partial_for_helios.py` **选择性 merge**：把 base-linear 的 LoRA 增量 + memory
  patch 烘焙进 base，但 **patch_embedding 的 LoRA 单独抽出**为 `patch_embedding_lora.safetensors`，训练时作为**活的 adapter 续训**（保留 17000 步）。
- 数据 `latents_cfr_int_30b_368x640`（429,634 latent）；**lr 1e-6 + warmup 500**；**gb128**（16×bs1×accum8）；6000 步（≈1.8 epoch）；
  random_drop 0.4、corrupt_history 0.9（= LoRA r128 版）。

**新代码（全部新 flag 门控，默认=旧行为，不影响 Round-1/2/LoRA）**：`model_config.full_finetune_scope="base_linear_lora_patch"`
+ `patch_embedding_lora_init_path`。要点：`add_adapter`（inject_adapter_in_model）会**重新冻结所有非 adapter 参数**，所以顺序必须是
**先 add_adapter(patch_embedding) → 再 requires_grad_(True) → 再冻结 norm + patch_embedding-base**；否则 base-linear 会被静默冻回去。

**Round-3 三处踩坑教训**：
1. **合并 LoRA 到全参起点必须先 merge**（否则丢掉 17000 步）；要"续训原 patch_embedding LoRA 而非新建"就做**选择性 merge**（不 fuse patch_embedding）。
2. **`is_full_finetune + use_ema` 之前从未一起用过**（Round-1 EMA off）：EMA 的 zero3 保存/恢复助手 `create_ema_zero3_lora.py` 是**纯 LoRA 写死**的——
   对全参 base 会用空 `target_modules` 崩溃、且只存 LoRA 会丢掉全量 base EMA 权重。**本轮先 use_ema:false 规避**；要开 EMA 需为该 scope 补一套
   全量 EMA 存/取（base 全量 + patch_embedding adapter 分开存、可 resume、最终再 fuse 出推理权重）。
3. **16 GPU 显存 OK**：H200 是 **141GB**（非 80GB），ZeRO-2 峰值 ~87GB，宽裕；smoke 已验证 scope 正确（norm/patch_embedding-base 0 可训、
   base-linear ~14.29B 全量、patch_embedding LoRA 0.66M、memory 23.94M）、无 ZeRO-2 None-grad 崩、checkpoint 正常。

> 结果待评估：跑完后按第 4 节稳定性预期对照 grad_norm/loss + 出片检查是否规避了 Round-1 的形变/抖动。
