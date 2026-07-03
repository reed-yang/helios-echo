# Helios Stage-1：INIT 与 POST 的区别（两阶段 two-phase 训练）

> 本文基于本 repo **签入的参考配置** `scripts/training/configs/stage_1_init.yaml` 与
> `stage_1_post.yaml` 的逐字段 diff，而非凭记忆或上游 README 描述。所有 `file:line`
> 指向本仓库真实文件。日期：2026-07-02。

---

## 0. TL;DR

Stage-1 的 Base 模型是**两阶段**产出的：先 `init`（高 LR、从裸 Wan 快速适配），
再 `post`（低 LR、在 init 结果之上精修）。两个阶段**训练的参数集完全相同**，
真正的差异只有 **3 处**：

| 维度 | `stage_1_init` | `stage_1_post` |
|---|---|---|
| **transformer 起点** | `Wan-AI/Wan2.1-T2V-14B-Diffusers`（裸 Wan，subfolder=`transformer`） | `BestWishYsh/Helios-Base` **subfolder=`transformer_init`**（= init 的 LoRA 已 merge 进 Wan 的中间权重） |
| **learning_rate** | `5e-5` | `3e-5` |
| seed / 命名 | 43 / `ablation_stage_1_init` | 44 / `ablation_stage_1_post`（无实质影响） |

**其余全部相同**：LoRA r128 / α128 / `all-linear`、`train_batch_size:2`、
`gradient_accumulation:1`、`lr_scheduler:constant` + `lr_warmup_steps:500`、
`checkpointing_steps:500`、`corrupt_history:true` (`noise`, p=0.9)、`is_random_drop:true`、
`use_ema:false`，以及下面这组关键的可训练开关（**init 与 post 一模一样**）：

```yaml
is_train_full_patch_embedding: false
is_train_lora_patch_embedding: true          # patch_embedding 走 LoRA
has_multi_term_memory_patch: true
is_train_full_multi_term_memory_patchg: true # 三段记忆 patch 走【全量】训练（非 LoRA）
is_train_lora_multi_term_memory_patchg: false
train_norm_layers: false
```

> ⚠️ **勘误**：`docs/HELIOS_STAGE1_TRAINING.md §7` 曾附注「上游 `stage_1_init`=全量记忆
> patch，`stage_1_post`=LoRA 记忆 patch」。在本 repo **签入的两份 yaml 里这不成立**——
> 两者 `is_train_full_multi_term_memory_patchg` 都为 `true`、`is_train_lora_*` 都为 `false`。
> 以本文件的逐字段 diff 为准。

逐字段 diff（去掉注释/空行后）只有：

```
model_config.transformer_model_name_or_path: Wan2.1-T2V-14B-Diffusers  →  BestWishYsh/Helios-Base
model_config.subfolder:                      (transformer)             →  transformer_init
training_config.learning_rate:               5e-5                      →  3e-5
seed:                                        43                        →  44
```

---

## 1. 概念：为什么要分两阶段，而不是一个 LoRA 从头训到底

整条 Stage-1 流水线是「**训练 → merge → 再训练 → merge**」：

```
        ┌─────────── INIT (lr 5e-5, ~5500 步) ───────────┐
裸 Wan2.1-14B  ──►  + 全新 r128 LoRA (+全量记忆patch)  ──►  merge LoRA/patch 进主干
(冻结主干)          训练：架构适配                            │
                                                            ▼
                                              transformer_init  ← 一个「新的冻结主干」
                                              (Helios-Base/transformer_init，
                                               适配已 bake 进全秩权重)
                                                            │
        ┌─────────── POST (lr 3e-5, ~7500 步) ───────────┐ │
transformer_init ──►  + 【又一个】全新 r128 LoRA      ──►  merge  ──►  transformer
(冻结主干)              训练：精修                                    (= Helios-Base 最终 Base)
```

- **INIT = 架构适配（architectural adaptation）**：裸 Wan 是一个**非**自回归、无分块历史、
  无 Multi-Term Memory 的 T2V 模型。INIT 用较高的 5e-5，让全新的 r128 LoRA + 全量记忆 patch
  Conv3d 快速学会「分块自回归 + 三段历史」这套结构。这是变化最大的一步，所以给高 LR。
- **两阶段之间：merge**（关键）。把 INIT 的 LoRA（+ 全量记忆 patch 增量）**合并进主干全秩权重**，
  得到 `transformer_init`。它是一个把适配「烤进去」的**新的冻结起点**。
- **POST = 精修（refinement）**：在 `transformer_init` 之上再开**一个全新的 r128 LoRA**，
  用较低的 3e-5 打磨质量。

### 为什么是「merge → 全新 LoRA」而不是「同一个 LoRA 继续降 LR 训」？

这是本文最核心的问题，也是官方 recipe 与「偷懒 continue」的分水岭：

1. **秩预算刷新**。一个 r128 LoRA 的表达力受秩 128 限制。INIT 那步「粗适配」会把这
   128 维方向大量占满。merge 之后适配以**全秩**烤进冻结主干，POST 的新 LoRA 又拿到一份
   **全新的、干净的 128 维预算**去学**残差方向**（细节/质量），而不是在已被占满的方向上继续挤。
   两个 r128 adapter 串联（各自 merge）≈ 比「一个 r128 训两遍」有效容量更大。
   这就是 progressive / stacked-LoRA 的标准套路。
2. **优化更稳**。merge 后 POST 的 loss landscape 是围绕「已适配主干」重新展开的，
   低 LR 精修不会被 INIT LoRA 里残留的大幅度方向拖动。
3. **可发布的干净中间产物**。`transformer_init` 本身就是一个自洽的中间 checkpoint
   （Helios 公开的 `Helios-Base/transformer_init` 子目录即此物），便于对比与复现。

> 实证：本地 `Helios-Base` 快照里**同时**存在 `transformer/` 与 `transformer_init/` 两个子目录
> （`ls .../models--BestWishYsh--Helios-Base/snapshots/<hash>/`），正是这条「init→merge→post→merge」
> 流水线的两个 merge 产物。

---

## 2. `transformer_partial.pth` 是什么，为什么 merge 需要它

我们的 checkpoint 目录里除了 `pytorch_lora_weights.safetensors` 还有一个
`transformer_partial.pth`（~96 MB）。原因：

- 配置里 `is_train_full_multi_term_memory_patchg: true` —— 三段记忆 patch（`patch_long/mid/short`
  Conv3d）是**全量权重**训练的，**不是** LoRA。
- 这些「非 LoRA 但可训练」的全量参数增量，`save_model_hook` 会单独存成 `transformer_partial.pth`
  （LoRA adapter 存 safetensors，全量可训练部分存 partial）。
- 所以**忠实 merge 必须同时合并两者**：LoRA safetensors + partial。对应工具是
  `tools/merge_lora_partial_for_helios.py`（而非只合 LoRA 的 `merge_lora_for_helios.py`）。

---

## 3. 映射到我们的 cfr368 run + POST 的具体做法

我们的实际 INIT run（`stage1_init_cfr368.yaml`）与官方 `stage_1_init` 对齐：从裸 Wan、
r128/α128、lr 5e-5 恒定、5500 步、全量记忆 patch。已完成于 `checkpoint-5500`
（`pytorch_lora_weights.safetensors` 2.5 GB + `transformer_partial.pth` 96 MB）。

差异点仅在于我们的数据/分辨率/global-batch（388k 的 368×640 latent、gb120），
这些不改变 init/post 的**结构关系**。

### 忠实 recipe 的 POST（推荐）

1. **merge**：用 `tools/merge_lora_partial_for_helios.py` 把 `checkpoint-5500` 的 LoRA + partial
   合并进裸 Wan 主干 → 产出我们自己的 `transformer_init`（一个物化的全权重 transformer 目录）。
2. **新 POST config**（clone `stage1_init_cfr368.yaml` 改 3 处，对齐官方 diff）：
   - `transformer_model_name_or_path` → 上一步 merge 出来的 `transformer_init` 目录；
     `subfolder` 相应设置（若 merge 输出为顶层则留空/`transformer`）。
   - `learning_rate: 5e-5 → 3e-5`。
   - `output_dir` / `wandb_name` → `stage1_post_cfr368`（新目录，保留 init run 不动）。
   - `max_train_steps` → 7500（post 规格）；`resume_from_checkpoint: latest`（在新目录内 crash-resume）。
   - 其余（r128/α128、bs、corrupt、记忆 patch 开关）**保持不变**。
3. 起训——POST 会在 `transformer_init` 之上开一个**全新** r128 LoRA。

### 「偷懒」变体（非官方，仅供权衡）

直接从 `checkpoint-5500` **continue 同一个 LoRA**、只把 lr 降到 3e-5、步数 +7500。
省掉 merge，但**偏离官方 recipe**（复用旧 adapter，拿不到「刷新秩预算」的收益）。

---

## 4. 一图速查（差异清单）

| 项 | INIT | POST | 是否影响结果 |
|---|---|---|---|
| transformer 起点 | 裸 Wan2.1-14B | `transformer_init`（init merge 产物） | **是（核心）** |
| 起点是否含 Helios 适配 | 否 | **是**（已 bake 进全秩主干） | **是** |
| LoRA | 全新 r128 | **又一个**全新 r128 | **是** |
| learning_rate | 5e-5 | 3e-5 | **是** |
| 训练参数集（LoRA all-linear + LoRA patch_embed + 全量记忆 patch，冻结 norm/主干） | 同 | 同 | 否 |
| bs / accum / scheduler / warmup / corrupt / random_drop / ema | 同 | 同 | 否 |
| 阶段间动作 | — | **merge（LoRA+partial）** 上一阶段 | 前置条件 |
| 语义 | 架构适配（大改） | 质量精修（小改） | — |

---

## 5. file:line 索引

- 官方参考配置：`scripts/training/configs/stage_1_init.yaml`、`stage_1_post.yaml`
  （逐字段 diff 见 §0）。
- 我们的 INIT：`scripts/training/configs/stage1_init_cfr368.yaml`
  （from-Wan、5e-5、5500 步、r128、`is_train_full_multi_term_memory_patchg:true`）。
- merge 工具：`tools/merge_lora_partial_for_helios.py`（LoRA+partial，忠实合并）；
  `tools/merge_lora_for_helios.py`（仅 LoRA）；`tools/merge_lora_for_wan.py`。
- LoRA-only 保存 / partial 保存逻辑：`train_helios.py` 的 `save_model_hook`
  （LoRA→safetensors，全量可训练→`transformer_partial.pth`）。
- 三段记忆 patch 结构：`helios/modules/transformer_helios.py:1065-1068`
  （`patch_long/mid/short` Conv3d，`history_sizes=[16,2,1]`）。
- 上游 stage 表：`../CLAUDE.md`「The 6 stage configs and how they chain」。
- 相关既有文档：`docs/HELIOS_STAGE1_TRAINING.md`（§7 附注已在本文 §0 勘误）、
  `docs/HELIOS_TRAINING_STAGES.md`。
