# Helios Stage 1 当前训练实况：输入/输出、Loss、分布式（DDP 非 FSDP）

> 本文档记录**当前实际在跑的 Stage-1 训练**（以 `stage1_video_single_24fps` 续训 run 为锚），覆盖：
> 启动方式与分布式策略、数据/输入、噪声与目标构造、loss、可训参数、推理路径，以及当前 config 的关键数与隐患。
> 概念层面的「Stage 1 为什么存在 / 架构适配」见 [`HELIOS_TRAINING_STAGES.md`](./HELIOS_TRAINING_STAGES.md) §1；本文偏**操作/实现**。
>
> 所有 `file:line` 基于本仓库当前 checkout。关键文件：`train_helios.py`、
> `helios/utils/utils_helios_base.py`（`prepare_stage1_noise_input:724`、`_flow_loss:20`）、
> `helios/dataset/dataloader_history_latents_dist.py`（`prepare_stage1_latent:136`）、
> `scripts/training/train_stage1_video_single_24fps.sbatch`、`scripts/training/configs/stage1_video_single_24fps.yaml`。

---

## 0. TL;DR

- **分布式：不是 FSDP，是普通 DDP**（`DistributedDataParallelKwargs(find_unused_parameters=True)`，`train_helios.py:147,163`）。
  Stage 1 全程**无** FSDP / DeepSpeed（DeepSpeed 仅 Stage-3 DMD 与 EMA 用）。能用 DDP 是因为**只训 LoRA**：14B 主干冻结（bf16 ~28GB），每卡放得下，无需分片。
- **输入**：预编码 latent（热循环不读视频）——一个干净 9 帧目标块 + 19 帧历史 + 文本 embed。
- **输出/Loss**：标准 flow-matching，模型预测速度场 $v=\text{noise}-\text{clean}$，loss = MSE。
- **推理**：`infer_helios.py` → 50 步、全分辨率、CFG 5.0、**自回归逐块**（无金字塔）。

---

## 1. 启动方式 & 分布式策略（FSDP 的答案）

启动不是 `accelerate launch`，而是 **`srun … python train_helios.py --config …`**，4 节点 × 8 = **32 GPU**
（c-node07 drain 时 3 节点 = 24 GPU），用环境变量 `RANK/WORLD_SIZE/LOCAL_RANK` 做 rendezvous
（`train_stage1_video_single_24fps.sbatch`）。

`train_helios.py` 构建 Accelerator（`:146-170`）：
```python
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)   # :147 → DDP
accelerator = Accelerator(
    gradient_accumulation_steps=1, mixed_precision="bf16",
    deepspeed_plugins=None,            # 仅 is_train_dmd 时才非 None（:158-161）
    kwargs_handlers=[kwargs, init_kwargs])
transformer, optimizer, lr_scheduler = accelerator.prepare(...)        # :889/899
```

- **无任何 FSDP / FullyShardedDataParallel**（全文件 grep 无命中）。
- DeepSpeed 仅当 `is_train_dmd`(Stage 3) 才建 `DeepSpeedPlugin`；Stage 1 不走。
- `find_unused_parameters=True`：大量参数冻结、部分前向分支（gan_heads / restrict_lora 等）不参与，DDP 需要它避免 all-reduce 报错。
- **为什么 14B 还能 DDP 不分片**：主干冻结、只训 LoRA(r128)+ 3 个多期记忆 Conv3d。每卡 = 完整 14B（bf16 冻结 ~28GB）+ LoRA + 少量优化器状态（仅可训参数）。H200 140GB 绰绰有余，故无需 FSDP/ZeRO。DDP 的梯度 all-reduce **只在可训参数上**发生。
- checkpoint：每 500 步一次，`save_model_hook` **只存 LoRA adapter + 可训模块**（不存冻结主干）；sbatch 注释提到 DCP（world-size 锁定，节点数变了不能直接 resume）。

## 2. 输入：预编码 latent（热循环不读视频）

`use_stage1_dataset:true` → `dataloader_history_latents_dist.py`。每个 `.pt` 是一条视频的 **VAE latent**，已切成 section：
`[num_sections, c=16, 9, h, w]`（9 = `latent_window_size`）。`prepare_stage1_latent`（`:136-196`）：

1. 前面**零填充 `history_window_size = sum([16,2,1]) = 19` 帧**（开头不足时历史为零）；
2. 按 seed 随机抽 `choice_idx`（预测哪个块）；
3. 切出 **`history_latent`（紧邻前 19 帧）** + **`target_latent`（9 帧目标块）**；
4. `x0_latent` = 视频首帧（全局锚点；抽到第 0 块则置零）。

随后 19 帧历史切成 **long(16)/mid(2)/short(1)+x0** 三段（→ `latents_history_long/mid/short`）并算 `indices_*`。
文本：每个 `.pt` 存 `prompt_embed_short` + `prompt_embed_long`，单提示场景**随机混用两版 caption**（`_pick_prompt_embed`）。

**喂给模型的张量**（non-navit 单块）：
```
model_input            = target_latent      [b,16,9,48,80]   ← 干净目标块（将被加噪）
latents_history_long   [b,16,16,48,80]      ← 远期（patch_long 4,8,8）
latents_history_mid    [b,16, 2,48,80]      ← 中期（patch_mid 2,4,4）
latents_history_short  [b,16, 2,48,80]      ← x0+近1帧（patch_short 1,2,2）
indices_hidden_states / indices_latents_history_*  ← RoPE 位置
prompt_embeds (UMT5)
```

### 2.1 视频 → 样本 → 并行（拆分粒度与 DP 分片）

**编码期（离线，`get_short-latents.py`）**：一条 `num_frame` 帧视频按 `frame_window_size=(9-1)*4+1=33` 帧一块逐块过 VAE，
每块 → 9 latent 帧；`num_sections = num_frame//33`，存成 `[num_sections,16,9,h,w]` 的一个 `.pt`（含多版 caption embed）。
`num_frame<121` 丢弃（`dataloader:103`）。例：161 帧 → 4 个 section。

**dataset 粒度**：`_build_folder_metadata` **每个 `.pt`（=一个视频）建一条 sample** → `len(dataset)=视频数`（**不是** section 数）。

**`__getitem__`/`prepare_stage1_latent`：每访问只随机产出 1 个 (history,target) 样本**——
把视频 section 拍平回 `[16, num_sections*9, h, w]`、前补 19 帧零、用 `seed=base_seed+epoch*1e6+idx` 随机抽
`choice_idx∈[0,num_sections)`：`history=序列[start:start+19]`、`target=序列[start+19:start+28]`（`start=choice_idx*9`），
`x0`=首帧（`choice_idx==0` 时置零=无历史）。
> 即一个含 `num_sections` 块的视频**有 `num_sections` 个可能样本**，但**每次只随机抽 1 个**；不同 epoch（seed 不同）抽不同块。
> **一个 epoch = 视频数个样本**（每视频随机 1 块）；多 epoch 才覆盖同一视频的不同块。本 ~1 epoch 的 vs24 run 基本每视频 1 块。

**collate**：`collate_fn` 把 `batch_size=2` 个样本 stack（同 bucket → shape 一致）→ `history[2,16,19,h,w]` 等。

**并行 = 纯数据并行（DDP）**（`BucketedSampler`，`num_sp_groups=world_size, sp_world_size=1`）：
① 按 `bucket_key=(num_frame,h,w)` 分桶（保证可 stack）；② epoch 级 seed 全局一致 shuffle；
③ **分片 `indices[rank::world_size]`** → 每 rank 拿互不重叠的 1/world_size 视频；④ rank 内按 2 切 batch。
每步：各 rank 跑自己的 2 样本 → 前向/backward → **DDP 仅在可训参数上 all-reduce 梯度** → 同步 optimizer.step。
全局 batch = `2×world_size` = 48(24GPU)/64(32GPU)。**无模型并行/序列并行（`sp_world_size=1`）、无 FSDP**。
`StatefulDataLoader` 支持中途恢复。

```
视频(161帧) ─编码→ [4 sect,16,9,h,w].pt(=1 dataset 条目)
   └ __getitem__: 拍平+前补19, 随机 choice_idx → 1 样本 (x0[1],hist[19],tgt[9],prompt)
        └ collate stack 2 → 一卡 batch [2,16,*]
             └ DDP: 24/32 卡各跑不同样本, 梯度 all-reduce → 全局 batch 48/64
```

## 3. 噪声构造 + 输出 + Loss（flow matching）

`prepare_stage1_noise_input`（`utils_helios_base.py:724-862`）：
```python
noise  = randn_like(model_input)
u      = logit_normal(mean=0, std=1);  sigma = scheduler.sigmas[u*1000]   # 噪声占比
# Easy Anti-Drifting：对三段历史加噪（corrupt_history=true, prob 0.9, 三段 ratio 1/3）
noisy_model_input = (1-sigma)*model_input + sigma*noise     # :845  σ=1 纯噪声, σ=0 干净
target            = noise - model_input                     # :846  flow-matching 速度场
```

模型 `HeliosTransformer3DModel`（non-navit 路径）吃 `noisy_model_input` + 历史 + timestep + 文本，**输出对每个 token 的预测速度**（unpatchify 回 `[b,16,9,48,80]`）。

**Loss**（`_flow_loss:20-89`）：

$$\mathcal L=\big\|\,f_\theta(\text{noisy},t,\text{hist},\text{text})-(\text{noise}-\text{clean})\,\big\|_2^2$$

`weighting_scheme=logit_normal`（此处权重≈1）。`accelerator.backward(loss)`，`clip_grad_norm=1.0`，AdamW。
**这是 Stage 1 唯一的 loss**（无 pyramid、无 DMD、无 GAN、无 ODE）。

## 4. 训练哪些参数（主干冻结）

| 部分 | 方式 | 开关（当前 config） |
|---|---|---|
| 所有 linear(q/k/v/out、ffn) | **LoRA r128** | `lora_layers:all-linear`，exclude down/up |
| `patch_embedding` | **LoRA** | `is_train_lora_patch_embedding:true` |
| `patch_short/mid/long`（多期记忆 Conv3d） | **全量** | `is_train_full_multi_term_memory_patchg:true` |
| 14B 主干 | **冻结** | `add_adapter` 后默认 |
| norm | 冻结 | `train_norm_layers:false` |

（`train_helios.py:394-423`）

## 5. 当前 run 关键数（`stage1_video_single_24fps.yaml`）

- **起点**：transformer 从 `Helios-Base/transformer_init`（本地快照）续训；vae/text_encoder/tokenizer 从本地 `Helios-Distilled` 目录（用本地目录避开 32-rank HF 快照竞争）。
- **数据**：`/mnt/beegfs/dataset/video_single_24FPS/latents_short_384x640`（~1.2M `.pt`，但 dataloader 丢弃 <121 帧的，仅 ~47–53 万合格）。
- **批/步**：per-GPU bs2，accum1 → 全局 batch **48(24GPU)/64(32GPU)**；`max_train_steps 9900` ≈ 1 epoch；每 500 步存 ckpt。
- **优化**：lr **3e-5** constant + warmup 500，AdamW，bf16，grad_checkpointing on，`max_grad_norm 1.0`。
- **抗漂移**：`corrupt_history:true` + `is_random_drop:true`(v2v/t2v 0.4/0.4)；`zero_history_timestep:true`，`guidance_cross_attn:true`，`restrict_self_attn:false`。
- **EMA off**；**built-in validation 关掉**（`validation_steps=1e8`），改由外部 8-GPU eval 节点(c-node08)对 checkpoint 推理评测。
- ⚠️ `validation_config.use_dynamic_shifting:true` 但训练 `false` —— 同 [`HELIOS_STAGE2_PYRAMID.md`](./HELIOS_STAGE2_PYRAMID.md) Finding A 的不一致（此 run 已禁用内建验证，影响不大；但若改回内建验证要注意）。

## 6. 推理（推理结果怎么来的）

Stage 1 / Base 推理走 `infer_helios.py` → `pipeline_helios.py:stage1_sample`：
- **50 步**全分辨率去噪、**CFG guidance 5.0**（有正负提示）、**无金字塔**（不加 `--is_enable_stage2`）；
- **自回归逐块**：每个 33 帧(=9 latent)块以 patchify 历史(long/mid/short)为条件，生成后接回历史、`restrict_self_attn` 时用 KV-cache，继续下一块；
- eval 用 `scripts/eval_live/run_eval_once.sh`(8 卡数据并行)对每个 checkpoint 跑 test/train 提示，产视频(+可选 HeliosBench 指标)，落 `helios_runs/eval_out*`。

## 7. 隐患 / 待确认

- **trainable 组合与上游不完全一致**：当前同时开 `is_train_lora_patch_embedding`(=_post 风格) + `is_train_full_multi_term_memory_patchg`(=_init 风格)，却用 3e-5 的 refine 学习率。上游 `stage_1_init`=全量记忆 patch + 5e-5、`stage_1_post`=LoRA 记忆 patch + 3e-5，两者都不完全等于本 config。需确认是有意为之（续训既想动记忆 patch 又想低 LR 稳）还是想对齐某一个。可用 `scripts/training/compare_yaml.py` 对比。
- **dynamic_shifting 训练/验证不一致**（见 §5）。
- **DCP world-size 锁定**：节点数变化后不能直接 resume（24↔32 GPU 切换需重起或转换 checkpoint）。

## 8. 关键 file:line 索引

| 内容 | 路径 |
|---|---|
| Accelerator/DDP 构建 | `train_helios.py:146-170`（`DistributedDataParallelKwargs` :147；DeepSpeed 仅 DMD :158-161） |
| LoRA + 可训参数 | `train_helios.py:394-423` |
| `accelerator.prepare` | `train_helios.py:889, 899` |
| Stage1 数据装配 | `helios/dataset/dataloader_history_latents_dist.py:136-196`（`_pick_prompt_embed:201`） |
| Stage1 噪声/目标 | `helios/utils/utils_helios_base.py:724-862`（noisy :845, target :846） |
| flow loss | `helios/utils/utils_helios_base.py:20-89` |
| 推理(Base) | `helios/pipelines/pipeline_helios.py:stage1_sample:483` |
| 启动 sbatch | `scripts/training/train_stage1_video_single_24fps.sbatch` |
| 当前 config | `scripts/training/configs/stage1_video_single_24fps.yaml` |
