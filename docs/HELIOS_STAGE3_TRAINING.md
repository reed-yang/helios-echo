# Helios Stage 3 训练实况：数据、两模型、DMD loss、与长视频（self-forcing rollout）机制

> 对称于 [`HELIOS_STAGE1_TRAINING.md`](./HELIOS_STAGE1_TRAINING.md)，本文记录 Stage 3（Distilled）**实际怎么训**：
> 多任务数据、两个 14B 模型、分布式（DDP + 双 DeepSpeed ZeRO-2）、loss、以及**长视频机制**。
> loss 的完整清单另见 [`HELIOS_TRAINING_STAGES.md`](./HELIOS_TRAINING_STAGES.md) §3.2.2。
>
> 关键文件：`train_helios.py`、`helios/utils/utils_helios_post.py`、`helios/dataset/dataloader_dmd.py`、
> `scripts/training/configs/stage_3_{ode,post}.yaml` 与 `stage_3_post_self-forcing_version.yaml`。

---

## 0. 重要澄清：长视频机制只在 self-forcing 变体，默认 `stage_3_post` 没开

| 长视频相关开关 | `stage_3_post`（默认/最小） | `stage_3_post_self-forcing_version`（**长视频版**） |
|---|---|---|
| `dmd_num_latent_sections_min/max`（rollout 段数） | **1 / 1**（不 rollout） | **3 / 44**（动态多段自回归） |
| `is_use_gt_history` | **true**（用真历史 = teacher forcing） | **false**（用**自生成**历史 = self-forcing） |
| `num_critic_input_frames` | 9（1 段） | 21 |
| `is_enable_cold_start` | false | **true**（少段热身→多段） |
| `is_use_gan` | false | true |

> 44 段 × 9 latent帧 ×4 = 44×33 = **1452 RGB 帧 ≈ 60s**：self-forcing 版在**一个 step 里自回归 rollout 最长 ~60 秒、用自己的输出当历史**——
> 这才是"在自身长程漂移上训练"的长视频机制。默认 `stage_3_post`（sections=1 + GT history）**不做** rollout。

## 1. 数据：三路 multi-task（`dataloader_dmd.py`）

一个样本可**同时**含三路（idx 对齐，`__getitem__:237`）：

| 数据路 | 内容 | 给谁用 |
|---|---|---|
| `gan_data_root` | 真实视频 VAE latent（+可选 GT 历史） | DMD 的 **real 分布参考** + GT-history 条件 |
| `ode_data_root` | 教师 ODE 轨迹对 | **stage_3_ode** 子阶段回归 |
| `text_data_root` | 纯文本 prompt embed（无视频） | generator **纯 t2v rollout**（无 GT 视频也能 self-forcing） |

**视频 → 样本**：gan 路仍用 `prepare_stage1_latent`（同 Stage 1：随机抽 `choice_idx` 切 1 chunk + 19 帧历史）；
`is_use_gt_history` 时多抽**第 2 个 chunk**（`target_latent_2`，给 rollout 的 GT 历史/连续性）。dataset 粒度仍"一个 `.pt`=一条样本"。

## 2. 两个子阶段 + 两个 14B 模型 + 分布式（与 Stage 1 不同）

- **stage_3_ode**（蒸馏初始化）：只有 ODE 回归 loss，**DDP**（单模型），EMA 开。
- **stage_3_post**（DMD2）：**两个 14B 模型** + `dfake_gen_update_ratio=5` 的 5:1 交替。

**分布式（关键区别）**：Stage 1 纯 DDP；**Stage 3-post 用 DeepSpeed ZeRO-2 × 2**（`train_helios.py:152-180`）：
```
accelerator         (generator)               ← DeepSpeedPlugin(zero2.json)   :889
critic_accelerator  (real_fake_score=critic)  ← 独立 DeepSpeedPlugin(zero2.json)  :180,895
```
- ZeRO-2 **分片优化器状态**（显存里有 generator + critic/real-score 两个 14B，比 Stage 1 紧张得多）；
- `dmd_is_low_vram_mode:true`：transformer / real_fake_score / vae 在 GPU↔CPU **换入换出**（`vram_manager.move_to_*`）；
- `train_batch_size:1`（rollout 长 + 双模型）。
- real-score 与 fake-score(critic) **共用一个 Helios-Base backbone**，靠 `disable/enable_adapters()` 切换（见 `HELIOS_TRAINING_STAGES.md` §3.2.1）。

## 3. 长视频机制详解（self-forcing 版的 rollout）

核心：`run_generator` / `_critic_loss` 的 rollout 循环（`utils_helios_post.py:735-903`）：
```python
num_rollout_sections = sample_dynamic_dmd_num_latent_sections(min,max,...)  # self-forcing: 3→44 动态
history_latents = zeros                       # 第 0 段无历史
for k in range(num_rollout_sections):         # 逐段自回归
    noisy = randn(...)                         # 该段从纯噪声起
    if not is_use_gt_history:                   # self-forcing: 用自己的历史
        latents_history_* = history_latents[-19:].split([16,2,1])   # 从【累积的自生成历史】切
    pred_x0 = denoise(noisy, history, text)    # 4 步 {1000,750,500,250}
    history_latents = cat([history_latents, pred_x0])   # :903 自己的输出接回历史 ← 闭环
    should_compute_grad = (k >= start_gradient_section_index)  # :821 梯度只压靠后段
```

三个让它"撑长"的设计：
1. **自生成历史闭环**（`is_use_gt_history=false`）：第 k 段以**前 k 段自己产出的（可能已漂移）历史**为条件 →
   模型被迫学会**在自身误差上继续生成而不崩**。这是 Stage 1/2（teacher-forcing，只见 GT 历史）**学不到**的。
2. **梯度只压靠后段**（`should_compute_grad`）：前几段允许漂，重点训"漂移之后仍产好结果"，直接对抗 exposure bias。
3. **Cold-start + 动态段数**（`is_enable_cold_start`，3→44）：先少段热身（一上来 44 段太不稳），随训练拉长 horizon。

配合 **DMD2 分布匹配**（每段输出分布贴教师）+ **首块加权**（`is_amplify_first_chunk`，首段无历史最难）共同撑住少步长程质量。

## 4. 并行期（每个 training step）

```
24/32 GPU 数据并行(DDP) + 每模型 ZeRO-2：
  if global_step % 5 != 0:   # 5/6 步：更新 critic
      critic rollout(自生成样本) → denoising 分数匹配 loss → critic_accelerator.backward
  else:                       # 1/6 步：更新 generator
      generator rollout(N 段,自历史) → DMD loss(s_fake−s_real) → accelerator.backward
  低显存：transformer / real_fake_score 在 GPU↔CPU 轮换
```
- 跨 GPU 仍是**数据并行**；卡内是 **ZeRO-2 分片** + **模型换入换出**；全局 batch = `1×world_size`。

## 5. Stage 1 vs Stage 3 速查

| | Stage 1 | Stage 3-post（self-forcing 版） |
|---|---|---|
| 一步训几个 chunk | 1（单块，teacher forcing） | **3→44 段自回归 rollout** |
| 历史来源 | GT（预编码） | **自生成**（`is_use_gt_history=false`） |
| 模型数 | 1 | **2**（generator + critic） |
| 分布式 | 纯 DDP | DDP + **双 DeepSpeed ZeRO-2** + CPU 换出 |
| loss | flow MSE | DMD(gen) + denoising(critic) |
| batch | 全局 48/64 | 全局 `1×world_size` |
| 长视频能力 | 弱（仅 `corrupt_history`） | **强（自身长程 rollout 训练）** |

## 6. 关键 file:line 索引

| 内容 | 路径 |
|---|---|
| dataloader 选择 | `train_helios.py:112-129`（`use_stage3_dataset` → `dataloader_dmd`） |
| 三路数据 `__getitem__` | `helios/dataset/dataloader_dmd.py:237-317` |
| Accelerator + 双 DeepSpeed | `train_helios.py:152-180`；generator/critic prepare :889/895 |
| rollout 循环 / 自历史闭环 | `helios/utils/utils_helios_post.py:735-903`（接回 :903，梯度门 :821） |
| 动态段数 / cold-start | `train_helios.py:1447-1471`（`sample_dynamic_dmd_num_latent_sections`） |
| DMD / critic loss | `utils_helios_post.py:_generator_loss:2141`, `_critic_loss:2788`, `compute_kl_grad:1615` |
| 长视频 config | `scripts/training/configs/stage_3_post_self-forcing_version.yaml` |
