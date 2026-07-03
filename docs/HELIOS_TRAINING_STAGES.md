# Helios 训练阶段详解：目的、算法与区别

> 本文档解析 `helios-team/`（上游公开 Helios 仓库）的**三阶段渐进式训练流程**，对应 6 个 stage 配置文件
> （`scripts/training/configs/stage_{1,2,3}_*.yaml`），逐阶段说明**目的、核心算法、关键配置差异**。
> 所有训练都通过 `accelerate launch train_helios.py --config <stage>.yaml` 启动；全部 stage 行为由
> **同一个 2603 行的 `train_helios.py` 中的配置分支**控制，而非独立脚本。
>
> 行号/路径基于本仓库当前 checkout。引用：`README.md:538`（官方三阶段概述）、各 stage YAML、
> `helios/utils/utils_helios_base.py`、`helios/utils/utils_helios_post.py`、`helios/scheduler/scheduling_helios.py`。

---

## 0. 总览：从 Wan2.1 双向模型到实时自回归视频生成器

Helios 的目标是把 **Wan2.1-T2V-14B**（一个双向、整段去噪的 DiT）改造成一个**自回归、分块（chunked）**
的视频生成器，能够**实时、分钟级**地连续生成长视频（官方宣称单卡 H100 上 19.5 FPS，`README.md:30`）。

这件事一次做不到，因此拆成三个**逐阶段叠加**的目标，每阶段产出一个公开模型：

| 阶段 | 产出模型 | 一句话目的 | 训练范式 | 推理步数 |
|---|---|---|---|---|
| **Stage 1 (Base)** | `Helios-Base` | **架构适配**：把双向模型变成自回归分块生成器 | Flow Matching | 50 |
| **Stage 2 (Mid)** | `Helios-Mid` | **token 压缩**：金字塔预测-校正，减少噪声 token | Flow + Pyramid | 金字塔多步 |
| **Stage 3 (Distilled)** | `Helios-Distilled` | **蒸馏加速**：50 步 → 4 步、且去掉 CFG | ODE 回归 → DMD2 对抗蒸馏 | 4（固定时间步） |

每个阶段又分 **`_init`（高学习率，快速收敛）** 与 **`_post`（低学习率，精炼）** 两个 phase（README 的 "two-phase" 说法），
通过 `transformer_model_name_or_path` + `subfolder` 把上一阶段产物作为本阶段起点串联起来。

```
Wan2.1-T2V-14B-Diffusers
        │  stage_1_init  (lr 5e-5, r128)
        ▼
   transformer_init ── stage_1_post (lr 3e-5, r128) ──► Helios-Base
        │  stage_2_init  (lr 1e-4, r256, pyramid)
        ▼
   transformer_init ── stage_2_post (lr 3e-5, r256) ──► Helios-Mid
        │  stage_3_ode   (lr 2e-6, ODE 回归, EMA)
        ▼
   transformer_ode ── stage_3_post (DMD2 对抗蒸馏) ──► Helios-Distilled
```

**贯穿所有阶段的不变量（不随阶段改变）：**
- **分块几何**：`latent_window_size = 9` 个 latent 帧/块 = `(9-1)*4 + 1 = 33` 个 RGB 帧（Wan VAE 时间 4×、空间 8× 下采样）。
- **多期记忆 Patchify（Multi-Term Memory）**：`history_sizes = [16, 2, 1]`（long/mid/short），由
  `patch_long`(k/s=4,8,8)、`patch_mid`(2,4,4)、`patch_short`(1,2,2) 三个独立 Conv3d 分别 patchify——
  远期历史强力池化、近期历史保留细节（`has_multi_term_memory_patch: true` 全程开启）。
- **全程 LoRA 训练，没有任何一个阶段对 14B 主干做全量 fine-tune**：14B Wan2.1 backbone **始终冻结**，
  只训 LoRA adapter（`lora_layers: all-linear`）。唯一被"全量训练"的是 Helios **新增的小模块**
  （多期记忆 Conv3d `patch_short/mid/long`，参数量极小），且**只在 Stage1 `_init` 这一段**如此——
  这跟"全量 fine-tune 主干"是两回事（详见 §1.3 的可训参数表）。仅保存 adapter；训练后用
  `tools/merge_lora_for_helios.py` 把 LoRA 合并回主干，得到最终 `transformer/*.safetensors`。
- **Dynamic Shifting**：把 noise schedule 按 latent 尺寸做时间步偏移，应用于所有 timestep 相关操作（`README.md:538`）。

---

## 1. Stage 1 — Base：架构适配（双向 → 自回归分块）

**配置**：`stage_1_init.yaml`（起点 `Wan2.1-T2V-14B-Diffusers`）→ `stage_1_post.yaml`（起点 `Helios-Base/transformer_init`）

### 1.1 目的
原始 Wan2.1 是**整段双向**去噪：一次看到全部帧。Helios 需要它**按块自回归**：每个 33 帧的块只能
看到**历史块**（已生成的过去）作为条件，逐块生成。Stage 1 就是教会模型这套新的条件化方式，README 称之为
三个机制的引入（`README.md:538`）：
- **Unified History Injection**（统一历史注入）：把历史 latent 作为条件喂入。
- **Easy Anti-Drifting**（轻量抗漂移）：通过对历史/输入加噪等手段缓解自回归长程误差累积。
- **Multi-Term Memory Patchification**（多期记忆 patchify）：上面提到的三粒度历史压缩。

### 1.2 算法：标准 Flow Matching（流匹配）
- 损失：`_flow_loss`（`helios/utils/utils_helios_base.py:20`），即标准的流匹配 MSE——模型预测从噪声到数据的
  速度场（velocity），对 ground-truth 速度做回归。这是 Stage 1 & 2 的默认 regime（既非 DMD、也非 ODE 回归）。
- 噪声输入构造：`prepare_stage1_noise_input`（`utils_helios_base.py:724`）。
- 时间步采样权重：`weighting_scheme: logit_normal`（`logit_mean 0, logit_std 1`）——是 Stage 1 区别于后续阶段
  （后续多为 `none`）的一个细节。

### 1.3 数据与正则（Stage 1 的"招牌"开关）
Stage 1 用**预编码的 history latents**（`use_stage1_dataset: true` → `dataloader_history_latents_dist.py`），
热循环里**不读原始视频**。关键正则项：
- `corrupt_history: true`：对历史 latent 加噪（noise 模式，`prob 0.9`，三粒度 `noise_corrupt_ratio_* = 1/3`）——
  **Easy Anti-Drifting 的核心**：训练时让模型适应"带噪的历史"，从而对推理期自身产生的不完美历史更鲁棒。
- `is_random_drop: true`（`v2v/t2v ratio = 0.4/0.4`）：随机丢弃历史，让模型既能 v2v（有历史续写）又能 t2v
  （无历史从头生成），统一两种任务。
- `zero_history_timestep: true`、`guidance_cross_attn: true`：历史以"干净"时间步注入、引导走 cross-attn。

#### Stage 1 到底训练哪些参数（`stage_1_init`，代码 `train_helios.py:394-422`）
> **再次强调：这不是全量 fine-tune。** 14B Wan2.1 主干**冻结不动**；下表才是真正参与梯度更新的部分。

| 参数部分 | 训练方式 | 配置开关 |
|---|---|---|
| 所有 linear 层（attn 的 q/k/v/out、ffn proj 等） | **LoRA r128**（低秩增量，主干权重不变） | `lora_layers: all-linear` |
| `patch_embedding`（输入 patchify） | **LoRA** | `is_train_lora_patch_embedding: true` |
| **`patch_short/mid/long`（多期记忆 Conv3d）** | **全量**（这几个**新增小模块**整权重可训） | `is_train_full_multi_term_memory_patchg: true` |
| Wan2.1 14B 主干权重 | **冻结** | （`add_adapter` 后默认 frozen） |
| norm 层 | 冻结 | `train_norm_layers: false` |

> 即 Stage 1 = **LoRA(r128) 微调** + **少量新增结构件（多期记忆 Conv3d）的全量训练**。那三个 Conv3d 是 Helios
> 为自回归改造**新加进 Wan2.1 的模块**（原模型没有），所以要全量学；但它们参数量极小，与"对 14B 主干做全量
> fine-tune"完全不是一回事。

### 1.4 `_init` 与 `_post`（两段**都做**，顺序串联，非二选一）
**先跑 `_init` 打基础，其产物再喂给 `_post` 精炼**——不是选其一：

```
Wan2.1-14B ──stage_1_init(lr 5e-5)──► Helios-Base/transformer_init ──stage_1_post(lr 3e-5)──► Helios-Base(发布)
```

| | `_init` | `_post` |
|---|---|---|
| **起点 checkpoint** | **`Wan-AI/Wan2.1-T2V-14B-Diffusers`**（原始双向模型，无 Helios 新部件） | **`Helios-Base/transformer_init`**（接 `_init` 产物，已含新部件） |
| 学习率 | **5e-5**（快速收敛） | **3e-5**（精炼） |
| seed | 43 | 44 |
| 多期记忆 Conv3d / patch_embedding / all-linear | full Conv3d + LoRA patch_embed + LoRA all-linear | **完全相同** |
| corrupt/random_drop/数据/其余全部 | … | **逐项相同** |

> **更正（按上游 `stage_1_{init,post}.yaml` 实测）**：init 与 post 的**可训参数配置完全一致**——两者
> `is_train_full_multi_term_memory_patchg: true`（多期记忆 Conv3d 都是**全量**训练）、`is_train_lora_patch_embedding: true`、
> all-linear LoRA r128，主干 14B 冻结。**早先"init 全量 Conv3d、post 改 LoRA"的写法是错误的**。
> 除超参（lr 5e-5→3e-5、seed）外，**唯一实质区别是起点 checkpoint**：`_init` 从**原始 Wan2.1-14B** 起步、
> 第一次构建并学出 Helios 新部件（三个 Conv3d 由 Wan `patch_embedding` 派生初始化，见 §1A Step 6）；
> `_post` 从 `_init` 产物接着低 LR 精炼。训练机制/训哪些参数两段相同。

> **关于 `_init` / `_post` 的来源（git 查证）**：`stage_1_post` / `stage_2_post` / `stage_3_post` 三个文件
> **都在上游唯一 commit `9862fbf init` 里被跟踪**，即 **`_init`/`_post` 是上游 Helios 原生的"两段式"训练，不是本地新增**。
> 本地新增（未 commit、`git status` 为 `??`）的是 `stage1_continue_ohv*` / `stage1_multievent_ohv*` /
> `stage1_video_single_24fps`，命名前缀是 `stage1_continue_` / `stage1_multievent_`，与上游 `stage_1_post` 无关。
>
> **`_init` vs `_post` 的含义**（README 的 "two-phase"）：每个 stage 的两段**都要跑、且顺序串联**（不是二选一）——
> - `_init`：**高学习率、快速收敛**，从一个**全新起点模型**开始（学新增架构部件），产物存为上一模型的 `transformer_init` 子目录；
> - `_post`：**低学习率、精炼**，**接着 `_init` 的产物继续训**，产物才是正式发布模型（Helios-Base/Mid）。
>
> 差异：**Stage1 init/post 在可训参数上相同**（都 full Conv3d + LoRA），区别只在起点 checkpoint + lr + seed
> （**主干 14B 始终冻结、没有全量 fine-tune**，见 §1.3）。注意 **Stage2 才有结构性差异**：`_init` 用 `sample_ratios [1,2,1]`、`_post` 用 `[1,1,1]`。

---

## 1A. 深入：Multi-Term Memory Patchification（多期记忆 Patchify）的完整数据流

> 这是 Stage 1 把双向模型改造成自回归生成器的**核心机制**，全程（Stage 1/2/3）开启（`has_multi_term_memory_patch: true`）。
> 代码路径：历史装配 `helios/dataset/dataloader_history_latents_dist.py:136-196`；切窗+随机丢弃
> `helios/utils/utils_helios_base.py:600-721`；三路 Conv3d patchify + 拼接
> `helios/modules/transformer_helios.py:1066-1068, 1200-1256`；权重初始化 `transformer_helios.py:1088-1106`。

### 问题背景：自回归长视频为什么需要"分粒度记忆"
每生成一个 33 帧的块，模型都要以**全部历史**为条件才能保持连贯。但历史会越来越长，若把所有历史帧都按当前块的高分辨率喂进
self-attention，token 数会爆炸、根本无法实时。**人看视频也是近处看细节、远处只记大概**——Multi-Term Memory 就是把这个直觉
做成结构：**离当前越近的历史保留越多细节，越远的历史压得越狠**，从而用**近似固定**的 token 预算装下很长的记忆。

### Step 1 — 历史装配（dataloader）
对随机抽到的目标块 `choice_idx`，取它**前面紧邻的 `history_window_size = sum([16,2,1]) = 19` 个 latent 帧**作为历史
（视频开头不足 19 帧时用零在**前面** padding）。得到一段连续的 `history_latent`（19 帧）+ `target_latent`（`latent_window_size=9` 帧）。
（`dataloader_history_latents_dist.py:144-192`）

### Step 2 — 按 `history_sizes` 切成 long / mid / short 三段
`history_sizes` 先**从大到小**排序为 `[16, 2, 1]`，沿时间维把 19 帧切成（`utils_helios_base.py:628-649`）：

```
时间轴 →  [ x0 ][ long = 16 帧 ][ mid = 2 帧 ][ short = 1 帧 ][ 目标块 = 9 帧 ]
            ↑      最远(最旧)        中期         最近(紧邻)        当前要生成的
          锚点
```

- **`latents_history_long`** = 最前的 16 帧（离当前最远）。
- **`latents_history_mid`** = 中间 2 帧。
- **`latents_history_1x`（short 主体）** = 最后 1 帧（紧贴目标块）。
- **`x0` 前缀**：视频第一帧 latent（`is_keep_x0`），拼到 short 前作为全局**锚点**，缓解长程漂移；若抽到的是第 0 块则 x0 置零。
  最终 `latents_history_short = cat([x0, latents_history_1x])`（`utils_helios_base.py:707-710`）。

> 注意"short 离当前最近、long 最远"——名字指的是**记忆的时间跨度/粒度**，不是序列里的物理位置。

### Step 3 — 三个独立 Conv3d 做"patchify + 下采样"
三段各过一个**独立的 `nn.Conv3d(in_channels, inner_dim, kernel_size=k, stride=k)`**（kernel=stride，即不重叠分块），
`k` 越大压得越狠（`transformer_helios.py:1066-1068`）：

| 窗口 | 帧数 | Conv3d (t,h,w) | 单 token 覆盖体素 | patchify 后 token 数(相对) | 角色 |
|---|---|---|---|---|---|
| `patch_short` | 1(+x0) | **(1,2,2)** | 4× | 多（高保真） | 近期细节 |
| `patch_mid`  | 2 | **(2,4,4)** | 32× | 中 | 中期 |
| `patch_long` | 16 | **(4,8,8)** | 256× | 少（强压） | 远期梗概 |

**关键巧思——token 预算平衡**：`long` 虽有 16 帧，但每 token 压 `4×8×8=256` 倍；`short` 只有 1 帧但每 token 只压 `1×2×2=4` 倍。
于是三段产出的 token 数量量级相当：**越久远的记忆"单帧 token 成本越低"**，所以能在近似固定的 token 预算里塞进很长的历史。
mid/long 进 Conv3d 前先 `pad_for_3d_conv` 补齐到 kernel 整除（`transformer_helios.py:1221,1241`）。

### Step 4 — RoPE 位置编码对齐
每段历史 token 都要拿到**正确的绝对时间位置**，才能让当前块"知道"哪段记忆离自己多远。索引在
`[1(x0) + 16 + 2 + 1 + 9]` 上切分（`utils_helios_base.py:637-647`）；mid/long 的 RoPE 频率还要
`center_down_sample_3d` 按 (2,2,2)/(4,4,4) 下采样，匹配池化后的 token 网格（`transformer_helios.py:1231-1232, 1251-1252`）。

### Step 5 — 拼成单序列过 self-attention
三段 patchify 结果依次 concat 到当前块 token 前，最终序列为 `[long, mid, short, 当前块]`
（`transformer_helios.py:1215,1235,1255`，每次把新历史 prepend）。整条序列过统一的 self-attention，
**当前块就能同时 attend 到粗粒度远景 + 细粒度近景**——这就是"Unified History Injection"。

### Step 6 — 权重初始化：复用 Wan 的 patch_embedding
三个 Conv3d 不是随机初始化，而是**从 Wan 原始 `patch_embedding` 的权重派生**（`initialize_weight_from_another_conv3d`，
`transformer_helios.py:1088-1106`）：`patch_short` 直接拷贝（取前 16 通道）；`patch_mid`/`patch_long` 把核
`einops.repeat` 平铺到更大尺寸后**分别除以 8 / 64**（= 平铺体素数），保证 patchify 后的激活尺度与预训练一致——
让新部件一开始就"约等于"预训练的 patch embedding，训练更稳。

### 配套机制（与多期记忆共同作用）
- **Easy Anti-Drifting（`corrupt_history`）**：训练时对 long/mid/short 三段**各自独立加噪**（`corrupt_history_latents`，
  `utils_helios_base.py:806-824`），三段 `noise_corrupt_ratio_* = 1/3`。让模型适应"带噪历史"，从而对推理期自身产生的不完美历史更鲁棒。
- **Random Drop（`is_random_drop`）**：以一定概率把历史**从最远端（long→mid→short 顺序）逐步置零**
  （`utils_helios_base.py:651-705`）——t2v 时全清（无历史从头生成）、v2v 时按窗口随机截断，统一 t2v/i2v/v2v 三种任务。

---

## 2. Stage 2 — Mid：金字塔预测-校正（token 压缩）

**配置**：`stage_2_init.yaml`（起点 `Helios-Base`）→ `stage_2_post.yaml`（起点 `Helios-Mid/transformer_init`）

### 2.1 目的
Stage 1 的模型每步都要处理大量"噪声 token"，计算量大。Stage 2 引入
**Pyramid Unified Predictor-Corrector（金字塔统一预测-校正）**，目标是**激进地减少噪声 token 数量**，
从而降低单块计算量、为实时性铺路（`README.md:538`）。

### 2.2 算法：Flow Matching + Pyramid 调度
- 仍是流匹配损失（`_flow_loss`），但启用 `is_enable_stage2: true` + `is_navit_pyramid: true`，
  噪声输入改由 `prepare_stage2_noise_input`（`utils_helios_base.py:1011`）构造。
- **Pyramid 调度器** `HeliosScheduler`（`helios/scheduler/scheduling_helios.py`）取代 Stage 1 的
  `FlowMatchEulerDiscrete`/`UniPC`：把去噪过程切成 `stage2_num_stages: 3` 段，
  `stage2_stage_range = [0, 1/3, 2/3, 1]`，`gamma = 1/3`——为每段预计算独立的 sigma/timestep 区间，
  对不同噪声水平的 token 用不同（金字塔式）的分辨率/token 密度处理。
- `weighting_scheme: none`（与 Stage 1 的 `logit_normal` 不同）。

### 2.3 `_init` vs `_post`（关键区别在采样比例）
| | `_init` | `_post` |
|---|---|---|
| 起点 | Helios-Base | Helios-Mid/transformer_init |
| 学习率 | **1e-4** | **3e-5** |
| `lr_warmup_steps` | 1000 | 500 |
| **`stage2_sample_ratios`** | **`[1, 2, 1]`** | **`[1, 1, 1]`** |
| seed | 45 | 46 |

- **LoRA 升到 r256**（`lora_alpha 256`），`train_batch_size 1`。
- `stage2_sample_ratios [1,2,1]`：`_init` 时给中间金字塔段**加倍采样**（重点学习中间噪声段），
  `_post` 时恢复均匀 `[1,1,1]` 做整体精炼。
- 仍保留 Stage 1 的全部抗漂移开关（`corrupt_history`、`is_random_drop` 等）——Stage 2 是在 Stage 1 能力之上**叠加**金字塔，而非替换。

---

## 3. Stage 3 — Distilled：对抗分层蒸馏（50 步 → 4 步，去 CFG）

Stage 3 是两步走：**先 ODE 回归做蒸馏初始化**（`stage_3_ode`），**再 DMD2 对抗蒸馏精炼**（`stage_3_post`）。
最终学生模型**锁定在 4 个时间步 `{1000, 750, 500, 250}`，且 CFG-free**（`guidance_scale=1.0`）。

### 3.1 Stage 3-ODE — 蒸馏初始化（`stage_3_ode.yaml`）

**目的**：直接对少步学生做对抗蒸馏很不稳定，先用**ODE 回归**把学生"对齐"到教师的去噪轨迹上，得到一个
稳定的初始化（产出 `Helios-Distilled/transformer_ode`）。

**算法**：
- 损失：`_ode_regression_loss`（`helios/utils/utils_helios_post.py:28`），开关
  `is_use_ode_regression: true` + `is_only_ode_regression: true`，`ode_regression_weight: 80.0`。
  让"少步学生"在若干离散时间步上**回归教师 ODE 轨迹**上的对应点。
- 数据：切换到 **Stage 3 数据集**（`use_stage1_dataset: false`、`use_stage3_dataset: true`
  → `dataloader_dmd.py`），`ode_data_root` 指向**预先用教师跑出来的 ODE 轨迹对**
  （`demo_data/vidprom_filtered_extended`，由 `tools/offload_data/get-ode-pairs` 离线生成）。
- 时间步：`dmd_denoising_step_list = [1000, 750, 500, 250]`，验证 `num_inference_steps: 6`。
- **EMA 开启**（`use_ema: true`, `ema_decay 0.99`, `ema_start_step 250`，用一个 zero3 deepspeed 配置维护 CPU 副本）。
- 优化器变化显著：`lr 2e-6`（极低）、`adam_beta1: 0.0`、`max_grad_norm 10.0`、`time_shift_type: linear`
  （前两阶段是 exponential）。
- 此阶段 `is_train_dmd: false`——只做 ODE 回归，不做对抗。

### 3.2 Stage 3-post — DMD2 对抗蒸馏（`stage_3_post.yaml`）

**目的**：在 ODE 初始化基础上，用 **DMD2（Distribution Matching Distillation v2）** 把少步学生的质量推到接近教师，
产出最终 **`Helios-Distilled`**。

**算法（核心是生成器/评论家交替）**：
- `is_train_dmd: true`：按 `global_step % dfake_gen_update_ratio == 0`（`dfake_gen_update_ratio: 5`）
  **交替更新 generator 与 critic**——即每更新 1 次生成器，先更新 5 次评论家（TTUR 双时间尺度）。
- 损失：`_generator_loss`（`utils_helios_post.py:2141`）+ `_critic_loss`（`utils_helios_post.py:2788`），
  DMD 梯度核心在 `compute_kl_grad`（`utils_helios_post.py:1615`），`real_guidance_scale 3.0`、`fake_guidance_scale 0.0`。
- 显存管理：`dmd_is_low_vram_mode: true` / `is_gan_low_vram_mode: true`（在 GPU↔CPU 间换入换出模型，
  README 称可在 80GB 内塞下最多四个 14B 模型）。

> **DMD2 三个角色、两个 backbone**——详见下面 §3.2.1。要点：显存里只有 **2 个 14B backbone**
> （① generator 学生 ② real_fake_score = Helios-Base），但有 **3 个逻辑角色**：real-score 与 fake-score
> **共用 Helios-Base backbone，只靠一个 LoRA 开关区分**。

#### 3.2.1 critic / fake-score 到底是做什么的（DMD 原理）

**目标**：让 4 步学生 G 的**输出分布**逼近 50 步 teacher 的分布。分布散度算不出来，但它对 G 输出的**梯度**
可用两个 score 函数（去噪器，估计 ∇log p）表示：

| 角色 | 含义 | 由谁提供 | 是否训练 |
|---|---|---|---|
| **real score** `s_real` | 真实/teacher 分布的分数 | **冻结的 Helios-Base**（`real_score_model_name_or_path`，`train_helios.py:345-360`，`requires_grad_(False)`） | 否，固定 |
| **fake score** `s_fake`（= critic） | **学生当前输出分布**的分数 | 同一个 backbone **+ 可训 critic LoRA**（`add_adapter(critic_lora r256)`，`train_helios.py:444-453`） | **是，持续在线更新** |

**为什么必须有 critic**：学生分布随 G 训练**不断变化**，所以需要一个**一直被重新拟合**的去噪器来估计"学生此刻
长什么样"——这就是 fake-score / critic。它通过对学生**刚生成的样本**做去噪分数匹配（`_critic_loss`）保持最新。

**DMD 给生成器的梯度**（`compute_kl_grad`，`utils_helios_post.py:1825-1834`，DMD 论文 eq.7-8，在 x0 空间）：
```
grad = pred_fake_image − pred_real_image            # s_fake − s_real（eq.7）
grad = grad / |x0_student − pred_real_image|.mean() # 归一化（eq.8）
```
直觉：**学生自己的分数（fake）与 teacher 分数（real）不一致处，就是学生分布"多出/缺失"的地方**，把输出往
缩小该差距的方向推。两分布一致时 `s_fake = s_real`，梯度归零。

**为什么 5:1 交替**：每更新一次 G，学生分布就漂移、critic 立即过时，于是先用新样本把 critic 重训 5 步再更新 G 一次
（`dfake_gen_update_ratio: 5`）。类似 GAN 的 min-max，但 critic 不是二分类判别器，而是**分数估计器**。

**省显存的关键实现**：real 与 fake **共用一个 Helios-Base backbone**，靠 `compute_kl_grad` 里
`disable_adapters()`（→ 跑出 real score）/ `enable_adapters()`（→ 跑出 fake score）切换
（`utils_helios_post.py:1716, 1789`）。该模型跑在独立的 `critic_accelerator`（`train_helios.py:180`）、
独立 `critic_optimizer`（`critic_learning_rate 4e-7`）、独立 zero2 deepspeed plugin（`"critic_model"`）上，
内部即代码所称的 `fake_score_model`（`train_helios.py:936`）。

**DMD2 vs DMD1**：DMD2 去掉了 DMD1 昂贵的"teacher 配对回归损失"（改由前一步 §3.1 的 ODE 回归做初始化），
并可选叠加一个 GAN 头进一步锐化细节（本配置 `is_use_gan: false`，脚手架就绪但默认关闭）。

**可叠加的附加项（各由独立 flag 控制）**：
- **GT-History**：`is_use_gt_history: true`, `use_gt_history_ratio: 1.0`——用真实历史而非自生成历史做条件，稳住蒸馏。
- **饱和度增广**：`is_add_saturation: true`（`ratio 0.3~1.7`）——本阶段新增的数据增广。
- **GAN 脚手架**：`Discriminator3DHead` 多层 hook（`gan_hooks [5,15,25,35]`）+ R1/R2 正则
  （`r1_weight 100`），但**本配置默认 `is_use_gan: false`**（脚手架就绪、默认关闭；
  `stage_3_post_gan_version.yaml` 是打开 GAN 的变体）。
- **Reward Model**：`reward_model_name_or_path` 指向 `videoalign/` 的 VLM 奖励模型（`is_use_reward_model` 时启用）。
- `is_amplify_first_chunk: true`：放大首块的权重（首块无历史、最难，单独加强）。

#### 3.2.2 Stage 3 完整损失清单（哪些 loss、谁在算、默认开关）

Stage 3 是 **GAN 式双模型交替**（generator 与 critic = fake_score_model，按 `dfake_gen_update_ratio=5`：每更新 1 次
generator 先更新 5 次 critic），**两个模型各有自己的 loss**。再加上一整套可选 loss 菜单。

**① 两个"必然存在"的核心 loss：**

| loss | 属于 | 公式（要点） | 代码 |
|---|---|---|---|
| **Denoising（去噪分数匹配）** | critic / fake-score | `MSE(fake_score(noisy_fake), critic_noise − generated)`——让 critic 持续拟合**学生当前分布**的分数 | `_critic_loss` `utils_helios_post.py:3137/3402` |
| **DMD 分布匹配** | generator | `grad = s_fake − s_real`（`compute_kl_grad:1615`）；`dmd_loss = 0.5·MSE(x0, (x0+grad).detach())` 的 MSE 代理，反传梯度恰为 `grad`；`is_decouple_dmd` 时拆 `ca+dm` 两项 | `_generator_loss` `:2043+` |

> 即便其它全关，Stage 3-post **至少有这两个**：generator 的 DMD loss + critic 的 denoising loss。

**② 可选 loss 菜单（各由 flag 控制）：**

| 可选 loss | 开关 | 加在 | 代码 |
|---|---|---|---|
| GAN-G | `is_use_gan` | generator | `:2133`（`gan_g_weight`） |
| GAN-D + R1/R2 正则 | `is_use_gan` | critic | `:3415`（`r1_weight/r2_weight`） |
| Reward model（vq/mq/ta 三项） | `is_use_reward_model` | generator | `:2655`（videoalign VLM） |
| Smoothness（相邻块 latent MSE） | `is_smoothness_loss` | generator | `:2443` |
| Mean-Var 正则（+x0/chunk 变体） | `is_mean_var_regular` 等 | generator | `:2152+`（KL 到目标 mean/var） |
| Consistency-align（多步 pred_x0 一致） | `is_consistency_align` | generator | `:894,2042` |
| ODE regression（可与 DMD 混） | `is_use_ode_regression` | generator | `:1486` |

**③ released `stage_3_post.yaml` 默认实际只开：**
- generator = **仅 DMD loss**；critic = **仅 denoising loss**（GAN/reward/smoothness/mean-var/consistency/ODE 全 `false`）。
- `is_use_gt_history`/`is_add_saturation`/`is_amplify_first_chunk` 为 true，但**不是独立 loss**——分别是 rollout 用 GT 历史、饱和度增广、首块加权。
- `real_guidance_scale 3.0`/`fake_guidance_scale 0.0` 是 DMD 内部 score 的 CFG 系数，也非额外 loss。

**④ 另一子阶段 `stage_3_ode`**：**只有 1 个 loss** = ODE regression（`MSE(学生少步 x0, 教师 ODE 轨迹 x0)`×`ode_regression_weight 80`，`_ode_regression_loss:28`）。

#### 3.2.3 附：什么是 consistency model（与 `consistency-align` 的关系）

**Consistency model（一致性模型，Song et al. 2023）** 是一种少步生成方法。

- **核心**：扩散采样 = 沿概率流 ODE(PF-ODE)从噪声 $x_T$ 解回数据 $x_0$。同一条轨迹上**每个点 $x_t$ 都对应同一个终点 $x_0$**。
  consistency function $f_\theta(x_t,t)\approx x_0$ 就是"不管你在轨迹哪个位置，直接映回终点"。
- **两个性质**：① 自一致性——同轨迹任意两点输出相同 $f_\theta(x_t,t)=f_\theta(x_{t'},t')$；② 边界 $f_\theta(x_0,0)=x_0$(skip 强制)。
- **训练**：取相邻时间点 $t_n<t_{n+1}$，对齐输出
  $\mathcal L=d\!\big(f_\theta(x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(x_{t_n},t_n)\big)$，目标用 stop-grad/EMA 副本 $f_{\theta^-}$。
  喂数据有 **CD**(用扩散 teacher 蒸馏) / **CT**(无 teacher 从零训) 两种。
- **推理**：1 步 $x_0=f_\theta(x_T,T)$；也可重加噪多步精炼。衍生：LCM / CTM / iCT。
- **vs DMD**：consistency = **回归对齐轨迹端点**；DMD = **分布匹配**($s_{\text{fake}}-s_{\text{real}}$)。两条不同的少步蒸馏路线。

**与 Helios 的关系**：Helios 主蒸馏是 **DMD2，不是 consistency model**。但 §3.2.2 那个可选的
`is_consistency_align`(默认关)借用了 CM 的"同轨迹各点该给同一 $x_0$"思想——在学生多步 rollout 时把**各步预测的 $\hat x_0$ 互相对齐**
($\tfrac12\text{MSE}(\hat x_0^{\text{前几步}},\hat x_0^{\text{末步}})$，`utils_helios_post.py:894`)，**当稳定 few-step 的辅助正则**，而非把模型变成 CM。

### 3.3 ODE vs DMD-post 速查
| | `stage_3_ode` | `stage_3_post` |
|---|---|---|
| 起点 | Helios-Mid | Helios-Distilled/transformer_ode |
| 范式 | ODE 回归（轨迹蒸馏） | DMD2 对抗蒸馏 |
| `is_train_dmd` | false | **true** |
| `is_use_ode_regression` | **true (only)** | false |
| 数据 root | `ode_data_root` | `gan_data_root` |
| 第二个 14B 模型 | 无 | real_fake_score（Helios-Base，一身兼 real-score+critic） |
| EMA start | 250 | 750 |
| `dfake_gen_update_ratio` | 5（未用） | 5（生成器/评论家交替） |

---

## 4. 三阶段横向对比（一张表看清区别）

| 维度 | Stage 1 (Base) | Stage 2 (Mid) | Stage 3 (Distilled) |
|---|---|---|---|
| **目的** | 双向→自回归架构适配 | token 压缩（金字塔） | 蒸馏到 4 步 + 去 CFG |
| **损失/范式** | Flow Matching | Flow + Pyramid | ODE 回归 → DMD2 对抗 |
| **核心损失函数** | `_flow_loss` | `_flow_loss` | `_ode_regression_loss` → `_generator/_critic_loss` |
| **调度器** | FlowMatchEuler / UniPC | `HeliosScheduler`（金字塔 3 段） | 金字塔 + 4 固定步 |
| **LoRA rank** | 128 | 256 | 256（+ critic 256） |
| **学习率** | 5e-5 → 3e-5 | 1e-4 → 3e-5 | 2e-6（极低） |
| **weighting_scheme** | logit_normal | none | none |
| **time_shift_type** | exponential | exponential | linear |
| **数据集 loader** | `dataloader_history_latents_dist`（预编码历史 latent） | 同 Stage 1 | `dataloader_dmd`（ODE 轨迹对 / GAN 数据） |
| **EMA** | 否 | 否 | **是**（decay 0.99） |
| **推理步数** | 50 | 金字塔多步 | **4**：`{1000,750,500,250}` |
| **CFG** | 有（guidance 5.0） | 有 | **无**（1.0，CFG-free） |
| **显存特点** | 单模型 14B | 单模型 14B | **2 个 14B backbone**（generator + real_fake_score；后者兼 real/fake 两角色） |
| **抗漂移开关** | corrupt_history / random_drop（引入） | 继承 | 继承 + saturation 增广 + GT-history |

---

## 5. 为什么是"叠加式"三阶段，而不是一步到位

1. **解耦难题**：架构改造（自回归条件化）、计算压缩（金字塔 token）、采样加速（少步蒸馏）是三个正交问题。
   一次同时优化会相互干扰、极不稳定，因此**逐层叠加、每层冻结上层能力（LoRA + 串联起点）**。
2. **蒸馏需要好教师**：Stage 3 的 DMD2 用 `Helios-Base` 当 real-score 教师、用 Stage 2 产物当学生初始化——
   没有前两阶段就没有可蒸馏的对象。
3. **稳定性递进**：Stage 3 内部还要先 ODE 回归"暖启动"再上对抗蒸馏，正是因为直接对抗会崩。
4. **抗漂移是地基**：`corrupt_history` 等开关从 Stage 1 引入并被后续阶段继承——自回归长视频的误差累积
   是贯穿始终的敌人，每阶段都在它之上工作。

---

## 6. 相关变体与扩展配置（本仓库额外存在）

- `correct.yaml`：上游 issue #38 的修正版（修 i2v 训练/推理不一致，完全启用 Easy Anti-Drifting）——
  复现改进配方时优先于 `stage_1_*`。
- `stage_3_post_gan_version.yaml` / `stage_3_post_self-forcing_version.yaml`：Stage 3-post 的 GAN 打开版 / self-forcing 变体。
- `stage1_continue_ohv*.yaml`、`stage1_multievent_ohv*.yaml`、`stage1_video_single_24fps.yaml`：
  **本地研究**新增的 Stage-1 续训配置（OpenHumanVid 单人续训、多事件提示切换、24FPS 单提示**全数据集**续训；
  这里"全数据集"指数据规模，训练方式仍是 LoRA 续训，非主干全量 fine-tune）——
  详见仓库根 `CLAUDE.md` 与记忆索引，**不属于上游三阶段发布流程**。

---

## 7. 关键文件索引

| 内容 | 路径 |
|---|---|
| 单文件训练入口 | `train_helios.py`（`main(args)` 在 line 2454） |
| 流匹配损失 / Stage1-2 噪声构造 | `helios/utils/utils_helios_base.py`（`_flow_loss:20`, `prepare_stage1_noise_input:724`, `prepare_stage2_noise_input:1011`） |
| ODE 回归 / DMD 损失 | `helios/utils/utils_helios_post.py`（`_ode_regression_loss:28`, `_generator_loss:2141`, `_critic_loss:2788`） |
| 金字塔调度器 | `helios/scheduler/scheduling_helios.py`（`HeliosScheduler`） |
| 核心模型 | `helios/modules/transformer_helios.py`（`HeliosTransformer3DModel:905`，多期记忆 patch 1065-1068） |
| 阶段配置 | `scripts/training/configs/stage_{1,2,3}_*.yaml` |
| 配置对比工具 | `scripts/training/compare_yaml.py` |
| 离线预编码 | `tools/offload_data/`（short/long latents、text embedding、ODE pairs） |
| 官方三阶段概述 | `README.md:538` |
