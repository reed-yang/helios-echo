# Helios Stage 2（Mid）金字塔预测-校正深度解析 + 训练效果不佳的成因分析

> 本文档是 [`HELIOS_TRAINING_STAGES.md`](./HELIOS_TRAINING_STAGES.md) §2 的展开版，专门、详尽地讲清楚
> **Stage 2 — Mid：Pyramid Unified Predictor-Corrector（金字塔统一预测-校正 / token 压缩）** 的算法，
> 并在**逐行阅读训练 + 推理两条代码路径**的基础上，分析「为什么我们训出来的 Stage 2 效果不好」的可能原因。
>
> 所有 `file:line` 基于本仓库当前 checkout（`helios-team/`，上游公开 Helios 的完整克隆，相关文件 `git status` 干净、未本地改动）。
> 关键文件：
> - 调度器 `helios/scheduler/scheduling_helios.py`
> - 训练噪声构造 `helios/utils/utils_helios_base.py`（`prepare_stage2_clean_input:868`、`prepare_stage2_noise_input:1011`、`_flow_loss:20`）
> - 训练主循环 `train_helios.py:1314-1384`
> - 模型前向（NaViT 打包）`helios/modules/transformer_helios.py`
> - 推理金字塔采样 `helios/pipelines/pipeline_helios.py:stage2_sample:617`

---

## 第一部分：Stage 2 到底在做什么

### 1.1 目的：把「噪声 token 的计算量」压下来

Stage 1（Base）已经把 Wan2.1 改造成自回归分块生成器，但它每个去噪步都要在**完整分辨率**上处理整块的噪声
token，计算量大、离实时还远。Stage 2 的目标是 **token 压缩**：借鉴 **Pyramidal Flow Matching（金字塔流匹配）**
的思想，把"从纯噪声到干净"的整条 flow 轨迹**按噪声水平切成若干段（stage）**，**噪声越大的段用越低的空间分辨率
（越少的 token）去算**，噪声越小的段才逐步升回高分辨率。

直觉：高噪声阶段图像本来就模糊，没必要用全分辨率算；把算力省下来留给低噪声的细节阶段。这样单块的总
token·步数预算大幅下降，为 Stage 3 的少步蒸馏和实时推理铺路。

"Predictor-Corrector / Unified" 指：每一段内部是一段独立的 rectified-flow（predictor），段与段之间通过
**升采样 + 重新加噪（renoise）**衔接（corrector，修掉升采样带来的 block artifact）；整个过程仍由同一个模型、
同一套 history 注入统一完成（unified）。

### 1.2 三段金字塔的几何（调度器 `HeliosScheduler`）

配置（`stage_2_*.yaml`）：

```yaml
is_enable_stage2: true
is_navit_pyramid: true
stage2_num_stages: 3
stage2_stage_range: [0, 1/3, 2/3, 1]     # 把 [0,1] 的 flow 轨迹三等分
stage2_scheduler_gamma: 1/3
stage2_timestep_shift: 1.0
```

`HeliosScheduler.init_sigmas_for_each_stage()`（`scheduling_helios.py:118`）做两件事：

1. **全局 sigma/timestep 轴**（`init_sigmas:100`）：`shift=1.0` 时就是 `sigma` 从 0.999 线性降到 0、
   `timestep = sigma*1000`（999→1）。这里 `sigma` 是**噪声占比**（1=纯噪声，0=干净）。
2. **每段的边界 sigma 与"修正起点" sigma**（`:130-149`）：
   - 段 `i_s` 的原始起点 `ori_start_sigmas[i_s] = sigmas[stage_range[i_s]*1000]`（段边界处的全局噪声水平 `s_b`）。
   - 对 `i_s>0`，起点 sigma 被**修正**（`:140-145`）：
     ```
     ori = 1 - s_b
     corrected = ori / (sqrt(1+1/gamma)*(1-ori) + ori)        # gamma=1/3 → sqrt(4)=2
     start_sigmas[i_s] = 1 - corrected = 2*s_b / (1 + s_b)
     ```
     这个修正是金字塔衔接的核心：它定义了"从上一段升采样上来后，本段开头应该处于多高的噪声水平"。

每段内部（v1）使用**归一化到 [0.999,0] 的 sigma 子轴**（`sigmas_per_stage[i_s]`，对所有段相同，`:185-186`），
但**时间步轴 `timesteps_per_stage[i_s]` 落在该段真实的全局区间**（`:176-184`，按 flow 弧长分配）：

| 段 | 角色 | 分辨率 | timestep 区间（shift=1 近似） | 段内 sigma 子轴 |
|---|---|---|---|---|
| `i_s=0` | 最高噪声 / 最粗 | 1/4 | ~999 → ~743 | 0.999 → 0（归一化） |
| `i_s=1` | 中噪声 / 中 | 1/2 | ~743 → ~384 | 0.999 → 0（归一化） |
| `i_s=2` | 低噪声 / 全分辨率 | 1/1 | ~384 → ~1 | 0.999 → 0（归一化） |

> 关键设计：**模型被告知的是该段的"全局时间步"**，但**输入实际混合用的是段内归一化 sigma**。
> 这就是金字塔的"分段重参数化"——只要训练和推理用**完全一致**的这套映射，就自洽（见 §3.3 的一致性验证）。

### 1.3 训练侧：每段的噪声/目标怎么构造（`prepare_stage2_clean_input:868`）

对一个干净的目标块 latent（`[b,c,t,h,w]`，t=`latent_window_size`=9）：

1. **金字塔干净列表**（`:878-888`）：逐级 `bilinear` 下采样 ×2，得到 `[1/4, 1/2, 1/1]`（低→高）。
2. **金字塔噪声列表**（`:898-907`）：从全分辨率高斯噪声逐级 `bilinear` 下采样 ×2，**每级乘 2**（补偿下采样导致的方差缩小），再反转成低→高。
3. **逐段构造 start/end 与训练对**（`:919-1006`）：对每段 `i_s`
   - `start_point`（段开头，高噪声端）：
     - `i_s=0`：纯噪声 `noise_list[0]`；
     - `i_s>0`：把**上一段的干净 latent `nearest` 升采样 ×2**（`:929-938`），再
       `start_sigma * noise + (1-start_sigma) * upsampled_last_clean`（`:939`）。
   - `end_point`（段结尾，低噪声端）：最后一段 = 干净 latent；否则 `end_sigma*noise+(1-end_sigma)*clean`（`:941-944`）。
   - 段内随机抽 `sigma`（来自归一化子轴）与对应**全局 timestep**（`:946-982`），合成：
     ```
     noisy = sigma * start_point + (1 - sigma) * end_point        # :1000
     target = start_point - end_point                              # :1006（rectified-flow 速度场，段内恒定）
     ```
   - `stage2_sample_ratios`（init `[1,2,1]` / post `[1,1,1]`）控制每段重复采样几次——
     **init 给中间段加倍监督**，post 回到均匀精炼。

> 注意：`use_dynamic_shifting` 分支（`:966-982`）在**训练时被关闭**（见 §2 Finding A），所以训练阶段
> timestep 就是 `timesteps_per_stage[i_s]` 原值，不做分辨率偏移。

### 1.4 NaViT 打包：多分辨率 + 每段历史，塞进一条 varlen attention

`is_navit_pyramid=True` 时，`prepare_stage2_noise_input`（`:1055-1064`）把上面那串**不同分辨率**的
`noisy_latents_list`（如 `[s0, s1a, s1b, s2]`，中段因 ratio=2 出现两次）**整体作为一个 list** 返回，
在 `_flow_loss:45-49` 里直接把这个 **list 当作 `hidden_states` 传给 transformer**。

模型前向 `transformer_helios.py:1317-1543` 的 NaViT 路径（已逐行核对，打包/解包是**自洽且正确**的）：

1. `process_input_hidden_states:1145-1175`：对 list 内每个分辨率各自 `patch_embedding` + 按**自身 H/W** 生成
   RoPE，然后**前插拼接**（顺序翻成 `[s2,s1b,s1a,s0]`，即 高→低）。历史 `short/mid/long` 只 patchify 一次（全分辨率），拼到最前。
2. `forward:1382-1429`：把历史段切下来，对**每个金字塔样本各复制一份历史**，拼成
   `[hist,s2, hist,s1b, hist,s1a, hist,s0]`；`timestep = timestep[::-1]`（`:1390`）使
   `s2↔t2 … s0↔t0` **配对正确**；每段 temb/timestep_proj 各自构造。
3. `create_navit_attention_masks`（`helios_kernels/attention_dispatch.py:46`）给出 varlen 的 `cu_seqlens`：
   **每个 `[hist_i, current_i]` 段是一个独立注意力块**，段间互不可见（这正是 NaViT「一个 batch 里塞多条变长序列」的做法）。
   默认 `restrict_self_attn=false`，所以走 `attn_varlen_func(q,k,v, mask[0])`（`:410-415`）的普通变长全注意力。
4. cross-attn（`guidance_cross_attn=true`，block `:822-864`）把历史临时剥掉、只让 current token 看文本、再插回。
5. 输出 `:1505-1543`：按段反向切出 current 部分、`proj_out`、unpatchify，再 `output[::-1]` 还原成
   `[s0,s1a,s1b,s2]`，与 `targets_list` 顺序一致；`_flow_loss:64-77` 逐段算 MSE 再平均。

> **结论：NaViT 多分辨率打包/解包、timestep 配对、注意力掩码都核对无误**，不是效果差的来源。

### 1.5 推理侧：金字塔采样 + renoise 衔接（`pipeline_helios.py:stage2_sample:617`）

推理是 §1.3 的「反过程」：从最低分辨率（i_s=0）开始逐段去噪，段间**升采样 + renoise**：

```python
for i_s in range(3):
    set_timesteps(steps, i_s)                       # 该段 timestep/sigma 子轴
    if i_s > 0:
        latents = interpolate(latents, mode="nearest") ×2          # :691 升采样
        # block-artifact 修正（与调度器 corrected_sigma 同一套公式）
        alpha = 1/(sqrt(1+1/gamma)*(1-o)+o);  beta = alpha*(1-o)/sqrt(gamma)   # :694-697, o=1-ori_start_sigmas[i_s]
        noise = sample_block_noise(...)                            # :700 带块内相关性的噪声
        latents = alpha*latents + beta*noise                       # :711 renoise
    for t in timesteps:
        noise_pred = transformer(latents, t, history...)           # 与训练同一个 forward
        latents = scheduler.step_unipc(noise_pred, t, latents)
```

`sample_block_noise`（`:443-481`）按 `gamma` 构造 2×2 块内协方差 `I*(1+gamma) - 1*gamma`，使「nearest 升采样后的
低分辨率信号 + 该块相关噪声」在分布上等价于高分辨率上的独立噪声——这是金字塔衔接不出 block 接缝的关键。

---

## 第二部分：为什么训练效果不好 —— 成因分析

> **先给结论（重要）**：本仓库这套 Stage 2 代码是**未改动的上游代码**，也就是**生产出公开 `Helios-Mid` 的同一套代码**
> （`git status` 干净）。我逐行核对了训练与推理两条路径，**没有发现会让模型学坏的「硬 bug」**：金字塔噪声构造、
> NaViT 打包、renoise 衔接、attention 掩码都自洽（§3.3 给出 renoise 的数学一致性证明）。
> 因此**「效果不好」更可能来自配置/评测/数据/起点，而非核心算法 bug**。下面按「我能在代码里确证的不一致」→
> 「设计层面的风险点」→「训练 setup 排查清单」的顺序，从高到低给出可操作的怀疑项。

### Finding A（**已确证的训练/评测不一致**，最高优先级）：验证集渲染开了 dynamic shifting，训练没开

这是我在代码里能 100% 确认的**唯一明确不一致**，而且最可能直接造成「看起来效果差」：

| 路径 | `use_dynamic_shifting` | 来源 |
|---|---|---|
| **训练**（构造 noisy/target） | **False** | `prepare_stage2_clean_input` 经 `get_config_value→training_config`（`utils_base.py:37-43`）；且 `train_helios.py:2548-2552` 有断言**强制** stage2 非 DMD 时它必须为 False |
| **训练中的 W&B 验证视频** | **True** | `log_validation` 的 `pipeline_args` 用的是 `validation_config.use_dynamic_shifting`（`train_helios.py:2170, 2383`），而 `stage_2_*.yaml` 里 `validation_config.use_dynamic_shifting: true` |
| **真·离线推理**（`infer_helios.py` / `helios-mid_t2v.sh`） | **False** | 脚本未传 `--use_dynamic_shifting`，pipeline 默认 False（`pipeline_helios.py:904`） |

后果：**训练时你在 W&B 里看到的 Stage 2 验证视频，用的是和训练分布不一致的时间步调度**
（`stage2_sample:716-731` 会按 latent 序列长度对 sigma 做 `apply_schedule_shift` 偏移），模型在**没见过的 timestep 分布**
上被采样，于是**验证视频会显著差于模型的真实水平**。而离线 `infer_helios.py` 反而是对的（False，与训练一致）。

- **如果你判断「效果不好」是看 W&B 验证视频** → 极可能就是这个假象。
- **修复**：把 `stage_2_*.yaml` 的 `validation_config.use_dynamic_shifting` 改成 `false`（与训练/离线推理对齐）；
  或者干脆用 `infer_helios.py`（不加 `--use_dynamic_shifting`）重新渲染同一 prompt 再判断。
- **验证实验**：固定 prompt/seed，分别用 `use_dynamic_shifting=true/false` 各跑一遍 `stage2_sample`，对比即可定位。

### Finding B（**设计层面风险**，需实验验证）：低分辨率金字塔段与全分辨率历史之间的 RoPE 尺度错配

这是**训练和推理一致**的行为（所以不是 A 那种不一致），但它可能**限制低分辨率段（i_s=0/1）的质量上限**：

- 每个金字塔段的 current token 用**自身分辨率**算 RoPE 空间坐标（`process_input_hidden_states:1160-1162`，
  低分辨率段坐标只到 `0..H/4`）。
- 但拼在它前面的**历史**是**全分辨率** patchify 的（坐标 `0..H_full`，`:1207-1256`，且三段历史每个金字塔段都共享同一份，
  推理 `stage2_sample:751-753` 也是每段传同一份全分辨率历史）。
- 于是在低分辨率段里，current chunk 的空间坐标只覆盖历史坐标系的**左上角约 1/4**——模型要从「左上角的历史」
  生成「整帧的低分辨率内容」，**位置对不齐**，难以做干净的位置对齐式 history 复制。

注意：因为训练/推理一致，模型**能学到一个自洽（但别扭）的映射**，所以这未必是致命 bug（公开 Helios-Mid 也用这套代码且能用）。
但它确实可能是「细节/运动一致性不够好」的来源。
- **验证**：单独看 stage 0/1 的预测（低分辨率段）是否结构/位置漂移明显大于 stage 2；
- **可试的改法**：给金字塔段的 RoPE 空间坐标做**与历史同尺度的归一化/缩放**，让 current 与 history 坐标范围对齐。

### Finding C（已核对：**不是 bug**，列出以排除）：renoise 系数 train/infer 一致

我一度怀疑「训练 `start_point` 的信号系数」与「推理 renoise 后的信号系数」对不上（差一个 `(1-s_b)` 因子）。
**逐项推导后确认是一致的**，差异只是因为「推理段间衔接时，上一段的输出本身就停在噪声水平 `s_b`、并非干净」。
完整推导见 §3.3。**所以 renoise 数学是对的，不要往这里改。** 但建议你跑一个 1-batch 数值自检（§3.3 末尾脚本）
来确认你这套权重/配置下两边分布确实吻合——这是排除性验证，成本极低。

### Finding D（**训练 setup 排查清单**，最可能的"非代码"原因）

既然核心算法没硬 bug、且同款代码能产出公开 Helios-Mid，那「我们训得不好」最可能落在 setup：

1. **数据规模**：`stage_2_*.yaml` 的 `instance_data_root` 是 `demo_data/ultravideo-long`（**玩具/演示数据**）。
   只在 demo 数据上训 Stage 2，效果差是预期的。确认你换成了真实规模的预编码 history-latents 数据。
2. **起点 checkpoint**：Stage 2 必须从一个**质量达标的 Helios-Base**（或官方 `BestWishYsh/Helios-Base`）接力。
   如果 `transformer_model_name_or_path` 指向的是**你自己尚未练好的 Stage 1 产物**，Stage 2 会把上游的缺陷继续放大。
   先单独验收 Stage 1（Base）的自回归续写质量，再上 Stage 2。
3. **两段式是否都跑**：`stage_2_init`（lr 1e-4, `sample_ratios [1,2,1]`）→ 产物存 `Helios-Mid/transformer_init` →
   `stage_2_post`（lr 3e-5, `[1,1,1]`）才是发布权重。只跑 init 不跑 post，或起点/`subfolder` 串错，都会明显掉质量。
4. **训练步数 / LoRA 容量**：Stage 2 把 LoRA 升到 r256（`lora_alpha 256`）专门学金字塔行为，需要足够步数收敛；
   `train_batch_size 1`，步数不足时金字塔衔接学不透，低分辨率段尤其欠拟合。
5. **分辨率单一**：`single_res: true, 384×640`。金字塔三段是在这一分辨率上做 ×4 缩放；若你的目标分辨率不同，
   段几何（token 预算）会变，需重新核对。
6. **per-stage loss 监控**：`_flow_loss:64-77` 是三段平均成一个标量。建议**改成分段记录 loss**
   （stage0/1/2 各一条曲线）——这是定位「哪一段没学好」最直接的手段，几乎必做。

### 建议的排查顺序（从便宜到贵）

1. 先排除 Finding A：把 `validation_config.use_dynamic_shifting` 设 `false` 或改用 `infer_helios.py` 重渲，看「效果不好」是否消失。
2. 跑 §3.3 的 renoise 数值自检（排除 Finding C）。
3. 给训练加 **per-pyramid-stage 的 loss 曲线**，看是哪段差（指向 Finding B / D-4）。
4. 核对 Finding D 的数据/起点/两段链路（D-1/2/3）。
5. 若 stage0/1 明显更差，再考虑 Finding B 的 RoPE 归一化改造。

---

## 第三部分：附录

### 3.1 init vs post 配置差异

| | `stage_2_init` | `stage_2_post` |
|---|---|---|
| 起点 | `Helios-Base` | `Helios-Mid/transformer_init` |
| 学习率 / warmup | 1e-4 / 1000 | 3e-5 / 500 |
| `stage2_sample_ratios` | `[1,2,1]`（中段加倍） | `[1,1,1]`（均匀） |
| `is_train_lora_patch_embedding` | false | **true** |
| `is_train_lora_multi_term_memory_patchg` | false | **true** |
| seed | 45 | 46 |

两段都：LoRA r256、`restrict_self_attn:false`、`corrupt_history:true`、`is_random_drop:true`、主干 14B 冻结。

### 3.2 训练 ↔ 推理 对照（应当逐项一致才不掉质量）

| 环节 | 训练 | 推理 | 是否一致 |
|---|---|---|---|
| 段几何 / sigma 子轴 | `sigmas_per_stage`（归一化） | 同 | ✓ |
| 段间升采样（信号） | `nearest` ×2（`:929-938`） | `nearest` ×2（`:691`） | ✓ |
| renoise 公式 | 调度器 `corrected_sigma`（`:140-145`）隐含在 `start_sigmas` | `alpha/beta`（`:694-711`） | ✓（§3.3 证明） |
| renoise 噪声类型 | 下采样×2 高斯（`noise_list`） | `sample_block_noise` 块相关 | ✓（设计上等价） |
| **时间步 dynamic shifting** | **False** | 验证 **True** / 离线 **False** | **✗（验证不一致 → Finding A）** |
| RoPE：current 用自身分辨率、history 全分辨率 | 是 | 是 | ✓（一致，但见 Finding B 的风险） |

### 3.3 renoise 一致性推导（确认 Finding C 不是 bug）

> **论文依据：** 这套 = **Pyramidal Flow Matching**（Jin et al., ICLR 2025, [arXiv:2410.05954](https://arxiv.org/abs/2410.05954)）；
> renoise = 论文的 **jump point**（Campbell et al. 2023）处理。论文要求**段间概率路径连续**：第 $k$ 段结束分布 = 第 $k{+}1$ 段起始分布。
> 推理只有低分辨率终点，要变换成下一段合法起点，需同时匹配两样：
> **① 均值 → 缩放系数 $\alpha$**（论文 "rescaling coefficient $s_k/e_{k+1}$"，代码 `alpha` :696）；
> **② 协方差 → 块相关校正噪声 $\beta n'$**（论文 "corrective noise"，代码 `beta·sample_block_noise` :697,700），
> 其块内协方差带**负**非对角 $\gamma=-\tfrac13$ 以抵消 nearest 升采样的块内正相关（论文取 $\gamma=-1/3$ 最大化去相关；
> 代码 `gamma=1/3`、`cov=I(1+γ)−𝟙γ` ⇒ 非对角 $-\tfrac13$，符号约定相反、数值一致）。
> 直观：**nearest 升采样把噪声成块复制(正相关)；校正噪声用 −1/3 负相关打散回该有的样子，α 再把信号缩放到正确强度。**

设段边界全局噪声水平 `s = ori_start_sigmas[i_s]`（噪声占比）。

**训练** `start_point`（`:939`，信号是**干净** low-res 升采样）：
```
start_sigmas[i_s] = 2s/(1+s)
信号系数 = 1 - 2s/(1+s) = (1-s)/(1+s)
噪声系数 = 2s/(1+s)
```

**推理** renoise（`:694-711`，输入是上一段**停在噪声水平 s** 的输出，记 `x_s ≈ (1-s)·clean + s·ε`）：
```
o = 1-s;  alpha = 1/(2-o) = 1/(1+s);  beta = (1-o)·sqrt(3)/(2-o) = s·sqrt(3)/(1+s)
renoised = alpha·x_s + beta·block_noise
         = alpha·(1-s)·clean + [alpha·s·ε + beta·block_noise]
信号系数（对 clean）= alpha·(1-s) = (1-s)/(1+s)   ← 与训练一致 ✓
```
信号系数两边相等。噪声侧：训练用「下采样×2 高斯」、推理用「`alpha·s·ε` + 块相关 `block_noise`」，
二者的总噪声协方差由 `gamma` 设计成分布等价（这正是金字塔不出接缝的原因）。**结论：renoise 一致，非 bug。**

**数值自检脚本**（建议实跑一次以排除你这套配置下的意外）：对同一 `clean`，
比较「训练 `prepare_stage2_clean_input` 在某段最高 timestep 处的 `start_point`」与
「推理 `stage2_sample` 进入该段 renoise 后的 `latents`」的**逐通道均值/方差**应当吻合（允许蒙特卡洛误差）。

### 3.5 数值 worked example（§1.2 几何 + §1.3 训练构造，带 LaTeX）

> 设定：$384\times640$、单块 9 latent 帧；VAE 空间 8× → latent $48\times80$；金字塔三段空间 $\times\tfrac12$ 递减
> （$48\times80\to24\times40\to12\times20$）。超参 $\gamma=\tfrac13$、`stage_range`$=[0,\tfrac13,\tfrac23,1]$、`shift`$=1$。

**(A) 调度器几何。** 全局轴 $\sigma_k\approx 1-\tfrac{k}{1000}$（$\sigma$=噪声占比）。边界原始噪声水平
$s_0=0.999,\ s_1=0.667,\ s_2=0.333,\ s_3=0$。终点 $\sigma^{\text{end}}_i=s_{i+1}$；起点修正（$a=\sqrt{1+1/\gamma}=2$）：

$$\sigma^{\text{start}}_i=\frac{a\,s_i}{a\,s_i+(1-s_i)}\xrightarrow{\gamma=1/3}\frac{2s_i}{1+s_i},\qquad \sigma^{\text{start}}_0=s_0\approx1.$$

| 段 $i$ | 分辨率 | $s_i$ | $\sigma^{\text{start}}_i$ | $\sigma^{\text{end}}_i$ | 全局 timestep 区间 |
|---|---|---|---|---|---|
| 0 | $12\times20$ | 0.999 | 0.999 | 0.667 | ~999→743 |
| 1 | $24\times40$ | 0.667 | **0.800** | 0.333 | ~743→384 |
| 2 | $48\times80$ | 0.333 | **0.500** | 0.000 | ~384→1 |

修正使 $\sigma^{\text{start}}_i>s_i$（0.667→0.8、0.333→0.5）：补偿 `nearest` 升采样引入的 $2\times2$ 块相关结构（方差因子 $a$）。

**(B) 训练构造（走 stage 1）。** 干净金字塔 $x^{(0)},x^{(1)},x^{(2)}$（`bilinear`↓2）；噪声金字塔
$\epsilon^{(i)}=2\cdot\mathrm{down}_2(\epsilon^{(i+1)})$。该段 start/end（$\tilde x^{(0)}=\mathrm{up}_2(x^{(0)})$）：

$$p^{\text{start}}_1=0.8\,\epsilon^{(1)}+0.2\,\tilde x^{(0)},\qquad p^{\text{end}}_1=0.333\,\epsilon^{(1)}+0.667\,x^{(1)}.$$

- $p^{\text{end}}$ = 本段流向的「交接点」(中分辨率、约 2/3 干净)，交给 stage 2；只有**末段**才到全干净。
- $p^{\text{start}}$ 用 **$\tilde x^{(0)}=\mathrm{up}_2(x^{(0)})$(升采样的 stage-0 干净) 而非 $x^{(1)}$** —— 因为推理时 stage 1 的输入只能来自
  **stage 0 输出(低分辨率)升采样**，训练必须模仿这一可得信息（这正是 §3.3 一致性的前提）。
- 该段是 latent 空间里 $p^{\text{end}}(\sigma{=}0)\!\to\!p^{\text{start}}(\sigma{=}1)$ 的**直线**，故速度 $v=dx/d\sigma=p^{\text{start}}-p^{\text{end}}$
  **沿整段恒定**(与 $\sigma$ 无关)——模型只需学「本段从干净指向噪声的方向」。

段内抽 $\sigma=0.6$（对应全局 $t\approx600$），合成（`utils_helios_base.py:1000,1006`）：

$$x_t=\sigma\,p^{\text{start}}_1+(1-\sigma)\,p^{\text{end}}_1=0.613\,\epsilon^{(1)}+0.12\,\tilde x^{(0)}+0.267\,x^{(1)},$$
$$v_{\text{target}}=p^{\text{start}}_1-p^{\text{end}}_1=0.467\,\epsilon^{(1)}+0.2\,\tilde x^{(0)}-0.667\,x^{(1)}\quad(\text{与 }\sigma\text{ 无关}).$$

损失 $\mathcal L=\|f_\theta(x_t,t,\text{history})-v_{\text{target}}\|_2^2$。三段拼接：
$[1.0\!\to\!0.667]_{12\times20}\xrightarrow{\uparrow2+\text{renoise}}[0.8\!\to\!0.333]_{24\times40}\xrightarrow{\uparrow2+\text{renoise}}[0.5\!\to\!0]_{48\times80}$。

### 3.6 NaViT 打包 worked example（§1.4 全流程，一步不省）

> 设定同 §3.5：batch=1，9 latent 帧、全分辨率 latent $48\times80$、16 通道，三段金字塔，`stage2_sample_ratios=[1,2,1]`；
> `patch_size=(1,2,2)`，token 维 inner_dim=5120；`patch_short/mid/long`=(1,2,2)/(2,4,4)/(4,8,8)。

**三个前置概念。**
- **token**：latent 块 patchify(把 帧×高×宽 切小块)后变成一串小块，每块 = 1 token，用 5120 维向量表示 → 张量 `[batch, N, 5120]`，N=序列长度。
- **打包(packing)**：想用一次前向同时训 3 个分辨率；但它们长度差很大(540 vs 8640)，padding 到最长太浪费。NaViT = 把多条序列首尾接成一条，再让注意力"别跨界互看"。
- **`cu_seqlens`**：那条长序列里各段的分界点；注意力靠它做"块对角"，实现段间隔离且无 padding。

**Token 数（本例）。** 当前块过 `patch_embedding`(1,2,2)：

| 段 | latent | ÷(1,2,2) | token |
|---|---|---|---|
| $s_2$ 全 | $9\times48\times80$ | $9\times24\times40$ | **8640** |
| $s_1$ 中 | $9\times24\times40$ | $9\times12\times20$ | **2160** |
| $s_0$ 粗 | $9\times12\times20$ | $9\times6\times10$ | **540** |

历史(全分辨率，patchify 一次)：short(`x0`+最近1帧=2帧,(1,2,2))=$2\times24\times40$=**1920**，mid(2帧,(2,4,4))=$1\times12\times20$=**240**，long(16帧,(4,8,8))=$4\times6\times10$=**240** → `history_context_length=2400`。
`ratios=[1,2,1]` ⇒ 中段抽两份 → 当前块 list $[s_0,s_1^a,s_1^b,s_2]$，token $[540,2160,2160,8640]$，合计 13500。

**STEP 1 — patchify 成 token 张量**（都 `[1,N,5120]`）：当前块 s2/s1a/s1b/s0；历史 H_long/H_mid/H_short(拼成 H=`[1,2400,5120]`)。每 token 按**自身分辨率**坐标算 RoPE。

**STEP 2 — 先拼成"一条"**（`process_input_hidden_states:1145-1256`，历史只有一份）：
```
[ H_long240 | H_mid240 | H_short1920 | s2 8640 | s1b2160 | s1a2160 | s0 540 ]
└────────── 历史 2400 ──────────┘ └──────────── 金字塔 13500 ──────────┘   → [1,15900,5120]
```

**STEP 3 — 把历史复制给每一段**（`forward:1382-1429`，历史变成 4 份）：切出历史(2400)与金字塔(13500)；对每段在前面粘一份新历史；再首尾接成一条：
```
seg0=[历史2400 | s2 8640]=11040   seg1=[历史2400 | s1b]=4560
seg2=[历史2400 | s1a]=4560        seg3=[历史2400 | s0 540]=2940
拼平 → [1, 23100, 5120]   (11040+4560+4560+2940)
```
> **为何复制 4 份**：这 4 段必须互相独立(同内容不同噪声/分辨率，互相 attend 会串扰)；独立序列无法共享历史，故每段各带一份。代价=历史 token ×4 显存。

**STEP 4 — `cu_seqlens` 块对角注意力**（`create_navit_attention_masks`+`attn_varlen_func`，`restrict_self_attn=false`）：
```
cu_seqlens = [0, 11040, 15600, 20160, 23100]
flash varlen 只在每区间内部算注意力：seg0/1/2/3 各自闭环，段间零泄漏 = 块对角
```
等价于跑 4 次独立 self-attention，但一次调用、无 padding(pad 到 max=4×11040=44160，~2× 浪费；varlen 仅 23100)。

**STEP 5 — 收尾（三件事，展开）**。手上是 `[1,23100,5120]`，4 段、每段 `[历史2400 | 当前块]`。

*5a. 每 token 配 timestep（`forward:1350-1429`）。* 传入 `timestep` 是 list `[t0,t1a,t1b,t2]`，`forward` 先 `timestep[::-1]`=`[t2,t1b,t1a,t0]` 对齐段序。造两种 embedding：**历史 token → timestep 0**(`zero_history_timestep`，`temb_t0` 扩到 2400)；**每段当前块 → 该段的 t**(循环算 `cur_temb`)。每段 temb=`[t0 历史2400 | t 当前cur]`，4 段拼成 23100、逐 token 对齐。用途：每个 block 里做 **AdaLN 调制**(scale/shift/gate)——历史按 t=0(当干净参考)、当前按该段 t。

*5b. cross-attn 剥历史→只当前块看文本→插回（block `:822-864`）。* 历史是过去、不该被当前 prompt 重新调。故每个 block：① 逐段剥掉历史(每段 2400)，把 4 段当前块拼成 `[1,13500,5120]`；② 这些当前 token 对文本做 cross-attn(mask 按段分组，段间隔离)；③ 把历史**原样插回**(历史不被文本调制)。

*5c. 输出解包（`forward:1505-1543`+`_flow_loss:64-77`）。* final norm 后逐段拆(序 $[s_2,s_1^b,s_1^a,s_0]$)：① 丢掉该段前 2400 历史，只留当前 cur 个 token(只预测当前块)；② `proj_out`:`5120→p_t·p_h·p_w·16=64`；③ **unpatchify** 摆回 latent，如 $s_2$ 的 8640=$9\times24\times40$→`[1,16,9,48,80]`，$s_0$ 的 540→`[1,16,9,12,20]`。得 $[s_2,s_1^b,s_1^a,s_0]$ → `[::-1]` 还原 $[s_0,s_1^a,s_1^b,s_2]$ 与 `targets_list` 对齐 → 逐段算加权 MSE(target=$p^{\text{start}}-p^{\text{end}}$)→ **3 段求平均**得标量 loss 反传。

### 3.4 关键 file:line 索引

| 内容 | 路径 |
|---|---|
| 金字塔调度器 / corrected_sigma | `helios/scheduler/scheduling_helios.py:118, 140-145, 216-260` |
| 训练噪声/目标构造 | `helios/utils/utils_helios_base.py:868-1008`（`prepare_stage2_clean_input`） |
| NaViT 返回打包 | `helios/utils/utils_helios_base.py:1055-1064` |
| flow loss（逐段） | `helios/utils/utils_helios_base.py:20, 45-77` |
| 训练主循环调用 | `train_helios.py:1314-1384` |
| NaViT 前向（打包/解包/掩码） | `helios/modules/transformer_helios.py:1145-1175, 1317-1543` |
| NaViT 注意力掩码 | `helios/modules/helios_kernels/attention_dispatch.py:46-119` |
| 推理金字塔采样 / renoise / block noise | `helios/pipelines/pipeline_helios.py:617-834, 443-481` |
| 验证 dynamic_shifting（Finding A） | `train_helios.py:2170, 2383`；断言 `2548-2552`；`get_config_value` `helios/utils/utils_base.py:37-43` |

---

## 第四部分：长视频 ~10s 后整体退化 / 渐变马赛克 的成因（原生 Helios-Mid）

> 观察：用**原生 Helios-Mid**(未微调) 跑长视频(我们这里 ~91s / 2178 帧 / 66 个 chunk)，
> 大约**十几秒(~7 个 chunk)后整段开始严重退化，并逐渐塌成马赛克/块状**。
> 结论先行：**这是自回归长视频的「误差累积 / exposure bias(暴露偏差)」**，不是某个静态 bug；
> Stage 2(Mid) 只有很弱的手工抗漂移，且从不在自身 rollout 上训练，所以无法纠正自己累积的误差；
> 金字塔的「粗分辨率段 + renoise 块噪声」会把这种漂移**放大成字面意义的马赛克**。
> 这也正是 **Helios-Distilled(Stage 3) 应当明显更耐长**的原因(见下，也是我们这次对照实验要验证的)。

### 4.1 时间线对应

24fps、每 chunk = 33 RGB 帧 = **1.375s/chunk**。「~10 秒」≈ **第 7 个 chunk** 左右开始崩——
即自回归走了约 7 步后误差累积越过临界点。91s = **66 个 chunk**，对漂移是极端考验。

### 4.2 主因：自回归暴露偏差（误差累积）

- **训练只见「干净历史」，推理喂「自生成历史」。** Stage 2 训练时预测第 N 块用的是
  **真实视频 VAE 编码出来的干净历史**(`dataloader_history_latents_dist.py` 预编码 latent)；
  推理时历史是**模型自己上几块的输出**，带误差。模型**从没在训练里见过自己的不完美输出当历史**
  → 条件分布偏移 → 每块的小误差进入下一块历史 → **逐块复利式累积**。约 7 块后历史 latent 已漂出训练分布，
  模型产出越来越 OOD 的 latent，VAE 解码 OOD latent 就表现为**块状/马赛克**。

- **Helios 的抗漂移(Easy Anti-Drifting = `corrupt_history`)是「弱且不对症」的。** 它对历史加**高斯噪声/降采样**
  (`utils_helios_base.py:265-`，prob 0.9)让模型对「带噪历史」鲁棒，但这种**手工噪声 ≠ 推理时真实的误差分布**
  (后者是内容相关、跨块自相关、且会复利累积的)。所以只能部分缩小 gap，**无法建模复利漂移**。

- **Mid 从不在多块 rollout 上训练。** 真正能治漂移的是「在模型自己的多步 rollout 上训练、让它学会纠正自身累积误差」——
  这套**只在 Stage 3(Distilled) 引入**：`num_rollout_sections`(`sample_dynamic_dmd_num_latent_sections`)、
  `is_use_gt_history`(GT 历史 vs 自生成历史混合)、以及 **DMD2 分布匹配**(把学生的**输出分布**拉回教师分布，直接对冲漂移)。
  **Stage 2 一个都没有** → 这就是 Mid 比 Distilled 更容易长程崩塌的根本差异。

### 4.3 金字塔特有的放大器（为什么是「马赛克」，且只在 Mid 这么明显）

1. **粗分辨率段最先吃到漂移。** 金字塔最低分辨率段(i_s=0，1/4 分辨率)负责从历史决定**全局粗结构**。
   一旦(已漂移的)历史在粗段就把结构带偏，`nearest` 升采样 + renoise 会把这个错误**往上抬**，高分辨率段无力挽回——
   而粗段主导全局结构，所以**崩得又快又整体**。
2. **renoise 的块相关噪声在 OOD 下不再抵消。** 段间衔接靠「nearest 升采样 + 2×2 块相关 `sample_block_noise`」，
   正常时块结构会被后续去噪抵消；可一旦 latent OOD，这层 2×2 块相关**不再被干净抵消** → **直接呈现马赛克/块缝**
   (与观察到的「马赛克」精确吻合)。
3. **RoPE 尺度错配(本文 Finding B)在漂移下更糟。** 低分辨率 current 块与全分辨率历史的位置坐标本就错配，
   漂移时粗段的「按位置抄历史」更不可靠。

### 4.4 次因（叠加，但非主因）

- **记忆视野有限且有损：** 多期记忆只留 `[16,2,1]`=19 latent 帧、远期被 4×8×8 强池化；高频细节与稳定全局锚点
  逐步丢失。`x0`(首帧)锚点有帮助，但单锚点压不住 66 块的缓慢色彩/纹理漂移；当**近期历史本身被污染**时，没有干净参照可重锚。
- **CFG + 长程：** Mid 用 guidance 5.0，高引导在长 rollout 上会逐块**过饱和/放大伪影**；Distilled CFG-free(1.0)更温和。
- **极端 horizon：** 66 个 chunk 远超常见 demo 长度，漂移随 horizon 指数式恶化。

### 4.5 需排除的「假退化」（先确认不是这些）

1. **dynamic shifting 不一致(Finding A)：** 若你看的是 W&B 验证或误开了 `--use_dynamic_shifting`，会叠加 off-distribution 退化。
   离线 `infer_helios.py` 默认不开，OK。
2. **VAE tiling/decode：** 确认不是分块解码的接缝(本仓库 `enable_tiling:false`)。
3. **精度：** bf16 长 rollout 累积误差——可试 fp32 VAE(`upcast_vae:true` 已开)。

### 4.6 这正是本次对照实验的意义

跑**原生 Mid vs 原生 Distilled** 同 prompt、同 91s：若 **Distilled 明显比 Mid 更耐长(更晚、更轻地退化)**，
就**实证**了「漂移的真正解药是 Stage 3 的 rollout 训练 + GT-history + DMD 分布匹配」，而 Mid 的 `corrupt_history`
不足以撑长程。若 **Distilled 也很快崩**，则问题更偏向我们的推理设置(dynamic shifting / 解码 / horizon)或这套权重本身，
需回到 §4.5 排除。

> 启发(给后续 multi-event / 续训用)：要做长视频/流式，**抗漂移必须靠 rollout 级训练**(self-forcing / DMD / GT-history 混合)，
> 单靠对历史加手工噪声(`corrupt_history`)不够。这条直接关系到我们 Stage-4 多事件切换的长程稳定性。

### 4.7 三阶段「是否在自身 rollout 上训练」对照（关键澄清）

| | 历史来源 | 单 chunk vs 自回归 rollout | 抗漂移 |
|---|---|---|---|
| **Stage 1 (Base)** | 预编码 **GT 历史 latent**(`dataloader_history_latents_dist`) | **单 chunk，teacher forcing**(`_flow_loss`) | 仅 `corrupt_history`+`is_random_drop` |
| **Stage 2 (Mid)** | 同上(GT 历史) | **单 chunk，teacher forcing**(金字塔 navit 仍一个目标块) | 同 Stage 1 **+ 金字塔(漂移放大器)** |
| **Stage 3 (Distilled)** | DMD（gan/text 多任务） | **多段自回归 rollout**（`num_rollout_sections`，仅 **self-forcing 变体** 3→44；默认 `stage_3_post`=1） | rollout 训练 + 自生成历史 + DMD 分布匹配 |

> **更正/细化**：自回归 rollout（用自身历史）只在 `stage_3_post_self-forcing_version.yaml` 开启（段数 3→44、`is_use_gt_history:false`、cold-start）。
> **默认 `stage_3_post.yaml` 其实 sections=1 且 `is_use_gt_history:true`（GT 历史）= 单段 teacher-forcing DMD，不做 self-rollout**。详见 [`HELIOS_STAGE3_TRAINING.md`](./HELIOS_STAGE3_TRAINING.md) §0/§3。

**Stage 3 rollout 代码证据(`utils_helios_post.py`)：** `for k in range(num_rollout_sections)`(:735) 逐段生成；
每段把**自己的输出**接回历史 `history_latents = cat([history_latents, pred_x0], dim=2)`(:903)，下一段从中切历史(:780-783)；
`should_compute_grad = k >= start_gradient_section_index`(:821) 让**梯度只压靠后段**(在自身漂移历史上 rollout 后仍要产好结果 = self-forcing)。

**为什么 Stage 1 没 rollout 却像是更耐长？**(常见误解，需澄清)
Stage 1 与 Stage 2 是**同一套 teacher-forcing**(GT 历史 + 仅 `corrupt_history`)，**Stage 1 并不具备更强的抗漂移训练，它一样会漂移**。
它"更耐久"是三个结构性原因，而非训练更好：
1. **Stage 2 = Stage 1 + 金字塔，金字塔是放大器(主因)**：Stage 1 全分辨率标准 flow，无「粗段 + nearest 升采样 + 块相关 renoise」，
   OOD 下不会爆马赛克，只表现为**渐变退化**(模糊/掉饱和/内容缓慢走样)。
2. **每 chunk 50 步全分辨率 → 单块更贴流形 → 累积更慢**(Mid 金字塔有效高分辨率步数更少)。
3. **`corrupt_history` 是三段共享的地基级抗漂移**：给**有界**容忍度("撑一段")，但建模不了复利式、内容相关的真实误差，所以**所有 teacher-forced 阶段(含 Stage 1)最终都漂**。

**诚实结论：** Stage 1 不是"解决漂移"，而是"无放大器 + 每块更干净 → 漂得更慢更优雅(不爆马赛克)"；推到 66 chunk 同样会退化。
**真正从根上治漂移的只有 Stage 3 的 rollout + GT-history + DMD**——README 的"实时长视频"主要靠 Distilled 成立。
(实证建议：补跑 `Helios-Base` 同 91s 看「优雅退化 vs 马赛克」对照。)
