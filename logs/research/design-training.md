# Helios 演化记忆(Evolving Memory)训练课程与 Trainer 改造设计（修订版）

修订说明：已逐条复核验证器报告并实读全部引用代码；blocker/major 均已通过修改设计解决。验证器唯一被驳回的子论断见脚注 [^1]。

## 0. 代码核验结论与先验偏差标记

以下与先验/上游报告不一致处，以代码为准：

- **偏差 F1（先验 #5）**：Echo **不存在**"静态 query 先行"或 teacher-forcing 预阶段。证据是配置与入口：两份已发布训练配置均从第一步启用 memory（`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:70-88`、`configs/echo_infinity-long.yaml:88-125`），README 只有两个 DMD 阶段（`README.md:123-143`）。另注意：`query_init` 在首次驱逐写入前是**潜伏**的——`get_kv()` 在 `has_history` 置位前返回 `None`（`model/query_memory.py:194-196`；置位在 :190），注意力注入也随之跳过（`wan/modules/causal_model.py:726-739`）。我们的 Stage A 是 Helios 数据通路强加的**新增**阶段，不是移植 Echo 课程。
- **偏差 F2**：`dmd_teacher_forcing` 是**损失侧死旗标但数据侧活跃**：它仅控制 dataset 的 `return_all_vae_latent`（`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:882-885`），dataloader 产出 `clean_all_latents`（`helios/dataset/dataloader_history_latents_dist.py:365-379`）但 trainer 热循环无任何消费者（:1343-1384 只取 prompt/history/target/x0 后 del batch）。它仍会改变切片、可 raise `Not enough sections`（dataloader:180-182）并增加 I/O。**不得复用该旗标名**承载新语义。
- **偏差 F3（修正表述）**：Python `random` 已被 `set_seed(args.seed)` 全局播种（`train_helios.py:230-232`），"未播种"说法不成立；真实缺陷是 `start_section_idx` 用全局 `random.randint`（dataloader:183）采样，与逐样本确定性的 `choice_idx`（seeded generator，:169-172）**解耦**，且输出 dict 不含 `start_section_idx`（:365-384），而 `prompt_embeds` 只绑定 `choice_idx` 的事件（:377）。修复必须是**返回采样到的起始索引并据其派生逐 section prompt**（见 D6），不是仅换 RNG 实现。
- **偏差 F4（修正表述）**：`use_deepspeed_optimizer` 仅当 DS json 含 `optimizer` 块才为真（`train_helios.py:820-823`），而本谱系使用的 `scripts/accelerate_configs/zero2.json` 无 optimizer/scheduler 块（全文仅 fp16/bf16/zero_optimization）——即使 DMD 双 DeepSpeed plugin（:154-165，Stage C 父配置 :230-231 指定 zero2.json）下，`torch.optim.AdamW` 也照常收到 param groups。真正的两个杀手是：(a) 含 optimizer/scheduler 块的 DS 配置（`get_optimizer` DummyOptim 只传全局 lr，`helios/utils/utils_base.py:76-87`）；(b) **`HELIOS_FORCE_LR=1` resume 覆写**把所有 param group 的 lr 与 scheduler `base_lrs` 压成单一标量（`train_helios.py:1086-1099`）——而所选 Stage A/B 祖先配置**明文依赖** FORCE_LR（`scripts/training/configs/stage1_lora_cfr_368_correct.yaml:95-97, 105`）。守卫见 D11/D13。
- 其余核验通过的关键承载点：可训练模块子串匹配 `train_helios.py:463-477`；optimizer 单 param-group :815-829；save/load hook :648-695/:696-768；`save_extra_components` 白名单 `helios/utils/utils_base.py:157-259`；Stage-3 rollout 段数采样与广播 :1571-1598、`USE_GT_HIST` 按比率抽签 :1561-1564；GT 模式单 section 断言 `helios/utils/utils_helios_post.py:2349`（generator）与 :2966（critic）；Echo gate 更新式 `query_memory.py:152-155`、`bptt_clips` 逐 chunk detach `streaming_training.py:212-217`。

## 1. 三阶段课程 → 配置体系映射

**D1 配置谱系分叉**：Stage A/B 从 `scripts/training/configs/stage1_lora_cfr_368_correct.yaml` 分叉（继承：r128 LoRA :44-51；LR/`constant_with_warmup`/步数 :95-107；I2V-drop 0.1 :142-146；corrupt random 0.8889 :156-166；saturation :167-171），命名 `stage1_lora_mem368_A.yaml` / `..._B.yaml`；Stage C 从 `stage_3_post_self-forcing_version.yaml` 分叉为 `stage3_mem_sf.yaml`。*弃选：从 `stage_1_init.yaml` 分叉——面向裸 Wan2.1 初始化且缺 correct 三项修正，与已训 lora368 检查点谱系断裂。*

**D2 新增 flags 与"单一完整构造字典"原则**：所有**影响参数形状的字段**（不止一个 bool）必须进 transformer 构造 kwargs 并注册进模型 config，且**同一份完整字典**流经全部重建路径：generator kwargs（`train_helios.py:310-325`）、DS resume 重建（:718-730）、EMA 重建（`helios/utils/create_ema_zero3_lora.py:37-44, 116-124`）、merge 工具（D12）、推理构造（D12）。critic 是独立 kwargs 字典（:334-347）——按 D8 决策 critic **不构造** memory 模块。另：`from_pretrained` 会**静默跳过**尺寸不匹配的键（`helios/modules/transformer_helios.py:1767-1776`），故加载后必须显式断言所有 `evolving_memory.*` 键已加载，禁止依赖宽容加载吞掉 N_Q sweep 的形状差异。

```yaml
# ---- Evolving Memory (arch-affecting keys go into transformer config) ----
is_enable_evolving_memory: false
memory_num_query_frames: 3        # N_Q; M = N_Q*920 tokens (fixed, full-res token density)
memory_enc_num_layers: 2
memory_gate_init_bias: 0.75       # §7, INFERRED heuristic; sweep {0.75, 2.0}
is_amplify_memory: false          # memory 自有 amp 参数（模块设计任务）
# ---- training-only ----
is_train_memory_module: false     # requires_grad 开关
memory_freeze_backbone: false     # Stage A: 冻结 LoRA/patch_*，只训 memory（D10）
memory_learning_rate: 5.0e-5
memory_bptt_sections: 1
memory_write_source: "t0_hidden"  # {t0_hidden, none}
memory_single_write_prob: 0.0
memory_tf_unroll: false
memory_unroll_sections: 1         # U; 传入 dataset num_rollout_sections（D6）
```

**D3 各阶段训练对象 / LR / 步数**：
- **Stage A**（TF+静态 query，含单写模拟）：只训 `evolving_memory.*`，backbone LoRA 与 patch_* 冻结（经 `memory_freeze_backbone`，D10）；memory LR `5e-5`、约 4k 步、gb 32（1 节点）。*弃选：Echo 式从头联合训练——lora368 已在无记忆分布上收敛，联合训练会让 LoRA 为未成熟记忆读数买单。*
- **Stage B**（TF unroll）：联合训练，LoRA `1e-5` + memory `5e-5`，U=4，`memory_bptt_sections=1`，约 3k 步、gb 32-64。
- **Stage C**（SF/DMD 校准）：generator LoRA `1e-5`、memory 降为 `2e-5`；critic **沿用父配置显式值 `critic_learning_rate: 4.0e-7`**（`stage_3_post_self-forcing_version.yaml:232`；dataclass 默认 2e-6 被父配置覆盖，不得误标为"沿用默认"）；约 2k 生成器步（`dfake_gen_update_ratio: 5`，yaml:233，即 10k trainer 迭代）。

## 2. Stage A：GT history 下的驱逐模拟与写入路径

**D4 驱逐帧的精确定义（已按 19 帧零前缀修正）**：时间线 `continue_source_latent = [19 zeros] + real frames`（dataloader:148-156）；目标 `choice_idx=k` 的 history 为 `[9k, 9k+19)`（:188-191）。由 k-1 推进到 k 时滚出的 9 帧是 `continue[9(k-1), 9k)`——即 k-1 段 history 的**最老 9 帧**，且因 19=2×9+1 的前缀偏移**永不与 chunk 边界对齐**。逐 k 情形（与 `docs/STAGE1_DATALOADER_FRAMES.md:92-103, 204-211` 的推演一致）：k=1 全零、**k=2 仍全零**（[9,18) 落在零区）、k=3 混合（1 零 + 8 实帧）、k≥4 全实。因此：(a) dataset 增加 `return_evicted_latent: true`，返回 `evicted_latent = continue[9(k-1) : 9k)`（9 帧）、其前置 19 帧切片（k=3 时左端越界 1 帧，左侧补零）、以及整型 `evicted_valid_frames = min(9, max(0, 9k-19))`；(b) 写入判定用 `evicted_valid_frames`：==0（k≤2）跳过写；1-8（仅 k=3）做**带 frame mask 的部分写**（Enc 只消费有效帧 token）；==9 全写。**禁止**硬编码 `choice_idx<=1` 边界。无需改 `tools/offload_data`。*弃选：离线预算写入特征——写源是模型 hidden state，随权重演化，离线缓存必然陈旧。弃选：只允许完整写（要求 k≥4）——白白丢弃 k=3 的首次写监督，且推理期首次驱逐同样是错位的。*

**D5 写源前向（需新增模型 API）**：TF 下没有去噪循环。以驱逐 9 帧为 X_Noisy、timestep=0、携带其前置 19 帧 history 做**一次 `no_grad` 干净前向**，取**最后一层、`norm_out`/`proj_out` 之前的目标 token hidden**。现有 forward 在 :1548 切掉 history token、:1550 投影后只返回 `(output, logits)`（`transformer_helios.py:1575-1578`）——**必须新增 `return_final_hidden=True` 接口**（在 :1548 处捕获 `hidden_states[:, -original_context_length:, :]` 返回），这是显式实现任务，不是现成集成点。形状：368×640 全分辨率桶下为 `[B, 8280, 5120]`（8280 = 9 帧 × 920 tokens/帧，920 = 23×40，`docs/STAGE1_DATALOADER_FRAMES.md:184-185`）[^1]；但 `single_res` 同时保留全/半/四分之一三种分辨率桶（dataloader:107-113），故 **Enc 必须接受可变长写源序列**，H/W 按桶推导；memory 状态 M = N_Q×920 为固定 token 数（N_Q=3 → 2760），与桶无关。梯度只经"写→状态→读→flow loss"进入 Enc/gate。此处与 Echo 的关系是**梯度阻断上的类比而非机制同构**：Echo 的 recache 前向确在 no_grad 下（`pipeline/self_forcing_training.py:171-176`），但其 update 消费的是持久 KV cache 中被驱逐的末 block K/V（`causal_model.py:822-828`），并非返回的 hidden——Helios 的 last-hidden→Enc 写路径是**新设计**。以 `memory_single_write_prob≈0.5` 混合"纯静态读"与"单写后读"样本。*弃选：patchify/layer-0 嵌入作写源——太浅（先验 #1）；弃选：当前 step 噪声前向中 history token 的深层 hidden——联合双向注意力污染（先验 #1）。*

## 3. Stage B：trainer 内多 section TF 展开

**D6 数据改造**：`memory_tf_unroll=true` 时：(a) trainer 向 dataset 构造传 `num_rollout_sections=memory_unroll_sections`——当前 `dataset_kwargs` 完全不传该参（`train_helios.py:875-890`），构造默认 3（dataloader:20），U=4 需要的 all-latent 切片为 19+9U=55 帧（:184-186），默认值下只有 46 帧、必然错配；(b) 输出 dict 增加 `start_section_idx` 与逐 section `prompt_embeds` 列表（复用 `_pick_prompt_embed(feature_data, start_section_idx+u)`，:282-303）；(c) 起始索引改用与 `choice_idx` 同纪律的 seeded generator（:169-172），并同时返回之（修复 F3 的对齐缺陷）；(d) 索引期过滤 `total_sections < U` 的样本，且 metadata cache（每 folder 的 `dataset_cache.pkl`，:52-64）**按 U 失效重建**；(e) 单元测试 `clean_all_latents` 形状恰为 `[B,16,19+9U,H,W]`。U=4 需 ≥4×33=132 可用 RGB 帧（离线编码器按 33 帧/chunk 切，`tools/offload_data/get_short-latents.py:214-232`），但 metadata 只保证 `num_frame>=121`（dataloader:102-104），**语料保留率必须先跑离线直方图扫描再做预算**（"绝大多数满足"不可从代码验证）。

**D7 展开循环、backward 重构与 detach 语义（三处修正）**：

```python
# NOTE: _flow_loss currently backwards internally (utils_helios_base.py:107).
# Refactor: add a section-level primitive that RETURNS the scalar loss
# (no internal backward); the outer accelerator.accumulate scope
# (train_helios.py:1484-1486) keeps ownership of sync boundaries.
state = mem.init_state(B)                       # from query_init, per-sample
for u in range(U):                              # forward in time
    hist = slice_tiers(clean_all, u)            # 16/2/1 + anchor
    hist = corrupt_history_latents(hist)        # existing anti-drift
    sync_this = accelerator.sync_gradients and (u == U - 1)
    ctx = nullcontext() if sync_this else accelerator.no_sync(transformer)
    with ctx:
        loss_u = flow_loss_section(..., memory_state=state) / U
        accelerator.backward(loss_u)            # frees activations per section
    # BPTT truncation happens BEFORE the next write, AFTER this read's
    # backward, so update(u) stays in graph until read(u+1) consumes it.
    if (u + 1) % memory_bptt_sections == 0:
        state = state.detach()
    if u + 1 < U:
        ev, n_valid = evicted(u + 1)            # slice [9j, 9j+9), D4 semantics
        if n_valid > 0:
            with torch.no_grad():
                h = transformer(ev, t=0, ..., return_final_hidden=True)
            state = mem.update(state, h, frame_mask(n_valid))  # grad kept
optimizer.step()
```

- **detach 次序（关键修正）**：detach 施加在**下一次写之前、本次读的 backward 之后**。若照初稿"update 后立即 detach"，`bptt_sections=1` 时 read(u+1) 只见 detach 后的状态，Enc/gate **永远收不到梯度**（写路径无学习信号）。修正后 `bptt_sections=1` 精确对应"Enc 梯度仅来自紧邻下一 section 的读"，语义对齐 Echo 的 `detach_state()`（`streaming_training.py:212-217`）。**验收：U=2、bptt=1 下单测 Enc/gate 梯度非零。**
- **backward/DDP 语义与显存均标记为实现假设**：现 trainer 无内嵌 `no_sync` 多次 backward 先例；祖先配置 `gradient_accumulation_steps: 2`（yaml:95-97）意味着内层 sync 判定必须与外层 `accelerator.accumulate` 的 `sync_gradients` 合取（如伪码）。显存推断锚点是 bs4 单 section ~90 GiB（`docs/STAGE1_SPEED_OPTIMIZATION.md:83-98`），bs2×U=4 在 H200 141GB 可行是**未测假设**——落地前先跑 1 节点 smoke profile。*弃选：U 段 loss 求和后一次 backward——激活 ×U 存留，必 OOM。*
- **DDP unused params**：写路径条件触发，保持 `HELIOS_DDP_FIND_UNUSED` 默认 true，或按 zero-touch 技巧（`utils_helios_base.py:100-105`）补零梯度。

## 4. Stage C：SF/DMD 末端校准

**D8 集成点（按实际控制流重写）**：入口是 `consistency_backward_simulation`（`helios/utils/utils_helios_post.py:1403-1498`），在 `is_enable_stage2=true`（父配置 :206）时分派到 `inference_with_trajectory_stage2`（:947-1400）；Stage-1 分支是死代码（:679 无条件 `raise NotImplementedError`）。**本设计将 Stage C 显式绑定 Stage-2 金字塔路径，不新开 Stage-1 rollout 实现任务。**具体机制：

1. **写源不能是"被选中的 exit 前向"**：exit timestep 是随机采样的 `denoising_step_list[init_exit_flag]`（每个金字塔 stage 各有独立 flag，:991-995、:1257），并非 t≈0；且所有 transformer 调用只取 `[0]`（flow pred，:1264-1324），不存在 hidden 返回。**改为**：在最终金字塔 stage 的 `pred_x0` 拼入 `history_latents`（:1350）之后，对该全分辨率 `pred_x0` 追加**一次 `no_grad` timestep=0 recache 前向**（携带该 section 当时的三层 history，`return_final_hidden=True`），输出喂 Enc——与 Stage A/B 写源分布一致（t=0、干净输入）。代价：每 section +1 前向。
2. **梯度结构**：前缀 section（`should_compute_grad=False`，:1160；exit 前向在 `torch.set_grad_enabled` 下随之关闭，:1300）的 `mem.update` 也置于 `no_grad` 且状态保持 detach；梯度后缀 section 的 update 留在图内，按 `memory_bptt_sections` 以 D7 的"写前 detach"次序截断。读侧梯度来自后缀 section 带 grad 的 exit 前向：DMD 伪 loss 经 `original_latent` 反传（:2044-2053）→ transformer → memory K/V → update 图 → Enc/gate。**验收：两 section 后缀下单测 Enc/gate 梯度非零。**
3. **状态所有权（修正为显式局部对象）**：memory state 作为 rollout 函数的**显式入参/返回值张量**，严禁持久 module 属性——generator-loss rollout（`train_helios.py:1653-1750`）与 critic-loss rollout（:1842-1916，其内部 generator unroll 全程 `torch.no_grad`，`utils_helios_post.py:2978-3023`）各自从 `query_init` 独立 reset，互不污染。`dmd_is_low_vram_mode: true`（yaml:222）下 state 张量随 generator 的设备搬移同步 `.to()`，加回归测试。
4. **score 模型保持 memory-blind（显式设计决策）**：DMD real/fake score 前向在非 GT 模式下 history 全传 `None`（:1657-1679；critic 只给裁剪出的生成片段打分，:3124-3136），且 `real_score_model` 是独立构造的冻结模型（`train_helios.py:330-353`）——它从未见过 memory 条件，强喂 memory K/V 反而破坏 real score 分布。第一期 critic/real score **均不建 memory 模块**；DMD 梯度仍会校准"memory 条件下的生成分布"。若后期 fake score 欠拟合再评估"冻结快照喂 critic"方案。*弃选：critic 同步维护 state——需 critic 侧构造模块与设备/EMA 全链路配套，现无证据必要。*
5. **critic LoRA 目标排除（新增必做）**：critic 的 target 列表复制自 generator all-linear 扫描、仅排除四个 patch 名（:517-527）；generator 的 all-linear 扫描本身发生在模块构造之后（:378-393），会把 memory 的 Linear 一并收进目标。两侧都必须过滤 `"evolving_memory" not in name` 并加入 `exclude_modules`，注入后断言不存在 `evolving_memory.*lora_*` 参数（D10/D12 的直拷语义依赖此点）。
6. **配置**：沿 `stage_3_post_self-forcing_version.yaml`，初期 `dmd_num_latent_sections_max` 44→**8**（:290-291）、`is_use_gan: false`（父配置 true，:256）。保留 `is_dmd_vae_decode: true`（:251）：其 decode 只对随机首/尾 21 帧做直通估计重建（:2424-2485）——因 score 是 memory-blind（第 4 点），无需为被评分裁剪窗维护 memory 快照。*弃选：stock `stage_3_post.yaml`（GT-history）——GT 路径被 `assert num_rollout_sections == 1` 锁死（`utils_helios_post.py:2349`、:2966），递归写入根本不被执行。*

**D9 为何 Stage C 不可省**：(a) gate 决定写幅度，其输入在 TF 下只见过 GT 统计的 t=0 hidden；推理时写源是自生成漂移分布，误差经 `g·s+(1-g)·p` 逐 section 复利——正是 Echo 批评的手写规则等价物（先验 #5）；(b) 本仓已有同类实证：I2V 训练/推理格式错配曾致真实退化（`docs/HELIOS_CORRECT_RECIPE.md:11-31`）；(c) Echo 的记忆写入在训练中始终消费自生成上下文（`causal_model.py:785-828`）——省略 Stage C 意味着移植版比原版少一个其第一步就具备的性质。

## 5. 参数组、LoRA 并行训练与持久化

**D10 模块归属与 Stage A 冻结（修正）**：`evolving_memory` 注册为 `HeliosTransformer3DModel` 顶层子模块（构造 kwargs 见 D2）。两处必改：(i) **LoRA 排除**——all-linear 目标发现会捕获 memory Linear（:378-393），按 D8-5 过滤+exclude；(ii) **Stage A 显式冻结通道**——`add_adapter` 先把 LoRA adapter 参数置为可训练（:456），其后的子串循环**只开不关**（:463-477），单靠 `trainable_modules.append("evolving_memory")` 实现不了"只训 memory"。新增：`memory_freeze_backbone=true` 时在 adapter 注入后把所有 `lora_` 与 patch_* 参数 `requires_grad=False`，再解冻 `evolving_memory`；optimizer 构造前**断言可训练参数名集合**恰为预期前缀集并落日志。*弃选：Echo 式 FSDP 外挂独立模块+手动 all-reduce（Echo `distillation.py:209-250`）——违反本 trainer 单模型 save-hook 类型断言（`train_helios.py:667-668`）与 EMA 位置序假设。*

**D11 optimizer param groups（修正）**：按**参数名**构组、名集互斥断言、**空组不构造**：

```python
mem_names = {n for n, p in transformer.named_parameters()
             if "evolving_memory" in n and p.requires_grad}
base = [p for n, p in transformer.named_parameters()
        if p.requires_grad and n not in mem_names]
mem = [p for n, p in transformer.named_parameters() if n in mem_names]
params_to_optimize = []
if base: params_to_optimize.append({"params": base, "lr": lr, "role": "base"})
if mem:  params_to_optimize.append({"params": mem, "lr": memory_lr, "role": "memory"})
```

（现源列表本就是"所有 requires_grad 参数"而非仅 LoRA，:815-818，命名构组同样覆盖。）配套：(a) 日志按 role 记录每组 LR——现 `_flow_loss` 只记 `get_last_lr()[0]`（`utils_helios_base.py:152-155`），Stage A 下组 0 就是 memory 组，索引假设必须去掉；(b) 守卫收窄（替代初稿"禁 full-FT"）：设定 `memory_learning_rate` ⇒ 运行时断言 `not use_deepspeed_optimizer and not use_deepspeed_scheduler`（:820-827）**且** `HELIOS_FORCE_LR != "1"`（:1086-1099 会压平组间比例；若必须 FORCE_LR，需先扩展该覆写为按 role 恢复各组 LR）；(c) `accelerator.prepare` 后冒烟断言两组 lr 不同。

**D12 持久化/推理/合并/EMA（从"自动覆盖"改为显式清单）**：`save_extra_components` 增第 5 节（`utils_base.py:259` `torch.save` 前）收取前缀 `evolving_memory.`，gate 条件用 **enable** 而非 train（Stage C 冻结变体也持久化）；`load_extra_components`（:260+）对称加载，且**当 checkpoint 含 memory 键而模块未构造时必须 fail loudly**。但仅此**不足**，以下为必做清单：
- **推理**：`infer_helios.py` 在 `from_pretrained` 时不传任何 additional kwargs（:238-242），`partial_path` 只人造四个既有 flag（:283-292）——需新增 CLI/配置项并**在构造期**传 memory kwargs；`pipeline_helios.py` 的整个 chunk 循环（:1205-1454）没有任何 state 传递/重置/写入调用——**推理侧状态机是独立实现任务**（每次生成 reset、去噪后每 section 恰好一次写、state 进每次 forward），归属推理设计文档。
- **merge 工具**：`tools/merge_lora_partial_for_helios.py` 构造 kwargs 硬编码无 memory 项（:58-74），`load_extra_components` 的 Namespace 同样缺 flag（:126-137）——两处补齐后，`evolving_memory.*` 直拷（全秩、无 LoRA 算术）才成立，且依赖 D8-5 的 LoRA 排除（否则 memory 的 adapter 键会被当作 base Linear delta 合并，:78-124）。
- **EMA**：覆盖**以构造同构为条件**，非"天然"。EMA `step` 是位置 zip 且长度不匹配时**静默截断**（`helios/utils/create_ema_zero3.py:245-250`）；resume/export 均从 `transformer_additional_kwargs` 重建模型（`create_ema_zero3_lora.py:37-44, 116-124`）。必做：EMA 初始化/resume 前比对双方有序 `(name, shape)` 列表、不匹配即 fail；memory kwargs 完整传入；模块设计约定**不引入需持久化的 buffer**（EMA 导出只收 named_parameters，:96-105）。deep-copy 起点（`train_helios.py:509-513`）在同构前提下成立。
- **DS resume**：从 kwargs 重建 generator/critic（:718-730），使用 D2 的同一份字典。

## 6. main() 断言与配置校验

**D13 断言块新增**（插入 :2656 "For Wan" 节附近）：`is_train_memory_module ⇒ is_enable_evolving_memory`；`is_enable_evolving_memory ⇒ is_enable_stage1 and has_multi_term_memory_patch`；`memory_tf_unroll ⇒ use_stage1_dataset and NOT use_stage3_dataset and memory_unroll_sections>=2 and not is_train_dmd and not use_error_recycling`——dataset 分派优先检查 `use_stage3_dataset`（`train_helios.py:112-129`），双真时仅断言 stage1 会静默放行错误数据路径，故**并加 dataset 选择 one-hot 断言**；`is_train_dmd and is_enable_evolving_memory ⇒ is_enable_stage2 and not is_use_gt_history`（Stage-1 rollout 死代码 + GT 单 section 断言）；`memory_bptt_sections>=1`；D11 的 DS/FORCE_LR 守卫；`validation_config.use_kv_cache ⇒ not is_enable_evolving_memory`（记忆 KV 第三组与 kv-cache 路径未验证兼容前封死）。**配置校验流程（修正表述）**：`compare_yaml.py` 的递归 diff 算法零改动（:39-58），但更新 `__main__` 默认对（:62-66，路径相对 `scripts/training`）本身就是代码改动，且**一次 A/B diff 不能保证全谱系显式含新 flag**——校验程序改为：枚举谱系内全部 YAML，逐一 OmegaConf merge 到 `Args` schema 并断言 memory 键显式存在；A/B diff 仅作人工复核辅助。

## 7. gate bias 定量换算（INFERRED 启发式，非代码推导的时间常数匹配）

名义换算：Echo causal block 3 帧（`echo_infinity.yaml:43-57`），单次保留率 `σ(2)=0.8808` → 每帧对数保留 `ln(0.8808)/3=−0.0423`；Helios 每写覆盖 9 帧 → `r=e^{−0.0423×9}=0.683` → `bias=logit(0.683)≈0.77`，取 **0.75**（状态半衰期 ≈1.8 次写 ≈16 latent 帧）。照抄 2.0 则每帧等效保留 `0.8808^{1/9}≈0.986`，周转慢约 3 倍，撞先验 #6"双稳定器冻结动态"风险。**必须标注的近似**：(a) Echo 实际按 `num_exited_frames>0` 触发更新、驱逐帧数可变（`causal_model.py:792-828`），训练 chunk 长度/overlap 也可变（`streaming_training.py:153-185`），"每 3 帧一写"只是名义节奏；(b) gate 是逐 token 逐 channel 的（`gate_linear` 输出 hidden_dim，`query_memory.py:85-87, 152-155`），`σ(bias)` 只是初始化均值。落地时实测 Helios rollout 的写节奏与初始 gate 分布，sweep `{0.75, 2.0}` 裁决。

## 8. 算力预算（全部为 INFERRED 预测，无一来自可执行基准）

可引用的实测锚点仅有：Stage-1 1 节点 bs2 throttled 13.1 s/it（`docs/STAGE1_SPEED_OPTIMIZATION.md:22`）、4 节点(32 GPU) gb64 ~28 s/it、gb128 ~45 s/it（:62-69）。Echo "+10.6% 记忆开销"**只出现在上游研究笔记，Echo 仓库内无基准佐证**，仅作数量级参考。
- **Stage A**：+1 次 no_grad 写前向 + 记忆读 ≈1.5× → ~20 s/it，1 节点×8，4k 步 ≈ 22h / ~180 GPU·h。
- **Stage B**（U=4）：≈4×(grad 前向+反传)+4×no_grad 前向 → 1 节点 bs2 ~70 s/it；2 节点 ≈40 s/it，3k 步 ≈ 33h / ~530 GPU·h。
- **Stage C**：rollout（3-8 段，后缀带 grad）+ 14B real score 双 pass + 14B critic + 每 section 1 次 recache 写前向，估 3-4× Stage-1 单步；4 节点×8，10k trainer 迭代 ≈ 3-4 天 / ~2.5k GPU·h。
- 含 N_Q{3,6}×bias{0.75,2.0} 缩减 sweep，总额 ≈ **4.5-5.5k H200·h**。参照系（同为投影）：一次 lora368_correct 续训 5000 步、gb160=5 节点×40 GPU（yaml:95-97），以 4 节点/32 GPU 实测 ~45 s/it 外推 ≈2-2.5k GPU·h——本课程约为其 2 倍。**所有乘数在 Stage A/B smoke profile（D7）后修订。**

---
[^1]: 验证器称 "9×920 与 8,280 不符"——算术上二者恒等：9×920=8280，920=23×40 为 368×640 全分辨率桶每 latent 帧 token 数（`docs/STAGE1_DATALOADER_FRAMES.md:184-185`；patchify (1,2,2)，`transformer_helios.py:982-1024`）。该 issue 的实质部分（现 forward 无 hidden 返回、多分辨率桶导致形状可变）成立，均已纳入 D5。
