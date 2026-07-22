# 设计任务 4/4：风险矩阵、消融矩阵与分阶段 Kill Criteria（含 lora368 主线协调）（修订版 v2）

前置核验声明：本节全部代码事实已在磁盘逐项复核（helios-team @ mid_training_xiangbo、Echo-Infinity）；本次修订另对 19 条 verifier issue 全量回查，均获代码证实并已吸收，无需反驳性脚注。两项关键 OBSERVED ABSENCE 置顶：
(1) **仓库内不存在任何 history 离散化/VQ 实现**——宽口径 grep `vq|vector quant|codebook|fsq` 的命中全部无关：VideoAlign 奖励维度 "VQ"（`helios/videoalign/prompt_template.py:12`、`helios/utils/train_config.py:383`、`helios/utils/utils_helios_post.py:2736-2744`、`helios/videoalign/data.py:20-43`）及调度器 docstring 中泛义 "discrete"（`helios/scheduler/scheduling_helios.py:305`）；针对 history 量化器/codebook/FSQ 的窄口径检索零实现命中，R6 按"纯规划态实验"处理；
(2) **评测侧无任何 within-video 时序斜率后端**——长评测只有 full/start15/end15 端点聚合（报表消费 `full_mean`/`drift_abs_mean`：`scripts/evaluation/make_eval_report.py:132-147`；逐视频 start/end 差值计算：`scripts/evaluation/summarize_helios_long_ratio_eval.py:92-110`），全仓库无 OLS/Theil-Sen 实现。本文所有标 [NEW] 的检测信号须先按设计节 3 §6 的 `external_command` 方案落地（`tools/long_video_eval/long_video_eval/runner.py:55-107`）才可执行——**检测手段本身是交付物，不是现货**。

## 0. R0：跨设计节机制冲突与两项 blocker 级仲裁（先于一切实验）

三份已验证设计节对同一机制给出互斥参数化：**读机制**（节 1 = token 插入、memory 获得 t=0 AdaLN；节 3 = 逐层 `to_k_mem/to_v_mem` 纯 KV 注入、AdaLN 不作用）；**M 定义**（节 1 mid-tier 390 token/帧、M=1170 @480×832；节 2 全分辨率 920 token/帧、M=2760 @368×640；节 3 960 token/帧、N_mem=2880 @384×640）；**RoPE 槽位**（节 1 (0,1) 分数 id vs 节 3 负 id）；**命名/API**（`helios_memory`/`evolving_memory`/`mem_encoder`；`capture_last_hidden`/`return_final_hidden`/`return_capture`）。

**D1（读机制/M 仲裁，写第一行实现代码前锁定）**：合成基线 = 节 1 的 token 插入读机制 + 节 2 的配置键/课程/trainer 集成 + 节 3 的 FiLM(σ_last) 写源条件化与评测协议；M 语义统一为"帧当量 N_Q × 主训练桶 mid-tier 网格"（368×640：full-res 23×40=920/帧，mid-tier 网格 12×20=240/帧，N_Q=3 → M=720），M 为固定形状（状态跨 section 持有必须定形），节 1 的 1170 与节 3 的 2880 是各自分辨率/密度下的换算，消融自变量统一为 N_Q 帧当量；RoPE 槽位交由消融 A9 裁决（分数 id 为默认）。*弃选：两种读机制并行实现再实验裁决——双份 diffusers 镜像移植成本翻倍，且 t=0 条件语义不可比；若 token 插入的 +FFN 开销实测不可接受，KV 注入是唯一回退项，由 A0 仲裁实验兜底。*

**D8（谱系仲裁，blocker——课程三阶段的父模型不同源）**：OBSERVED：Stage A/B 父系 = Helios-Base `transformer_init` + LoRA rank 128（`scripts/training/configs/stage1_lora_cfr_368_correct.yaml:34-48`），而节 2 提议的 Stage C 父配置 = Helios-Distilled `transformer_ode` + LoRA rank 256（`scripts/training/configs/stage_3_post_self-forcing_version.yaml:35-53`）；trainer 从配置路径重建 generator（`train_helios.py:310-325`）并按配置 rank 新建 adapter（`train_helios.py:397-405`），checkpoint hook 将 PEFT adapter 与 extra 全秩组件分开保存（`train_helios.py:662-694`）。rank-128 adapter 无法载入 rank-256 构造，"A/B 直通 C"不存在形状兼容的交接。裁决四条：(a) **memory 全部可训练参数（queries M、Enc、gate、读路径新增投影）一律实现为 PEFT adapter 之外的全秩 named modules**，走 `save/load_extra_components` 通道——形状只依赖 config dim（Base 与 Distilled 同为 dim 5120/40 层，同构可互载）；(b) Stage A/B 仍在 Base@ckpt-19500 rank-128 上训（保 C2 可比性），结束后仅将 memory 组件移植进 Stage C 的 distilled rank-256 generator；(c) 承认移植后 backbone 失配：Stage C 前置强制 **C0 warm-up 子阶段**（distilled generator + memory 加载、backbone 冻结或低 lr 的短程 TF 回归 ~500-1000 步），预算并入 C；(d) 落地 checkpoint-conversion 测试：把完整 A/B memory+读路径 state 加载进 Stage C 构造器，任何 missing/mismatched key 即 fail——列入 K0。*弃选一：A/B 直接在 distilled rank-256 上训——丢失与主线 `_correct` 谱系的横比与 C2 冻结基线，且 few-step 生成器上 TF flow-matching 的 σ 采样语义未经验证；弃选二：merge rank-128 adapter 进全模型再重跑 ODE 蒸馏过渡——重蒸馏成本与风险远超 C0 warm-up。*

**D9（CP 兼容，v1 硬约束）**：v1 **禁止 memory 与 `--enable_parallelism` 同时开启**（入口 `infer_helios.py:312-324`）。依据：diffusers CP plan 仅按序列维切分 `hidden_states`/`rotary_emb`（`helios/diffusers_version/transformer_helios_diffusers.py:558-573`），不携带全局偏移标注；processor 用 rank-local 序列长减全局 `original_context_length` 推 history 长度（`transformer_helios_diffusers.py:132-139`）——任何"前 M 个 rank-local key 属于 memory"的切片在 CP 下会在不同 rank 命中不同的 [memory|history|current] 片段。启用前必须扩展 CP plan 携带全局偏移并逐 backend 通过单卡 vs CP 数值等价测试。*弃选：v1 直接改 CP plan——上游 diffusers 侵入面大，且 memory 单卡收益未证前不值。*

## 1. 风险矩阵

### R1 双稳定器过稳 / 动态度冻结
- **机制（修订）**：x0/image anchor 在默认 rollout 中持续存在——T2V 于第一个生成 chunk 后由 `image_latents = latents[:, :, 0:1]` 建立（`helios/pipelines/pipeline_helios.py:1447-1450`）并在后续 section 前缀拼接（`pipeline_helios.py:1294-1298`），I2V 自始存在；但它**并非不可拆**：`is_keep_x0` 是 API 参数、默认 True（`pipeline_helios.py:906`），False 时走无 anchor 分支（`pipeline_helios.py:1175-1177,1299-1309`），故 anchor×memory 交互可单变量消融（已并入 A6）。history key 放大则是**可用但当前关闭**的机制：冻结基线显式 `is_amplify_history: false`（`stage1_lora_cfr_368_correct.yaml:173-174`），参数仅在构造期按 flag 分配（`helios/modules/transformer_helios.py:518-528`，构造默认 False：`transformer_helios.py:1006-1009`）；`infer_helios.py:290` 的 `is_amplify_history = True` 只是 `lora_path+partial_path` 分支里给 `load_extra_components` 的 loader shim（`infer_helios.py:279-292`），transformer 早已构造完毕（`infer_helios.py:238-242`），不会激活运行时放大。因此本风险的静态前缀组合是 **anchor + t=0 memory（含 amp_mem）**，不含 history 放大；若实验要启用 history 放大，须构造期开启并配对训练，作为独立 paired baseline。注意力质量向静态内容集中的风险不变。基线已有静动权衡实证：step-0 dynamic_degree 0.30 vs 续训 0.55–0.90（`docs/VS24_EVAL_ANALYSIS.md:35-59`）——向冻结方向的回退空间只有约 0.25。
- **检测信号**：[NEW] dynamic-degree / motion-amplitude 逐 chunk（33 帧对齐）斜率，OLS+Theil-Sen 双报；阈值 = memory-on 相对 no-KV 均值降 >0.1 或斜率负向劣化 >30%（对齐节 3 Stage B ③）；扰动实验（`perturb_at_chunk=30`）恢复达标但恢复后 dynamic-degree 掉出 [0.55, 0.90]，判"以冻结换稳定"，同样触发。
- **缓解**：gate bias 下调（0.75 → 更低，加快状态周转，节 2 §7 换算）；`memory_key_scale` 从中性 init −4.0 起步且可冻结（节 1 §1）；**D2（修订）：训练期 memory-dropout p≈0.1，实现为 attention 列 mask 或 gated bypass**（阻止 current query attend memory K/V，保持 compile-stable 形状），**禁止用零值 token 冒充关断**——AdaLN 有加性 shift（`transformer_helios.py:792-820`）且 `to_q/to_k/to_v` 均 bias=True（`transformer_helios.py:488-490`），全零 token 经投影后仍是非零 Q/K/V 参与无 mask 注意力（`transformer_helios.py:410-415` 的 `attn_varlen_func` 调用），no-memory 对照必须是真移除前向并做数值等价验证。*弃选：不加 dropout 全靠 Stage C 校准——C 最贵且只校准写分布，不治读依赖。*
- **残余风险**：斜率指标未经校准，首个 campaign 须先测 no-KV 的自然斜率分布定 ±1σ 带；anchor 默认开启下双前缀交互仍需靠 A6 的 `is_keep_x0` arm 分离。

### R2 TF-only 写门净负资产（含 Stage C 可训练性前提）
- **机制**：gate 输入在 TF 下只见 GT 统计的 t=0 hidden；推理写源是自生成漂移分布，`g·old+(1−g)·proj`（Echo 更新式 `Echo-Infinity/model/query_memory.py:152-155`）逐 section 复利放大失配——先验 #5 判定无末端校准则净负。
- **Stage C 可训练性前提（新增，源自代码事实）**：拟用父配置 `dmd_num_latent_sections_min: 3`（`stage_3_post_self-forcing_version.yaml:289-295`）且 cold-start 强制取该最小值（`train_helios.py:1571-1581`）；[16,2,1]+x0 窗口 19 latent 帧、每 section 追加 9 帧（`pipeline_helios.py:1294-1298,1452-1454`），首次含真实帧的驱逐写最早发生在第 2 个 section 末、其读收益落在下一 section——3-section rollout 在任何 loss 读到写入状态前即结束，Enc/gate 得不到读回报梯度。**强制项：memory 开启时断言 `dmd_num_latent_sections_min >= 4`（含 cold-start 分支），并附最短 rollout 上 Enc/gate 梯度非零的单元测试（列入 K0）。**
- **检测信号**：(a) 训练日志逐步记录 gate sigmoid 均值/分位数，TF 与自 rollout 双通道对比；(b) capture 特征统计（norm、与 TF 均值 cos）在 rollout 下漂移 >2σ；(c) 裁决性信号 = Stage B checkpoint 直接跑 91s 自生成 rollout：若 memory-on 在 ≥2 项斜率指标劣于 no-KV 而 Stage C 后翻正 → 风险实锤且缓解有效（这同时就是消融 A5a）。
- **缓解**：Stage C 必做（节 2 D9），谱系交接按 D8 执行（C0 warm-up 吸收 backbone 失配）；Stage B 混入低比例自生成 history / 对 capture 加噪模拟漂移（廉价前置缓解）；D13 同 σ_last 捕获保证训推同分布（节 3，σ 取值见 R4 修订）。
- **残余风险**：Stage C 自身脆弱——critic lr 4e-7、`dfake_gen_update_ratio: 5`、GAN 初期需关（`stage_3_post_self-forcing_version.yaml:232-233,256`）——"校准失败"与"写门本质不可救"不可区分；C 占全课程预算约一半。

### R3 粒度失配（Enc 容量 / 写稀疏）
- **机制（修订）**：Echo 的写是**驱逐驱动**而非逐 block 无条件——block 大小 3 帧（`Echo-Infinity/configs/echo_infinity-long.yaml:55`），但 `query_memory_encoder.update` 仅在局部窗推进产生 `num_exited_frames > 0` 时触发（`Echo-Infinity/wan/modules/causal_model.py:799-828`），窗满前与不推进全局 cache 的调用不写；稳态下仍是高频小写（每次 ~3 帧）。Helios 每次驱逐写 E≤9×920=8280 token（368×640，多分辨率桶可变，`helios/dataset/dataloader_history_latents_dist.py:107-113`）压进 M≈720–2760 query，压缩率 3–11×。**写频修订**：dataloader 从 [0, total_sections−U] 均匀采样 `start_section_idx`（`dataloader_history_latents_dist.py:176-186`），晚起点的 U=4 展开有 3 次全真实驱逐，仅绝对起点附近的样本是 ≤2 次真实写——写频/梯度稀疏度是 start-index 分布的函数，非常数；且当前 output dict 不返回 `start_section_idx`（`dataloader_history_latents_dist.py:365-379`），驱逐 mask 无法从绝对 section 位置推导——**数据契约必须新增该字段（并入节 2 的 dataloader 改造），写稀疏与成本假设在 smoke profile 后按实测起点分布重算**。
- **检测信号**：每次写的 ‖ΔM‖/‖M‖ 时序（塌向 0 = 死写，发散 = 不稳）；Enc cross-attn 对写源的注意力熵；Stage B ② 一致性增益缺失但 gate 未饱和 → 指向容量而非门。
- **缓解**：N_Q 上调（A1）；Enc `ffn_mult=4` 保留（降 2 只作显存备选）；9 帧写拆 3×3 帧子写（回退向 Echo 粒度，写开销 ×3 仍 <0.5% 单次前向，节 1 §8）；U 4→6（须先跑语料帧数直方图，节 2 D6）。
- **残余风险**：`bptt_sections=1` 使 Enc 只从紧邻 section 的读回报学习，>1 section 后才兑现的写价值在梯度上不可见；sweep 只覆盖 {1,2}。

### R4 蒸馏 few-step 写入源质量
- **机制（修订）**："distilled σ_last≈0.5 vs Mid σ≈0.001、固定 500× 差距"**不成立为常数**——0.5/0.001 是 pre-shift 排布值（per-stage 原始 sigmas `np.linspace(0.999,0,...)`：`helios/diffusers_version/scheduling_helios_diffusers.py:165-166`；dmd 截断：`scheduling_helios_diffusers.py:244-246`），而 `set_timesteps` 末尾会对 `self.sigmas` 施加 dynamic shifting（`scheduling_helios_diffusers.py:248-256`），拟用 Stage C 父配置正开启 `use_dynamic_shifting: true`（`stage_3_post_self-forcing_version.yaml:128-134`）——transformer 实际消费的是 post-shift σ。定性结论保留：同一 Enc 面对跨数量级的写源噪声差异。
- **检测信号**：分模式 gate 均值，distilled <0.5× Mid 判写源退化（节 3 D3）；capture 逐通道方差对 σ 的标定曲线——**标定与 FiLM 条件一律采用 `set_timesteps` 后调度器实际给出的 σ_last 并逐 forward 落日志，按模式分别建曲线，不用 linspace 端点**。
- **缓解**：FiLM(σ_last) 必选；`memory_capture_extra_t0`（显式 t=0 commit 前向）仅在检测触发后启用——**该路径在两仓库均不存在，其 "+30% 成本" 与"必然击穿 ≤12% 吞吐线"均降级为 INFERRED、待端到端 profile**（现行 distilled launcher 为 3 stage × 2 步：`scripts/inference/helios-distilled_t2v.sh:14-16`；推理循环：`helios/diffusers_version/pipeline_helios_diffusers.py:674-720`），不作为确定性 kill 条件；Stage C 直接在 distilled σ_last 捕获训练（D13）。
- **残余风险**：若 profile 证实 extra_t0 开销不可接受，才讨论 Mid-memory / distilled-no-KV 分裂配置（同为 INFERRED 备案）。

### R5 与 correct.yaml Easy Anti-Drifting：共存，非替代
- **机制**：`corrupt_history_latents` 作用于输入侧 latent history——先以 `noise_corrupt_clean_prob`（=0.1）掷 clean，未中才按 `noise_mode_prob`（=0.8889）在 noise/downsample 间二选（`helios/utils/utils_helios_base.py:310-318`；配置及其自注释 "P(noise)≈0.8/P(blur)≈0.1/P(clean)=0.1"：`stage1_lora_cfr_368_correct.yaml:156-166`，上游依据 `scripts/training/configs/correct.yaml:18-26`）；memory 作用于状态侧驱逐抽象——目标变量不同。真实交互点：Stage B 展开的 loss 前向保留 corrupt（节 2 D7 伪码），若 capture 前向也吃 corrupted history，Enc 学的是"被人为损坏的场景摘要"，而推理 history 无 corruption → 写源训推错位。
- **D3**：capture 前向一律用未 corrupt 的 history（写源干净化），loss 前向保留 corrupt。*弃选：capture 也 corrupt——主动制造写源训推不一致，与 D13 同分布原则矛盾。*
- **检测信号（修订）**：消融 A7 的自变量改为**总损坏概率** `noise_corrupt_clean_prob_history` ∈ {0.1, 0.5}（损坏率 90%→50%，条件 noise/blur 比例 0.8889 保持不动）——原稿把 0.8889 当损坏概率是语义错误，改它只会把 noise/blur 配比从 ~80/10 变 ~45/45 而总损坏仍 90%。显著更优 = 双机制打架；无差异 = 共存成立。
- **缓解/残余**：correct.yaml 三项是主线已付学费的修正（`docs/HELIOS_CORRECT_RECIPE.md:11-31`），v1 绝不移除；"memory 替代 anti-drifting"仅作 Tier-2 实验。残余：corrupt 随机性使 Stage B 各 section 损坏水平独立采样，写-读两通路见到的 history 不一致是常态，需分层统计。

### R6 与 history 离散化（VQ）的先后与交互
- **现状**：OBSERVED ABSENCE（见前置声明），纯路线图排序问题。
- **机制/交互（修订，INFERRED）**：静态代码上 memory 不改写 history latents 张量，但 **rollout 级耦合是双向的**：memory 改变去噪输出，生成 chunk 直接 `torch.cat` 进 `history_latents`（`pipeline_helios.py:1452-1454`）并被切片为后续 section 的 history 输入（`pipeline_helios.py:1294-1298`）——未来 VQ 若作用于该递归 history，其输入分布随 memory 开关而变；反向亦然（VQ 改 patchify 输入分布 → 写源分布 → Enc/gate 全重校准）。
- **D4（排序，理由修正）**：memory Stage A/B 先行；VQ 排在 memory M1 之后、独立分支单独验证；禁止同一 run 首次引入两个未验证机制。排序依据不再是"零副作用"（该说法撤销），而是重校准成本不对称：memory 侧重校准已知且较小（一次 Stage A ≈7.5 GPU·天），VQ 先行则立刻废掉 memory 全部 Stage A 校准。**无论何序，预算里都预留一次联合重校准。***弃选：VQ 先行——重校准代价方向相反且更大。*
- **残余风险**：若主线强推 VQ 优先，memory 预算需预留一次 Stage A 重训（≈7.5 GPU·天，可接受）。

### R7 Wan2.1-14B vs Wan2.2 目标底座
- **现状**：Wan2.2-TI2V-5B 移植已 smoke 验证（load/48ch encode/30 步训练/推理视频，`docs/WAN22_PORT.md:98-103`），但无任何"替换 14B"政策文件（OBSERVED ABSENCE）；主线 lora368 全谱系仍锚定 Wan2.1/Helios-Base（`stage1_lora_cfr_368_correct.yaml:32-48`）。
- **D5**：memory v1 锚定 Wan2.1-14B——VS24 全部基线与 step-5000 锚点（DOVER drift 0.069、dynamic 0.75，`docs/VS24_EVAL_ANALYSIS.md:61-71`）都在 14B 上；Wan2.2-5B 仅作机制级 smoke 沙盒（梯度连通/形状/BPTT 正确性，单节点、约便宜 3×）。*弃选：在 5B 上跑全课程再迁回——dim 5120→3072、40→30 层、VAE 16×（`docs/WAN22_PORT.md:6-29`），gate/amp 校准不可迁移，等于白做。*
- **缓解**：模块全部维度从 config 推导（节 2 D2 单一构造字典，与 D8(a) 全秩组件原则同构），未来切底座代码零改、只重训。**残余**：主线若在 Stage C 前切底座，约 104 GPU·天的 C 段投入重做——由协调条款 C5 冻结该决策。

## 2. 消融矩阵

成本口径：单位 = H200 单卡·天。实测锚点：1 节点 bs2 throttled 13.06 s/it、4 节点 gb64 ≈28 s/it、bs4 峰值 ≈90 GiB（`docs/STAGE1_SPEED_OPTIMIZATION.md` §3 杠杆表）。课程基价（节 2 §8，全 INFERRED，smoke profile 后修订）：Stage A ≈180 GPU·h ≈7.5 天；B ≈530 ≈22 天；C ≈2500 ≈104 天（含 D8 的 C0 warm-up）。评测 arm（VS24-long 20 视频生成+全指标）≈2–4 天/组 [INFERRED]。注："Echo +10.6% 开销"在 Echo 仓库无佐证（节 3 §5 已撤销其代码事实地位），吞吐验收以实测为准。

| # | 变量与组 | 依赖 | 成本(GPU·天) | 判读方式 | 优先级 |
|---|---|---|---|---|---|
| A0 | 读机制 {token 插入, KV 注入} | 2×Stage A | 15 | Stage A ①②③ + FLOPs 实测；并列取 token 插入（t=0 语义完整）。仅在 D1 被质疑时启动 | 备用 |
| A1 | N_Q ∈ {1,3,6} 帧当量 | 3×(A+B) | 90（含默认组） | Stage B ② 增益-开销曲线拐点；full-res 上界仅在胜出组 +1 arm(+30) | **Tier-0** |
| A2 | gate bias {0.75, 2.0} | 2×B（共享 A ckpt，gate bias 重 init） | 44 | gate 均值轨迹 + R1 动态度斜率；0.75 若动态更好且一致性不降 → 确认稀疏大写重标定 | **Tier-0** |
| A3 | 写入源层深 {40 末层, 30, 20} | 3×A 短程(2k 步) | 12 | flow loss 劣化 + 读注意力质量(D10 诊断路径) + 线性 probe(capture→驱逐帧 latent 的 R²) | Tier-1 |
| A4 | amp_mem {on(init −4), off(无参数)} | 2×B | 44 | memory 注意力质量分布 + ②；off 无损则删参简化 | Tier-1 |
| A5a | 去 Stage C（B ckpt 直接 rollout 评测） | 评测 only | 3 | 先验 #5 的裁决实验：B 劣于 no-KV 而 C 翻正 = 课程论成立 | **Tier-0** |
| A5b | 去 Stage A（直接 B 联训） | 1×B | 22 | 对照 A+B：验证 warm-start 必要性 | Tier-2 |
| A5c | Stage C 后 extra_t0 开关（distilled，先 profile 后启用，R4） | 评测 only | 3 | distilled gate 均值 + C-⑤ 吞吐实测 | Tier-1 |
| A6 | 推理消融 {on, no-KV, frozen-M₀, shuffled-M, **is_keep_x0=False×{on,no-KV}**} + eventswitch 切换策略 {no-reset, reset, recache, decay} | 评测 only（共享 C ckpt） | 12–17 | 节 3 §6/§7 全套斜率 + C-⑥；x0 arm 分离 anchor×memory 交互（R1，走 `pipeline_helios.py:906` 现成参数） | **Tier-0** |
| A7 | `noise_corrupt_clean_prob_history` {0.1, 0.5}（R5，总损坏率 90%→50%，mode 配比不动） | 1×B 加组 | 22 | 双机制交互判定（见 R5） | Tier-2 |
| A8 | bptt_sections {1,2} | A1 胜出组 +1 arm | 22 | ② + ‖ΔM‖/‖M‖ 稳定性 | Tier-1 |
| A9 | RoPE 槽位 {分数 id, mid-tier 池化频谱, id=0} | 3×A 短程 | 12 | Stage A 即可判读（读注意力质量 + flow loss），胜者进 B | **Tier-0** |

Tier-0 合计 ≈ 主课程(134) + A9(12) + A1 增量(≈60) + A2 增量(22) + A5a(3) + A6(15) ≈ **246 GPU·天 ≈ 5.9k GPU·h**，与节 2 预算（4.5–5.5k）同量级、略超——A2 可与 A1 胜出组串行裁剪至 ~5.3k。
**D6（消融纪律）**：全部 arm 单变量、共享 seed/prompt 集、判读一律斜率优先；v1 评测一律单卡或数据并行，禁用 CP（D9）。*弃选：全因子 sweep——A1×A2×A8 即 12 arm ≈260 天，超总预算。*

## 3. 分阶段 Kill Criteria

- **K0（实现前，第 0–1 周）**：D1/D8 仲裁未签字；或 **D8(d) checkpoint-conversion 测试未通过**（完整 A/B memory+读路径 state 载入 Stage C 构造器，missing/mismatched key 即 fail）；或 **memory 开启下 `dmd_num_latent_sections_min>=4` 断言与最短 rollout Enc/gate 梯度非零测试未落地**（R2，cold-start 分支 `train_helios.py:1571-1581` 一并覆盖）；或 Stage B smoke（1 节点 bs2×U=4，显存参照 bs4 单 section ≈90 GiB，`docs/STAGE1_SPEED_OPTIMIZATION.md`）OOM 且降 U=2 仍不可行 → 不进入训练，回炉模块设计。凡 kill 均产出验尸报告归档 `logs/`。
- **K1（Stage A 末，预算 ≤15 GPU·天含一次返工）**：节 3 Stage A ①②③（flow loss 劣化 ≤+1%、memory 读注意力质量 >1%、91s DOVER 回退 ≤0.02）。**D10（前置交付物）**：现行两条 attention 路径都只返回 attended tensor、不暴露注意力概率（主训练 dispatch：`helios/modules/helios_kernels/attention_dispatch.py:132-167`；diffusers processor：`helios/diffusers_version/transformer_helios_diffusers.py:141-157`）——须实现诊断专用采样层路径（采样 3–5 层，fp32 下由 Q/K 显式 softmax 计算 current-query→memory 质量，明确 query/layer/head 聚合定义，不替换生产注意力），与 [NEW] 时序脚本同列 Stage A 期间交付；**未就绪则 1% 门从 K1 移除，K1 仅以 flow loss + DOVER 判读**。一次修复迭代后读通路仍死 → kill，保留 no-KV。*弃选：全局换 eager attention 取概率——训练吞吐不可接受。*
- **K2（Stage B 末，预算累计 ≤70 天含 A2 返工）**：gate 均值 ∈[0.05,0.6]；subject-consistency ≥+0.005 且 boundary-LPIPS-excess ≤−5%（vs no-KV）；R1 冻结检测不触发。gate 双端饱和且 bias 重调无效 → kill；冻结触发且 memory-dropout+降 bias 无效 → kill。
- **K3（Stage C 末，预算累计 ≤130 天；C 不重跑，一次即裁决）**：节 3 C-①..⑥，其中 **drift 判据改为配对相对口径**：与同 `checkpoint-19500`、同 seed/prompt 的 no-KV 基线（C2 重跑产物）对比，memory-on 的 DOVER drift 不劣于该基线置信区间；**0.069 仅作历史参照**——它属另一 campaign 的 step-5000（`docs/VS24_EVAL_ANALYSIS.md:61-71`），而 memory 分支冻结基座是 `_correct/checkpoint-19500`（`stage1_lora_reweight_368.yaml:100-107`；`train_stage1_lora_rwtag.sbatch:29-35`），该谱系无 0.069 实测，绝对阈值会仅因祖先不同而误杀改进模型。其余：saturation |斜率| ≤0.7× no-KV、扰动恢复 ≤0.7×、dynamic ∈[0.55,0.90]、吞吐 ≤12%、eventswitch CLIP 不劣。≥2 项主斜率仍劣于配对 no-KV → 判"TF-only 写门净负"成立（先验 #5 被证实），发布 no-KV，项目转验尸。
- **K4（全局）**：累计 >6k GPU·h，或占用主线节点 >2 周，或 C5 冻结决策被推翻 → 无条件暂停、重新立项。
- **D7（判读前置条件）**：每个 K 点的数据必须来自 [NEW] 时序指标脚本与 D10 诊断路径——Stage A 期间并行落地、Stage B 判读前完成 no-KV 基线校准；未就绪则 K2 顺延而非放行。*弃选：用现有端点均值凑合判读——端点均值正是先验 #6 明令禁止的口径。*

## 4. 与 lora368 主线的资源/检查点协调

- **C1（节点，修订）**：主线 `_correct` 占 5 节点（mc-node01,c-node03–06，`train_stage1_lora_cfr_368_correct.sbatch:3-5`）至 step 22000（monitor 上限 `MAXSTEPS=22000`：`scripts/training/monitor_lora368c.sh:13`）；descendants 中 reweight 复用同一批 5 节点（`train_stage1_lora_reweight.sbatch:3-5`），而 **rwtag 已把 c-node07 整节点独占 pin 死**（`train_stage1_lora_rwtag.sbatch:2-8`，理由即该节点跨节点 NCCL wedge、节点内 NVLink 可用：`train_stage1_lora_rwtag.sbatch:18-20`）——c-node07 **不是**结构性无竞争资源。调度规则改为条件式：memory 单节点作业仅在 rwtag 窗口外使用 c-node07，或以 Slurm dependency 与 rwtag 显式串行；同时提名一个备选单节点（使用前必须先过单节点 NVLink smoke），预算中不得把 c-node07 记为常态空闲。Stage C 需 4 节点 ×3–4 天，与主线排程窗口协商，建议卡在主线两个 5000 步区段之间。*弃选：Stage C 抢占 5 节点主线窗口——主线 monitor 会按名字自动检测并重提（`monitor_lora368c.sh:30-47`），排队冲突双输。*
- **C2（基座 checkpoint 冻结）**：Stage A 分叉点钉死 `_correct/checkpoint-19500`——与 reweight/rwtag 同源（`stage1_lora_reweight_368.yaml:100-107`），memory 结果可与主线 descendants 横比；并在**同一 ckpt** 上重跑一次 no-KV VS24 基线——该基线同时就是 K3 的配对对照组（step-5000 的 0.069 属另一 campaign 谱系，只作锚不作对照组）。*弃选：追最新 ckpt——移动基线毁掉全部消融可比性。*
- **C3（checkpoint 卫生）**：memory run 独立 `output_dir` + symlink 播种模式复用（`train_stage1_lora_rwtag.sbatch:29-35`）；500 步/limit 30 沿用（`stage1_lora_cfr_368_correct.yaml:98-100`）；恢复前必须先落地 D11 的 FORCE_LR 按 param-group 扩展——主线 `HELIOS_FORCE_LR=1` 会把 memory lr×5 压平为单一标量（`train_helios.py:1090-1099`），这是复用主线脚本的第一颗雷。
- **C4（工具链先行合并）**：`save/load_extra_components` 第 5 节、merge 工具与 EMA 的 memory 分支（节 2 D12）、**D8 的全秩组件跨谱系载入路径与 conversion 测试**须在 Stage B 前合入——对主线是纯加法（默认 `is_enable_evolving_memory: false` 门控），可先行合并不阻塞主线；否则 Stage C 与评测无 checkpoint 可消费。
- **C5（决策冻结）**：请主线在 memory M1（Stage B 通过）前书面冻结两项：底座不切 Wan2.2（R7）、VQ 不先行（R6/D4）；任一提前变更触发 K4。
- **C6（评测共建）**：[NEW] 时序指标脚本与 D10 注意力诊断按 `external_command` 注册为共同资产——主线 checkpoint 同样受益（现 campaign 的 "drift" 也只是端点差：`make_eval_report.py:132-147`、`summarize_helios_long_ratio_eval.py:92-110`），落地成本由两条线分摊。

（全文已同步写入 `/mnt/beegfs/siyuan/workspace/helios-echo/logs/research/design-risks.md`）
