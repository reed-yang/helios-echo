# 设计任务 4/4：风险矩阵、消融矩阵与分阶段 Kill Criteria（含 lora368 主线协调）

前置核验声明：本节全部代码事实已在磁盘逐项复核（helios-team @ mid_training_xiangbo、Echo-Infinity）。两项关键 OBSERVED ABSENCE 置顶：
(1) **仓库内不存在任何 history 离散化/VQ 实验**——对 `docs/`、`helios/` 全量 grep `vq|vector quant|discret|codebook|fsq`，命中仅为 VideoAlign 奖励维度 "VQ"（`helios/videoalign/prompt_template.py:12`）与 FlowMatchEulerDiscrete 调度器命名（`helios/scheduler/scheduling_helios.py:305`），R6 按"纯规划态实验"处理；
(2) **评测侧无任何 within-video 时序斜率后端**——长评测只有 full/start15/end15 端点聚合（`scripts/evaluation/make_eval_report.py:91-111`；`summarize_helios_long_ratio_eval.py:68-112`）。本文所有标 [NEW] 的检测信号须先按设计节 3 §6 的 `external_command` 方案落地（`tools/long_video_eval/long_video_eval/runner.py:55-107`）才可执行——**检测手段本身是交付物，不是现货**。

## 0. R0：跨设计节机制冲突（blocker 级，先于一切实验）

三份已验证设计节对同一机制给出互斥参数化：**读机制**（节 1 = token 插入、memory 获得 t=0 AdaLN；节 3 = 逐层 `to_k_mem/to_v_mem` 纯 KV 注入、AdaLN 不作用）；**M 定义**（节 1 mid-tier 390 token/帧、M=1170 @480×832；节 2 全分辨率 920 token/帧、M=2760 @368×640；节 3 960 token/帧、N_mem=2880 @384×640）；**RoPE 槽位**（节 1 (0,1) 分数 id vs 节 3 负 id）；**命名/API**（`helios_memory`/`evolving_memory`/`mem_encoder`；`capture_last_hidden`/`return_final_hidden`/`return_capture`）。

**D1（仲裁，写第一行实现代码前锁定）**：合成基线 = 节 1 的 token 插入读机制 + 节 2 的配置键/课程/trainer 集成 + 节 3 的 FiLM(σ_last) 写源条件化与评测协议；M 语义统一为"帧当量 N_Q × 主训练桶 mid-tier 网格"（368×640：full-res 23×40=920/帧，mid-tier 网格 12×20=240/帧，N_Q=3 → M=720），M 为固定形状（状态跨 section 持有必须定形），节 1 的 1170 与节 3 的 2880 是各自分辨率/密度下的换算，消融自变量统一为 N_Q 帧当量；RoPE 槽位交由消融 A9 裁决（分数 id 为默认）。*弃选：两种读机制并行实现再实验裁决——双份 diffusers 镜像移植（节 1 偏差 D）成本翻倍，且 t=0 条件语义不可比；若 token 插入的 +FFN 开销在实测中不可接受，KV 注入是唯一回退项，由 A0 仲裁实验兜底。*

## 1. 风险矩阵

### R1 双稳定器过稳 / 动态度冻结
- **机制**：x0/image anchor 前缀恒在（`helios/pipelines/pipeline_helios.py:1447-1450`）且推理强制 `is_amplify_history=True`（`infer_helios.py:290`；可学习 key 放大 `helios/modules/transformer_helios.py:518-532`）；memory 作为第二个 t=0 静态前缀再叠加 `amp_mem`，注意力质量向静态内容二次集中。基线已有静动权衡实证：step-0 dynamic_degree 0.30 vs 续训 0.55–0.90（`docs/VS24_EVAL_ANALYSIS.md:35-59`）——向冻结方向的回退空间只有约 0.25。
- **检测信号**：[NEW] dynamic-degree / motion-amplitude 逐 chunk（33 帧对齐）斜率，OLS+Theil-Sen 双报；阈值 = memory-on 相对 no-KV 均值降 >0.1 或斜率负向劣化 >30%（对齐节 3 Stage B ③）；扰动实验（`perturb_at_chunk=30`）恢复达标但恢复后 dynamic-degree 掉出 [0.55, 0.90]，判"以冻结换稳定"，同样触发。
- **缓解**：gate bias 下调（0.75 → 更低，加快状态周转，节 2 §7 换算）；`memory_key_scale` 从中性 init −4.0 起步且可冻结（节 1 §1）；**D2：训练期 memory-dropout p≈0.1**（读路径随机置零 memory tokens，防结构性依赖）——*弃选：不加 dropout 全靠 Stage C 校准——C 最贵且只校准写分布，不治读依赖*；缩小 N_Q。
- **残余风险**：斜率指标未经校准，首个 campaign 须先测 no-KV 的自然斜率分布定 ±1σ 带；anchor 前缀是既有架构不可拆，双前缀交互无法被单变量消融完全分离。

### R2 TF-only 写门净负资产
- **机制**：gate 输入在 TF 下只见 GT 统计的 t=0 hidden；推理写源是自生成漂移分布，`g·old+(1−g)·proj`（Echo 更新式 `Echo-Infinity/model/query_memory.py:152-155`）逐 section 复利放大失配——先验 #5 判定无末端校准则净负。
- **检测信号**：(a) 训练日志逐步记录 gate sigmoid 均值/分位数，TF 与自 rollout 双通道对比；(b) capture 特征统计（norm、与 TF 均值 cos）在 rollout 下漂移 >2σ；(c) 裁决性信号 = Stage B checkpoint 直接跑 91s 自生成 rollout：若 memory-on 在 ≥2 项斜率指标劣于 no-KV 而 Stage C 后翻正 → 风险实锤且缓解有效（这同时就是消融 A5a）。
- **缓解**：Stage C 必做（节 2 D9）；Stage B 混入低比例自生成 history / 对 capture 加噪模拟漂移（廉价前置缓解）；D13 同 σ_last 捕获保证训推同分布（节 3）。
- **残余风险**：Stage C 自身脆弱——critic lr 4e-7、`dfake_gen_update_ratio: 5`、GAN 初期需关（`scripts/training/configs/stage_3_post_self-forcing_version.yaml:232-233,256`）——"校准失败"与"写门本质不可救"不可区分；C 占全课程预算约一半。

### R3 粒度失配（Enc 容量 / 写稀疏）
- **机制**：Echo 每 causal block（3 帧）高频小写；Helios 每 section 一次写入 E≤9×920=8280 token（368×640，多分辨率桶可变，`helios/dataset/dataloader_history_latents_dist.py:107-113`）压进 M≈720–2760 query，压缩率 3–11×；且 U=4 展开每样本真实写 ≤2 次（k≤2 零写、k=3 部分写，节 2 D4），Enc 梯度信号稀疏。
- **检测信号**：每次写的 ‖ΔM‖/‖M‖ 时序（塌向 0 = 死写，发散 = 不稳）；Enc cross-attn 对写源的注意力熵；Stage B ② 一致性增益缺失但 gate 未饱和 → 指向容量而非门。
- **缓解**：N_Q 上调（A1）；Enc `ffn_mult=4` 保留（降 2 只作显存备选）；9 帧写拆 3×3 帧子写（回退向 Echo 粒度，写开销 ×3 仍 <0.5% 单次前向，节 1 §8）；U 4→6（须先跑语料帧数直方图，节 2 D6）。
- **残余风险**：`bptt_sections=1` 使 Enc 只从紧邻 section 的读回报学习，>1 section 后才兑现的写价值在梯度上不可见；sweep 只覆盖 {1,2}。

### R4 蒸馏 few-step 写入源质量
- **机制**：distilled 最后一次 transformer 调用在 stage-local σ≈0.5（`helios/diffusers_version/scheduling_helios_diffusers.py:234-246` 推导，节 3 偏差 1），Mid 为 σ≈0.001——同一 Enc 面对跨 ~500× 的写源噪声。
- **检测信号**：分模式 gate 均值，distilled <0.5× Mid 判写源退化（节 3 D3）；capture 逐通道方差对 σ 的标定曲线。
- **缓解**：FiLM(σ_last) 必选；`memory_capture_extra_t0`（显式 t=0 commit 前向，distilled +30% 成本）仅在检测触发后启用；Stage C 直接在 distilled σ_last 捕获训练（D13）。
- **残余风险**：若 extra_t0 成为必需，吞吐验收线 ≤12% 在 distilled 必然失守 → 只能发布 Mid-memory / distilled-no-KV 分裂配置，旗舰路径收益折损。

### R5 与 correct.yaml Easy Anti-Drifting：共存，非替代
- **机制**：`corrupt_history_latents` 作用于输入侧 latent history（clean 0.1 → 噪声/模糊按 0.8889 分流，`helios/utils/utils_helios_base.py:283-318`；配置 `stage1_lora_cfr_368_correct.yaml:156-166`，上游依据 `scripts/training/configs/correct.yaml:18-26`），训练去噪器对退化 history 的鲁棒性；memory 作用于状态侧驱逐抽象——目标变量不同。真实交互点：Stage B 展开的 loss 前向保留 corrupt（节 2 D7 伪码），若 capture 前向也吃 corrupted history，Enc 学的是"被人为损坏的场景摘要"，而推理 history 无 corruption → 写源训推错位。
- **D3**：capture 前向一律用未 corrupt 的 history（写源干净化），loss 前向保留 corrupt。*弃选：capture 也 corrupt——主动制造写源训推不一致，与 D13 同分布原则矛盾。*
- **检测信号**：消融 A7（memory-on 下 corrupt 概率 0.8889→0.5）——显著更优 = 双机制打架；无差异 = 共存成立。
- **缓解/残余**：correct.yaml 三项是主线已付学费的修正（I2V 训推错配曾致真实退化，`docs/HELIOS_CORRECT_RECIPE.md:11-31`），v1 绝不移除；"memory 替代 anti-drifting"仅作 Tier-2 实验。残余：corrupt 的随机性使 Stage B 展开各 section 损坏水平独立采样，写-读两条通路见到的 history 不一致是常态，需在日志中分层统计。

### R6 与 history 离散化（VQ）的先后与交互
- **现状**：OBSERVED ABSENCE（见前置声明），纯路线图排序问题。
- **机制/交互（INFERRED）**：VQ 若落在 history latents（patchify 之前），则三层 patchify 输入分布改变 → 写源（末层 hidden）分布随之改变 → Enc/gate 全部重校准；量化误差同时是新的"驱逐前损耗"，memory 的保全价值反而上升。反向无耦合：memory 不改 history latents 本身，对 VQ 无影响——**交互是单向的**。
- **D4（排序）**：memory Stage A/B 先行；VQ 排在 memory M1 之后、独立分支单独验证；禁止同一 run 首次引入两个未验证机制。*弃选：VQ 先行——会使 memory 全部 Stage A 校准作废重来，而 memory 先行对 VQ 零副作用。*
- **残余风险**：若主线强推 VQ 优先，memory 预算需预留一次 Stage A 重训（≈7.5 GPU·天，可接受）。

### R7 Wan2.1-14B vs Wan2.2 目标底座
- **现状**：Wan2.2-TI2V-5B 移植已 smoke 验证（load/48ch encode/30 步训练/推理视频，`docs/WAN22_PORT.md:98-103`），但无任何"替换 14B"政策文件（OBSERVED ABSENCE）；主线 lora368 全谱系仍锚定 Wan2.1/Helios-Base（`stage1_lora_cfr_368_correct.yaml:32-48`）。
- **D5**：memory v1 锚定 Wan2.1-14B——VS24 全部基线与 step-5000 锚点（DOVER drift 0.069、dynamic 0.75，`docs/VS24_EVAL_ANALYSIS.md:61-71`）都在 14B 上；Wan2.2-5B 仅作机制级 smoke 沙盒（梯度连通/形状/BPTT 正确性，单节点即可、约便宜 3×）。*弃选：在 5B 上跑全课程再迁回——dim 5120→3072、40→30 层、VAE 16×（`docs/WAN22_PORT.md:6-29`），gate/amp 校准不可迁移，等于白做。*
- **缓解**：模块全部维度从 config 推导（节 2 D2 单一构造字典），未来切底座代码零改、只重训。**残余**：主线若在 Stage C 前切底座，约 104 GPU·天的 C 段投入重做——由协调条款 C5 冻结该决策。

## 2. 消融矩阵

成本口径：单位 = H200 单卡·天。实测锚点：1 节点 bs2 throttled 13.06 s/it、4 节点 gb64 ≈28 s/it、bs4 峰值 ≈90 GiB（`docs/STAGE1_SPEED_OPTIMIZATION.md` §3 杠杆表）。课程基价（节 2 §8，全 INFERRED，smoke profile 后修订）：Stage A ≈180 GPU·h ≈7.5 天；B ≈530 ≈22 天；C ≈2500 ≈104 天。评测 arm（VS24-long 20 视频生成+全指标）≈2–4 天/组 [INFERRED]。注："Echo +10.6% 开销"在 Echo 仓库无佐证（节 3 §5 已撤销其代码事实地位），吞吐验收以实测为准。

| # | 变量与组 | 依赖 | 成本(GPU·天) | 判读方式 | 优先级 |
|---|---|---|---|---|---|
| A0 | 读机制 {token 插入, KV 注入} | 2×Stage A | 15 | Stage A ①②③ + FLOPs 实测；并列取 token 插入（t=0 语义完整）。仅在 D1 被质疑时启动 | 备用 |
| A1 | N_Q ∈ {1,3,6} 帧当量 | 3×(A+B) | 90（含默认组） | Stage B ②增益-开销曲线拐点；full-res 上界仅在胜出组 +1 arm(+30) | **Tier-0** |
| A2 | gate bias {0.75, 2.0} | 2×B（共享 A ckpt，gate bias 重 init） | 44 | gate 均值轨迹 + R1 动态度斜率；0.75 若动态更好且一致性不降 → 确认稀疏大写重标定 | **Tier-0** |
| A3 | 写入源层深 {40 末层, 30, 20} | 3×A 短程(2k 步) | 12 | flow loss 劣化 + 读注意力质量 + 线性 probe(capture→驱逐帧 latent 的 R²) | Tier-1 |
| A4 | amp_mem {on(init −4), off(无参数)} | 2×B | 44 | memory 注意力质量分布 + ②；off 无损则删参简化 | Tier-1 |
| A5a | 去 Stage C（B ckpt 直接 rollout 评测） | 评测 only | 3 | 先验 #5 的裁决实验：B 劣于 no-KV 而 C 翻正 = 课程论成立 | **Tier-0** |
| A5b | 去 Stage A（直接 B 联训） | 1×B | 22 | 对照 A+B：验证 warm-start 必要性 | Tier-2 |
| A5c | Stage C 后 extra_t0 开关（distilled） | 评测 only | 3 | distilled gate 均值 + C-⑤ 吞吐 | Tier-1 |
| A6 | 推理消融 {on, no-KV, frozen-M₀, shuffled-M} + eventswitch 切换策略 {no-reset, reset, recache, decay} | 评测 only（共享 C ckpt） | 10–15 | 节 3 §6/§7 全套斜率 + C-⑥ | **Tier-0** |
| A7 | corrupt 概率 {0.8889, 0.5}（R5） | 1×B 加组 | 22 | 双机制交互判定（见 R5） | Tier-2 |
| A8 | bptt_sections {1,2} | A1 胜出组 +1 arm | 22 | ② + ‖ΔM‖/‖M‖ 稳定性 | Tier-1 |
| A9 | RoPE 槽位 {分数 id, mid-tier 池化频谱, id=0} | 3×A 短程 | 12 | Stage A 即可判读（读注意力质量 + flow loss），胜者进 B | **Tier-0** |

Tier-0 合计 ≈ 主课程(134) + A9(12) + A1 增量(≈60) + A2 增量(22) + A5a(3) + A6(13) ≈ **245 GPU·天 ≈ 5.9k GPU·h**，与节 2 预算（4.5–5.5k）同量级、略超——A2 可与 A1 胜出组串行裁剪至 ~5.3k。
**D6（消融纪律）**：全部 arm 单变量、共享 seed/prompt 集、判读一律斜率优先。*弃选：全因子 sweep——A1×A2×A8 即 12 arm ≈260 天，超总预算。*

## 3. 分阶段 Kill Criteria

- **K0（实现前，第 0–1 周）**：D1 仲裁未签字，或 Stage B smoke（1 节点 bs2×U=4，显存参照 bs4 单 section ≈90 GiB，`docs/STAGE1_SPEED_OPTIMIZATION.md`）OOM 且降 U=2 仍不可行 → 不进入训练，回炉模块设计。凡 kill 均产出验尸报告归档 `logs/`。
- **K1（Stage A 末，预算 ≤15 GPU·天含一次返工）**：节 3 Stage A ①②③（flow loss 劣化 ≤+1%、memory 读注意力质量 >1%、91s DOVER 回退 ≤0.02）。一次修复迭代后读通路仍死 → kill，保留 no-KV。
- **K2（Stage B 末，预算累计 ≤70 天含 A2 返工）**：gate 均值 ∈[0.05,0.6]；subject-consistency ≥+0.005 且 boundary-LPIPS-excess ≤−5%（vs no-KV）；R1 冻结检测不触发。gate 双端饱和且 bias 重调无效 → kill；冻结触发且 memory-dropout+降 bias 无效 → kill。
- **K3（Stage C 末，预算累计 ≤130 天；C 不重跑，一次即裁决）**：节 3 C-①..⑥ 全套（drift ≤0.069、saturation |斜率| ≤0.7× no-KV、扰动恢复 ≤0.7×、dynamic ∈[0.55,0.90]、吞吐 ≤12%、eventswitch CLIP 不劣）。≥2 项主斜率仍劣于 no-KV → 判"TF-only 写门净负"成立（先验 #5 被证实），发布 no-KV，项目转验尸。
- **K4（全局）**：累计 >6k GPU·h，或占用主线节点 >2 周，或 C5 冻结决策被推翻 → 无条件暂停、重新立项。
- **D7（判读前置条件）**：每个 K 点的数据必须来自 [NEW] 时序指标脚本——Stage A 期间并行落地、Stage B 判读前完成 no-KV 基线校准；未就绪则 K2 顺延而非放行。*弃选：用现有端点均值凑合判读——端点均值正是先验 #6 明令禁止的口径。*

## 4. 与 lora368 主线的资源/检查点协调

- **C1（节点）**：主线 `_correct` 占 5 节点（mc-node01,c-node03–06，`train_stage1_lora_cfr_368_correct.sbatch:3-5`）至 step 22000（monitor 上限，`scripts/training/monitor_lora368c.sh:14`），descendants（reweight→31140 / rwtag→20784 / neworg）继续占用。memory Stage A/B 全按 1 节点设计，优先调度 c-node07（单节点 NVLink 可用、跨节点 NCCL 挂死已被主线排除，`train_stage1_lora_rwtag.sbatch:18-20`）——与主线零竞争。Stage C 需 4 节点 ×3–4 天，与主线排程窗口协商，建议卡在主线两个 5000 步区段之间。*弃选：Stage C 抢占 5 节点主线窗口——主线 monitor 会按名字自动重提（monitor_lora368c.sh:10-16），排队冲突双输。*
- **C2（基座 checkpoint 冻结）**：Stage A 分叉点钉死 `_correct/checkpoint-19500`——与 reweight/rwtag 同源（`stage1_lora_reweight_368.yaml:102-103`），memory 结果可与主线 descendants 横比；并在**同一 ckpt** 上重跑一次 no-KV VS24 基线（step-5000 的 0.069 属另一 campaign 谱系，只作锚不作对照组）。*弃选：追最新 ckpt——移动基线毁掉全部消融可比性。*
- **C3（checkpoint 卫生）**：memory run 独立 `output_dir` + symlink 播种模式复用（sbatch:28-36）；500 步/limit 30 沿用（`stage1_lora_cfr_368_correct.yaml:98-100`）；恢复前必须先落地 D11 的 FORCE_LR 按 param-group 扩展——主线 `HELIOS_FORCE_LR=1` 会把 memory lr×5 压平为单一标量（`train_helios.py:1086-1099`），这是复用主线脚本的第一颗雷。
- **C4（工具链先行合并）**：`save/load_extra_components` 第 5 节、merge 工具与 EMA 的 memory 分支（节 2 D12）须在 Stage B 前合入——对主线是纯加法（默认 `is_enable_evolving_memory: false` 门控），可先行合并不阻塞主线；否则 Stage C 与评测无 checkpoint 可消费。
- **C5（决策冻结）**：请主线在 memory M1（Stage B 通过）前书面冻结两项：底座不切 Wan2.2（R7）、VQ 不先行（R6）；任一提前变更触发 K4。
- **C6（评测共建）**：[NEW] 时序指标脚本按 `external_command` 注册为共同资产——主线 checkpoint 同样受益（现 campaign 的 "drift" 也只是端点差，`docs/VS24_EVAL_ANALYSIS.md:61-71`），落地成本由两条线分摊。
