# Findings — Echo-Infinity → Helios 迁移研究

**日期**: 2026-07-22 | **状态**: 研究完成，设计文档已交付
**主交付物**: `docs/echo-to-helios-migration-design.md`（总纲 + 四章，119KB，全部 file:line 锚定）

## 核心结论

1. **迁移可行，但研究文档的四个设计承诺被代码现实推翻/修正**（详见总纲第二节偏差表）：
   - "KV 拼接第三组" → **token 插入**（Helios 无跨层共享 KV 先例，t=0 AdaLN 只作用于序列 token）
   - "t≈0 写入源" → distilled 最后一次调用在 **stage-local σ≈0.5**（非 t≈0）→ Enc 的 **FiLM(σ_last) 必选**
   - 单一代码路径 → **双路径**：发布推理走独立 `helios/diffusers_version` 镜像，必须 M1（训练链）/M2（镜像）双移植
   - "Echo 静态 query 先行课程" → Echo 无此阶段（两个 DMD 阶段从第一步启用 memory）；Stage A 是 Helios 数据通路强加的新阶段

2. **统一设计基线**（第四章 R0/D1 仲裁后）：token 插入读机制 + M = N_Q×mid-tier 网格（720@368×640）+ 最后调度步 last-layer hidden 写源 + Enc 2 层 cross-attn/gate bias 0.75 + (0,1) 分数 RoPE id（A9 消融裁决）+ transformer 正式子模块全秩归属 + 三阶段课程（A: TF 静态 → B: TF unroll U=4 → C0 warm-up → C: SF/DMD ≥4 sections）。

3. **Stage C 不可省的代码级论证**：TF-only 写门 = Echo 批评的手写规则等价物；且现有 DMD 基建三个硬约束——Stage-1 rollout 死代码（`utils_helios_post.py:679`）、GT 模式断言锁单 section（`:2349,:2966`）、出口步随机采样——必须用 self-forcing 变体 + `dmd_num_latent_sections_min≥4` + σ_last 捕获的专用 rollout。

4. **工程雷区**（复用主线脚本前必须处理）：
   - `HELIOS_FORCE_LR=1` resume 覆写压平 param-group lr（`train_helios.py:1086-1099`），祖先配置明文依赖
   - LoRA all-linear 扫描会捕获 memory 的 Linear → 必须 `exclude_modules` + 注入后断言
   - `from_pretrained` 静默跳过形状不匹配键（`transformer_helios.py:1767-1776`）→ N_Q sweep 需显式加载断言
   - EMA step 位置 zip 静默截断（`create_ema_zero3.py:245-250`）→ 初始化前比对 (name, shape) 列表
   - rank-128 (A/B) → rank-256 (C) adapter 不可互载 → memory 组件全秩 + C0 warm-up 交接
   - detach 次序：必须"下一次写之前、本次读 backward 之后"，否则 bptt=1 下 Enc 永无梯度

5. **两项 OBSERVED ABSENCE 影响路线图**：
   - 仓库无任何 history 离散化/VQ 实现（研究文档 Phase 0 是纯规划态）；仲裁：memory 先行，VQ 排 M1 后独立分支
   - 评测侧无 within-video 斜率后端（只有端点聚合）→ 4 类新时序指标以 `external_command` 挂入 long_video_eval，检测手段本身是交付物

6. **预算与协调**：Tier-0 ≈ 5.3–5.9k H200·h；c-node07 被 rwtag pin 死仅可条件复用；基座冻结 `_correct/checkpoint-19500`；K3 采用配对相对口径（历史 0.069 属另一 campaign 谱系，绝对阈值会误杀）。

## 事实修正记录（实现期新增）

- **设计 D11 被代码证伪**：utils_helios_post.py 有 4 处对 transformer 返回值的二元组解包（:329/:3212/:3295/:3350，DMD/GAN 路径），"恒定三元组、所有调用点索引 [0]" 不成立 → 实现改为条件返回（仅 `capture_last_hidden=True` 时三元组），存量调用点零改动。
- **测试陷阱**：`init_weights()` 将 `proj_out` 权重与 bias 全部清零（transformer_helios.py:1599 + 基础循环）→ 新构造模型输出恒零，任何基于输出的断言（等价性/梯度）都会空洞通过或假失败。tiny 模型测试必须重随机化 proj_out。
- **GPU 测试要点**：flash-attn3 只接受 CUDA + fp16/bf16 张量；测试在 CUDA 可用时自动上卡转 bf16（顺带覆盖了 bf16 计算 + fp32 state 不变量的组合）。
- **决策记录**：真模型测试的环境与 ckpt 选型见 `docs/specs/2026-07-22-real-model-test-checkpoint-decision.md`（P1 冒烟 → Helios-Base + A1@19500 装配臂；P2 anti-drift → Distilled 为主）。

## 事实修正记录（研究期）

- 早期勘察误读：`infer_helios.py:290` 的 `is_amplify_history=True` 只是 loader shim（transformer 已构造完毕），**不激活**运行时放大；冻结基线实际 `is_amplify_history: false`。
- Echo "+10.6% 吞吐开销"在 Echo 仓库无基准佐证（源自论文转述），已撤销"代码事实"地位，验收以实测 ≤12% 为线。

## 文件索引

- `docs/echo-to-helios-migration-design.md` — 主设计文档（总纲 + 四章）
- `logs/research/design-{module,training,inference-eval,risks}.md` — 四章修订版单行本
- `logs/research/design-risk-experiments.md` — 风险节修订前草稿（已被 design-risks.md v2 取代，留档）
- `logs/research/read-*.md` — 6 份深读报告（英文，file:line 锚定）：echo-mem / echo-train / echo-infer / helios-model / helios-train-infer / xiangbo
- `plan.md` — 研究计划（已执行完毕）

## 研究方法记录

22-agent 工作流（Run ID `wf_e0612e6f-c7e`）：6 并行深读 → 3+1 设计（风险节基于前三节成稿）→ 每节 2 路对抗验证（代码锚定核查 + 可行性反驳）→ 修订。验证轮共发现 26+ 项 issue（含多个 blocker），全部在修订版解决或以脚注反驳。跨越多次进程重启，靠 journal 缓存 + 脚本手术（空返回 fallback/retry）无损续跑。

## GPU 冒烟结论（2026-07-22 深夜批次）

- P1-B 臂（Helios-Base 直载）13/13 PASS；P1-A1 臂（transformer_init + lora368@19500 + partial 装配）9/9 PASS → 生产加载与训练谱系装配两条路径均与 memory 模块兼容，off-path 等价在真权重成立。
- rollout 冒烟 run1：状态机全对（3 写/队列[3,4]/逐条目σ），开销 +5.6%（验收线 12%），NaN 根因 = 冒烟自身调用偏离验证采样配置（8步+固定mu=1+空负提示），产品代码无缺陷（Sol worker 对照 log_validation/infer 双基准 + diff 复核）。
## 基础设施与数据资产发现（2026-07-23 凌晨）

- **368×640 语料失踪（硬阻塞）**：旧路径在 xiangbo reorg 的 delete_list.sh 中；canonical tree yaml 计划了 `human_single/latents`（n=168431 硬链接）但目录从未创建；demo latents 同灭；原始视频+manifest 完好可重编码。恢复由用户操办。
- **mc-node01 闲置真因**：非仅 GPU2 硬件——`nvidia_uvm` 不干净卸载（07-22 20:08）致整节点 CUDA init 失败；rmmod/modprobe 修复后 7/8 卡可用。nvidia-smi 正常≠CUDA 可用（NVML 不经 uvm）。
- **Slurm 24.05.7**：partition 级 `DefMemPerGPU/DefCpuPerGPU` + reconfigure 两次复现全节点注册 INVAL（配置同步与否无关→参数触发）；无公开 bug 票；替代 = job_submit.lua（调查报告入库）。恢复程序：配置一致 → 逐节点重启 slurmd → resume。
- **trainer 配置快照校验**：output_dir 里的快照会在重启时逐键比对，schema 演进后旧 scratch 目录必须清理。

## 实现期对抗审查发现（2026-07-23，Stage A Task 1/2 批次）

- **上游既有缺陷（值得同步主线）**：stage-1 dataset 的 `_epoch` 是普通 int，persistent DataLoader workers（谱系配置 8 workers）持 fork 副本，主进程 `set_epoch` 永不到达——`choice_idx` 的逐样本 seeded 流跨 epoch 塌缩为创建时 epoch 的固定值（审查 worker 用 StatefulDataLoader 实测复现；影响 xiangbo 现行训练的跨 epoch 数据增广）。本 fork 修复：epoch 放 `multiprocessing.Value`（fork 继承下 set_epoch 实时可见，spawn pickling 降级为冻结 int 不劣于现状）。
- D4 审查 blocking（已修 e064ad9）：低分辨率桶驱逐帧曾取全分辨率 source timeline，而写前向的 X_Noisy 必须随桶分辨率；修为双 timeline 角色分离。审查同时确认：既有 history 路径本就是 full-res 条件契约（有意的混合分辨率训练）。
- D6 审查 minor（已修）：非事件样本 U=4 rollout 有 ~98.4% 概率中途切换 caption 版本（逐 section 独立全局随机抽）→ 裁决为一次 rollout 只抽一次并复用；事件样本逐 chunk 映射不变。
- D6 审查证实项：torch.randint 首抽与旧行为逐位等价（Torch 2.10 实测）；`num_frame` 文件名字段确为 RGB 帧数、`//33` 与离线 chunk 数一致；filter 后 bucket 重建无 stale-index。
- Task 3a 审查：CLEAN（逐语句等价核对；memory_tokens=None 与缺省同路径；唯一调用点关键字调用不受损）。
- Task 3b 审查 blocking（已修）：单写 helper 的全零 early-return 是逐 rank 本地决策——多卡下 rank 间 wrapped-forward 次数不一致 ⇒ buffer-broadcast collective 错位 ⇒ 训练卡死，单卡测试不可见。修复双管齐下：写掷币改为 (seed, global_step) 确定性导出（全 rank 一致 + resume 稳定）；early-return 移除（写步全 rank 跑捕获前向，全零 batch 由 blend 丢弃）。**smoke 门升级：必须 ≥2 卡 DDP 且覆盖 rank 间混合全零/有效驱逐批次。**
- Task 3b 审查反驳项（重要排险）：`find_unused_parameters=True` 与图外 update **不是**失败模式——DDP 自 forward 输出反向遍历会沿 `memory_tokens` 输入边继续到 Enc/query_init 上游（审查 worker 2-rank 最小复现确认梯度可达）；D4 mask 方向、capture 契约、blend dtype/device、trainer 作用域、逐 micro-step reset 语义均核对通过。

- **rollout 冒烟 run2（终局）：8/8 ALL PASS**（job 5552, c-node06, ~48 min）——修正采样配置后双臂 latents 均有限，NaN 根因判定实锤；状态机与 run1 逐项一致；开销 +5.5%、峰值 43.3 GiB；σ_last=0.122≠0 实测确认 FiLM(σ_last) 写条件化为必要设计（呼应总纲 §二 σ_last 偏差项）。verdict: `logs/research/rollout-smoke-verdict-run2.md`。**管线状态机 GPU 门全绿，M1 训练链四批次（模块/transformer/trainer/管线）全部收口。**

## 数据恢复与训前门收口（2026-07-23 晚）

- **语料备份存在且可用**：R2 `openhumanvid-backup/Xiangbo_july_8/helios_organized/` 是 xiangbo reorg canonical tree 的完整备份（human_single: latents.tar 2.53TB=168,431 clips + mp4_cfr.tar 167GB + captions.jsonl + dataset.yaml，provenance 指向已删除的原路径）。恢复通道 = rclone r2: remote 流式 `cat|tar`；tar 顺序格式支持 `--count/--offset` 切片与断点扩容,无需全量下载。
- **子集规模决策**：BeeGFS 97% 满（2.3T 剩）→ 40GB 头部切片（2,869 clips ≈ 语料 1.7%）。实测 tar 头部即含 121–501 帧混布,sections 直方图 3:830/4:993/≥5:1,046——全零驱逐(3 sections, k≤2)与有效驱逐(≥4 sections, k≥3)两侧充足,足够 smoke/DDP/小 pilot；**真 Stage A 训练必须全量恢复后 REPOINT**（主 YAML 有醒目注释）。
- **训前门全关**：②单卡 smoke（3 步,checkpoint 84 partial=78 memory+6 patch,query_state 零泄漏,v2 cache 生产原子落盘）③双卡 DDP（5 步零 NCCL 错误）+ 确定性重放证据（step 3/4/5 跨 rank 混合驱逐 5 对,双次重放逐字节一致,logs/research/ddp-smoke-draw-replay.md）——3b 审查设立的 DDP 验证条件全部实证。
- **latent .pt 载荷结构**（实测）：`{vae_latent (num_sections,16,9,46,80), prompt_embed (512,4096), prompt_embed_short (512,4096), prompt_raw, first_frames_image}`——离线预切 section、双 caption 版本内嵌（caption_version 抽签的物理来源）。
- **v2 cache 缺陷回移**：上游移植的对抗审查发现 `_validate_cache_payload` 只验 `samples[0]`（混合合法/非法载荷漏过→后续 single_res 过滤 TypeError 而非静默重建）；fork 同缺陷已修（878a6f0）+ 上游 PR 版本已含全样本校验。

## P2 Task 0 门拦获:Stage-2 采样路径缺 memory 捕获契约(2026-07-27)

- **缺陷**:训练侧 `pipeline_helios.py` 的 capture 三元组契约只在 Stage-1 `sample` 实现(:644-646);`stage2_sample` 裸返 latent(:884),而 section 循环在 `mem_enabled` 时无条件三元组解包(:1511-1512)→ Distilled 配方(is_enable_stage2)12 采样步跑完后 `ValueError: not enough values to unpack`。
- **为何此前全绿**:rollout run2 走 50 步验证配置 = Stage-1 路径;P2 Task 0 的 GPU dry-run 门(job 5667, mc-node01, run1 日志 `results/p2_task0_dryrun_run1.log`)正是为在 66-section 长 rollout 前暴露配方映射缺口而设 —— 按设计拦截。
- **通过项**(7/9):Distilled 6 shard 无 memory tensor、模块存在、load 后 query_state=None、reset 后 = M₀、training-side pipeline、linear dynamic shifting、proj_out 非零有限。
- **修复方向**:stage2_sample 镜像条件返回契约(仅 capture_last_step=True 时三元组;捕获 = 最后 stage 最后调度步 + 该步 σ),非捕获调用方零改动。

- **整节点内存记账再次实证(2026-07-27,c-node08)**:4 个单卡 A/B 作业不带 `--mem` 提交 → 首作业按 DefMemPerNode=UNLIMITED 吃满节点内存记账,其余 3 个 PEND(Resources/ReqNodeNotAvail),8 卡节点被 1 卡作业锁死;显式 `--mem=200G --cpus-per-task=16` 重发后 4 作业即刻并行(5677-5680)。job_submit.lua 默认配额方案的直接依据。另:后台包装脚本不 `wait` 会孤儿化 srun(仍存活但失去完成通知),包装必须 `wait`。

## 2026-07-27 晚:c-node08 泄漏 Prohibited GPU 导致 CVD 错位(jobs 5719-5721 三连败根因)

**结论**:c-node08 有一块 Prohibited 计算模式的卡(nvidia-smi 显示 4 MiB 常驻)不在 Slurm gres 管辖内,却会漏进每个作业的 device cgroup;cgroup 内重编号后它常占 index 0,而 Slurm 一律设 `CUDA_VISIBLE_DEVICES=0`,于是分到物理 GPU 2/3/4 的作业实际全指向坏卡 → `cudaErrorDevicesUnavailable`(5719/5720/5721,EXIT_CODE=1)。分到物理 GPU 0 的 5718 因坏卡排在其后而幸免,并证明与 xiangbo 同卡叠加(41.7+50=91.8GB)完全可行。
**修复**:sbatch 内按 `compute_mode==Default` 重映射 CVD(`84df58e`),补发 5722-5724 全部 `REMAPPED cvd=1` 起跑。
**附带事实**:mc-node02 也有一块同签名 Prohibited 卡(GPU5, 4 MiB)——疑似逐节点屏蔽坏卡的管理惯例;scancel 在 c-node08 可触发 "Kill task failed" 自动 drain(本日第二次,resume 程序同 c-node05)。
**排除**:非 VRAM 不足(余量 ~99GB)、非 Exclusive 模式(实测 Default)、非 xiangbo 的 Slurm 占用(其任务在 Slurm 外,账面 idle)。

## 2026-07-28:P2 协议 prompt 合规性缺陷(用户质询证实)+ 基座谱系澄清

**结论 1(prompt 不合规)**:训练 caption 与团队标准推理集均为结构化格式(`<header>/<event>/<role>/<Background>`,见 captions.jsonl 首条与 `eval_prompts_rep50/3000.csv`),而 P2 r1/r2 所用 prompt 为 `eval_prompts_vs24_long` test split 的 **raw 裸句列**(eval_norm/long/prompt.txt,20 条)——r1 计划期因 2178 帧参考时长精确匹配 66-section 协议而选中,未执行结构化改写规范,r2 为保种子/prompt 配对继承之。**影响**:三臂同 prompt 同 seed,内部相对结论(静态塌缩消除、饱和斜率翻转)仍有效;但全部臂处于 backbone prompt-OOD 状态,绝对漂移量级不作数,正式 Stage A 验收必须换 `eval_prompts_rep50`(实测 50 个结构化 case,id 3000+)。
**结论 2(基座谱系)**:Stage A 基座 = Helios-Base + `stage1_lora_cfr_368_correct/checkpoint-19500` LoRA 合并(设计 D8/主线协调明文冻结,保 C2 配对可比);用户提及的 27500 属 `stage1_lora_reweight_368`(同样自 _correct@19500 分叉的 rwtag 重加权 campaign,现已完结于 **checkpoint-31140-final**)。迁基座 = 设计变更,候选时机为 Stage B 选基或 Stage A-v2。

## 2026-07-29:ckpt 质量判定体系(四层)与已知盲区

**结论:memory ckpt 的质量只能由行为 A/B 判定,训练侧信号(loss)在本设定下无判定力,且短时域结果不能外推长时域。**

四层判定(由浅到深,前两层是前提而非质量):
1. **结构完整性**(每 ckpt 落盘即查):LoRA 814 tensors 零 memory 键泄漏 / `transformer_partial.pth` 84 键(evolving 38 + patch 6 + per-head key_scale 40)/ `query_state` 缺席 / 权重全有限。只排除保存损坏。至今 40+ ckpt(pilot 9 + slice512 9 + full 24)零破坏。
2. **训练信号(弱)**:flow-matching 逐批 σ 方差 + 冻结骨干 ⇒ loss 全程 ~0.08-0.09 平台,**无判定力**;有用的是相邻 ckpt 的 evolving rel-L2 移动量(每 500 步 0.3-0.8%)与 gate/key_scale 演化(gate 由"对输入零响应"变为内容依赖)——只回答"在学",不回答"学得好"。
3. **行为 A/B(现役主力)**:三协议 × 种子配对 × 三类对照(off / untrained / 各 ckpt)⇒ ① 病理计数(冻结 motion→0、爆冲 sat 失控)= 一票否决;② sat Theil-Sen 斜率距零度(给出步数演化曲线);③ 8min 回复力(偏移是否末段回收)= 目前区分度最高的单项。
4. **交叉验证**:同步数 × 不同语料横比(pilot/slice512/full @4000)暴露"多样性 vs 复读"结构;不同时域(58s/90s/8min)暴露漂移形态的时域依赖。

**已知盲区(按重要性)**:① 身份/背景一致性未量化——记忆的核心价值只能人眼看站点,补法 = CLIP 相似度轨迹(v5 协议);② **过拟合不可见**——pilot 44 epochs 的完美斜率可能绑定其 2,869 clips 分布,现成检法 = rep50 尚未使用的 case 8-15 做 held-out 泛化检验(≈2.7 GPU·h,可直接仲裁 pilot vs full 选基);③ 观感质量无分数——团队现成 `scripts/evaluation/metrics/`(DOVER/HPSv3/PickScore)可对已有 130+ 支视频离线补算,无需新推理;④ 可塑性(哪个 ckpt 进 Stage B 后训得更好)只有 Stage B 能答。

**跨协议durable 结论:短时域指标不预测长时域行为**。两次独立证实:r3(90s,off 温和去饱和 −0.137)vs r5(8min,off 中后段过饱和 +0.219);full@8000 切换协议优于基线(−0.263)但 8min 劣于基线(+0.297,3/5 失控)。⇒ **8min 协议从"可选"升级为选基必测项**(判定见 `logs/research/p2-longhorizon-verdict-r5-2026-07-28.md` 附录)。

## 2026-07-29 深夜:抗漂移机制定位 + FramePack 离散化真相 + pinghe 节点守卫缺口

**结论:漂移的因果链是历史 latent 误差逐段复利,与记忆的"补窗口外信息"是两条独立链路;记忆提供 onset 之后的回复力而不推迟 onset。设计契约见 `docs/specs/2026-07-29-history-projection-design.md`。**

1. **记忆不预防漂移(onset 细算证实)**:pilot@4000 的 onset 中位 173s(5/5 触发)比 off 209s(3/5 触发)**更早**,但 0/5 失控、5/5 末段回收(off 2/5 单调失控)。用户观测的"3min 起明显漂移"在量级上成立、作为普适阈值不成立(单 case 109–404s,off 有 2/5 全程不触发,full@8000 系统性更早 ~124s)。全部数据与 r5 判定表零分歧:`logs/research/read-r5-drift-onset-analysis.md`。新判据:主 = onset 定义(a),副 = 末段回收比(off 0.801 / pilot@4000 **0.511** / full@8000 0.809)。
2. **FramePack 的 history discretization 是训练期操作**:仅记载于 arXiv 2504.12626 **v3(2025-10 修订)**,无独立 "P1" 论文、**从未放出代码或权重**(issue #738 无人应答)。机制 = 在预计算 latent 上离线拟合 K-means 码本 Ω,训练时把每个 history 帧替换为最近质心重建(Eq.6),位置在 VAE encode 后、patchify 前,只作用条件帧。原文明文 "during training";骨干必须微调才能吃量化历史 ⇒ **免训练直接量化是新的未验证变体,不是复现论文**。超参只有 K:正文推荐 128 而 Table 2 用 256(自相矛盾,K-sweep 未公开)。消融:Δ_M(前/后 15% 帧差)全面改善(ΔClarity 3.18→2.30、ΔAnatomy 18.05→14.11),ELO 1030–1092 → 1139–1225,且运动动态范围优于 inverted anti-drifting。取证:`logs/research/read-framepack-p1-history-discretization.md`。
3. **均匀标量量化对系统性偏置无抑制作用(机制证伪,拦在 GPU 花费之前)**:按 dither 理论,偏置 b<step/2 会让约 b/step 比例的像素跳一整格,均值位移原样保留;实测 19.9% 元素跳整格。⇒ 我原先"死区吃掉漂移增量"的推理错误,quantize 从候选降级为**对照臂**(若它也有效则机制不是统计量复原)。回归护栏:`tests/test_history_projector.py::test_quantize_does_not_remove_a_sub_step_bias`。
4. **继承的 `AdaptiveAntiDrifting` 按构造治不了慢漂移**(`helios/utils/utils_base.py:743-815`,默认关闭且评测从未开启):参照是 ρ=0.9 的 EMA(≈10 chunk 记忆),慢蠕变把参照一起带走;`and` 双阈值再降灵敏度;"修正"是加白噪声(抬方差不复位均值)且作用于 `latents`、在 append 之前 ⇒ 污染可见输出。不复用。
5. **语料 latent 已在模型空间(码本拟合的最大坑)**:离线编码器存盘前就做了 `(z_raw−mean)/std`(`tools/offload_data/get_short-latents.py:104-107,229-232`),loader 原样传递。二次归一化会让码本评分从 0.5304 恶化到 0.8853。语料逐通道 std 0.65–0.86(聚合 0.823)、均值 −0.70…+0.59,**不严格等于 0/1 先验**(VAE 常量是全局的,本语料是单人子集)。k256 码本 held-out L2 1.135 / 保留方差 84.7% ⇒ α=1 投影约改动向量模长的 **37%**,属显著 OOD,必须 α sweep。
6. **pinghe 节点守卫缺口(操作风险,已修)**:实测 pinghe 在 **c-node03/04/06/07 全部 32 卡**上跑 Slurm 外进程(每卡约 57GB 常驻、**每卡余 ~86GB**),而 Slurm 报这四台 `idle`。评测 launcher 的守卫只看"Default 模式 + 60GB 余量"、**不看属主**,在 pinghe 卡上会照样放行——此前未出事仅因 launcher 硬编码了 `--nodelist=c-node08`。已在 r3/r4 两个 launcher 加显式属主守卫:GPU 进程属主含 pinghe 即 `EXIT_CODE=44` 拒跑。另:c-node08/mc-node02 上是 xiangbo(每进程约 41GB)+ 我们的作业,属规则允许的叠加。

## 2026-07-30:FramePack-P1 一手来源核实(官方仓库逐文件)+ Stage B 就绪审计

**结论 1(代码确实不存在,出处更正)**:先前记录说离散化"仅记载于 arXiv v3"不准确——官方 README 与 P1 结果页同样记载,且**明说未发布**。逐项硬证据(2026-07-30 实查):仓库 `lllyasviel/FramePack` 总大小 **76 KB**、最后推送 2025-10-16、分支只有 `main`/`windows`、无 tag;文件构成仅 `demo_gradio.py`(FramePack)+ `demo_gradio_f1.py`(F1)+ `diffusers_helper/*`,**无训练代码**;全部 10 个 py 文件对 `kmeans|codebook|centroid|quantiz|discret|cluster|vq_|torch.round` **零命中**;全 GitHub 无 `FramePack-P1` 仓库(其余命中皆 ComfyUI wrapper/F1 时代 fork)。README News:"2025 June 26 … The FramePack-P1 **will be** the next version of FramePack with two designs: Planned Anti-Drifting and History Discretization";结果页:"models and paper will be uploaded soon"。⇒ K 的正文/表格矛盾(128 vs 256)无代码可作唯一真相,免训练移植仍是我们的新变体。

**结论 2(P1 是两机制分治,只有一半可移植)**:结果页原文——"Planned Anti-Drifting predicts sections that are far away from the next section before generating nearby sections"(减少**端点之间**的漂移);"History Discretization converts all history to discretization tokens (directly apply K-Mean to the entire dataset)"(减少**跨端点**漂移,"the endpoints themselves will not drift")。**Planned Anti-Drifting 是非因果的**(先远后近),与 Helios 流式自回归 rollout 及记忆 k−2 逐出写入语义根本冲突 ⇒ 不可移植;可移植的只有离散化。另:其公开证据为 >2100 帧、6 个通用 prompt、**无基线无量化指标**的纯视频页,证据等级低于我们现行逐 chunk 指标对照。

**结论 3(Stage B 不能只加 YAML,且有静默陷阱)**:四部件三就绪——数据集已能吐 U 个连续 section(`memory_unroll_sections`,切 `19+9U`,按 U 过滤;`dataloader_history_latents_dist.py:349-362,596-623`,8/8 PASS);可微状态原语已在(`helios_memory.py:171-182` 用赋值而非原地写、不自动 detach,唯一截断 `detach_state()` :133-137,13/13 PASS);`capture_last_hidden` 逐 call 无状态可调 U 次(`transformer_helios.py:1365-1401`)。**缺 trainer unroll 循环**:`train_helios.py:1405-1435` 从不消费 `clean_all_latents`/`section_prompt_embeds`,`:1608-1634` 仍是单次 `_flow_loss` + `optimizer.step()`,无逐段 backward/state carry/TBPTT detach。**陷阱**:`train_helios.py:1575-1607` 的 memory 写入分支显式要求 `not memory_tf_unroll` ⇒ 今天打开 U=4 会正常跑完并落盘,但 `memory_tokens` 全程 `None`,**训出无记忆参与的 ckpt**(训练侧的静默 no-op)。配置 schema 与校验已在(`train_config.py:469-474,506-542`),但 `scripts/training/configs/` 下无 Stage B YAML。计算:U=4 每单元约 7-8 次 transformer forward,`gradient_checkpointing` 已启用,`offload` 与 stage1 dataset 互斥(`train_helios.py:2739-2744`)不可用作解法。

## 2026-07-30:判据修正(signed slope → |slope|)+ full@12000 切换协议判读

**结论:signed 平均斜率会让正负漂移互相抵消,不能当"距零漂移距离"用。改用 mean|slope| + 逐 case 配对比较后,全量谱系"更多步数能否救回"的答案是明确的"不能",且其自身最优点是 @4000 而非 @12000;pilot@4000 仍是唯一单调收敛到最小 |slope| 的谱系。**

**判据修正(方法论,适用于全部既往与后续切换协议判读)**:signed mean 只回答"是否存在系统性漂移方向";距零距离必须用 |slope|。历史表述更正一处:此前称 pilot@4000 "零净漂移(−0.002)"——该 0 部分来自正负 case 抵消,其 mean|slope| 实为 0.244(仍是全臂最小,结论不变,但"零净漂移"夸大)。既往 verdict 的 signed 数字不改写(保留历史),以本条为口径修正。

**切换协议 mean|slope|(n=8,种子跨臂配对,off 基线 0.661 / median 0.634)**:

| 臂 | mean\|slope\| | median | 配对更近零 |
|---|---|---|---|
| untrained | 1.302 | 0.716 | 4/8 |
| pilot@1500 → @2000 → **@4000** | 0.498 → 0.392 → **0.244** | 0.439 → 0.333 → **0.214** | 5/8 |
| slice512@2000 / @4000 | 0.593 / 0.857 | 0.549 / 0.878 | 6/8 / 3/8 |
| full@4000 / @8000 / **@12000** | 0.331 / 0.328 / **0.496** | 0.210 / 0.231 / 0.586 | **8/8** / 5/8 / 6/8 |

1. **pilot 谱系单调收敛**(0.498→0.392→0.244);**全量谱系 4000 步后掉头变差**(0.331→0.328→0.496)⇒ 跑满 2.28 epoch 不能救回全量谱系,其最优 ckpt 是 @4000。
2. **full@4000 是唯一 8/8 全 case 都比基线更近零的臂**(pilot@4000 为 5/8)⇒ 多样性语料给"普遍小幅改善",复读语料给"少数 case 大幅改善"。这是两种不同的记忆行为,不是同一指标上的强弱。
3. 单 seed、n=8、off 臂本身 range [−1.254,+1.089] ⇒ 均值差 <0.05 不作主张;上表只支持"pilot@4000 最优""full 谱系 @12000 劣于 @4000/@8000"这两条量级结论。

## 2026-07-30:held-out 泛化检验 —— pilot 优势不泛化(盲区②已闭合)

**结论:记忆的收益是"驯服极端 case",不是"普遍降低漂移";在未见 case 上驯服与扰乱大致打平。** rep50 case 8-15(训练与既有评测均未用过)× 三臂 × 切换协议,24/24 零失败。mean|slope|:off 0.706(域内 0.661,难度可比,构成有效对照)/ pilot@4000 **0.531**(域内 0.244,边际缩水约 60%,配对 4/8,中位数 0.384 略差于基线 0.351)/ full@12000 **0.768**(劣于无记忆基线,配对 3/8)。逐 case 证据:pilot 把 case 8(−1.643→−0.438)、case 13(−1.571→−0.330)驯服,却把 case 9(−0.156→−0.893)、case 12(+0.209→−1.285)弄坏。
⇒ ① 既往"pilot@4000 近乎完美校准"的表述仅在域内成立,不可外推;② 基座选择不变(pilot@4000 唯一在 8min 上 0/5 失控);③ Stage B 的学习目标应精确为"条件化干预强度(何时干预、何时放手)",而非继续压平均斜率;④ 判定文档:`logs/research/stage-a-terminal-verdict-2026-07-28.md` 附录 5。
