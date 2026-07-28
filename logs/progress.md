# Progress Log

## 2026-07-21

- [x] 通读研究文档 `agent-research/Claude_export_Memory...md`（211 行，六大迁移挑战与 Phase 0/1 实验设计）
- [x] 勘察工作区：Echo-Infinity 与 helios-team 均在 `../`；helios-team 当前即 `mid_training_xiangbo` 分支（领先 origin/main 5 commits）
- [x] 建立 plan.md，启动 22-agent 工作流（Run ID `wf_e0612e6f-c7e`）
- [x] Phase Read：6 份深读报告落盘 `logs/research/read-*.md`（~250KB，全部 file:line 锚定）

## 2026-07-22

- [x] Phase Design：3 核心设计节起草 → 每节 2 路对抗验证（共 26+ issues，含 blockers）→ 修订
- [x] Phase Risks：基于修订后三节起草风险与消融节 → 验证（19 issues 全部证实）→ 修订 v2
- [x] 工作流跨 4 次进程重启无损续跑（journal 缓存 + 空返回 fallback 脚本手术，详见 findings 方法记录）
- [x] 最终汇总：`docs/echo-to-helios-migration-design.md`（总纲 + 四章，119KB）
- [x] `logs/findings.md` 关键发现汇总

## 2026-07-22（工作仓库建立）

- [x] helios-echo 初始化为 git 仓库：`team` → 本地 `../helios-team`、`origin` → `Visko-Platform/helios-team.git`
- [x] 继承 `mid_training_xiangbo`（34f5a99），建工作分支 `echo-memory`；本地 `mid_training_xiangbo` 分支追踪 `team/`，用于持续同步
- [x] 研究语料入库（commit 6e1cc7d，16 files / 2703 行）
- [x] 调研 `../Human-Replacement` 文档管理体系（CLAUDE.md 实时记录规则、docs/superpowers specs+plans 日期命名、logs 双账本、my-docs 本地区、postmortem 规范），适配写入本仓库 CLAUDE.md「Documentation conventions」节；.gitignore 增 `my-docs/`、`results/`

## 2026-07-22（K0 基础落地，plan: docs/plans/2026-07-22-k0-memory-foundations-plan.md）

- [x] `helios/modules/helios_memory.py`：HeliosMemoryEncoder（2 层 cross-attn Enc + 门控 EMA + fp32 state 普通属性 + ctx k/v 投影 + FiLM(σ_last) 零初始化恒等 + frame mask）— commit c4d0328
- [x] `tests/test_helios_memory.py` 13 项：核心行为 5 + K0 验收（Enc/gate/query_init 梯度非零、detach 截断、BPTT=1 第二窗口可学）+ mask/FiLM + conversion-lite 严格互载 — commit 9afd50e
- [x] `train_config.py` 演化记忆 13 个 flag（默认全关，存量配置零行为变化）+ `validate_evolving_memory_config()`（D13 断言含 `dmd_num_latent_sections_min>=4`）接入 train_helios.py "For Wan" 校验区；`tests/test_memory_config.py` 6 项 — commit 592f9a3
- 冒烟门通过：`unittest discover` 19/19 OK（登录节点 CPU，团队环境 /mnt/beegfs/yuheng/miniconda3/envs/helios）
- 遗留到下一 plan：transformer 注册（设计 ch.1 D9：正式子模块、PEFT exclude、extra_components 第 5 节）、`is_enable_stage1` 断言待对照真实 Stage-C 配置

## 状态：研究阶段完成

下一步（待决策后启动实现）：
1. K0 前置项：D1/D8 仲裁签字、checkpoint-conversion 测试、`dmd_num_latent_sections_min≥4` 断言 + Enc/gate 梯度单测、Stage B smoke profile
2. 评测工具先行：4 类时序斜率指标 external_command 脚本 + D10 注意力诊断路径（Stage A 期间交付）
3. 工具链先行合并：save/load_extra_components 第 5 节、merge/EMA memory 分支（默认关闭门控，不阻塞主线）
4. 与主线协调：C2 基座冻结 checkpoint-19500 + no-KV 基线重跑；C5 书面冻结（不切 Wan2.2、VQ 不先行）

## 2026-07-22（transformer 读路径集成，plan: docs/plans/2026-07-22-transformer-memory-integration-plan.md）

- [x] `transformer_helios.py` 手术：evolving_memory 子模块注册（config kwargs 6 项）、token 插入读路径（分数 RoPE + t=0 AdaLN 复用 + guidance 分支 prefix 切分）、三分记账透传、独立 memory_key_scale（init −4→scale≈1.16）、capture_last_hidden 条件三元组返回 — commit 555f7ba
- [x] 设计 D11 修正：4 处二元组解包调用点（utils_helios_post.py）→ 条件返回替代恒定三元组
- [x] 集成测试 8 项：注册/旧 ckpt 加载/关闭路径逐位等价/读路径生效/捕获 API/端到端梯度（含 memory_key_scale）
- [x] 冒烟门双通过：CPU（SDPA, fp32）27/27；H200 c-node03 单卡（flash-attn3, bf16）27/27
- [x] 调试记录：proj_out 零初始化致输出恒零的测试盲区（见 findings）
- [x] 决策记录 `docs/specs/2026-07-22-real-model-test-checkpoint-decision.md`：环境 = xiangbo env（yuheng/envs/helios）；P1 冒烟 = Helios-Base + A1@19500 装配臂；P2 anti-drift = Distilled 为主（候选 A1/A2/A3/B/C 全列，含 revisit 条件）
- 下一步：trainer 接线（PEFT exclude / trainable_modules / extra_components 第 5 节 / param groups）→ P1 真权重冒烟 → pipeline 状态机

## 2026-07-22（AFK 批次：trainer 接线 + P1 真权重冒烟 + 管线状态机）

- [x] trainer 接线（plan: docs/plans/2026-07-22-trainer-wiring-plan.md）：extra_components 第 5 节（enable 门控 + fail-loud 加载）、build_transformer_param_groups 角色化双组、DS DummyOptim 守卫、FORCE_LR 按角色恢复、all-linear 过滤 + 注入后断言、memory_freeze_backbone 通道 — commit 43a32c1
- [x] P1 真权重冒烟（tests/smoke_real_weights.py，真 Helios-Base 14B @ c-node04 单卡）：13/13 PASS——加载容忍（缺键仅 memory）、真权重关闭路径逐位等价、720-token 读路径、捕获 [1,8640,5120]、Stage-A 式梯度穿 14B，峰值 58.3 GiB
- [x] P1 抓到并修复真 bug：fp32-kept memory_key_scale × bf16 key 的 dtype 提升 → flash-attn q/k 失配（tiny 测试因整模型 .to(bf16) 而失明；已加强制 fp32-scale 回归测试）— commit fe4063f
- [x] 管线状态机（plan: docs/plans/2026-07-22-pipeline-state-machine-plan.md）：stage1/2_sample 捕获契约（末调度步、金字塔仅末 stage、逐条目自带 σ）、__call__ 的 M₀ 初始化/k−2 驱逐写/状态导出 — commit ce62bcc
- [x] 环境归属定论（用户问询）：yuheng 名下的 env 即 xiangbo 全部作业的运行环境（其 sbatch 与 eval_env.sh 均指向之），无需切换；评测采纳其 eval_env.sh 变量 — 决策记录已修订
- 冒烟门：CPU 套件 39/39；下一批：GPU 端到端 rollout 冒烟（真权重 + enable_evolving_memory ≥5 sections，验证 k=2 首写/队列/导出）+ P2-interim 漂移 A/B 脚手架 + A1@19500 装配臂

## 2026-07-22 深夜（GPU 冒烟批次 + 文档规范化，pre-compact 快照）

- [x] rollout 冒烟 run1：状态机全过（3 写/队列[3,4]/逐σ）、开销 +5.6%；NaN 根因 = 冒烟调用偏离验证采样配置（Sol worker 取证：产品代码无缺陷、plain 路径未变）— 修正后 run2 运行中（孤儿 srun，log: results/rollout_smoke_run2.log）
- [x] A1 装配臂冒烟 9/9 PASS（814 LoRA tensors / 6 partial 键 / memory fresh 共存）→ P1 决策记录两臂全绿 — commit 86bd487
- [x] CLAUDE.md 精简重写（145→57 行）+ context stewardship 四纪律入册；verdict/运行日志归位（logs/research 与 results/）— commit ba9ca82
- [x] 调度反思与准则：agent-research/2026-07-22-agent-orchestration-retrospective.md、context-stewardship-rules-draft.md
- 恢复入口：agent-research/2026-07-22-echo-memory-progress-tracking.md 顶部 POST-COMPACT 节

## 2026-07-22 深夜（post-compact 恢复批次）

- [x] rollout run2 终局 **8/8 ALL PASS**（+5.5% 开销 / 43.3 GiB 峰值 / 双臂有限）→ verdict `logs/research/rollout-smoke-verdict-run2.md`；管线状态机 GPU 门关闭
- [x] 恢复期根因修复：run2 日志"消失"= srun stdout 重定向按规范整改前的旧路径（agent-research/）武装，全部哨兵盯新路径（results/）而永久静默；BeeGFS 拒绝 rename 打开中的文件 → 清 6 旧哨兵、重架真路径哨兵（终态自动复制到 results/）。教训：长作业跨越路径规范变更时，用 /proc/<pid>/fd 核实真值通道实际落点，不信"应有路径"
- [x] P2-interim 漂移 A/B 脚手架计划 + Stage A 实质改造计划（D4-D7）两份计划文档 — commit a02c365
- [x] P2 Task 1 时序斜率指标脚本 — commit 382fe09（对抗验证抓到 motion 分辨率不可比 blocker，已修：Farneback 前缩放到固定 384×640 + 跨分辨率不变性测试；边界指标 L2 默认因环境无 lpips）
- [x] Stage A dataloader 证据卡片 11/11 → logs/research/read-stage-a-dataloader-anchors.md（stage1 dataset = dataloader_history_latents_dist.py）— commit 7618eb7
- [x] D13 断言补差 + 加固（commit fe1d7a5 → 7618eb7）：stage1 蕴含 / dataset XOR one-hot / kv-cache 互斥；对抗审查抓到 validation_config 可选默认的静默跳过漏洞，改为必需参数 + TypeError 测试；config 套件 11 项、全套 46 绿
- [x] ultracode 工作流（wf_de941cfd-8ce，4 agent）：三工作流并行 produce→verify，两个真实缺陷（motion 不可比、可选参数漏洞）在进 GPU 验证前被独立对抗视角抓到（兑现 stewardship 纪律 C）
- 待做（顺序）：Stage A Task 1-3 主刀实现（D4 驱逐切片 / D6 展开参数 / D7 section 级 loss 原语，证据卡片已备）→ 对抗审查 → 1 节点 smoke profile；P2 Task 0 配方映射 GPU dry-run 可交错

## 2026-07-23 凌晨（Stage A Task 1/2 落地批次）

- [x] Task 1 / D4 驱逐切片 — commit 7f422b6（TDD：红灯测试先于实现；纯静态 helper `_compute_eviction`；k∈0..4 手算序列全验证；默认关闭路径含 RNG 状态逐位纯净）
- [x] D4 对抗审查抓到 blocking：低分辨率桶的驱逐帧取自全分辨率 source timeline，而 D5 中驱逐帧是写前向的 X_Noisy、必须随桶分辨率——修为双 timeline 角色分离（X_Noisy=桶 `continue_vae_latent`，history=全分辨率 source，与既有 history 条件契约一致）— commit e064ad9；其余审查角度全 CLEAN
- [x] Task 2 / D6 展开参数 — commit 3729bed（start_section_idx 并入逐样本 seeded 流并返回，修 F3；首抽与旧行为逐位等价有 parity 测试；`return_rollout_metadata` 门控 start+逐 section prompts；载入后内存过滤 sections=num_frame//33——有意偏离设计的"cache 按 U 失效"：共享数据目录的 cache 会被不校验的旧读者误载，改为不写 U 特化 cache；trainer 接线三 flag）
- [x] Task 3 实现决策预先固化（计划文档 Task 3 节）：批级写决策+样本级 blend、token 级 mask 展开、TF 写 σ_last=0、驱逐分辨率角色
- [x] D6 对抗审查闭环 — commit 09faaf3：major=persistent workers 收不到 `_epoch`（上游既有缺陷，已入 findings 待同步主线；修为 mp.Value 共享 epoch）；minor=U-rollout 中途切 caption（修为一次 rollout 抽一次并复用）；七个反驳角度全确认
- [x] trainer batch-prep 证据 7 卡全中 → logs/research/read-trainer-batch-prep-anchors.md（tier: long=[:16]/mid=[16:18]/short=[x0,1x]；写前向必须复用 `prepare_stage1_clean_input_from_latents`；t=0 需显式传入）
- [x] Task 3a：`_flow_loss_section` 原语抽取 + `memory_tokens` 贯通 — commit 4929521（无内部 backward，caller 掌管 sync 边界；`_flow_loss` 行为等价保留）
- 测试面：全套 61 绿
- [x] Task 3b：单写模拟 helper `_memory_single_write` + trainer 接线 — commit e36d659（no_grad t=0 捕获前向 + 图内门控 update；批级写决策+样本级 blend+token 级 mask 全兑现）
- [x] 3a/3b 对抗审查闭环 — commit 48132c1：3a 等价性 CLEAN；3b 抓到 blocking=全零 early-return 逐 rank 决策致多卡 collective 错位卡死（修：确定性掷币 + 写步全 rank 跑前向）；审查同时反驳了 find_unused_parameters 担忧（DDP 沿输入边遍历，2-rank 复现梯度可达）
- 测试面：全套 64 绿
- [x] Task 5 配置分叉 `stage1_lora_mem368_A.yaml` — commit f843df8（Sol worker；程序化逐键 diff 恰 16 字段变更；13 个 memory 字段全显式；worker 抓到 max_train_steps 绝对语义 23500=19500+4000，主 agent 独立核验 train_helios.py:1154 确认）
- 测试面：全套 69 绿。**Stage A 实现面（Task 1-5）全部完成并经对抗审查。**
- 待做（训练前验证梯子，依次）：② 单卡真权重微型训练 smoke → ③ ≥2 卡 DDP smoke → ④ 用户拍板 Stage A 真训练。3c U-展开（Stage B）与 P2 Task 0 可交错。

## 2026-07-23 凌晨后半（smoke 迭代 + 基础设施夜）

- [x] 分叉机制修正 + merge 基座产出：`_merged/stage1_lora368_correct_merged19500`（MERGE RESULT: OK，28.6GB bf16，fp32 融合）— commit 6962644 + 工具 merge_lora_full_for_helios.py
- [x] smoke 迭代抓真问题 ×2：run1'=force_rebuild 不变量；run3'=**D2 缺口**（trainer 构造字典漏 memory 键→模块未建→零可训练参数）— commit 43b7d88（memory_frame_hw 升格 config 字段，14 键断言）
- [x] Slurm 事故 ×2 + 恢复 ×2（postmortem：logs/2026-07-23-postmortem-slurm-reconfigure-inval.md）：DefMemPerGPU/DefCpuPerGPU 在 24.05.7 触发全节点注册 INVAL，部署顺序假设被第二次试验推翻；调查报告（logs/research/slurm-defcpupergpu-inval-investigation.md）推荐 job_submit.lua 替代路线，待用户拍板
- [x] mc-node01 修复：GPU2 硬件问题之外，nvidia_uvm 不干净卸载致**整节点** CUDA 死（这才是它闲置的真因）；rmmod/modprobe 后 7/8 卡可用
- [x] 388k 扫描修复（用户批准）：v2 cache — commit be8ce09；对抗审查 3 blocking+2 major 全修 — commit 4108ab1（原子发布/自愈全覆盖/只读降级/断言范围化/配置翻转激活）；测试面 79 绿
- [!] **硬阻塞（需用户/xiangbo 决策）**：368×640 latent 语料已从盘上消失——重组只写了 yaml 计划（human_single/latents, n=168431）但目录从未落地，demo latents 同灭；原始视频（videos_cfr_int + bprime）与 manifest 仍在，可重编码。smoke run4-6 均未及数据关（分别死于 CUDA/配置快照/排队+取消）。选项：A 问 xiangbo 数据去向（S3/他处）；B tools/offload_data 重编码（先子集起 smoke，后全量）；C wan22 704×1280 语料不可行（C5 冻结 + 谱系不匹配）

## 2026-07-23 晚（R2 数据恢复 + 训前门②③关闭 + 上游 PR 就绪）

- [x] **语料恢复（硬阻塞解除）**：R2 备份 `openhumanvid-backup/Xiangbo_july_8/helios_organized/human_single/` 验明正身（dataset.yaml provenance 指向原路径；latents.tar 2.53TB / 168,431 clips）。BeeGFS 97% 满（剩 2.3T）→ 全量放不下，**决策 = 40GB 头部流式切片**（`rclone cat --count | tar -x`，tar 顺序性支持日后 `--offset` 断点扩容）→ `/mnt/beegfs/siyuan/dataset/human_single_368x640_subset/latents`：2,869 files（删 1 截断尾）+ captions.jsonl + dataset.yaml。验证：sections 直方图 3:830/4:993/≥5:1,046（121–501 帧），torch.load 三点抽检 OK（vae_latent (k,16,9,46,80) 离线预切、双 caption 版本、UMT5 512×4096）— 混合驱逐两侧样本充足
- [x] 两份 Stage A YAML 指向子集（正式配置带 REPOINT 注释：真训练前必须换全量）— commit be4d654；smoke scratch 清理
- [x] **smoke run7（门②）ALL PASS**：mc-node01 单卡，3/3 步 loss/grad 有限，EXIT_CODE 0；checkpoint 结构精确（partial 84 = 78 memory 键 + 6 patch conv；**query_state 不在 state_dict**）；v2 cache 生产首落盘且无 tmp 残留（原子发布实证）— log: results/smoke_stage_a_run7.log
- [x] **DDP smoke run1（门③）ALL PASS**：mc-node01 双卡 5/5 步，EXIT_CODE 0，零 NCCL/Traceback，checkpoint 结构与单卡一致 — log: results/smoke_stage_a_ddp_run1.log（job 5626）
- [x] 门③证据不靠概率：Sol worker 确定性重放（同 seed/采样器，双次逐字节一致）→ R0=[0,8,0,0,0,9,8,0,9,9] / R1=[0,9,0,0,9,9,0,9,0,0]，**step 3/4/5 存在跨 rank 零/非零混合驱逐**（5 个并发 pair）→ 3b blocker 的验证条件实证闭环 — logs/research/ddp-smoke-draw-replay.md
- [x] 上游 PR 就绪（等用户 push）：worktree helios-upstream-pr 分支 fix/stage1-dataset-epoch-and-cache-v2（73b2d47 epoch + 0f27099 cache，基于 team@34f5a99）；7-agent 工作流 + 双镜头对抗审查 + 两轮修复 PASS + 主 agent 亲核 diff（4 文件 +457/−22 零泄漏）；PR 正文 logs/research/upstream-pr-dataset-fixes.md — commit 972bde2
- [x] 审查反哺 fork：`_validate_cache_payload` 只验 samples[0] 的缺陷回移修复 + 回归测试 — commit 878a6f0（cache 套件 10/10）
- **状态：训前验证梯子全部完成（①单测 79 绿 ②单卡 smoke ③DDP smoke + 重放证据）。Stage A 真训练只差：全量语料恢复 + 用户拍板。**
- [x] **上游 PR 已提交**（用户批准）：https://github.com/Visko-Platform/helios-team/pull/1（base mid_training_xiangbo；GitHub 尖端核验 = 本地基点 34f5a99，无混入）

## 2026-07-27 · 10h 自主窗口:双轨训练 + 语料扩容

- **Gate 4 放行**:用户指令"完成所有预期任务并推进得到实验训练结果"+ 存储已清理、多节点空闲、可酌情下载数据。
- **Pilot(保底出结果)**:job 5666 @ c-node06 8×H200,`stage1_lora_mem368_A_pilot.yaml`(= 主配方,独立 output_dir,40GB 子集,4000 步 ≈ 44.6 epochs)。日志 `results/stageA_pilot_run1.log`,watcher 双通道(失败签名 + 15min 步数/ckpt 巡检)。
- **带宽实测**:R2 单流 25.9 MiB/s,4 流 40 MiB/s,12 流 25.6 MiB/s(不扩展,园区出口硬顶)→ 全量 2.53TB ≈ 17-26h 窗口内不可行。
- **扩容决策**:流式 512GiB 头部切片(`rclone cat --count | tar -x`,~5.6h,≈34k clips = 12× pilot 语料)→ `/mnt/beegfs/siyuan/dataset/human_single_368x640_slice512g/`(独立目录,不混写子集)。尾部截断 member 为预期终态,恢复后删截断尾 + torch.load 抽检再启用。`stage1_lora_mem368_A_slice512.yaml` 已备好(c-node04 待发)。
- **上游 PR#1**:OPEN,无人工 review,Copilot CI 因 GitHub Actions 账单未启动 —— 无可操作项。

### 2026-07-27 · P2 Task 0 门关闭(先拦缺陷后全绿)
- run1(job 5667)按设计拦获 stage-2 捕获契约缺失 → 修复 `b671ba6`(4 行条件返回 + 3 回归测试,TDD 红灯复现,82/82 CPU 全绿,双镜头审查 1 轮闭环)→ run2 门重跑:1-section 9/9 + 5-section 双臂 17/17 ALL PASS(`results/p2_task0_dryrun_run2.log`)。
- 实证:σ_last 逐 section 非零([0.391, 0.657×4],首节 amplify 效应)→ FiLM(σ_last) 设计必要性落地;writes=sections−2;off 臂真 no-KV;harness 入库 `0711773`。
- 教训:workflow 内 GPU 执行 agent 被 structured-output 终止时后台 srun 连带 CANCELLED(job 5668)→ GPU 长任务改由主会话后台直跑,workflow 只做实现+审查。

### 2026-07-27 · P2 Task 2 GPU 全量完成
- A/B driver 审查闭环后彩排(3-section 双臂)→ 全量 66-section 2 prompts × 2 arms 4 进程并行(jobs 5677-5680, c-node08):4/4 EXIT=0,2178 帧全有限,on 臂 writes=64、queue=[64,65]、66 σ 齐全,双臂 pair-seed 对齐。driver 入库。
- 事故与修正:不带 --mem 提交 4 作业被整节点内存记账串行化(1 跑 3 等)→ scancel 后显式 --mem=200G 重发即并行;实证已记 findings(支持 job_submit.lua 方案)。
- wall-clock 受同节点竞争污染(on/off: 484/516s 与 1089/473s)→ 开销验收引用 run2 洁净测量 +5.5%,Task 3 verdict 中如实标注。

### 2026-07-27 · P2 Task 3 收口(P2-interim 全链完成)
- Metrics jobs 5681-5684 4/4 COMPLETED 0:0;四 JSON 66 chunks/65 boundaries 完整有限;硬门全过(writes=64×2、开销继承 run2 +5.5%)。
- Parity caveat 根因判定(orchestrator 时间线分析,见 verdict addendum):写前 3 chunk 双臂同量级 → 管线/RNG 无 bug;off 臂经典漂移(motion↑ + 去饱和 175→24-57),on 臂未训练记忆反馈回路致静态塌缩(motion→0)但去饱和更缓。R1 = 未训练记忆长时域基线,非效果参考;无需脚手架修复。
- P2-interim 三任务(0/2/3)全部收口:verdict `logs/research/p2-interim-verdict-r1.md`。

### 2026-07-27 · 512GiB 切片落位 + slice512 扩量 run 启动
- 恢复终态:RESTORE_EXIT=2(预期截断 EOF),36,674 文件/512G;目录重排(tar 内层 latents/ 前缀上提);删 1 截断尾(torch.load 实证损坏)→ **36,673 clips**;首/中/尾抽检 LOAD_OK(schema: vae_latent/first_frames_image/prompt_raw/prompt_embed_short);captions.jsonl + dataset.yaml 随行。
- 扩量 run 启动:c-node04 8×H200,`stage1_lora_mem368_A_slice512.yaml`(≈12.8× pilot 语料,~1.05 epoch/4000 步 → 实为 ~3.5 epochs@36.7k),日志 `results/stageA_slice512_run1.log`,watcher 双通道。
- 全量语料(2.53TB)剩余部分:园区出口 ~26MiB/s 下窗口内不可达;断点扩容需 tar member 边界重对齐(header 扫描法)或全量重流,留给用户决策。

### 2026-07-27 · 窗口收口
- 双跑健康:pilot 854+ 步(ckpt-500 结构全验证)、slice512 320+ 步;loss CSV + run verdict 入库(`logs/research/stage-a-first-runs-verdict-2026-07-27.md`)。
- 新 session checkpoint:`logs/session-ckpts/2026-07-27-session-checkpoint.md`(在跑作业、完成矩阵、待决策、环境事实)。

### 2026-07-27 · 节点礼仪整改 + 会话重启事故 + sbatch 化
- 用户纠正:c-node04/06 上叠了 pinghe 的 Slurm 外直跑作业(sinfo idle 的盲区)——迁移方案执行中恰逢 Claude 进程重启,后台 srun 连带阵亡(pilot 死于 ~980 步,ckpt-1000 未落,存 ckpt-500;slice512 死于 ~450 步,无 ckpt,段落损失)。两节点已让出。
- **纠正措施(用户指令)**:训练启动全部改 sbatch(作业与登录会话解耦):`scripts/training/sbatch_stage1_mem368_slice512.sbatch`(已提交 job 5692 @ c-node08,全集群唯一真空节点)+ `sbatch_stage1_mem368_pilot_resume.sbatch`(备好未提交——Slurm 看不见 pinghe,盲目 pin 节点会再次叠加;等真空节点后填 -w 提交,自动从 ckpt-500 续)。
- 礼仪规则固化到长期 memory:任何 srun/sbatch 前先 `ssh <node> nvidia-smi --query-compute-apps` + 进程属主检查;pinghe 节点禁停;yuheng 节点可叠但需显存余量核算(本配方 ~120GB/卡,H200 141GB 放不下与 yuheng 53-75GB 叠加)。

### 2026-07-27 · c-node08 OOM 事故 + 转入排队制
- job 5692(slice512 @ c-node08)首步 OOM:属主检查(0 进程)与训练起步之间,节点被外来 Slurm 外进程占走 79.42GB/卡(79+60>140);随后 ssh c-node08 无响应。实证:属主检查存在竞态窗口,pin 节点不可靠。
- **新方案(5693 slice512 / 5694 pilot-resume)**:不 pin 节点,`--exclude` pinghe(c-node04/06)/rwtag(c-node07)/失联 c-node08/drained c-node03,正规排队在 yuheng 的 Slurm 作业之后;sbatch 内置前哨检查(分配到的节点若有任何外来 GPU 进程 → PREFLIGHT_ABORT exit 42 报警,拒绝叠加)。节点腾出即自动开跑,无人值守安全。
- 用户规则入长期 memory:pinghe 任务保持安宁;yuheng/xiangbo 可叠加(显存核算前提);nvidia-smi 属主检查前置。

### 2026-07-27 · 数据源切换至 2026-07-15 备份 + 全量续填启动(用户指令)
- 用户指定改用 `r2:openhumanvid-backup/xiangbo_backup_2026-07-15_helios_organized/human_single/`。勘察:**散文件结构**(latents/ 168,431 对象 2.300TiB,非 tar),dataset.yaml provenance = 消失的 canonical 路径;captions.jsonl 更新为 148.7MB 清洗版;另有 latents_text_v3/(仅 1,211 个,实验小集,未取)。
- 一致性实证:已下载的 36,673 文件与 07-15 备份**逐字节相同**(cmp 抽检)→ 同目录 `--ignore-existing` 续填即通向全量,tar 断点难题消解。
- 全量同步已以 setsid 脱离会话启动(`/mnt/beegfs/siyuan/dataset/r2_full_sync.sh`,log `results/r2_full_sync_run1.log`,剩余 ~1.8TiB 预计 ~20h);loader 只认 *.pt,rclone .partial 临时文件对扫描不可见;陈旧 v2 cache 已删(排队作业起跑时按当时快照重建),同步完成后脚本再次失效 cache 供全量重扫。
- captions.jsonl / dataset.yaml 已刷新为 07-15 版本。

### 2026-07-27 晚 · 双跑就位 + 全量同步(pre-compact 收口)
- 5693(slice512,57,741-clip 快照)@ mc-node01 RUNNING;5694(pilot 自 ckpt-500 resume)@ c-node05 RUNNING(节点 "Kill task failed" drain 按标准程序 resume 后派发);均 sbatch + 前哨检查,零警报。
- 全量同步 ~71k/168,431 @ ~34 MiB/s(transfers 并发实测 8/32/64 = 26/34/33,出口硬顶),ETA 明晨;完成自动失效 cache。
- 数据源已切 07-15 散文件备份(逐字节一致实证);checkpoint §六/§七 补录晚间全部事件与实验速览。

- 2026-07-27 晚(压缩后):slice512(job 5693)checkpoint-500 落盘并通过结构不变量验证 —— LoRA 814 tensors 零 memory 键泄漏;transformer_partial.pth 84 键(evolving 38 + patch 6 + blocks 40);query_state 缺席;evolving 权重全有限。与 pilot ckpt-500 结构逐项一致(57,741-clip 语料下复现)。9.0G。

- 2026-07-28 凌晨:P2-interim r2 收口——4 支训练后记忆长视频(jobs 5718/5722-5724,90.75s each,13.7 min/臂)全绿 + metrics 三方对照完成。**核心结果:静态塌缩消除(motion 全程存活 vs 未训练臂 →0.02)、去饱和漂移翻转(sat 斜率 −1.2~−2.4 → ≈0~+0.85)**。判定:`logs/research/p2-interim-verdict-r2.md`。驱动器三连修:9f33b4e / 9f8c74e / 84df58e。

- 2026-07-28:用户裁定 vs24_long raw prompt 的 r1/r2 推理结果全部作废(不合规:训练 caption 与标准集均为结构化格式)。P2 r3 重建启动:rep50 前 8 case(段 0)× 三臂(off / on-untrained / on-pilot@1000)= 24 个单卡作业(5725-5748)@ c-node08 叠加。驱动器 --prompt-set rep50 支持已入库。

- 2026-07-28:P2 r3 收口——rep50 结构化三臂 24/24 全绿(5739 OOM 补发 5749),metrics 齐,判定 `logs/research/p2-interim-verdict-r3.md`:分布内基线漂移温和(r1/r2 OOD 伪影证实)、未训练记忆仍有害(3/8 冻结 + 1 爆冲)、pilot@1000 无害化达成(8/8 存活,指标持平基线)。预览站重建为 r3-only(24 卡,192MB)。

- 2026-07-28:P2 r4 Event-Switch 收口——rep50 全 6 段硬切换三臂 24/24 零失败(14 卡双节点一波半),metrics 齐。判定 `logs/research/p2-eventswitch-verdict-r4.md`:切换场景下未训练记忆病理转为系统性饱和爆冲(+1.29),pilot@1500 完全压制贴合基线(−0.30);冻结在切换场景消失(三臂 0/8)。预览站改版:Event-Switch 主页 + r3 静态子页。
