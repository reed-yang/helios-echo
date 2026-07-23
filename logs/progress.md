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
- 待做（Task 3 剩余）：3b 写前向 + Stage A 单写混合接线（决策已固化于计划 Task 3 节：批级写决策+样本级 blend、token 级 mask、σ_last=0、留存 evicted 字段于 batch 删除之前）→ 3b 对抗审查 → 3c U-展开循环（Stage B）→ Task 5 配置分叉 → 1 节点 smoke profile 门
