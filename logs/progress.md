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
