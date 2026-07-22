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

## 状态：研究阶段完成

下一步（待决策后启动实现）：
1. K0 前置项：D1/D8 仲裁签字、checkpoint-conversion 测试、`dmd_num_latent_sections_min≥4` 断言 + Enc/gate 梯度单测、Stage B smoke profile
2. 评测工具先行：4 类时序斜率指标 external_command 脚本 + D10 注意力诊断路径（Stage A 期间交付）
3. 工具链先行合并：save/load_extra_components 第 5 节、merge/EMA memory 分支（默认关闭门控，不阻塞主线）
4. 与主线协调：C2 基座冻结 checkpoint-19500 + no-KV 基线重跑；C5 书面冻结（不切 Wan2.2、VQ 不先行）
