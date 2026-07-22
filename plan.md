# Echo-Infinity → Helios 迁移：精细设计研究计划

**日期**: 2026-07-21
**目标**: 基于 (1) agent-research 中的 Memory/Drift/Stability 研究文档、(2) `../Echo-Infinity` 代码库、(3) `../helios-team` 的 `mid_training_xiangbo` 分支，产出 Echo 记忆机制迁移到 Helios 的精细设计与实现方案。

## 输入源

| 来源 | 路径 | 状态 |
|---|---|---|
| 研究文档（六大挑战框架） | `agent-research/Claude_export_Memory, drift and stability...md` | 已通读 |
| Echo-Infinity 代码 | `/mnt/beegfs/siyuan/workspace/Echo-Infinity` | 待深读 |
| Helios mid-train 分支 | `/mnt/beegfs/siyuan/workspace/helios-team` (mid_training_xiangbo) | 待深读 |

## 研究文档已定下的设计承诺（待代码验证）

1. **写入源**: 缓存每 section 末次去噪步 (t≈0) X_Noisy 的 last-layer hidden states，作为 Echo "last-layer evicted KV" 的对应物
2. **注入点**: 注意力 KV 拼接 [K_Noisy, K_Hist, K_Q]，memory 配独立 amp（代码中已确认存在 `is_amplify_history` 机制可扩展）
3. **位置编码**: 复用 Helios Relative RoPE，Q 占固定 id 槽（first-frame anchor 之后、long-term 之前），timestep=0
4. **粒度适配**: section 级稀疏写入 → gate bias 重调、Enc 吃长驱逐序列、section 内 BPTT 穿透/跨 section detach
5. **训练课程**: Stage 1 TF 静态 token → Stage 1.5 TF unroll → 末端 SF 校准（必不可省）
6. **风险**: 双稳定器（anchor+Q）冻结动态度；验收用斜率三件套而非终点均值

## 执行方案（多智能体工作流）

- **Phase Read**（并行 6 agent）: Echo 记忆内核 / Echo 训练管线 / Echo 流式推理 / Helios 模型架构 / Helios 训练+推理流 / xiangbo 分支增量与文档
- **Phase Design**（4 agent）: 模块设计 / 训练课程与代码改造 / 推理集成与评测 / 风险与消融
- **Phase Verify**（每设计 2 agent 对抗验证 + 修订）: 代码锚定核查 + 可行性反驳
- **综合**: 主会话整合为最终中文设计文档

## 产出物

- `logs/research/*.md` — 各 agent 深读报告（英文，含 file:line）
- `logs/findings.md` — 关键发现汇总
- `logs/progress.md` — 进度
- `docs/echo-to-helios-migration-design.md` — 最终精细设计文档（中文）
