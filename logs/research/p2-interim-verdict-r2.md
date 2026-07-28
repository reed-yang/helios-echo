# P2-interim r2 判定:Stage A 训练后记忆的首次长视频 A/B(2026-07-28 凌晨)

> **【作废 VOID,2026-07-28 用户裁定】**:本判定基于 vs24_long raw 裸句 prompt,不符合结构化改写规范;结果作废,由 rep50 结构化三臂重建(p2-rep50-r3-*,见 p2-interim-verdict-r3)取代。

**结论:仅训练 500-1000 步的记忆组件,已同时消除 r1 观察到的两种长时域退化——未训练记忆的静态塌缩(motion→0)与基线的去饱和漂移(饱和度斜率由 −1.2~−2.4 翻转为 ≈0~+0.85)。Stage A 课程的有效性得到首个下游实证。**

## 运行矩阵

| Run | Job | 记忆权重 | sha256 前 16 位 |
|---|---|---|---|
| r2-pilot1000 × p0/p1 | 5718 / 5722 | pilot checkpoint-1000 `transformer_partial.pth` | `02a21d3296b3e7c6` |
| r2-slice512-500 × p0/p1 | 5723 / 5724 | slice512 checkpoint-500 `transformer_partial.pth` | (manifest 记录) |

协议与 r1 完全一致(66 sections = 2178 帧 = 90.75s @384×640,Distilled 基座 DMD [2,2,2]×3,base-seed 7 种子配对,on 臂),唯一差异 = `--memory-partial` 载入训练后记忆组件(evolving_memory 38 键 + per-head memory_key_scale 40 键;**patch convs 6 键按设计排除**——Stage A 冻结零移动且属训练基座,载入会污染 Distilled 的历史 patchify,见 `9f8c74e`)。off 与 on-untrained 臂复用 r1 工件(种子配对可直接对比)。

## 指标对照(timeseries metrics,l2 backend;mot/sat = 前 10 / 末 10 chunk 均值)

| 臂 | p | mot 首→末 | sat 首→末 | sat 斜率(Theil-Sen) |
|---|---|---|---|---|
| r1-off | 0 | 0.14 → **1.71** | 175 → **62** | **−1.94** |
| r1-off | 1 | 0.21 → 0.43 | 173 → **55** | **−2.43** |
| r1-on-untrained | 0 | 0.05 → **0.018**(冻结) | 135 → 83 | −1.74 |
| r1-on-untrained | 1 | 0.07 → **0.028**(冻结) | 181 → 128 | −1.15 |
| **r2-pilot@1000** | 0 | 0.11 → 0.13(存活稳定) | 97 → 110 | **+0.03** |
| **r2-pilot@1000** | 1 | 0.15 → 1.83(存活,末段高) | 180 → 181 | **+0.06** |
| **r2-slice512@500** | 0 | 0.07 → 0.54(存活) | 100 → 160 | **+0.85**(过冲) |
| **r2-slice512@500** | 1 | 0.12 → 0.30(存活) | 175 → 209 | **+0.66**(过冲) |

## 解读

1. **静态塌缩消除**:r1 未训练记忆把生成拖向冻结(motion 0.018-0.028);r2 全部 4 支 motion 全程存活。训练让记忆从"噪声先验"变为"无害且有信息"。
2. **去饱和漂移翻转**:所有基线臂饱和度斜率 −1.15~−2.43;r2 全部非负。pilot@1000 近乎完美持平(+0.03/+0.06);slice512@500 过冲上行(+0.66/+0.85,末段 sat 160-209 偏高)——记忆在主动对抗去饱和,早期 ckpt 校准不足属预期。
3. **pilot@1000 vs slice512@500**:更多步数的 pilot 校准更好;但语料与步数双变量混杂,不能归因。
4. **早段起点差异**(如 sat 97 vs 175):记忆 token 自 section 1 即被读取(M₀ 不同则首 chunk 即分化),写入自 k−2 驱逐起;与 r1 的臂内 parity 验证不矛盾。

## 保留意见

- n=2 prompts × 1 seed,无视觉 QC(pilot p1 末段 motion 1.83 与 r1-off 漂移端点同量级,需人眼确认是内容运动还是不稳定伪影)。
- 训练基座(merged Stage-1)≠ 评测基座(Distilled)的迁移混杂仍在;方向性结论稳健,幅度不作数。
- 早期 ckpt(gate 仍近初始化)已现效果,4000 步终态 ckpt 需重跑本协议获得正式结论。

## 过程事故(均已修复入档)

- 首批 5714-5717:partial 的 patch convs 会覆盖 Distilled 同名权重(rel_l2 至 1.17)→ 取消重发,driver 排除骨干键(`9f8c74e`)。
- scancel 触发 c-node08 "Kill task failed" 自动 drain → resume 后重发。
- 5719-5721 三连败:c-node08 泄漏 Prohibited 坏卡进 cgroup、抢占重编号 index 0,Slurm 盲设 CVD=0 → `cudaErrorDevicesUnavailable`;sbatch 内按 compute_mode 重映射(`84df58e`)。叠加 xiangbo 的 Default 卡实测可行(41.7+50 GB 共存)。

## 产物

- 视频+manifest:`results/p2_interim/p2-interim-r2-{pilot1000,slice512-500}/on/prompt_0{0,1}.{mp4,manifest.json}`
- metrics:同目录 `metrics/on_prompt_0{0,1}.json`;单臂全程 13.7 分钟(load 217s + rollout 501s + encode 101s)。
- 驱动器变更:`9f33b4e`(--memory-partial)+ `9f8c74e`(骨干键排除)+ `84df58e`(CVD 重映射)。

## 下一步

1. 视觉 QC 4 支视频(尤其 pilot p1 末段)。
2. 双训练 run 到 4000 步后,以终态 ckpt 重跑本协议(可加 prompt/种子扩 n)= Stage A 正式验收。
3. slice512 过冲现象随训练步数的演化值得追踪(每 1000 步一测,13.7 min/臂成本可承受)。

## 补充 caveat(2026-07-28,用户质询后)

本判定所有 prompt 为 vs24_long raw 裸句,**不符合团队"推理 prompt 需结构化改写"规范**(训练 caption 与 rep50 标准集均为 `<header>/<event>/<role>/<Background>` 结构体)。三臂同 prompt 同 seed ⇒ 相对结论(塌缩消除/斜率翻转)有效;绝对量级与正式验收需在 `eval_prompts_rep50`(50 结构化 case)上重建三臂。详见 findings 2026-07-28 条目。
