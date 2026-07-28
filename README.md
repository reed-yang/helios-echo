# Helios-Echo

**Echo-Infinity 演进记忆 → Helios 移植工作库**(`../helios-team` @ `mid_training_xiangbo` 的工作 fork,工作分支 `echo-memory`)。

Helios 是基于 Wan2.1-T2V-14B 的自回归分块视频 DiT(上游:[PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios))。其固定历史窗口(`[16,2,1]` + x0 锚,19 latent 帧)之外的内容被永久遗忘,长视频出现漂移与不一致。本 fork 为其移植 Echo-Infinity 的**演进记忆状态 M**:以固定 token 预算持续携带窗口外信息。

## 记忆机制一览

| 环节 | 设计 | 位置 |
|---|---|---|
| 读 | token 插入前缀 `[mem\|long\|mid\|short\|current]`,分数 RoPE id,共享 t=0 AdaLN,per-head `memory_key_scale` 温和起步 | `helios/modules/transformer_helios.py` |
| 写 | `HeliosMemoryEncoder`(2 层 cross-attn)+ 门控 EMA;最后调度步捕获 hidden states,FiLM(σ_last) 补偿捕获噪声水平 | `helios/modules/helios_memory.py` |
| 时机 | section k 生成后、k−2 内容滑出历史窗口前写入——与窗口零冗余 | `helios/pipelines/pipeline_helios.py` |
| 课程 | Stage A(冻结骨干只训记忆)→ B(U=4 unroll)→ C(联合微调) | `docs/echo-to-helios-migration-design.md` |

设计权威:`docs/echo-to-helios-migration-design.md`(含统一基线、先验 vs 代码偏差表、D 系仲裁)。

## 仓库导航

阅读顺序:① 设计总纲 ② `logs/findings.md`(证据台账)③ `logs/progress.md`(执行台账)④ `logs/research/`(深读报告与 run 判定)。目录约定见 `CLAUDE.md`(文档类型由目录决定,日期前缀定时序)。

本 fork 新增(相对上游):记忆模块与读路径改造、trainer 配置组(`memory_*` 系列 + `validate_evolving_memory_config`)、pipeline 状态机、CPU 单测 `tests/test_*.py` + GPU 冒烟 `tests/smoke_*.py`、评测驱动 `scripts/evaluation/run_p2_interim_drift_ab.py`(rep50 结构化 prompt 三臂协议)与预览站生成器 `scripts/evaluation/build_p2_site.py`。

## 快速开始

```bash
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios          # 团队共用环境,勿装包

# CPU 单测(82+)
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' $ENV/bin/python -m unittest discover -s tests

# Stage A 训练(sbatch,禁止会话内 srun 前台跑)
sbatch scripts/training/sbatch_stage1_mem368_slice512.sbatch

# 长视频漂移 A/B(90.75s × 66 sections,单卡 ~14 min/臂)
sbatch scripts/evaluation/sbatch_p2_r3_infer.sbatch on 0 <run-id> <transformer_partial.pth|none>

# 结果预览站(静态生成 + Cloudflare 隧道)
$ENV/bin/python scripts/evaluation/build_p2_site.py
```

集群注意事项(节点礼仪、Prohibited 坏卡 CVD 重映射、整节点内存记账)见 `logs/findings.md` 对应日期条目;上游训练/推理/数据管线文档在 `docs/`(UPPERCASE 命名,保持原样)。
