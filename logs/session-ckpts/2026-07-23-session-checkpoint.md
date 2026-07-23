# Session Checkpoint — 2026-07-22/23(echo-memory 实现期第 3-4 批 + 基础设施夜)

> 用途:压缩重载后恢复上下文的单一入口。读序:本文 → `CLAUDE.md` → `logs/progress.md` 末三节 → `logs/findings.md`。设计权威:`docs/echo-to-helios-migration-design.md`。

## 一、硬阻塞(恢复后第一件事)

> **【2026-07-23 晚已解除】** R2 备份(`openhumanvid-backup/Xiangbo_july_8/helios_organized/human_single/latents.tar`, 2.53TB)恢复了 40GB 子集(2,869 clips → `/mnt/beegfs/siyuan/dataset/human_single_368x640_subset/latents`);本节动作序列 ①②③ 已全部执行完毕(YAML 已指向子集 be4d654;smoke run7 3/3 PASS;DDP smoke 5/5 PASS + 重放证据)。**剩余 = ④ 用户拍板**,且真训练前须全量恢复语料并 REPOINT 主 YAML。详见 `logs/progress.md` "2026-07-23 晚" 节。原文如下留档。

**368×640 latent 语料从盘上消失,等用户恢复数据**(用户已知,说"数据稍后才恢复")。
- 证据:旧路径 `/mnt/beegfs/dataset/video_single_24FPS/latents_cfr_int_30b_368x640` 在 xiangbo `helios_runs/scripts/reorg/delete_list.sh` 删除清单中;canonical tree(`helios_organized/human_single/dataset.yaml`)计划了 `latents/` 子目录(n=168431,硬链接方案)但从未落地;demo latents 同灭;原始视频与 manifest 完好。
- **数据恢复后的动作序列**(无需重新决策):① 确认新语料路径 → 更新两份 YAML 的 `instance_data_root`(`scripts/training/configs/stage1_lora_mem368_A{,_smoke}.yaml`)→ ② smoke run7(单卡,命令在 smoke YAML 头部注释;先 `rm -rf /mnt/beegfs/siyuan/helios_runs/smoke_stage_a` 清配置快照)→ ③ ≥2 卡 DDP smoke(必须覆盖 rank 间混合全零/有效驱逐——3b 审查 blocker 的验证项)→ ④ 用户拍板 Stage A 真训练。

## 二、代码面状态(全部提交,工作树干净)

分支 `echo-memory`,HEAD `f087e3b`,测试面 **79 全绿**(CPU;登录节点带 GPU 时需 `CUDA_VISIBLE_DEVICES=''`,已知套件小坑)。

| 单元 | 状态 | 关键提交 |
|---|---|---|
| M1 训练链(模块/transformer/trainer/管线) | ✅ 全绿 + GPU 冒烟(P1 13/13、A1 9/9、rollout run2 8/8 +5.5%) | ~ce62bcc, 320821d |
| Stage A Task 1 D4 驱逐切片 | ✅ + 审查修复(桶分辨率角色分离) | 7f422b6, e064ad9 |
| Task 2 D6 展开参数 | ✅ + 审查修复(mp.Value 共享 epoch、caption 钉定) | 3729bed, 09faaf3 |
| Task 3a loss 原语 | ✅ 审查 CLEAN | 4929521 |
| Task 3b 单写模拟 | ✅ + 审查修复(rank 对称掷币/前向) | e36d659, 48132c1 |
| Task 5 配置分叉 | ✅ 修正为 merged 基座 + 4000 步 fresh | f843df8, 6962644 |
| D2 缺口(构造字典漏 memory 键) | ✅ smoke 抓到并修(memory_frame_hw 升格 config) | 43b7d88 |
| v2 数据集 cache(388k 扫描修复) | ✅ + 审查加固(原子发布/自愈/降级/断言范围化) | be8ce09, 4108ab1 |
| P2 Task 1 指标脚本 | ✅(motion 分辨率不变性修复) | 382fe09 |
| **Stage A merged 基座** | ✅ 产出于 `/mnt/beegfs/siyuan/helios_runs/_merged/stage1_lora368_correct_merged19500/transformer`(28.6GB bf16) | 工具 tools/merge_lora_full_for_helios.py |

未做(按序):3c U-展开循环(Stage B 期)、P2 Task 0 Distilled 配方 GPU dry-run、P2 Task 2/3、EMA/M2 镜像(后期)。

## 三、等用户决策的事项

1. **语料恢复**(本 checkpoint 第一节)——用户在办。
2. **Slurm 默认配额**:`DefMemPerGPU/DefCpuPerGPU` 在 24.05.7 两次触发全节点 INVAL(postmortem: `logs/2026-07-23-postmortem-slurm-reconfigure-inval.md`;调查报告 `logs/research/slurm-defcpupergpu-inval-investigation.md` 推荐 **job_submit.lua** 替代)。lua 脚本可代写,部署等用户。当前配置已回滚,全节点留有 `/etc/slurm/slurm.conf.bak-20260723`。
3. **上游 PR**(用户已批准路线):打包 persistent-worker epoch 修复(09faaf3 的 dataset 部分)+ v2 cache(be8ce09+4108ab1)发 team 仓库独立 PR;PR 描述带复现与等价性测试。执行时从 `team/mid_training_xiangbo` 开新分支 cherry-pick。
4. **mc-node01 GPU2 硬件**(用户已知)与 **slurmdbd 连不上 database**(记账丢数据)——infra 侧待办。
5. 主线协调旧项:C2 no-KV 基线重跑排期、C5 书面冻结。

## 四、今晚学到的环境事实(已存 memory 的除外)

- mc-node01 = 主 slurmctld;`nvidia_uvm` 不干净卸载曾致整节点 CUDA 死(已 rmmod/modprobe 修复,7/8 卡可用);它长期闲置的真因即此。
- 集群非 configless;恢复 INVAL 的唯一路径 = 配置一致后逐节点重启 slurmd(清 INVALID_REG)→ `scontrol update state=resume`(清 DRAIN);`resume` 对 INVALID_REG 状态无效。
- trainer 会在 output_dir 存配置快照并于重启时校验("Key ... missing in existing config")——schema 变更后必须清 scratch 目录。
- `single_res` 强制 `force_rebuild` 的断言已范围化(仅 dmd/mp4 数据集);stage-1 走 v2 cache。
- 集群礼仪与节点选择已存长期 memory(`cluster-gpu-etiquette`)。

## 五、Session 元信息

- 时间跨度:2026-07-22 晚 — 07-23 凌晨(pre-compact ×2:上一次在 rollout run2 期间)。
- 本段新增提交:`5f59fcc..f087e3b` 约 20 个(feat 6 / fix 7 / docs 7)。
- 对抗审查记录:6 轮审查,抓获 blocking×5、major×4、minor×2,全部闭环——审查存档在各 commit message 与 findings。
- 委托单元:~15 个 subagent/workflow(实现 4、审查 6、取证 4、web 调查 1);全部收口,无在飞任务。
- agent-research 规范(本 session 定稿):`YYYY-MM-DD-<core-concept>.md`,append-only,日期前缀承担排序。
