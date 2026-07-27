# Session Checkpoint — 2026-07-27(10h 自主窗口:双轨 Stage A 真训练 + P2-interim 全链收口)

> 压缩重载/用户回归的单一入口,取代 2026-07-23 checkpoint。读序:本文 → `CLAUDE.md` → `logs/progress.md` 末四节 → `logs/findings.md`。

## 一、正在跑的东西(回来第一眼)【晚间更新,取代下午版本】

| 任务 | Job/位置 | 状态 |
|---|---|---|
| Stage A **slice512**(57,741-clip 快照) | **sbatch job 5693** @ mc-node01 | RUNNING(run3,log `results/stageA_slice512_run3.log`,48h 限额) |
| Stage A **pilot resume**(自 ckpt-500) | **sbatch job 5694** @ c-node05 | RUNNING(run2,log `results/stageA_pilot_run2.log`,48h 限额) |
| **全量语料同步**(07-15 备份散文件) | 登录节点 setsid 脱离进程(`/mnt/beegfs/siyuan/dataset/r2_full_sync.sh`) | ~71k/168,431 文件,~34 MiB/s(transfers=64 = 出口顶),ETA 07-28 晨;完成后自动删 v2 cache 供全量重扫 |
| watcher ×2(双作业 + 同步) | 本 session Monitor | session 结束即失效,重载后需重新布防 |

**下午的 5666/5685 已死**(session 重启连带杀 srun,见 §六);现役作业均为 **sbatch**(会话无关):`scripts/training/sbatch_stage1_mem368_{slice512,pilot_resume}.sbatch`,内置 exclude 名单(c-node03/04/06/07/08)与前哨检查(节点有外来 GPU 进程即 exit 42 拒跑)。
运行判定(下午快照):`logs/research/stage-a-first-runs-verdict-2026-07-27.md`;loss CSV `results/stageA_*_loss.csv`(pilot run1 855 点)。

## 二、本窗口完成矩阵(全部已提交,提交号见 git log 2026-07-27)

1. **上游 PR#1** 已提交(前序 session):OPEN、无人工 review、Copilot CI 因 GitHub Actions 账单未跑 —— 等 xiangbo。
2. **P2-interim 全链收口(Task 0/2/3)**:
   - Task 0 门先拦获真缺陷:训练侧 stage-2 采样路径缺 memory 捕获契约 → 修复 `b671ba6`(条件三元组,TDD 红灯复现,82/82 CPU 全绿,双镜头审查)→ 门重跑 1-section 9/9 + 5-section 双臂 17/17 ALL PASS;harness 入库 `0711773`。
   - Task 2 driver `5beaebf`(scripts/evaluation/run_p2_interim_drift_ab.py):66-section 双臂 × 2 prompts 4/4 绿(jobs 5677-5680)。
   - Task 3 metrics(jobs 5681-5684)+ verdict `logs/research/p2-interim-verdict-r1.md`:**脚手架 PASS**;parity caveat 根因判定 = 未训练记忆长时域行为基线(off 臂经典漂移 motion↑/去饱和 175→24-57;on 臂静态塌缩 motion→0 但去饱和更缓;写前 3 chunk 双臂同量级证明管线/RNG 无 bug)—— 非 bug,是 Stage A-C 课程必要性的实证;R1 不作效果参考。
3. **512GiB 语料切片恢复**:36,673 clips 验证落位(`/mnt/beegfs/siyuan/dataset/human_single_368x640_slice512g/`,删 1 截断尾,抽检 LOAD_OK);全量 2.53TB 因园区出口 ~26-40 MiB/s 硬顶窗口内不可达。
4. **双轨 Stage A 真训练**(gate 4 = 本窗口用户指令放行):pilot ckpt-500 结构全验证(LoRA 814 零泄漏 / memory 全秩 38 键 / query_state 缺席);v2 cache 36k 生产首演 1 build + 7 load。
5. 配置新增:`stage1_lora_mem368_A_pilot.yaml`、`stage1_lora_mem368_A_slice512.yaml`(主 YAML 仍留给全量恢复后的正式 run)。

## 三、等用户决策

1. **全量语料剩余 ~2TB**:选项 ① tar header 扫描断点续传(需小工具,~21h 网络);② 全量重流 ~26h;③ 维持 512GiB 切片。建议:若 slice512 曲线满意可暂缓。
2. **两条 run 的 24h 到期处理**:延长 TimeLimit(管理员)或到期后 resume 重发(无人值守缺口 = watcher 失效期)。
3. **Slurm 默认配额**:新增实证 —— c-node08 四作业无 `--mem` 被整节点内存记账串行化(1 跑 3 等),显式 `--mem=200G` 即并行;job_submit.lua 方案(调查报告在 `logs/research/slurm-defcpupergpu-inval-investigation.md`)待拍板。
4. PR#1 是否 ping xiangbo 人工 review;GitHub Actions 账单(Copilot CI)。
5. 旧项:mc-node01 GPU2 硬件、slurmdbd 记账、C2/C5 主线协调。

## 四、本窗口新增环境事实

- R2 出口带宽硬顶 ~26-40 MiB/s(单流 25.9,4 流 40,12 流 25.6 —— 并行不扩展)。
- 整节点内存记账:无 `--mem` 的作业按 DefMemPerNode=UNLIMITED 吃满节点 ⇒ 同节点多作业必须显式 `--mem`(8 卡训练独占节点则无妨)。
- Workflow 内 GPU 执行 agent 被 structured-output 强杀会连带 CANCEL 其后台 srun(job 5668)⇒ GPU 长任务由主会话后台直跑,workflow 只做实现+审查。
- 后台包装脚本不 `wait` 会孤儿化 srun(作业存活但失去完成通知)。
- `latents.tar` 内层带 `latents/` 前缀,解包需重排或 strip;截断尾 member 由 mtime 最新 + torch.load 失败识别。
- tqdm 显示的 lr 是最后一个参数组(memory 组),勿误判 warmup。

## 五、Session 元信息

- 窗口:2026-07-27 ~08:20-16:30 UTC(用户预告 10h 后回归)。
- 提交序列(本窗口):`806a0a2` 配置 → `0521f13` 计划 → `1c5c079` 缺陷记录 → `b671ba6` stage-2 修复 → `0711773` harness → `e39d2b9` 门收口 → `5beaebf` driver → `c196251`/`c7544a8` P2 文档 → `8a93f71` 切片落位 → 本 checkpoint 批次。
- 编排:4 个 Workflow(Task 0 harness / stage-2 修复 / Task 2 driver / 上游 PR 移植 wf_507fb2c3 前序完成)+ 约 8 个独立 subagent;对抗审查 6+ 轮全闭环;严重事故 0,GPU 空转 0。

## 六、晚间事件补录(用户回归后,~17:00-20:30 UTC)

1. **节点礼仪整改(用户两次纠正,已入长期 memory)**:pinghe 的任务必须保持安宁(其作业在 Slurm 外直跑,sinfo idle 不可信,必须 `ssh <node> nvidia-smi` 查属主);yuheng/xiangbo 的 job 允许叠加(需先核算显存:本配方 ~120GB/卡)。
2. **Session 重启事故**:下午的 srun 挂在会话后台,进程重启全灭——pilot 死于 ~980 步(ckpt-1000 未落,存 ckpt-500),slice512 run1 死于 ~450 步(无 ckpt,段落损失)。用户定性"训练 CLI 没放 tmux/srun 是严重失误"→ **全部转 sbatch**。
3. **c-node08 OOM 事故**:重启后 slice512 pin 到勘察时全空的 c-node08,起步瞬间被 xiangbo 的 Slurm 外进程(79.42GB/卡)占先 → 首步 OOM;且该节点 ssh 间歇失联。结论:pin 节点存在竞态,改为**排队制**(不 pin + exclude 名单 + sbatch 内前哨检查)。
4. **数据源切换(用户指令)**:改用 `r2:openhumanvid-backup/xiangbo_backup_2026-07-15_helios_organized/human_single/`——**散文件结构**(latents/ 168,431 对象 2.300TiB,dataset.yaml provenance = 消失的 canonical 路径;另有 latents_text_v3/ 仅 1,211 个实验小集未取)。抽检与已下载数据**逐字节相同** → 同目录 `--ignore-existing` 续填全量,tar 断点难题消解。并发实测:transfers 8→32→64 = 26→34→33 MiB/s,~34 为出口硬顶。
5. **双跑就位**:5693(slice512)在 yuheng 腾出 mc-node01 后自动派发,57,741-clip 快照重建 cache 训练;5694(pilot resume)按用户指令定向 c-node05——该节点 "Kill task failed" 自动 drain,物理核净后按标准程序 `scontrol resume` 清除,即刻派发,自 ckpt-500 续训。
6. **旧 v2 cache 已删**:同步完成脚本会再删一次,保证下次启动全量 168k 重扫;运行中作业不受影响(启动时快照语义)。

## 七、实验设计速览(供向用户汇报;详细依据见各文档)

- **在跑的两个实验**:同一 Stage A 配方(merged 基座 + 冻结骨干 + memory 组件训练,bs2×accum2×8 卡=全局 32,4000 步,memory lr 5e-5 warmup 500,单写概率 0.5)的**语料规模对照**:pilot=2,869 clips(~44.6 epochs,过拟合侦察)vs slice512=57,741 clips(~2.2 epochs,真实训练面)。产出:ckpt 链(每 500 步)+ loss/grad 曲线 + memory 组件权重演化,为 Stage B(U=4 unroll)选基。
- **今日已收口的 P2-interim**:Distilled 冻结基座上 66-section 双臂 A/B(memory-on vs true no-KV),脚手架全绿;副产物 = 未训练记忆的长时域行为基线(off 臂漂移/on 臂静态塌缩)——Stage A-C 课程必要性的实证。设计权威:`docs/plans/2026-07-22-p2-interim-drift-ab-plan.md` + `logs/research/p2-interim-verdict-r1.md`。
- **总设计**:`docs/echo-to-helios-migration-design.md`(token 插入读 + 最后调度步捕获写 + FiLM(σ_last) + 三阶段课程 A/B/C)。
