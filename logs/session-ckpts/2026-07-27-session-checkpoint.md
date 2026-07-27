# Session Checkpoint — 2026-07-27(10h 自主窗口:双轨 Stage A 真训练 + P2-interim 全链收口)

> 压缩重载/用户回归的单一入口,取代 2026-07-23 checkpoint。读序:本文 → `CLAUDE.md` → `logs/progress.md` 末四节 → `logs/findings.md`。

## 一、正在跑的东西(回来第一眼)

| 任务 | Job/位置 | 状态与到期 |
|---|---|---|
| Stage A **pilot**(40GB 子集) | job **5666** @ c-node06 | 24h 限额 ~07-28 08:15 到期(届时 ~step 2800/4000);ckpt 每 500 步,`resume_from_checkpoint: latest` 重发同命令即续 |
| Stage A **slice512**(36,673 clips) | job **5685** @ c-node04 | 24h 限额 ~07-28 13:05(~step 2600);同上可续 |
| watcher ×2 | 本 session Monitor | 失败签名 + 15min 巡检;session 结束即失效,续训需重新布防 |

续训命令 = 各配置 YAML 头部注释里的 srun 原命令(输出目录已有 ckpt 会自动 resume);或管理员 `scontrol update jobid=<id> TimeLimit=48:00:00` 延长现有作业。
运行判定(快照):`logs/research/stage-a-first-runs-verdict-2026-07-27.md`;loss CSV 在 `results/stageA_*_loss.csv`。

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
