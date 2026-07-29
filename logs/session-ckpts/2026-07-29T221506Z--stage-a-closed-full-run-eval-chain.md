# Session Checkpoint: Stage A 收官 + 全量 run 终态评测链

- Created (UTC): `2026-07-29T22:15:06Z`
- Workspace root: `/mnt/beegfs/siyuan/workspace/helios-echo`
- Session ID: `ad5ee0ce-7d48-4e06-af32-5d9f440892cd`
- Branch / revision: `echo-memory` @ `d67c785`
- Working tree: clean(0 changes;staged/unstaged/untracked 均无)
- Supersedes: `logs/session-ckpts/2026-07-29-session-checkpoint.md`(同一任务的上一份,内容仍有效但状态已过时)

## Resume First

1. 读 `CLAUDE.md`(§Context stewardship rules + §Evolving-memory work map 末两条 = 评测工具链与预览站)。
2. 读本文件 §Canonical Document Map 按序读文档;**不要重跑已完成战役**。
3. 查活口:`squeue -u siyuan`(训练 5823 + 依赖链 5872-5885)与 `tr '\r' '\n' < results/stageA_full_run1.log | grep -oE "[0-9]+/12000" | tail -1`。
4. 第一个可执行动作:**判读 full@12000 依赖链产物**——`ls results/p2_interim/p2-rep50-{r4evsw,r5long}-on-full12000/metrics/*.json`(应各 8/5 份),按 `logs/research/stage-a-terminal-verdict-2026-07-28.md` 附录格式做臂间对照,写入该文件与 `p2-longhorizon-verdict-r5-2026-07-28.md` 的新附录,commit 后 `git push private echo-memory`。
5. 若链条未完成:仅重布 watcher(session 内 Monitor 全部随会话失效),勿重发作业。

## Task and Motivation

- Goal:把 Echo-Infinity 演进记忆移植进 Helios 并用证据支撑课程推进;当前边界 = Stage A 已收官,正在为 Stage B 选基。
- Original motivation:Helios 的固定历史窗口(`[16,2,1]`+x0)之外内容永久遗忘 ⇒ 长视频漂移与不一致;记忆状态 M 以固定 token 预算携带窗口外信息。
- Success condition(当前阶段):Stage A 验收 PASS(已达成)+ 一个有证据支撑的 Stage B 基座选择(待 full@12000 数据)。
- Constraints:回复中文;代码注释/commit 英文;不自动推 origin(仅 `private` = github.com/reed-yang/helios-echo);训练一律 sbatch;pinghe 节点不可叠加(yuheng/xiangbo 可,需核显存);不装包进共享 env;凭据不外泄。

## Current State

### Completed

- **Stage A 双轨训练 + 终态验收 PASS**:`logs/research/stage-a-terminal-verdict-2026-07-28.md`(pilot 谱系切换斜率单调收敛 −0.30→−0.09→−0.002;slice512@4000 切换欠校准 −0.624)。
- **三协议证据链**:r3 静态 90.75s / r4 切换 57.75s / r5 8min(各自 verdict 见文档图)。
- **全量 run cadence 追踪**:full@4000(−0.193)、full@8000(−0.263)、**full@8000 × 8min(+0.297,劣于基线,3/5 失控)** ⇒ Stage B 选基回到 pilot@4000(证据:两份 verdict 的附录)。
- **ckpt 质量判定体系四层 + 盲区**:`logs/findings.md` 2026-07-29 条目。
- **预览站自动收录**:127 卡两页,`p2-rep50-*` 免登记上站;URL `results/p2_site/.current_url`。
- 战役统计:127 支推理,连续零失败;GPU 峰值 30 卡。

### In Progress

- **全量训练 5823**:11,691/12000(检查点时刻),24 ckpt 已验证,零错误;ETA ~40 分钟。
- **Slurm 依赖链(脱离会话)**:5872-5879(切换 × full@12000 × 8 case)+ 5880-5884(8min × 5 case)+ 5885(metrics 批处理 + 自动重建站点),全部 `afterany:5823`,PD 待训练结束自动派发。

### Remaining

1. 判读 full@12000 双协议结果 → 两份 verdict 新附录 + Stage B 选基定论(pilot@4000 vs full@12000)。
2. (待用户拍板)held-out 泛化检验:rep50 case 8-15 × {pilot@4000, full 最优 ckpt} ≈ 2.7 GPU·h — 直接仲裁 pilot 过拟合疑虑。
3. (待用户拍板)Stage B(U=4 unroll)设计与启动;验收目标已定为"8min 回复力→完全稳定器"。
4. (可选,无需新推理)DOVER/HPSv3/PickScore 对现有 130+ 支视频离线补算(`scripts/evaluation/metrics/`)。

### Blockers and Open Decisions

- **Stage B 选基**:pilot@4000(校准最优但过拟合未测)vs 全量谱系(多样性但 8min 失控)——需 full@12000 数据 + 可选 held-out 检验;决定权在用户。
- **永久域名**:Cloudflare Pages 需用户提供 API token(Pages:Edit);当前 trycloudflare URL 随隧道进程重启而变。
- 旧项:上游 PR#1 是否 ping xiangbo / GitHub Actions 账单;`job_submit.lua`;mc-node01 GPU2 硬件;slurmdbd。

## Canonical Document Map

| Priority | Path | Section or role | Why read it | Current state |
|---|---|---|---|---|
| 1 | `CLAUDE.md` | Evolving-memory work map(末两条)+ Context stewardship | 评测工具链/建站/依赖链约定;主 context 纪律 | updated |
| 2 | `logs/progress.md` | 末 6 条(07-28 ~ 07-29) | 执行台账:训练完成、验收、cadence、依赖链 | updated |
| 3 | `logs/findings.md` | 2026-07-27 ~ 07-29 三条 | ckpt 质量判定体系、prompt 合规缺陷、Prohibited GPU 泄漏 | updated |
| 4 | `logs/research/stage-a-terminal-verdict-2026-07-28.md` | 主体 + 附录 2/3 | Stage A 验收结论与 full@4000/@8000 横比 | unchanged(本轮附录已在) |
| 5 | `logs/research/p2-longhorizon-verdict-r5-2026-07-28.md` | 主体 + full@8000 附录 | 8min 判读:回复力 vs 失控 | unchanged |
| 6 | `logs/research/p2-eventswitch-verdict-r4.md` / `p2-interim-verdict-r3.md` | 全文 | 切换/静态协议基线与 untrained 病理 | unchanged |
| 7 | `docs/echo-to-helios-migration-design.md` | §一 基线 / D8 谱系仲裁 / 三阶段课程 | Stage B 设计前必读(基座 = _correct@19500 合并) | unchanged |
| 8 | `logs/session-ckpts/2026-07-29-session-checkpoint.md` | §二 Stage A 闭环 / §三 环境事实 | 环境事实全集(坏卡重映射、drain 处置、双节点 accelerate) | unchanged |

## Working State

### Code and Artifact Pointers

| Path | Symbol, section, or artifact | State / reason |
|---|---|---|
| `scripts/evaluation/run_p2_interim_drift_ab.py` | `--prompt-set/--segments/--sections/--memory-partial`;`load_rep50_prompts`;`load_pipeline` 排除 patch convs | changed(本轮稳定,勿回退骨干键排除) |
| `scripts/evaluation/sbatch_p2_r3_infer.sbatch` / `sbatch_p2_r4_eventswitch.sbatch` | CVD 重映射(max-free Default + 60GB 门槛) | changed |
| `scripts/evaluation/sbatch_p2_metrics_batch.sbatch` | CPU metrics 批处理 + 自动 rebuild 站点 | new(依赖链尾节点 5885 使用) |
| `scripts/evaluation/build_p2_site.py` | `R4_META`/`R3_META` + `PAGES[*]["auto"]` 自动收录 | changed |
| `scripts/training/configs/stage1_lora_mem368_A_full.yaml`、`sbatch_stage1_mem368_full_2node.sbatch` | 168k × 12k 步 / 16 卡双节点 | changed |
| `/mnt/beegfs/siyuan/helios_runs/stage1_lora_mem368_A_{pilot,slice512,full}/checkpoint-*` | ckpt 链(9 / 9 / 24) | external artifact,全部结构验证通过 |
| `results/p2_interim/p2-rep50-*/` | 127 视频 + manifest + metrics | external artifact(results/ 不入库) |
| `results/p2_site/` + `.current_url` | 两页站点 + 脱离会话的 server/tunnel/keepalive | generated,运行中 |

### Working-Tree Details

- Staged: None
- Unstaged: None
- Untracked: None(`results/` 按约定不跟踪)
- User-owned or unrelated changes to preserve: None

### Verification

| Command or artifact | Result | What it proves |
|---|---|---|
| `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' $ENV/bin/python -m unittest discover -s tests` | pass,`Ran 82 tests ... OK`(切换协议改造后跑过) | CPU 侧记忆读写/pipeline 契约未回归;不证明 GPU 数值 |
| ckpt 不变量脚本(torch.load + safetensors 计数) | pass,40+ ckpt:LoRA 814 / leak 0 / partial 84 / query_state 0 / finite | 保存路径正确(含 16 卡多节点);与生成质量无关 |
| `results/p2_interim/*/logs` EXIT_CODE 统计 | 127/127 = 0 | 五轮战役零失败;不含视觉质量判定 |
| `results/stageA_full_run1.log` | step 11,691/12000,err0 | 全量 run 健康;终态待 EXIT_CODE 行 |
| `curl .../index.html` `.../static.html` | 200 / 200 | 站点两页公网可访问(隧道 URL 随进程重启变化) |

## Tracking and Decision Pointers

- Active plan/tracker:`logs/progress.md` 末条(依赖链待判读)+ 本文 §Remaining。
- External IDs:上游 PR `Visko-Platform/helios-team#1`(OPEN,待 xiangbo review);Slurm jobs 5823 / 5872-5885;私有仓库 `github.com/reed-yang/helios-echo`。
- Key decision rationale:`docs/echo-to-helios-migration-design.md` D8(谱系仲裁,基座冻结 @19500);`logs/research/stage-a-terminal-verdict-2026-07-28.md` §Stage B 衔接建议。
- Failed approaches that must not be repeated:vs24_long raw prompt(prompt-OOD,r1/r2 已 VOID);`--memory-partial` 载入 patch convs(污染 Distilled 历史 patchify);srun 挂会话后台(session 重启杀作业);sbatch 内 `-w` pin 节点(与 Slurm 外进程竞态);首个 Default 卡选择(忽略占用致 OOM)。

## Resume Guardrails

- 依赖链 5872-5885 **已排好且脱离会话**——重载后先查队列/产物,**不要重发**。
- 所有 session 内 Monitor 随会话失效:需要盯防时重布(全量 run 日志、依赖链完成、站点可用性)。
- 站点 URL 可能已变:读 `results/p2_site/.current_url`,勿引用旧 URL。
- 节点占用需实测:`ssh <node> nvidia-smi --query-compute-apps` + 属主;pinghe 不可叠。
- Stage B 启动、held-out 检验、Cloudflare token 均需用户授权后再动。

## Coverage Audit

| Source checked | Meaningful information routed to |
|---|---|
| 本轮用户请求(全量训练启动、8min 推理、全部结果上站、ckpt 质量判定、依赖链) | `CLAUDE.md`(工具链)、`logs/progress.md` 07-29 条、`logs/findings.md` 07-29 条、本检查点 §In Progress/§Remaining |
| 会话内 todo/计划 | 本检查点 §Remaining(有序)+ `logs/progress.md` |
| Git status/diff/commits | 本检查点 §Working-Tree Details(clean)+ §Code and Artifact Pointers;提交序列见 `git log` |
| 验证结果与产物 | 本检查点 §Verification;判定文档附录(full@4000/@8000/8min);站点 127 卡 |

## Not Persisted

- 逐 case 原始指标数值(留在 `results/p2_interim/*/metrics/*.json` 与各 verdict 表格,不重复抄录)。
- 例行 watcher 心跳、集群瞬时占用快照(易过期,重载时实测)。
- 隧道/凭据细节(仅记 `.current_url` 路径,不记 token)。
- 1 个提交尚未推送(`d67c785`);`git push private echo-memory` 由下一会话或用户执行。

## Post-Compact Resume Prompt

```text
Context compaction has just completed. Read the session checkpoint at `/mnt/beegfs/siyuan/workspace/helios-echo/logs/session-ckpts/2026-07-29T221506Z--stage-a-closed-full-run-eval-chain.md`, then follow its `Resume First` and `Canonical Document Map` sections to read CLAUDE.md, logs/progress.md, logs/findings.md and the three verdict documents it points to. Recover the task motivation (Echo-Infinity evolving memory ported into Helios; Stage A closed,选 Stage B 基座), the constraints, the decisions, the tracking IDs, the completed campaigns, the verification evidence and the open decisions. Reconcile the checkpoint against current files, `squeue -u siyuan` and the latest logs before acting; do not repeat completed work and do not re-dispatch the queued Slurm dependency chain (5872-5885). Continue from the first remaining action: judge the full@12000 dependency-chain results (`results/p2_interim/p2-rep50-{r4evsw,r5long}-on-full12000/metrics/`), append the arm comparison to the two verdict documents, commit, and push `private echo-memory` — if the chain has not finished yet, only re-arm watchers and report status. 回复用中文。
```
