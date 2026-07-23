# Slurm 24.05.7：`DefCpuPerGPU` / `DefMemPerGPU` 后全节点 INVAL 调查

**范围**：只做公开资料与上游源码调查；未连接 `mhcluster`、未改集群配置。  
**对象**：Slurm 24.05.7，`select/cons_tres`、`CR_CORE_MEMORY`、cgroup/v2，节点静态拓扑 `CPUs=192 Boards=1 SocketsPerBoard=2 CoresPerSocket=48 ThreadsPerCore=2`，有 `CpuSpecList` 与 `Gres=gpu:h200:8`。  
**已知复现实验（由提问方提供）**：仅在 PartitionName 增加 `DefMemPerGPU=250000 DefCpuPerGPU=16` 后 `scontrol reconfigure`，所有节点转 INVAL，理由为 `Low socket*core*thread count, Low CPUs`；控制端独有配置和预先同步至所有节点均复现；回滚并重启 `slurmd` 恢复。因此“先后部署顺序/节点拿到旧 slurm.conf”已被该实验排除。

## 结论先行

1. **CONFIRMED：未找到把 Partition `JobDefaults=DefCpuPerGPU/DefMemPerGPU` 与该拓扑 INVAL 签名直接关联的公开 SchedMD bug、邮件线程、24.05.x NEWS 或 24.11.x NEWS 条目；因此不能把它称为已有已定位的 Def* bug，也没有可据以宣称的“修复版本”。** 搜索覆盖 SchedMD support/Bugzilla、slurm-users 旧/新归档、上游 NEWS/CHANGELOG 与 GitHub mirror；查到的是同一 *症状类* 的 `CPUSpecList` 已知 bug（24.05.3 已修），而不是 Def* JobDefaults bug。[24.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS)；[24.11 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.11/NEWS)；[Ticket 3608](https://support.schedmd.com/show_bug.cgi?id=3608)；[Ticket 8199](https://support.schedmd.com/show_bug.cgi?id=8199)
2. **CONFIRMED：24.05.7 包含 24.05.3 的 `CPUSpecList` 修复。** 该条目的原文是：`slurmd - Fix node going into invalid state when using CPUSpecList and setting CPUs to the # of cores on a multithreaded node`；发布公告日期为 2024-08-28。它与本集群的 `CpuSpecList`、超线程以及 INVAL 有实质相关性，但其公开描述没有提及 partition JobDefaults、GPU defaults 或 `DefMemPerGPU`。故它是应向 SchedMD 提供的强关联先例，不是对当前 24.05.7 复现的已确认解释。[24.05.3 NEWS/公告](https://groups.google.com/g/slurm-users/c/XT46zkWMJKw)
3. **CONFIRMED：这个 INVAL reason 是 controller 在 node-registration 中对“节点本次上报的 topology/CPU”与“controller 的 node config”所做的比较失败，而不是该检查本身直接读取 partition JobDefaults。** 上游 `node_mgr.c` 在 `threads1 < configured sockets×cores×threads` 时追加 `Low socket*core*thread count`，在 `reg_msg->cpus < config_ptr->cpus` 时追加 `Low CPUs` 并返回 `EINVAL`；同一实现段没有 `job_defaults_list`、`DefCpuPerGPU` 或 `DefMemPerGPU` 引用。[node_mgr.c L3132–3156](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/slurmctld/node_mgr.c#L3132-L3156)
4. **INFERRED：触发点很可能是 reconfigure 后 controller/slurmd 的 node-registration 或 CpuSpec/topology 状态被错误重建、重解释或短暂失配，而不是 `250000` MiB 或 `16` CPU 的正常调度含义本身。** 依据是：Def* 解析后被存为 partition/global `job_defaults_list`，而拓扑 INVAL 验证路径比较的是 registration message 与 node `config_ptr`；两路径在上游源文件中分离。这个推断不能在无 controller debug log、reconfigure 前后 `scontrol show node -d`、每节点 `slurmd -C` 和 `slurmd -C`/daemon log 的条件下升级为根因结论。[Def* partition parsing](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/common/read_config.c#L1520-L1538)；[global Def* parsing](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/common/read_config.c#L4171-L4188)；[registration comparison](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/slurmctld/node_mgr.c#L3132-L3156)
5. **生产首选：不要在该 24.05.7 集群继续用 partition `DefCpuPerGPU`/`DefMemPerGPU` 触发 reconfigure；改用一个极小、同步、快速的 `job_submit/lua` 在 controller 端只为目标 GPU partition 的“未显式请求”作业补入等价的 CPU/内存每 GPU 请求，并保留用户显式值。** 它绕开当前已复现的 JobDefaults/reconfigure 触发面，并且 server-side 覆盖 `sbatch`、`salloc`、`srun`、`slurmrestd`。该插件必须在 staging 验证后上线；不得在持锁回调中做网络/数据库/慢文件 I/O。[job_submit plugin 文档](https://slurm.schedmd.com/job_submit_plugins.html)

---

## 1. 证据与已知问题的精确归类

### 1.1 CONFIRMED：`Low socket*core*thread count, Low CPUs` 的含义

上游 controller 在节点注册时先得到节点上报的 sockets/cores/threads 与 `reg_msg->cpus`，再以 controller 读取的 node configuration 计算 `configured sockets × cores × threads` 并比较。任一“上报低于配置”都会把 error code 设为 `EINVAL`，并串接与现场相同的两个 reason 字符串；这解释了 `_slurm_rpc_node_registration ...: Invalid argument` 的同现关系。[node_mgr.c L3132–3156](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/slurmctld/node_mgr.c#L3132-L3156)

SchedMD Ticket 3608（已 RESOLVED FIXED，但针对 16.05.4）也将同一 reason 归为配置 topology 大于 node 注册 topology；其答复明确说触发条件是 configured sockets×cores×threads 大于节点报告值，并认为重启 `slurmd` 很可能清掉了节点 NUMA-mode 差异。该票**不**提及 DefCpuPerGPU、DefMemPerGPU 或 reconfigure。[Ticket 3608](https://support.schedmd.com/show_bug.cgi?id=3608)

**影响**：不能把这条 reason 解读为“250 GB/GPU 超过内存”或“16 CPU/GPU 的排程不可满足”。它是注册时的硬件/配置视图失败签名。

### 1.2 CONFIRMED：最接近的 24.05 已修回归是 CPUSpecList，不是 Def*

24.05.3 修复了“使用 `CPUSpecList` 且在超线程节点把 `CPUs` 设为 core 数时进入 invalid state”。24.05.7 比 24.05.3 新，故标准 24.05.7 上游发行内容已含此修复。[24.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS)；[24.05.3 release announcement](https://groups.google.com/g/slurm-users/c/XT46zkWMJKw)

本案同时具备 `CpuSpecList`、192 logical CPUs、2 thread/core、`CR_CORE_MEMORY` 和该 reason，因而应把以下两类情况并列为待证伪假设：

- **H1（INFERRED）**：24.05.7 后仍存在另一条 `CpuSpecList`/core-vs-thread/reconfigure 交互路径，未被 24.05.3 修复覆盖。
- **H2（INFERRED）**：JobDefaults 变更触发的 reconfigure 使已有/持久化 node configuration 或 slurmd registration 走到该交互路径；JobDefaults 只是触发器，并非被 topology 比较直接使用。

两假设均不能仅凭公开资料定案。要向 SchedMD 开 ticket，需要保存一次失败复现的：controller `SlurmctldDebug=debug2`/`DebugFlags=...` 日志、至少一台 node 的 `slurmd -C`、`slurmd -Dvvv` 或服务日志、reconfigure 前后 `scontrol show node -d <node>`、`scontrol show config`、完整 anonymized `slurm.conf`/`gres.conf`，以及是否由 systemd 启动 slurmd、`SlurmdParameters`、`CpuSpecList` 的精确值。

### 1.3 CONFIRMED：未命中“Def* JobDefaults 导致 INVAL”的公开修复

- 24.05.0–24.05.7 NEWS 中没有 `DefCpuPerGPU`、`DefMemPerGPU` 或 `JobDefaults` 的 INVAL/registration/reconfigure 修复条目；24.05.4 只有 typed GRES 加 `--[cpus|mem]-per-gpu` 的 memory-leak 修复，非节点注册问题。[24.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS)
- 24.11 NEWS 也没有该 Def* / registration 组合；它包含多项不相同的 reconfigure 并发、死锁、systemd CoreSpec 限制和 cgroup 行为修复，不能外推为本案已经修复。[24.11 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.11/NEWS)
- Ticket 8199 讨论的是 DefCpuPerGPU 的正常“下限”语义和展示/文档问题，结论为 INFOGIVEN；没有 reconfigure 或 INVAL。[Ticket 8199](https://support.schedmd.com/show_bug.cgi?id=8199)
- 历史上 DefMemPerGPU 有其他调度/共享缺陷：20.02 用户报告未显式 `--mem` 的 GPU 作业会阻塞其他作业共享节点，并引用 Bug 8527；临时 workaround 是改为 DefMemPerCPU。该线程不涉及 registration 或当前版本，故只能说明该 defaults 家族过去有独立问题，不能证明本案。[slurm-users 2020 thread](https://lists.schedmd.com/pipermail/slurm-users/2020-March/005101.html)

**判定**：`known bug?` 的严格答案是 **“没有找到此精确 bug 的公开报告/修复版本；找到一个已在 24.05.3 修复的强相关 CPUSpecList INVAL bug。”**

---

## 2. `scontrol reconfigure`：新实现、已知限制与本案含义

### 2.1 CONFIRMED：所谓“restartless 新实现”起于 23.11，不是 24.05 首次引入

23.11.0 NEWS 明载：`slurmctld - Rework 'scontrol reconfigure' to avoid race conditions that can result in stray jobs`。24.05 继承这一代实现；因此“24.05 改写 reconfigure”更准确的表述是“24.05 运行在 23.11 重构后的实现之上”。[23.11.0 NEWS entry](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS#L1235-L1255)

### 2.2 CONFIRMED：24.05/23.11 系列仍陆续修复 reconfigure 缺陷

上游 NEWS 记录过：failed reconfigure 后后续 reconfigure hang、`slurmd -c` 的 `scontrol reconfigure` 问题、`-b` 启动的 slurmd 每次 reconfigure 看似 reboot、slurmctld reconfigure 期间 crash/hang 等。它们证实 reconfigure 是有风险的复杂路径，但没有一条把 JobDefaults 与本案 reason 对上。[23.11.x/24.05 branch NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS#L1197-L1221)；[24.05.5 reconfigure fix](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS#L61-L68)

### 2.3 CONFIRMED：官方一般规则不是“Def* 必须重启 slurmd”

`slurm.conf` 文档的一般规则是配置修改在 daemon restart、SIGHUP 或 `scontrol reconfigure` 时生效（除非选项另有说明）；DefCpuPerGPU/DefMemPerGPU 没有文档化的“必须重启所有 slurmd”例外。因此不能把“重启 slurmd”包装为官方要求；它只是本集群已证实可恢复的事故处置/临时绕行。[slurm.conf](https://slurm.schedmd.com/slurm.conf.html)

Slurm FAQ 对“增加/删除 nodes”给 24.05 及更旧版本的保守程序是：停 controller、各节点更新 config、重启全部 slurmd、再起 controller；24.11+ 才给“各节点更新后 `scontrol reconfigure`”的简化程序。该 FAQ 是 node-membership 变更说明，**不能**直接推演为 partition JobDefaults 变更的必须步骤，但说明官方确实对旧版本的某些 topology/config 变更采用更保守的 daemon-restart 操作。[FAQ](https://slurm.schedmd.com/faq.html)

### 2.4 对本案的操作结论

- **INFERRED**：既然“预同步全部节点”也失败，单纯改部署顺序或只重启 slurmd 后再 reconfigure 不应被当作可靠生产方案；它可能掩盖、重置或规避 race/state，但没有消除触发条件的证据。
- **CONFIRMED**：在 24.05.7 上，继续为同一个 JobDefaults 目的重复 controller reconfigure 已经是一次全节点可用性风险；先换机制比继续试命令序列更稳健。

---

## 3. 不用 partition JobDefaults 的机制比较

目标是每张 H200 默认 **16 CPU + 250000 MiB memory**，8 GPU 节点上全占时为 128 CPU 与 2,000,000 MiB；是否适合实际 `RealMemory`、是否允许 CPU-only 作业占余下 64 logical CPU，属于本地调度策略问题，未在本调查中假定。

### (a) `DefMemPerCPU`（global 或 partition）

**CONFIRMED 语义**：DefMemPerCPU 是每个 *usable allocated CPU* 的默认 memory（MiB），一般用于 `SelectType=select/cons_tres` 的 individual CPU 分配；DefMemPerCPU、DefMemPerGPU、DefMemPerNode 互斥。超线程而又请求 `--threads-per-core=1` 时，不能使用的 sibling thread 不计入 memory-per-CPU。[slurm.conf DefMemPerCPU](https://slurm.schedmd.com/slurm.conf.html#OPT_DefMemPerCPU)

**优势（INFERRED）**：它走的是传统 `def_mem_per_cpu` 配置字段，而非 Def*GPU 被打包到 `job_defaults_list` 的路径；因而是一个与当前触发器不同的候选路径，值得在隔离 staging 上测试。

**限制（CONFIRMED/INFERRED）**：它不按 GPU 数计量。若想让每 GPU 等价 250000 MiB，需让每 GPU 恰好请求 16 CPU 且将 DefMemPerCPU 设为 15625 MiB；一旦用户请求不同 CPU 数、CPU-only 作业运行，或 thread/core 可用数变化，memory 不再是“250000 MiB/GPU”。因此它不能单独、可靠地实现需求；全局设高值还会强制抬高所有 CPU-only 作业的默认内存。[slurm.conf DefMemPerCPU](https://slurm.schedmd.com/slurm.conf.html#OPT_DefMemPerCPU)

**结论**：仅作为 staging A/B 或“明确固定 16 CPU/GPU 且 CPU-only 策略可接受”时的次选；不建议作为生产首选。

### (b) `job_submit/lua` 在 controller 端补入请求

**CONFIRMED**：job_submit 在 slurmctld 上的每次 job submit 调用（包括 `salloc`、`sbatch`、`slurmrestd`）发生在 Slurm job defaults 之前；job descriptor 是输入/输出，可修改提交参数。官方警告 slurmctld 在持内部锁时运行它，单个脚本实例串行，故脚本必须快速且不做慢/阻塞工作。[job_submit plugin 文档](https://slurm.schedmd.com/job_submit_plugins.html)

**建议设计（INFERRED，需在 24.05 staging 做 API/字段单元测试）**：

1. 只匹配 GPU partition（最好用固定 partition 名/allowlist；多 partition 作业需明确策略或拒绝，避免已知跨 partition DefCpuPerGPU 语义复杂性）。
2. 只在用户没有显式 CPU-per-GPU / CPU-per-task / memory (`--mem`、`--mem-per-cpu`、`--mem-per-gpu`) 请求时，注入等价的 `--cpus-per-gpu=16` 与 `--mem-per-gpu=250000` 的 job descriptor 请求；显式请求一律保留，或按书面策略拒绝低于站点最小值。
3. 使用本集群的 Slurm 24.05 Lua descriptor ABI/官方 contrib 样例确认字段与 memory-per-TRES flag；提交后以 `scontrol show job` 和 `sacct --allocations` 验证 `ReqTRES/AllocTRES`，并做 1/2/8 GPU、typed/untyped GRES、sbatch/salloc/srun/slurmrestd、显式 override、多个 partition 的矩阵。
4. 脚本及目录由 Slurm 管理用户拥有、不可被普通用户写；保持纯计算/常数读取，并写最小审计日志。

**结论**：实现/运维风险最低、策略强制力最高，且避免触碰已复现出故障的 partition JobDefaults reconfigure 面；为本调查的首选。

### (c) `cli_filter` / `cli_filter.lua`

**CONFIRMED**：cli_filter 在 `salloc`、`sbatch`、`srun` 客户端中运行；`setup_defaults` 可设默认，`pre_submit` 在 CLI/environment/directive 解析后、contact controller 前可读写或 unset options。官方同时明确它不是安全边界：用户可提供替代 `slurm.conf` 禁用插件；推荐配合 job_submit 强制策略。[cli_filter plugins](https://slurm.schedmd.com/cli_filter_plugins.html)

**结论**：可用于给用户即时显示“默认已加 16 CPU/GPU、250 GiB/GPU”的体验、早期提醒或本地提交便利；不能单独用于生产强制执行。若采用，应与 (b) 同时部署，(b) 才是权威。

### (d) 升级 Slurm

**CONFIRMED**：当前 24.05.7 已包含最接近的 CPUSpecList INVAL 修复（24.05.3）。24.11 的 NEWS 未给出“DefCpuPerGPU/DefMemPerGPU JobDefaults 触发 Low CPUs INVAL”的修复条目，因而**没有证据能指定一个升级版本为本案修复**。[24.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS)；[24.11 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.11/NEWS)

后续主线确有 DefMemPerGPU 的其他修复（例如 25.05 NEWS 的“setting job requested memory 时考虑 DefMemPerGPU”），但这属于 requested-memory 语义，不是 registration/topology INVAL，不能作为本案 fix version。[25.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-25.05/NEWS)

**结论**：中期应升级到供应商支持的较新稳定系列（并让 SchedMD 根据复现确认是否有私有/未索引修复），但升级不是当前生产回避此触发器的已证实修复。升级前需 staging 跑完整 reconfigure 和 GPU accounting/cgroup/v2 regression matrix。

---

## 4. 生产 8 节点集群的风险排序与决策

| 排名 | 方案 | 风险 | 推荐结论 |
|---|---|---|---|
| 1 | **Controller `job_submit/lua` 注入（b）** | 低至中：小脚本若慢会影响全局提交，故必须极小、staging 验证、保留回滚 | **现在采用。** 无需为了 GPU 默认值再 reconfigure partition JobDefaults；server-side，无法由用户绕过。 |
| 2 | **(b) + cli_filter（c）** | 中：增加客户端部署/版本兼容面；但 controller 仍是最终强制 | 推荐作用户体验层，不是替代品。 |
| 3 | **升级至受支持新系列后，在 staging 重测原 JobDefaults（d）** | 中至高：升级面广，且没有针对本故障的已确认修复版本 | 作为中期治理；拿最小复现与 SchedMD support case 结果决定是否恢复 JobDefaults。 |
| 4 | **DefMemPerCPU（a）并额外要求 16 CPU/GPU** | 中至高：GPU 与 CPU/memory 脱钩，改变 CPU-only 作业语义；仍需一次 reconfigure 测试 | 仅在固定资源比例与 CPU-only policy 明确时试验；非首选。 |
| 5 | **继续使用 partition DefCpuPerGPU/DefMemPerGPU，并用“同步 config + 重启 slurmd”操作顺序赌恢复** | 高：已两次造成所有节点 INVAL；无公开证据证明该程序消除了根因 | 不推荐用于生产。若为了支持 case 必须复现，只在可回滚的维护窗口/隔离环境做。 |

### 上线前最小安全门

1. **冻结当前可用配置**，保留已知可恢复的 rollback 包、每节点服务操作与 node state 恢复 runbook；不要在工作日重复 JobDefaults reconfigure。
2. 用一台 staging controller + 至少一个同拓扑 GPU node，优先验证 job_submit 方案；提交矩阵必须含 1/2/8 GPU、`--gres=gpu:h200:N` 与 `--gpus=N`、用户显式 CPU/memory、CPU-only、异构/多 partition 请求以及 `salloc`/`sbatch`/`srun`。
3. 验收原则：默认 GPU 作业的 `ReqTRES`/`AllocTRES` 为每 GPU 16 CPU 和 250000 MiB；显式 policy 行为一致；无 `slurmctld` 提交延迟增长；不需要 `scontrol reconfigure` 才能新增/修改策略脚本（脚本更新仍按官方 job_submit 安全加载/回滚机制操作）。[job_submit plugin 文档](https://slurm.schedmd.com/job_submit_plugins.html)
4. 若向 SchedMD 升级为正式 defect，请报告为：**“24.05.7 / CPUSpecList / 2-thread core node：仅添加 partition DefCpuPerGPU+DefMemPerGPU 并 `scontrol reconfigure`，所有 byte-identical nodes 报 Low socket*core*thread count, Low CPUs；预同步 slurm.conf 仍复现；24.05.3 CPUSpecList fix 已包含。”** 附带 §1.2 的完整证据包，并要求确认是否存在 24.05.7 后的 private patch 或推荐 fix release。

---

## 5. 来源审计与不确定性

### 已查来源

- SchedMD 上游 [24.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.05/NEWS)、[24.11 NEWS](https://github.com/SchedMD/slurm/blob/slurm-24.11/NEWS)、[25.05 NEWS](https://github.com/SchedMD/slurm/blob/slurm-25.05/NEWS) 与 `slurm-24.05` 源码的 [node registration validation](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/slurmctld/node_mgr.c#L3132-L3156)、[partition Def* parsing](https://github.com/SchedMD/slurm/blob/slurm-24.05/src/common/read_config.c#L1520-L1538)。
- SchedMD support：[Ticket 3608](https://support.schedmd.com/show_bug.cgi?id=3608)、[Ticket 8199](https://support.schedmd.com/show_bug.cgi?id=8199)。
- slurm-users：[24.05.3 announcement](https://groups.google.com/g/slurm-users/c/XT46zkWMJKw)、[DefMemPerGPU historical thread](https://lists.schedmd.com/pipermail/slurm-users/2020-March/005101.html)、[multi-partition DefCpuPerGPU discussion](https://groups.google.com/g/slurm-users/c/Vd2BTCg3vgM)。
- 官方文档：[slurm.conf](https://slurm.schedmd.com/slurm.conf.html)、[cons_tres](https://slurm.schedmd.com/cons_tres.html)、[job_submit plugins](https://slurm.schedmd.com/job_submit_plugins.html)、[cli_filter plugins](https://slurm.schedmd.com/cli_filter_plugins.html)、[FAQ](https://slurm.schedmd.com/faq.html)。
- 旁证（非 SchedMD 官方，不能作为根因证明）：[StackHPC CPUSpecList/CoreSpec INVAL issue](https://github.com/stackhpc/ansible-role-openhpc/issues/199)；它说明改变 CpuSpecList 后 controller/slurmd 不一致可导致 `CoreSpec differ`，未提及 Def*GPU。

### 明确未知

- 没有 `mhcluster` 当次失败的 controller/node debug registration 字段，因此不知道失败时 `reg_msg->{cpus,sockets,cores,threads}` 与 controller `config_ptr` 各自具体为何值。
- 没有 SchedMD 私有 support case/补丁访问权；公开未命中不等于不存在私有已知 defect。
- 不能仅从当前两个复现实验区分“DefCpuPerGPU 单独触发”、“DefMemPerGPU 单独触发”还是两者组合触发；生产上不应为区分它们再做破坏性二分试验。
