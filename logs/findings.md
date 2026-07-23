# Findings — Echo-Infinity → Helios 迁移研究

**日期**: 2026-07-22 | **状态**: 研究完成，设计文档已交付
**主交付物**: `docs/echo-to-helios-migration-design.md`（总纲 + 四章，119KB，全部 file:line 锚定）

## 核心结论

1. **迁移可行，但研究文档的四个设计承诺被代码现实推翻/修正**（详见总纲第二节偏差表）：
   - "KV 拼接第三组" → **token 插入**（Helios 无跨层共享 KV 先例，t=0 AdaLN 只作用于序列 token）
   - "t≈0 写入源" → distilled 最后一次调用在 **stage-local σ≈0.5**（非 t≈0）→ Enc 的 **FiLM(σ_last) 必选**
   - 单一代码路径 → **双路径**：发布推理走独立 `helios/diffusers_version` 镜像，必须 M1（训练链）/M2（镜像）双移植
   - "Echo 静态 query 先行课程" → Echo 无此阶段（两个 DMD 阶段从第一步启用 memory）；Stage A 是 Helios 数据通路强加的新阶段

2. **统一设计基线**（第四章 R0/D1 仲裁后）：token 插入读机制 + M = N_Q×mid-tier 网格（720@368×640）+ 最后调度步 last-layer hidden 写源 + Enc 2 层 cross-attn/gate bias 0.75 + (0,1) 分数 RoPE id（A9 消融裁决）+ transformer 正式子模块全秩归属 + 三阶段课程（A: TF 静态 → B: TF unroll U=4 → C0 warm-up → C: SF/DMD ≥4 sections）。

3. **Stage C 不可省的代码级论证**：TF-only 写门 = Echo 批评的手写规则等价物；且现有 DMD 基建三个硬约束——Stage-1 rollout 死代码（`utils_helios_post.py:679`）、GT 模式断言锁单 section（`:2349,:2966`）、出口步随机采样——必须用 self-forcing 变体 + `dmd_num_latent_sections_min≥4` + σ_last 捕获的专用 rollout。

4. **工程雷区**（复用主线脚本前必须处理）：
   - `HELIOS_FORCE_LR=1` resume 覆写压平 param-group lr（`train_helios.py:1086-1099`），祖先配置明文依赖
   - LoRA all-linear 扫描会捕获 memory 的 Linear → 必须 `exclude_modules` + 注入后断言
   - `from_pretrained` 静默跳过形状不匹配键（`transformer_helios.py:1767-1776`）→ N_Q sweep 需显式加载断言
   - EMA step 位置 zip 静默截断（`create_ema_zero3.py:245-250`）→ 初始化前比对 (name, shape) 列表
   - rank-128 (A/B) → rank-256 (C) adapter 不可互载 → memory 组件全秩 + C0 warm-up 交接
   - detach 次序：必须"下一次写之前、本次读 backward 之后"，否则 bptt=1 下 Enc 永无梯度

5. **两项 OBSERVED ABSENCE 影响路线图**：
   - 仓库无任何 history 离散化/VQ 实现（研究文档 Phase 0 是纯规划态）；仲裁：memory 先行，VQ 排 M1 后独立分支
   - 评测侧无 within-video 斜率后端（只有端点聚合）→ 4 类新时序指标以 `external_command` 挂入 long_video_eval，检测手段本身是交付物

6. **预算与协调**：Tier-0 ≈ 5.3–5.9k H200·h；c-node07 被 rwtag pin 死仅可条件复用；基座冻结 `_correct/checkpoint-19500`；K3 采用配对相对口径（历史 0.069 属另一 campaign 谱系，绝对阈值会误杀）。

## 事实修正记录（实现期新增）

- **设计 D11 被代码证伪**：utils_helios_post.py 有 4 处对 transformer 返回值的二元组解包（:329/:3212/:3295/:3350，DMD/GAN 路径），"恒定三元组、所有调用点索引 [0]" 不成立 → 实现改为条件返回（仅 `capture_last_hidden=True` 时三元组），存量调用点零改动。
- **测试陷阱**：`init_weights()` 将 `proj_out` 权重与 bias 全部清零（transformer_helios.py:1599 + 基础循环）→ 新构造模型输出恒零，任何基于输出的断言（等价性/梯度）都会空洞通过或假失败。tiny 模型测试必须重随机化 proj_out。
- **GPU 测试要点**：flash-attn3 只接受 CUDA + fp16/bf16 张量；测试在 CUDA 可用时自动上卡转 bf16（顺带覆盖了 bf16 计算 + fp32 state 不变量的组合）。
- **决策记录**：真模型测试的环境与 ckpt 选型见 `docs/specs/2026-07-22-real-model-test-checkpoint-decision.md`（P1 冒烟 → Helios-Base + A1@19500 装配臂；P2 anti-drift → Distilled 为主）。

## 事实修正记录（研究期）

- 早期勘察误读：`infer_helios.py:290` 的 `is_amplify_history=True` 只是 loader shim（transformer 已构造完毕），**不激活**运行时放大；冻结基线实际 `is_amplify_history: false`。
- Echo "+10.6% 吞吐开销"在 Echo 仓库无基准佐证（源自论文转述），已撤销"代码事实"地位，验收以实测 ≤12% 为线。

## 文件索引

- `docs/echo-to-helios-migration-design.md` — 主设计文档（总纲 + 四章）
- `logs/research/design-{module,training,inference-eval,risks}.md` — 四章修订版单行本
- `logs/research/design-risk-experiments.md` — 风险节修订前草稿（已被 design-risks.md v2 取代，留档）
- `logs/research/read-*.md` — 6 份深读报告（英文，file:line 锚定）：echo-mem / echo-train / echo-infer / helios-model / helios-train-infer / xiangbo
- `plan.md` — 研究计划（已执行完毕）

## 研究方法记录

22-agent 工作流（Run ID `wf_e0612e6f-c7e`）：6 并行深读 → 3+1 设计（风险节基于前三节成稿）→ 每节 2 路对抗验证（代码锚定核查 + 可行性反驳）→ 修订。验证轮共发现 26+ 项 issue（含多个 blocker），全部在修订版解决或以脚注反驳。跨越多次进程重启，靠 journal 缓存 + 脚本手术（空返回 fallback/retry）无损续跑。

## GPU 冒烟结论（2026-07-22 深夜批次）

- P1-B 臂（Helios-Base 直载）13/13 PASS；P1-A1 臂（transformer_init + lora368@19500 + partial 装配）9/9 PASS → 生产加载与训练谱系装配两条路径均与 memory 模块兼容，off-path 等价在真权重成立。
- rollout 冒烟 run1：状态机全对（3 写/队列[3,4]/逐条目σ），开销 +5.6%（验收线 12%），NaN 根因 = 冒烟自身调用偏离验证采样配置（8步+固定mu=1+空负提示），产品代码无缺陷（Sol worker 对照 log_validation/infer 双基准 + diff 复核）。
## 基础设施与数据资产发现（2026-07-23 凌晨）

- **368×640 语料失踪（硬阻塞）**：旧路径在 xiangbo reorg 的 delete_list.sh 中；canonical tree yaml 计划了 `human_single/latents`（n=168431 硬链接）但目录从未创建；demo latents 同灭；原始视频+manifest 完好可重编码。恢复由用户操办。
- **mc-node01 闲置真因**：非仅 GPU2 硬件——`nvidia_uvm` 不干净卸载（07-22 20:08）致整节点 CUDA init 失败；rmmod/modprobe 修复后 7/8 卡可用。nvidia-smi 正常≠CUDA 可用（NVML 不经 uvm）。
- **Slurm 24.05.7**：partition 级 `DefMemPerGPU/DefCpuPerGPU` + reconfigure 两次复现全节点注册 INVAL（配置同步与否无关→参数触发）；无公开 bug 票；替代 = job_submit.lua（调查报告入库）。恢复程序：配置一致 → 逐节点重启 slurmd → resume。
- **trainer 配置快照校验**：output_dir 里的快照会在重启时逐键比对，schema 演进后旧 scratch 目录必须清理。

## 实现期对抗审查发现（2026-07-23，Stage A Task 1/2 批次）

- **上游既有缺陷（值得同步主线）**：stage-1 dataset 的 `_epoch` 是普通 int，persistent DataLoader workers（谱系配置 8 workers）持 fork 副本，主进程 `set_epoch` 永不到达——`choice_idx` 的逐样本 seeded 流跨 epoch 塌缩为创建时 epoch 的固定值（审查 worker 用 StatefulDataLoader 实测复现；影响 xiangbo 现行训练的跨 epoch 数据增广）。本 fork 修复：epoch 放 `multiprocessing.Value`（fork 继承下 set_epoch 实时可见，spawn pickling 降级为冻结 int 不劣于现状）。
- D4 审查 blocking（已修 e064ad9）：低分辨率桶驱逐帧曾取全分辨率 source timeline，而写前向的 X_Noisy 必须随桶分辨率；修为双 timeline 角色分离。审查同时确认：既有 history 路径本就是 full-res 条件契约（有意的混合分辨率训练）。
- D6 审查 minor（已修）：非事件样本 U=4 rollout 有 ~98.4% 概率中途切换 caption 版本（逐 section 独立全局随机抽）→ 裁决为一次 rollout 只抽一次并复用；事件样本逐 chunk 映射不变。
- D6 审查证实项：torch.randint 首抽与旧行为逐位等价（Torch 2.10 实测）；`num_frame` 文件名字段确为 RGB 帧数、`//33` 与离线 chunk 数一致；filter 后 bucket 重建无 stale-index。
- Task 3a 审查：CLEAN（逐语句等价核对；memory_tokens=None 与缺省同路径；唯一调用点关键字调用不受损）。
- Task 3b 审查 blocking（已修）：单写 helper 的全零 early-return 是逐 rank 本地决策——多卡下 rank 间 wrapped-forward 次数不一致 ⇒ buffer-broadcast collective 错位 ⇒ 训练卡死，单卡测试不可见。修复双管齐下：写掷币改为 (seed, global_step) 确定性导出（全 rank 一致 + resume 稳定）；early-return 移除（写步全 rank 跑捕获前向，全零 batch 由 blend 丢弃）。**smoke 门升级：必须 ≥2 卡 DDP 且覆盖 rank 间混合全零/有效驱逐批次。**
- Task 3b 审查反驳项（重要排险）：`find_unused_parameters=True` 与图外 update **不是**失败模式——DDP 自 forward 输出反向遍历会沿 `memory_tokens` 输入边继续到 Enc/query_init 上游（审查 worker 2-rank 最小复现确认梯度可达）；D4 mask 方向、capture 契约、blend dtype/device、trainer 作用域、逐 micro-step reset 语义均核对通过。

- **rollout 冒烟 run2（终局）：8/8 ALL PASS**（job 5552, c-node06, ~48 min）——修正采样配置后双臂 latents 均有限，NaN 根因判定实锤；状态机与 run1 逐项一致；开销 +5.5%、峰值 43.3 GiB；σ_last=0.122≠0 实测确认 FiLM(σ_last) 写条件化为必要设计（呼应总纲 §二 σ_last 偏差项）。verdict: `logs/research/rollout-smoke-verdict-run2.md`。**管线状态机 GPU 门全绿，M1 训练链四批次（模块/transformer/trainer/管线）全部收口。**
