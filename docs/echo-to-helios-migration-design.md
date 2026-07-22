# Echo-Infinity → Helios 迁移精细设计（总纲）

**日期**: 2026-07-22
**输入**: ① 研究文档 `agent-research/Claude_export_Memory, drift and stability...md`（六大迁移挑战先验）；② `/mnt/beegfs/siyuan/workspace/Echo-Infinity`（全量深读）；③ `/mnt/beegfs/siyuan/workspace/helios-team` @ `mid_training_xiangbo`（全量深读，含分支增量与团队文档）。
**产出方式**: 22-agent 多阶段工作流（6 深读 → 4 设计 → 每节 2 路对抗验证 → 修订），全部代码事实在磁盘逐行复核（共 422 次工具调用）。本总纲整合四个设计章节并吸收第四章的跨节仲裁（R0/D1/D8/D9）。

---

## 一、结论摘要（统一设计基线）

把 Echo 的可学演化记忆迁移到 Helios，统一后的 v1 设计是：

1. **读机制 = token 插入（非 KV 注入）**。Enc 维护的 query state `[B, M, 5120]` 作为第四个 tier 前缀插入序列，布局 `[mem | long | mid | short | current]`；memory 由此获得 t=0 AdaLN 条件（纯 KV 注入无法做到，这是仲裁的决定性理由）；`memory_key_scale` 逐头独立（init −4.0 → scale≈1.16 中性起步），复用 `is_amplify_history` 的映射模式。memory 块输出被现有后缀切片丢弃、不回写状态。
2. **M 语义 = N_Q 帧当量 × 主训练桶 mid-tier 网格**（368×640：240 token/帧，N_Q=3 → M=720），固定形状、跨 section 持有、fp32 常驻。
3. **写入源 = 每 section 最后一次被调度的去噪前向的 last-layer hidden**（`norm_out`/`proj_out` 之前、X_Noisy 切片）。关键修正：t≈0 只在 Mid 严格成立（σ≈0.001）；Base σ≈0.02–0.04；**distilled 是 stage-local σ≈0.5**——三模式写源噪声差异跨数量级，故 **Enc 的 FiLM(σ_last) 条件化为必选**，显式 t=0 commit 前向仅作 ablation 开关。
4. **写入路径 = Echo Enc 忠实移植 + 适配**：2 层 cross-attention（RMSNorm/40头/128维），新增 `ctx_k_proj/ctx_v_proj`（Echo 直接吃 KV 缓存、Helios 写源是 hidden states）；门控 EMA `new = g·old + (1−g)·proj`；**gate bias 从 Echo 的 +2.0 重标定为 0.75**（section 级 9 帧稀疏写 ≈ 3 次 Echo block 写，0.881³≈0.684→logit≈0.77；sweep {0.75, 2.0}）。
5. **驱逐语义**：窗口 19 = 2×9+1 帧，append section k 时离窗帧组为 `[9k−19, 9k−11]`（k=2 首次含真实帧 8 帧、k≥3 整 9 帧、永不与 chunk 边界对齐）；捕获入 FIFO 队列（capture 即 detach），驱逐事件触发 Enc 写入，单 section 延迟可见。首个非平凡 memory 生效于 section 3。
6. **RoPE 槽位**：默认 (0,1) 开区间分数 id（N_Q=3 → {0.25, 0.5, 0.75}，anchor=0 之后、long=1 之前，既有 id 布局零扰动）；负 id 与 id=0 共列消融 A9 裁决（分数 id 的"模型见过分数相位"论据已被验证撤回，此项必须实验裁决）。
7. **归属 = transformer 正式子模块**（非 Echo 式 `object.__setattr__` 外挂）：全秩 named module、加入 PEFT `exclude_modules`、走 `save/load_extra_components` 第 5 节持久化、独立 param group（memory lr ×5）。Helios 训练器全链路走 `named_parameters`，外挂即全部遗漏。
8. **训练课程（三阶段 + 交接）**：
   - **Stage A**（TF + 静态 query + 单写模拟，~4k 步）：只训 memory，backbone LoRA/patch 冻结；从 `stage1_lora_cfr_368_correct.yaml` / `checkpoint-19500` 分叉。
   - **Stage B**（TF unroll U=4，~3k 步）：dataloader 供给连续 section（19+9U 帧）、逐 section prompt；trainer 新增 `_flow_loss_section_unroll` 状态机；**detach 施加在下一次写之前、本次读的 backward 之后**（否则 Enc 永远无梯度——初稿 bug 已由验证修正）；corrupt 只进 loss 前向、capture 前向用干净 history。
   - **C0 warm-up**（~500–1000 步）：memory 全秩组件移植进 distilled rank-256 generator（rank-128 adapter 无法直接载入，谱系交接靠全秩组件 + 短程回归）。
   - **Stage C**（self-forcing DMD 校准，~2k 生成器步，不可省）：绑定 Stage-2 金字塔路径（Stage-1 rollout 是死代码）；强制 `dmd_num_latent_sections_min ≥ 4`（否则 Enc/gate 在任何 loss 读到写入前 rollout 已结束、梯度恒零）；捕获在与推理相同的 σ_last；critic/real score 保持 memory-blind。
9. **推理集成**：训练链（`transformer_helios.py` + `pipeline_helios.py`）先行 = M1；**发布推理走独立的 `helios/diffusers_version` 镜像**，必须双移植 = M2。段内只读、段间写入；每次 `__call__` 重置为可学 M₀（恒在，主动偏离 Echo 冷启动语义，换 compile 形状恒定）；v1 禁用 CP（CP plan 无 memory 概念，rank-local 切片会错位）；开销解析估算 ≈ +10.5%（N_mem 帧当量 3、全分辨率序列），验收线：实测吞吐损失 ≤12%。
10. **评测 = 斜率优先**：vs24_long（90.75s）主战场 + ultralong（181.5s，~130 次写入）+ eventswitch；新增 4 类时序指标（motion-amplitude 斜率、saturation/chroma 曲线、boundary LPIPS-excess、event-aware CLIP-over-time）以 `external_command` 挂入 long_video_eval，零核心改动——**评测侧现无任何斜率后端，检测手段本身是交付物**；扰动再漂移测试（`perturb_at_chunk=30` 持久型状态扰动，度量恢复时间与再漂移斜率）；推理消融五组 {memory-on, no-KV, frozen-M₀, shuffled-M, is_keep_x0=False}（注意 M=0 不是"无记忆"——投影带 bias，no-KV 必须真移除前向）。

## 二、研究先验 vs 代码现实：关键偏差（本研究最有价值的发现）

研究文档的六大挑战框架方向正确，但四个设计承诺在代码现实前需要修正：

| # | 先验承诺 | 代码现实 | 设计修正 |
|---|---|---|---|
| 1 | memory KV 拼接为注意力"第三组" | Helios 所有非文本 K/V 均来自当层 hidden 经块内 to_k/to_v；无跨层共享 KV 先例；t=0 AdaLN 只作用于序列内 token | 改为 token 插入（第一章偏差 A） |
| 2 | 写入源 = "t≈0 last-layer hidden" | 无 commit 前向；distilled 最后一次调用在 stage-local σ≈0.5，非 t≈0 | 取"最后调度步前向"+ FiLM(σ_last) 必选（第三章偏差 1） |
| 3 | 单一模型代码路径 | 发布推理走独立 `diffusers_version` 镜像，与训练侧模块完全分离 | M1 训练链 / M2 镜像双移植（第一章偏差 D） |
| 4 | Echo 课程含"静态 query 先行"阶段 | Echo 两个 DMD 阶段从第一步即启用 memory；无 TF 预阶段 | Stage A 是 Helios 数据通路强加的新增阶段，非 Echo 移植（第二章偏差 F1） |
| 5 | Echo 实测 +10.6% 吞吐开销 | Echo 仓库无此基准（源自论文转述） | 撤销"代码事实"地位；以实测裁决，验收 ≤12%（第三章§5） |
| 6 | 现有训练可扩展出 memory 训练 | `_flow_loss` 单前向、无 section 循环、无驱逐事件；`dmd_teacher_forcing` 为损失侧死旗标；GT 模式断言锁单 section；Stage-1 rollout 死代码 | dataloader/trainer 实质改造 + 专用 memory rollout（第二章 D6/D7/D8） |
| 7 | （工程盲点） | `HELIOS_FORCE_LR=1` resume 覆写把所有 param-group lr 压平为单一标量，而 Stage A/B 祖先配置明文依赖它 | D11 守卫 + 按 role 恢复扩展（第二章偏差 F4） |
| 8 | （工程盲点） | 冻结基线 `is_amplify_history: false`；`infer_helios.py:290` 只是 loader shim，不激活运行时放大 | R1 的静态前缀组合是 anchor+memory，history 放大须构造期开启并配对训练（第四章 R1） |

另有两项 OBSERVED ABSENCE 影响路线图：仓库内无任何 history 离散化/VQ 实现（研究文档的 Phase 0 VQ 实验是纯规划态，与 memory 的先后由第四章 R6/D4 裁决：memory 先行）；评测侧无 within-video 斜率后端（先验 #6 要求的斜率验收需先建设工具）。

## 三、里程碑与裁决门（K0–K4 摘要，详见第四章）

- **K0（实现前）**：D1/D8 仲裁签字；checkpoint-conversion 测试（A/B memory state 载入 Stage C 构造器零 missing key）；`dmd_num_latent_sections_min≥4` 断言 + 最短 rollout Enc/gate 梯度非零单测；Stage B smoke（bs2×U=4 显存）。
- **K1（Stage A 末，≤15 GPU·天）**：flow loss 劣化 ≤+1%；memory 读注意力质量 >1%（需先落地 D10 诊断路径）；91s DOVER 回退 ≤0.02。
- **K2（Stage B 末，累计 ≤70 天）**：gate 均值 ∈[0.05,0.6]；subject-consistency ≥+0.005 且 boundary-LPIPS-excess ≤−5%（vs no-KV）；动态度冻结检测不触发。
- **K3（Stage C 末，累计 ≤130 天，一次即裁决）**：配对相对口径（同 checkpoint-19500 的 no-KV 重跑基线，历史 0.069 只作锚不作对照）；≥2 项主斜率仍劣 → 判"TF-only 写门净负"成立，发布 no-KV。
- **K4（全局）**：累计 >6k GPU·h 或占主线节点 >2 周 → 无条件暂停。

**预算**：Tier-0（主课程 + A9 RoPE + A1 N_Q + A2 gate bias + A5a 去C消融 + A6 推理消融）≈ 5.3–5.9k H200·h。**主线协调**：c-node07 已被 rwtag pin 死，仅条件复用；基座冻结 `_correct/checkpoint-19500`；工具链（extra_components 第 5 节、merge/EMA memory 分支、conversion 测试）默认关闭门控、可先行合并；请主线书面冻结"底座不切 Wan2.2、VQ 不先行"至 M1。

## 四、文档结构

- **第一章 模块架构**（design-module.md）：参数清单、写入源 tap 点、注入与三分记账、RoPE、t=0、驱逐语义、修改清单、N_Q 规模
- **第二章 训练课程与 trainer 改造**（design-training.md）：配置谱系、Stage A/B/C 逐项、param groups、持久化清单、断言、gate 换算、算力预算
- **第三章 推理集成与评测协议**（design-inference-eval.md）：捕获挂钩、写入时序、状态生命周期、三模式交互、兼容性、开销、评测协议、go/no-go
- **第四章 风险矩阵与消融计划**（design-risks.md）：跨节仲裁 R0、七大风险 R1–R7、消融矩阵 A0–A9、Kill Criteria K0–K4、主线协调 C1–C6
- **附录 深读报告**：`logs/research/read-{echo-mem, echo-train, echo-infer, helios-model, helios-train-infer, xiangbo}.md`（英文，全部 file:line 锚定）

注意第一、二、三章各自成文于仲裁前，其间对读机制/M 定义/RoPE/命名的互斥参数化以**第四章 R0/D1 的仲裁为准**（本总纲第一节已按仲裁后基线陈述）。各章内部的行号锚点、验证脚注均保留原貌以便核查。

---

# 第一章 模块架构


所有关键锚点已在两个仓库当前代码上逐行复核(helios-team 与 Echo-Infinity 均以磁盘现状为准);本版针对验证轮全部 blocker/major 完成设计级修复(非措辞软化),minor 逐条修正;两处验证意见经复核不成立,以脚注注明。

## 0. 与先验承诺的偏差(重要)

**偏差 A(承诺 #2 的实现方式修正)**:承诺 #2 要求"memory KV 拼进 joint attention 作第三组"。但 Helios 中除文本外所有 K/V 均来自当层 hidden state 经块内 `to_k/to_v` 投影(`helios/modules/transformer_helios.py:488-490,227`);Echo 式"跨层共享固定 KV 注入"在 Helios 无先例,且承诺 #2 自身要求的 t=0 AdaLN 条件在纯 KV 注入下无实现路径(AdaLN 只作用于序列内 token,`transformer_helios.py:809,1444-1446`)。故本设计以 **token 插入**实现:Enc 维护的 query state `[B,M,5120]` 作为第四个 tier 插入输入序列;"第三组 K"由块内投影自然产生,`amp_mem` 为独立参数作用于 memory K 切片。注意这与 Echo 机制**并不等价**:Echo 的 memory 是只读 K/V——K/V 布局为 `[sink, memory, local]` 而 Q 仅来自当前 token(`Echo-Infinity/wan/modules/causal_model.py:283-300`),且在因果局部缓存窗口内(`causal_model.py:124-168`);Helios token 插入在非限制模式(全注意力、无 mask,`helios/modules/helios_kernels/attention_dispatch.py:132-149`)下 memory 的 Q 与全序列互见。这是 token 插入的必然结果、有意偏离,memory 块输出被丢弃不回写(§3 D6)。
**偏差 B(承诺 #7)**:token 插入下 N_Q=3 全分辨率伪帧(4680 tokens)使序列 17,966→22,646(+26.0%),自注意力+FFN 口径估算 FLOPs +40%,且超过全部 history tokens(3,926)。默认改为 **mid-tier 分辨率伪帧(390 tokens/帧,N_Q=3 → M=1,170)**,全分辨率仅作 sweep 上界(§8)。
**偏差 C(承诺 #1 的细化)**:Helios 无 Echo 式 `context_noise=0` 的 commit 前向(`helios/pipelines/pipeline_helios.py:517-615` 循环后直接返回,OBSERVED ABSENCE)。写入源取**最后一次被调度的去噪前向**——其 t 由 `scheduler.timesteps`(`pipeline_helios.py:1323-1345`)或 DMD 稀疏 timesteps 决定(最后一步经 `convert_flow_pred_to_x0` 直接出 x0,不再有 t=0 调用,`pipeline_helios.py:577-595`),**不保证 t≈0**;可选的显式 t=0 commit 前向保留为少步蒸馏配置下的 ablation 开关(每 section +1 次 14B 前向)。
**偏差 D(新增,双代码路径)**:发布推理入口 `infer_helios.py:25-27` 实际 import `helios.diffusers_version` 下的独立 transformer/pipeline,而非训练侧模块;两条链路必须**双移植**(§7 第 6 条),否则旗舰蒸馏推理路径完全不经过 memory。

## 1. 参数清单(新文件 `helios/modules/helios_memory.py`,类 `HeliosMemoryEncoder`)

Helios-14B 实测维度(`transformer_helios.py:985-1002`):`num_layers=40`,`num_attention_heads=40`,`attention_head_dim=128`,`inner_dim D=5120`,`ffn_dim=13824`。480×832 下每 latent 帧 patch (1,2,2) 后 30×52=1560 tokens/帧。

- **memory queries**:`query_init: nn.Parameter [1, M, 5120]`,`M = N_Q × tokens_per_mem_frame`;默认 N_Q=3、每伪帧 390 tokens(15×26),M=1170。初始化 `Normal(0,1)×0.014`(对齐 Echo,`Echo-Infinity/model/query_memory.py:68,83`)。跨层共享单组(不移植 `num_query_groups`)。运行态 `query_state: [B,M,5120]` 以普通 Python 属性持有(不注册 buffer、不进 state_dict)。**[本设计新增决策,非 Echo 继承]** state 常驻 fp32、进 Enc/主模型前 cast bf16,gate 融合在 fp32 完成(`projected` 上转后再做 `g·old+(1−g)·projected`),以避免递归误差累积——Echo 的 reset 用调用方 dtype(`query_memory.py:110-117`;`model/streaming_training.py:63` 传模型 dtype),并未建立 fp32 惯例。
- **写入编码器 Enc**:2 层 `MemoryCrossAttentionLayer` 忠实移植(`query_memory.py:8-34`):RMSNorm → `q`+`norm_q` → cross-attn(40 heads×128)→ `o`+残差 → RMSNorm → FFN(GELU-tanh,ffn=4D=20480)+残差。与 Echo 的关键差异:Echo 的 Enc 直接消费缓存 K/V、无 K/V 投影(`query_memory.py:21-28` 只有 `self.q`);Helios 写入源是 hidden states `[B,E,5120]`,须新增 `ctx_k_proj / ctx_v_proj: Linear(5120,5120)`(reshape 至 `[B,E,40,128]`)。参数量:2×262.1M(层)+52.4M(ctx k/v)+52.4M(connector)+52.4M(gate)+6.0M(query_init@M=1170)≈ **687M ≈ 14B 的 4.9%**;对照 Echo 入库配置(`configs/echo_infinity.yaml:70-84`:hidden_dim=1536、2 层、M=3×1560、12 heads)按 `query_memory.py:69-95` 的定义逐项计数约 **68.6M ≈ 1.3B 的 5.3%**(原稿"61M/4.7%"无出处,已改),比例相近。Enc 的 AdamW fp32 m/v ≈8GB 须计入训练预算;ffn=2D 版本 ≈477M 备选。
- **写门 gate**:公式照搬 `query_memory.py:148-160`:`projected = connector(state)`(Linear→GELU→Linear→RMSNorm,`query_memory.py:85`);`g = sigmoid(Linear_{2D→D}([old;projected]))`;`new = g·old+(1−g)·projected`。**bias 重标定(启发式,非代码证明的逐帧衰减模型)**:Echo 的写粒度是每个 block/chunk 3 个 latent 帧——`num_frame_per_block` 注入注意力(`causal_model.py:555-559`)、streaming 循环按该尺寸分块(`pipeline/streaming_switch_training.py:26-29`)、更新由离开缓存窗口的帧驱动(`causal_model.py:797-828`)——bias=+2.0 → 每次 block 写保留 σ(2.0)=0.881。Helios 每 section(9 帧)一次写 ≈ 3 次 Echo block 写,按 0.881³≈0.684 → `bias = logit(0.684)≈0.77`,取 **`gate_init_bias = 0.75`**,并列入超参 sweep。
- **注入投影**:不新建 W_k/W_v(偏差 A);Echo 的输出 `to_k/to_v`(`query_memory.py:88-89`)无对应物,复用块内 `attn.to_k/to_v`(`transformer_helios.py:489-490`)。
- **memory 放大参数**:每 block 的 `attn1` 新增 `memory_key_scale: nn.Parameter(full(heads, -4.0))`,映射沿用 `1+sigmoid(raw)×9`(`transformer_helios.py:518-539`)。初始 raw=−4.0 → scale≈1.16(近中性),不同于 `history_key_scale` 的 raw=1→7.58:后者给已训练 history 加压,新 memory 从中性起步。加入 `_keep_in_fp32_modules`(`transformer_helios.py:954-961`,现含 `history_key_scale`)。
- **归属(修复 blocker)**:`HeliosMemoryEncoder` 注册为 transformer 的**正式子模块** `self.helios_memory`(§7 D9),不采用 Echo 的 `object.__setattr__` 外挂。原稿引用有误:`query_memory.py:106` 只是 reset 内的形状诊断信息;Echo 真实外挂在 `trainer/distillation.py:241-252`(`object.__setattr__` 挂到解包后的 inner model;初始化占位在 `causal_model.py:547-549`),配合独立 `encoder_optimizer`(`distillation.py:295-299`)与手工梯度 all-reduce hook(`distillation.py:233-237`)。Helios 的 dtype 转换(`train_helios.py:595-602`)、可训练选择(`train_helios.py:463-476`)、partial ckpt(`helios/utils/utils_base.py:157-262`)全部基于 `named_parameters/named_modules`,外挂参数会被全部遗漏。

**D1**(token 插入而非 KV 拼接):被否 = Echo-literal 跨层共享 KV 注入;理由:Helios 无跨层固定 KV 先例、t=0 AdaLN 无处施加、需重复建设投影。
**D2**(单组跨层共享 state):被否 = 每层/分组独立 state;理由:token 插入下逐层 K/V 已天然分化,分组只增参数显存。
**D3**(Enc 新增 ctx k/v 投影):被否 = 让 Enc 直接吃 14B 的 K/V 缓存;理由:Helios 无持久 KV 缓存,section 内 kv_cache(`transformer_helios.py:383-388`)是步间优化且默认关闭(`pipeline_helios.py:928`)。
**D4**(gate bias 0.75):被否 = 沿用 +2.0;理由:section 级稀疏写下记忆刷新过慢,长程一致性收益延迟。

## 2. 写入源最终方案

**tap 点**:`transformer_helios.py:1486`(block 循环结束)与 `:1488`(输出 AdaLN)之间:

```python
if capture_last_hidden:  # 仅最后调度步、cond 前向
    mem_tap = hidden_states[:, -original_context_length:, :].detach()
    # [B, T_cur*H_p*W_p, 5120];按当前 post-patch 维度 reshape 为 [B, T_cur, H_p*W_p, 5120]
```

即 GAN hooks 已用的切片模式(`transformer_helios.py:1485-1486`),取 X_Noisy 的末块原始 hidden state(AdaLN/proj_out 之前)。**形状按当前 `T,H,W` 参数化**(修复 major):token 数由 post-patch 维度动态推导(diffusers 侧 `transformer_helios_diffusers.py:681-696` 同理);480×832 全分辨率 9 帧时为 `[B,14040,5120] → [B,9,1560,5120]`(patchify 为 T-major flatten,`transformer_helios.py:1187-1189`)。
**触发(修复 blocker:两个 sampler、两条代码路径)**:
- stage1:训练侧 `stage1_sample`(`pipeline_helios.py:517-542`)与 diffusers 侧(`pipeline_helios_diffusers.py:513-564`)在最后调度步的 cond 前向传 `capture_last_hidden=True`;CFG 时只取 cond 支路。
- stage2(金字塔):`__call__` 在 `is_enable_stage2` 时只走 `stage2_sample`(训练侧 `pipeline_helios.py:1362-1364`/`617-820`;diffusers 侧 `pipeline_helios_diffusers.py:1266-1294`/`624-`),而发布蒸馏脚本正是该路径(`scripts/inference/helios-distilled_t2v.sh:14-15`,pyramid 2/2/2)。捕获条件定义为 **`i_s == pyramid_num_stages−1 且 idx == 该 stage 最后一步`**:最终 stage 已逐级上采样回全分辨率(`pipeline_helios_diffusers.py:653-666,695-715`),低分辨率 stage 一律不捕获(**D10**;被否 = 各 stage 都捕获再重采样,理由:需新增分辨率对齐投影,且低分辨率 hidden 与全分辨率写源分布不一致)。`is_amplify_first_chunk` 只倍增步数,不改变"最终 stage 最后一步"语义。
**返回契约(修复 major)**:训练侧 `forward` 在 `return_dict=False` 时现返回二元组 `(output, logits)`(`transformer_helios.py:1575-1578`),各调用点均 `return_dict=False` 且索引 `[0]`(`pipeline_helios.py:528-542,741-755`)。改为**恒定三元组 `(output, logits, last_hidden)`**(未捕获时 `last_hidden=None`):现有 `[0]` 调用点零改动,捕获调用点读 `[2]`;`Transformer2DModelOutput` 同步增字段。diffusers 侧 `(output,)`(`transformer_helios_diffusers.py:822-823`)改为 `(output, last_hidden)`。(**D11**;被否 = 模块属性暂存捕获张量,理由:与 torch.compile / context parallel 的交互不可控。)
**缓存生命周期**:pipeline 持有 `mem_frame_queue`(FIFO,元素 = 单帧 `[B,1560,5120]`+全局帧号);入队/消费次序与容量上界见 §6(consume-then-push,峰值 19 帧 ≈ **289.5 MiB bf16/样本**,可 CPU offload)。capture 即 `.detach()`(训练时梯度只经读路径回传;Echo 对缓存同样按 BPTT 边界 detach,`streaming_training.py:238-246`,语义一致)。
**被否方案**(与先验承诺一致):layer-0 patchify 嵌入(太浅,无语义压缩价值);去噪中 history token 的深层 hidden(被 joint bidirectional attention 污染,`transformer_helios.py:227-230,410-415` 无掩码);新增 t=0 commit 前向(每 section +1 次 14B 前向,保留为少步采样 ablation)。

## 3. 注入点与 is_amplify_history 扩展

**D5(物理布局)**:memory 作最外层前缀,canonical 布局 `[mem | long | mid | short | current]`。被否 = 插在 short 与 current 之间;理由:**非 NAViT 路径**的现有代码以"前缀 history、后缀 original"约定组织(`transformer_helios.py:213,408,866-869,1548`),最外层前缀改动面最小、输出裁剪(`:1548`)零改动。此说法不适用于 NAViT——NAViT 把 history 前缀按 packed 序列逐个复制(`:1385-1406`)并按 per-seq 起点切片(`:250-276`);**v1 在 forward 入口断言区(`:1290-1306`)即 `assert not (use_memory and enable_navit)` 拒绝**,避免进入任何 NAViT 前缀/mask/t0 路径。
**注入位置**:`process_input_hidden_states`(`transformer_helios.py:1133`)在 long 分支(`:1241-1259`)之后追加第四段 prepend:`hidden_states = cat([mem_tokens, hidden_states], 1)`、`rope_freqs = cat([mem_rope, rope_freqs], 1)`,返回新增 `memory_context_length=M`。
**长度记账(修复 major,统一约定)**:引入 `prefix_len = memory_context_length + history_seq_len`;`history_seq_len` 改为 `(total − original − mem_len) // len(list)`,只表示 latent history。三处必须同步使用 `prefix_len`:
1. **非限制模式 amp 三分**(`transformer_helios.py:394-408` 由二分改三分):

```python
if attn.is_amplify_history and history_seq_len > 0:
    k_mem  = key[:, :mem_len] * attn.get_scale_memory()      # per-head [1,1,40,1]
    k_hist = key[:, mem_len:prefix_len] * scale_key
    key = torch.cat([k_mem, k_hist, key[:, prefix_len:]], dim=1)
```

2. **限制模式(`restrict_self_attn=True`)**:processor 新增 `memory_context_length` 形参,cache 复用时按 `prefix_len` 丢弃前缀(现按 `history_seq_len`,`:224-225`)、首步前缀切分用 `prefix_len`(`:301-318`)、kv_cache 缓存含 memory 的完整前缀(`:383-388`)。若只改 `history_seq_len` 推导而不引入 `prefix_len`,memory 会漏进"current"Q 路径且不被缓存——原稿"cache 原样兼容"不成立,已修正。memory 在 section 内冻结,缓存前缀语义仍成立。
3. **guidance_cross_attn**(`transformer_helios.py:823-881`):该分支**自行**以比值公式重推 `history_seq_len`(`:824`)再按前缀切分(`:866-869`);必须同步传入 `memory_context_length` 并用 `prefix_len` 切分,否则在新记账下 memory/history 尾部会漏入文本 cross-attn——已发布训练配置均开启该开关(`scripts/training/configs/stage_1_init.yaml:178`)。排除 memory 的依据是现有行为:history 前缀本就整体不做文本 cross-attn(`:823-881`);"memory 是内容态非指令态"的语义理由降级为 INFERRED。(**D7**;被否 = memory 参与文本 cross-attn,理由:破坏现有 guidance 分支前缀语义,NAViT mask 公式改动面大。)
**可见性**:非限制模式无 mask、走全注意力(`attention_dispatch.py:132-149`),memory 的 Q/K/V 与全序列互见;这与 Echo 的只读 K/V 注入不同(偏差 A),"双方都无专门 tier 掩码"不构成机制等价。memory 的块输出被 `:1548` 后缀切片丢弃、**不回写** query_state(**D6**;被否 = 末层 memory hidden 回写状态;理由:引入去噪步内递归,与 Echo"仅经 Enc 更新"语义冲突且不稳定)。

## 4. RoPE 槽位设计

现行 id 布局(`helios/utils/utils_helios_base.py:655-666`;推理逐 section 重建 `pipeline_helios.py:1265-1273,1284-1292`):x0 前缀=0,long=1..16,mid=17..18,newest=19,current=20..28,窗口平稳(每 section 重置)。**D8:memory 用 (0,1) 开区间分数 id**,N_Q=3 → `{0.25,0.5,0.75}`,每 section 固定(满足承诺 #3"anchor 之后、long 之前")。**数值可行性(OBSERVED)**:`HeliosRotaryPosEmbed.forward` 将 `frame_indices` cast 为 float32 后直接与频率相乘(`transformer_helios.py:701,683-685`),分数 id 可执行。**建模有效性(未验证,列为必做 ablation)**:mid/long 的"非格点"频率是整数格点 RoPE 的平均池化(`:1233-1241,1253-1261`),cos/sin 的平均一般不等于分数 t* 处的单位模 RoPE,故不能据此认为模型已见过分数时间相位——原稿此论据撤回。ablation 备选:(a) 分数 id;(b) memory 复制 mid-tier 式池化频谱;(c) 共享 id=0。空间 id 按 mid-tier 惯例:`self.rope(mem_ids, height=30, width=52)` 后 `center_down_sample_3d(·,(1,2,2))` 得 15×26,使 memory 空间相位与 mid-tier token 同分布。被否方案:整体右移 id(破坏预训练已学的全部相对距离);负 id(语义上"比 anchor 更老"错误且完全未训练区)。

## 5. timestep 条件(t=0)

memory token 作为前缀成员获得 t=0 AdaLN,但**构造条件必须改写**(修复 major):现 t0 张量仅在 `indices_hidden_states is not None and self.zero_history_timestep` 时创建并拼接(`transformer_helios.py:1353-1366,1444-1446`)。改为:t0 构造条件 = `(indices_hidden_states is not None and zero_history_timestep) or memory_context_length > 0`,span 从 `history_context_length` 扩为 `history_context_length + memory_context_length`,拼接点公式不变;块内六路调制(`:792-806,809,896-900`)按位置自动对齐。**v1 约束**:`use_memory=True` 时要求 history 输入齐备(与 `:1290-1306` 的全有/全无断言合并),memory-only 前向显式 assert 拒绝——驱逐事件只在 history 模式存在,该约束无功能损失。NAViT 已在 §3 入口拒绝,不触及其 t0/mask 路径(`:1370-1406`)。这条 token 化的 t=0 路径正是纯 KV 注入做不到的(偏差 A 的核心论据)。

## 6. 驱逐语义与写入节奏

**定义(以 append 事件为基准,修复 major)**:section k(0 起)生成帧 `9k..9k+8`;生成前读取尾部 19 帧窗口 = 帧 `[9k−19, 9k−1]`(负数为零填充;`pipeline_helios.py:1295-1298`,初始零 history `:1180`;diffusers 侧 `:1215-1239`),生成后 append(`pipeline_helios.py:1453`;diffusers 侧 `:1327-1329`)。**append section k 导致离开窗口的帧组 = `[9k−19, 9k−11]`**:k=2 时 = `[−1,7]` → 8 个真实帧(0..7)+1 个 padding 槽(原稿"k=2 驱逐 9 真实帧"已改);k≥3 → 9 个真实帧 = section k−3 的最后 1 帧 + section k−2 的前 8 帧。section j 的最后一帧 `9j+8` 在 append section j+3 时离开,即**自 section j+4 的窗口起完全缺席**¹。被驱逐帧此前处于 long tier(26 tokens/帧的 4×8×8 压缩),正是 memory 要保全的信息。x0 前缀双重存在:持久 x0/image prefix 并非恒为物理帧 0——无输入图像时取首个生成 section 的第一帧、`is_skip_first_section` 时取第二个 section 的第一帧、I2V 时为输入图像 latent(`pipeline_helios.py:1447-1450,1280-1282`)——与 memory 内容部分重叠,可接受,记录之。
**写入节奏与队列次序(修复 major,consume-then-push)**:在 section k 结束、append 之后:(1) k≥2 时从队列 consume 刚离开的帧组(按全局帧号过滤 padding)→ `memory.update(evicted_hidden)`,`evicted_hidden = cat(组内各帧) → [B, ≤9×1560=14040, 5120]`;(2) push 本 section 捕获的 9 帧。更新对 section k+1 首步可见(单 section 延迟,类比 Echo 的单前向延迟)。**容量上界推导**:push 后队列恰为下一 section 窗口的 19 帧 → 峰值 = 19×1560×5120×2B ≈ **289.5 MiB bf16/样本**;若 push 后延迟到下一 section 开头才 consume,则瞬时峰值为 28 帧(≈427 MiB)²。首次真实写入发生在 section 2 结束(帧 0..7),首个带非平凡 memory 的生成是 section 3。训练:section 内 BPTT(update 参与图),跨 section `detach_state()`(移植 `query_memory.py:224-229`;Echo 按 `chunk_count % bptt_clips` detach,`streaming_training.py:214-217`,入库配置 `bptt_clips: 1`,`echo_infinity.yaml:85`)。**[本设计选择]** `bptt_sections=1` 为默认,{1,2} 进 sweep。

¹ 脚注(反驳验证者更正文):其"section j 在 section j+3 开始时完全缺席"差一——j+3 开始时窗口 = `[9j+8, 9j+26]` 仍含 section j 的末帧 `9j+8`(尾 19 切片 `pipeline_helios.py:1295-1298` + append `:1453`),完全缺席自 section j+4 始。
² 脚注(部分反驳):验证者称"本方案队列上界应为 19、28 无根据"——在原稿"下一 section 开始前才 consume"的次序下,push(k) 与 consume(k+1) 之间队列确实持有帧 `[9k−19..9k+8]` 共 28 帧,28 是该次序下的真实瞬时峰值;本版改为 consume-then-push 后上界取 19,两个数字各自成立于各自次序。验证者关于"缺推导、次序未定义"的批评成立,已补。

## 7. 修改清单(签名级)与训练路径

1. **新增 `helios/modules/helios_memory.py`**:`HeliosMemoryEncoder.__init__(dim=5120, num_heads=40, head_dim=128, n_layers=2, ffn_mult=4, n_mem_frames=3, mem_frame_hw=(15,26), gate_init_bias=0.75, initializer_range=0.014)`;方法 `reset(B, device)`、`update(evicted_hidden: Tensor[B,E,5120]) -> None`、`get_tokens() -> Tensor[B,M,5120]`、`detach_state()`。
2. **归属(D9,修复 blocker)**:注册为 `HeliosTransformer3DModel.__init__`(`transformer_helios.py:982-1022`)的正式子模块 `self.helios_memory`(受新增 config `use_helios_memory / memory_frames / is_amplify_memory` 门控)。被否 = Echo 式 `object.__setattr__` 外挂 + 独立 optimizer(`distillation.py:241-252,295-299`);理由:Helios 训练器全链路走 `named_parameters`,外挂即全部遗漏(§1)。注册带来的连锁须显式处理:(a) **PEFT**:`all-linear` 目标发现遍历 `named_modules` 会捕获 Enc 全部 Linear(`train_helios.py:378-384`)→ 把 `"helios_memory"` 加入 `LoraConfig.exclude_modules`(`train_helios.py:399-406`),Enc 走全参训练而非被 LoRA 包裹;(b) **冻结-解冻**:全局 `requires_grad_(False)`(`train_helios.py:357-362`)之后,`trainable_modules` 字符串清单(`train_helios.py:463-476`)增 `"helios_memory"` 与 `"memory_key_scale"`;(c) **dtype**:`named_parameters` 转换循环(`train_helios.py:595-602`)自然覆盖,`memory_key_scale` 入 `_keep_in_fp32_modules`(`transformer_helios.py:954-961`);(d) **优化器**:参数经 requires_grad 进入现有 optimizer,单列 param group lr×5(对齐 Echo `encoder_lr_multiplier: 5.0`,`echo_infinity.yaml:85`);(e) **ckpt**:`save_extra_components / load_extra_components`(`utils_base.py:157-262,266-363`,现只枚举 patch 模块、restrict LoRA、history scale、GAN)增第 5 节 `helios_memory.*` 与 `memory_key_scale`(gated by 新增 `is_train_helios_memory`),`save/load_model_hook`(`train_helios.py:649-760`)同步;`query_state` 为普通属性不入 state_dict;DeepSpeed full-finetune 路径因子模块注册天然被 engine 覆盖。
3. `process_input_hidden_states(..., memory_tokens=None, memory_rope=None)`(`:1133`):最外层 prepend,返回值追加 `memory_context_length`。
4. `HeliosTransformer3DModel.forward(..., memory_tokens=None, capture_last_hidden=False)`(`:1271`):断言按 §5 约束放宽/收紧、NAViT×memory 入口拒绝、t0 条件改写(§5)、`:1486` 后条件捕获、**恒定三元组返回**(§2)。
5. `HeliosTransformerBlock.forward` / `HeliosAttention.forward` / `HeliosAttnProcessor.__call__` 增 `memory_context_length=0`,按 `prefix_len` 三分与缓存(§3);`HeliosAttention.__init__`(`:462`)增 `memory_key_scale` 与 `get_scale_memory()`(复制 `:518-539` 模式,init −4.0)。
6. **双路径移植(D14,修复 blocker)**:训练/验证链 = `helios/modules/transformer_helios.py` + `helios/pipelines/pipeline_helios.py`;**发布推理链** = `infer_helios.py:25-27` 实际使用的 `helios/diffusers_version/transformer_helios_diffusers.py`(processor `:100-157` 为无限制模式简化版、forward `:661-825`)与 `pipeline_helios_diffusers.py`(chunk 循环 `:1184-1329`、stage1 `:513`、stage2 `:624`)。同一 memory API/状态机镜像移植,并复验 `enable_compile` 与 context parallel。里程碑:M1 训练链跑通即可训练与验证,M2 完成镜像移植后才能用发布脚本评测。被否 = 把 `infer_helios.py` 入口改指训练模块;理由:放弃 diffusers 侧 offload/CP/compile 生态,成本更高。
7. **训练路径(修复 blocker:现有 flow 训练无法训练演化 memory)**:OBSERVED 现状——`_flow_loss` 对预置静态 history 单次调用 transformer(`helios/utils/utils_helios_base.py:20-61`),dataloader 每样本一个目标 chunk + 19 帧 history(`helios/dataset/dataloader_history_latents_dist.py:136-196`),无 section 循环、无驱逐事件,`prepare_stage1_*`(`utils_helios_base.py:627-741,744-`)只是张量整形——仅扩签名不可能获得 BPTT/写语义。新增:
   - **dataloader 变体**:每样本供给 S≥4 个连续 section 的 GT latents(S=4 恰好覆盖首次真实驱逐 + ≥1 个带非平凡 memory 的 loss section)。
   - **`_flow_loss_section_unroll`**(新函数;train_helios.py 新分支持有状态机):

```python
memory.reset(B); queue.clear()
for s in range(S):                          # teacher-forcing 展开(Stage A/B)
    hist = roll_gt_history(s)               # GT latents 滚动构造 [16,2,1] 窗口
    out = transformer(noisy(gt[s], t~rand), memory_tokens=memory.get_tokens(), ...)
    loss += flow_loss(out[0], target_s)     # 读路径梯度入口
    with torch.no_grad():                   # TF 下无“最后去噪步”,须显式补一次小 t 捕获前向
        _, _, tap = transformer(noisy(gt[s], t_min), ..., capture_last_hidden=True)
    if s >= 2: memory.update(queue.consume_leaving(s))  # 梯度进 Enc 权重/state;写源已 detach
    queue.push(tap.view(B, 9, 1560, 5120))
    if (s + 1) % bptt_sections == 0: memory.detach_state()
```

   梯度语义:写源 detach 与 Echo 一致(Echo 亦按 BPTT 边界 detach 缓存 K/V,`streaming_training.py:238-246`);Enc 的训练信号来自**后续 section 的读路径 loss → memory_tokens → update 链**,不依赖写源梯度。TF 阶段每 section +1 次 no-grad 捕获前向是写语义的固有成本(承诺 #5 Stage A/B)。Stage C(self-forcing/DMD)复用现有 `is_train_dmd` rollout,捕获条件与推理一致(最后 DMD 步,`pipeline_helios.py:577-595`)。
8. pipeline(两侧):section 循环持有 encoder 状态机与 frame queue;`stage1_sample`/`stage2_sample` 按 §2 传 `capture_last_hidden`;推理伪代码 = §6 次序(生成→append→consume→push→get_tokens 供下一 section)。

## 8. N_Q 规模分析

基准序列 17,966 tokens(long 416 + mid 390 + short 3,120 + current 14,040,已按 patch 尺寸核算;为 480×832、9 帧全分辨率情形,stage2 低分辨率 stage 按 §2 不参与 memory 记账)。以下比例为**自注意力+FFN+QKVO 投影口径**(公式:ΔF = M×每 token 线性成本 + 二次注意力项增量;不含每 block 必然执行的文本 cross-attn(`transformer_helios.py:882-893`)与 Enc,故非"总 FLOPs"):
- **默认 N_Q=3 @ mid-res**,M=1170;序列 19,136(+6.5%);该口径 FLOPs ≈ **+9.5%**,计入文本 cross-attn 后 ≈ **+8.6%**;memory tokens 激活显存 ≈11.4MB/前向,query_state fp32 常驻 ≈24MB。
- **上界 N_Q=3 @ full-res**,M=4680;序列 22,646(+26.0%);≈ **+40%**(计入 cross-attn ≈ +36%),且超过全部 history tokens(3,926)——仅作容量上限实验。
- **sweep 集合**:M ∈ {390, 1170, 2340}(mid-res 1/3/6 伪帧),替代承诺 #7 字面的全分辨率 {3,6}。
- **Enc 写开销(量化,修复 major,替代原稿"可忽略")**:每次 section 更新 ≈ **3.6 TFLOPs**(E=14040、M=1170:ctx K/V 投影 2×2ED²≈1.47T;每层 cross-attn score+value 4MED≈0.34T + FFN≈0.49T + q/o≈0.12T,两层≈1.9T;connector+gate≈0.25T)。对照单次全分辨率 40-block 前向 ≈0.7 PFLOPs(线性≈0.43P + 自注意力≈0.26P),Enc ≈ 其 **0.5%**,且每 section 仅一次,摊到 ≥4 次去噪前向后 <0.15%。训练侧主开销是 Enc 优化器态(≈8–10GB,ffn_mult=2 降至 ≈6GB)与瞬态帧队列(≤289.5 MiB bf16/样本,可 offload)。

**关键 ABSENT 重申**(设计前提):Helios 现无任何 memory 参数、写门、Enc、memory RoPE 约定、三段长度记账、末层 hidden 返回 API(训练侧 `transformer_helios.py:1290-1306,1353-1366,1575-1578` 与 diffusers 侧 `transformer_helios_diffusers.py:822-823` 均为二段/二元结构),也无任何 section 级 rollout 训练路径(`utils_helios_base.py:20-61` 为单前向)——以上全部为新建,无可"顺路启用"的潜伏机制。

---

# 第二章 训练课程与 Trainer 改造


修订说明：已逐条复核验证器报告并实读全部引用代码；blocker/major 均已通过修改设计解决。验证器唯一被驳回的子论断见脚注 [^1]。

## 0. 代码核验结论与先验偏差标记

以下与先验/上游报告不一致处，以代码为准：

- **偏差 F1（先验 #5）**：Echo **不存在**"静态 query 先行"或 teacher-forcing 预阶段。证据是配置与入口：两份已发布训练配置均从第一步启用 memory（`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:70-88`、`configs/echo_infinity-long.yaml:88-125`），README 只有两个 DMD 阶段（`README.md:123-143`）。另注意：`query_init` 在首次驱逐写入前是**潜伏**的——`get_kv()` 在 `has_history` 置位前返回 `None`（`model/query_memory.py:194-196`；置位在 :190），注意力注入也随之跳过（`wan/modules/causal_model.py:726-739`）。我们的 Stage A 是 Helios 数据通路强加的**新增**阶段，不是移植 Echo 课程。
- **偏差 F2**：`dmd_teacher_forcing` 是**损失侧死旗标但数据侧活跃**：它仅控制 dataset 的 `return_all_vae_latent`（`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:882-885`），dataloader 产出 `clean_all_latents`（`helios/dataset/dataloader_history_latents_dist.py:365-379`）但 trainer 热循环无任何消费者（:1343-1384 只取 prompt/history/target/x0 后 del batch）。它仍会改变切片、可 raise `Not enough sections`（dataloader:180-182）并增加 I/O。**不得复用该旗标名**承载新语义。
- **偏差 F3（修正表述）**：Python `random` 已被 `set_seed(args.seed)` 全局播种（`train_helios.py:230-232`），"未播种"说法不成立；真实缺陷是 `start_section_idx` 用全局 `random.randint`（dataloader:183）采样，与逐样本确定性的 `choice_idx`（seeded generator，:169-172）**解耦**，且输出 dict 不含 `start_section_idx`（:365-384），而 `prompt_embeds` 只绑定 `choice_idx` 的事件（:377）。修复必须是**返回采样到的起始索引并据其派生逐 section prompt**（见 D6），不是仅换 RNG 实现。
- **偏差 F4（修正表述）**：`use_deepspeed_optimizer` 仅当 DS json 含 `optimizer` 块才为真（`train_helios.py:820-823`），而本谱系使用的 `scripts/accelerate_configs/zero2.json` 无 optimizer/scheduler 块（全文仅 fp16/bf16/zero_optimization）——即使 DMD 双 DeepSpeed plugin（:154-165，Stage C 父配置 :230-231 指定 zero2.json）下，`torch.optim.AdamW` 也照常收到 param groups。真正的两个杀手是：(a) 含 optimizer/scheduler 块的 DS 配置（`get_optimizer` DummyOptim 只传全局 lr，`helios/utils/utils_base.py:76-87`）；(b) **`HELIOS_FORCE_LR=1` resume 覆写**把所有 param group 的 lr 与 scheduler `base_lrs` 压成单一标量（`train_helios.py:1086-1099`）——而所选 Stage A/B 祖先配置**明文依赖** FORCE_LR（`scripts/training/configs/stage1_lora_cfr_368_correct.yaml:95-97, 105`）。守卫见 D11/D13。
- 其余核验通过的关键承载点：可训练模块子串匹配 `train_helios.py:463-477`；optimizer 单 param-group :815-829；save/load hook :648-695/:696-768；`save_extra_components` 白名单 `helios/utils/utils_base.py:157-259`；Stage-3 rollout 段数采样与广播 :1571-1598、`USE_GT_HIST` 按比率抽签 :1561-1564；GT 模式单 section 断言 `helios/utils/utils_helios_post.py:2349`（generator）与 :2966（critic）；Echo gate 更新式 `query_memory.py:152-155`、`bptt_clips` 逐 chunk detach `streaming_training.py:212-217`。

## 1. 三阶段课程 → 配置体系映射

**D1 配置谱系分叉**：Stage A/B 从 `scripts/training/configs/stage1_lora_cfr_368_correct.yaml` 分叉（继承：r128 LoRA :44-51；LR/`constant_with_warmup`/步数 :95-107；I2V-drop 0.1 :142-146；corrupt random 0.8889 :156-166；saturation :167-171），命名 `stage1_lora_mem368_A.yaml` / `..._B.yaml`；Stage C 从 `stage_3_post_self-forcing_version.yaml` 分叉为 `stage3_mem_sf.yaml`。*弃选：从 `stage_1_init.yaml` 分叉——面向裸 Wan2.1 初始化且缺 correct 三项修正，与已训 lora368 检查点谱系断裂。*

**D2 新增 flags 与"单一完整构造字典"原则**：所有**影响参数形状的字段**（不止一个 bool）必须进 transformer 构造 kwargs 并注册进模型 config，且**同一份完整字典**流经全部重建路径：generator kwargs（`train_helios.py:310-325`）、DS resume 重建（:718-730）、EMA 重建（`helios/utils/create_ema_zero3_lora.py:37-44, 116-124`）、merge 工具（D12）、推理构造（D12）。critic 是独立 kwargs 字典（:334-347）——按 D8 决策 critic **不构造** memory 模块。另：`from_pretrained` 会**静默跳过**尺寸不匹配的键（`helios/modules/transformer_helios.py:1767-1776`），故加载后必须显式断言所有 `evolving_memory.*` 键已加载，禁止依赖宽容加载吞掉 N_Q sweep 的形状差异。

```yaml
# ---- Evolving Memory (arch-affecting keys go into transformer config) ----
is_enable_evolving_memory: false
memory_num_query_frames: 3        # N_Q; M = N_Q*920 tokens (fixed, full-res token density)
memory_enc_num_layers: 2
memory_gate_init_bias: 0.75       # §7, INFERRED heuristic; sweep {0.75, 2.0}
is_amplify_memory: false          # memory 自有 amp 参数（模块设计任务）
# ---- training-only ----
is_train_memory_module: false     # requires_grad 开关
memory_freeze_backbone: false     # Stage A: 冻结 LoRA/patch_*，只训 memory（D10）
memory_learning_rate: 5.0e-5
memory_bptt_sections: 1
memory_write_source: "t0_hidden"  # {t0_hidden, none}
memory_single_write_prob: 0.0
memory_tf_unroll: false
memory_unroll_sections: 1         # U; 传入 dataset num_rollout_sections（D6）
```

**D3 各阶段训练对象 / LR / 步数**：
- **Stage A**（TF+静态 query，含单写模拟）：只训 `evolving_memory.*`，backbone LoRA 与 patch_* 冻结（经 `memory_freeze_backbone`，D10）；memory LR `5e-5`、约 4k 步、gb 32（1 节点）。*弃选：Echo 式从头联合训练——lora368 已在无记忆分布上收敛，联合训练会让 LoRA 为未成熟记忆读数买单。*
- **Stage B**（TF unroll）：联合训练，LoRA `1e-5` + memory `5e-5`，U=4，`memory_bptt_sections=1`，约 3k 步、gb 32-64。
- **Stage C**（SF/DMD 校准）：generator LoRA `1e-5`、memory 降为 `2e-5`；critic **沿用父配置显式值 `critic_learning_rate: 4.0e-7`**（`stage_3_post_self-forcing_version.yaml:232`；dataclass 默认 2e-6 被父配置覆盖，不得误标为"沿用默认"）；约 2k 生成器步（`dfake_gen_update_ratio: 5`，yaml:233，即 10k trainer 迭代）。

## 2. Stage A：GT history 下的驱逐模拟与写入路径

**D4 驱逐帧的精确定义（已按 19 帧零前缀修正）**：时间线 `continue_source_latent = [19 zeros] + real frames`（dataloader:148-156）；目标 `choice_idx=k` 的 history 为 `[9k, 9k+19)`（:188-191）。由 k-1 推进到 k 时滚出的 9 帧是 `continue[9(k-1), 9k)`——即 k-1 段 history 的**最老 9 帧**，且因 19=2×9+1 的前缀偏移**永不与 chunk 边界对齐**。逐 k 情形（与 `docs/STAGE1_DATALOADER_FRAMES.md:92-103, 204-211` 的推演一致）：k=1 全零、**k=2 仍全零**（[9,18) 落在零区）、k=3 混合（1 零 + 8 实帧）、k≥4 全实。因此：(a) dataset 增加 `return_evicted_latent: true`，返回 `evicted_latent = continue[9(k-1) : 9k)`（9 帧）、其前置 19 帧切片（k=3 时左端越界 1 帧，左侧补零）、以及整型 `evicted_valid_frames = min(9, max(0, 9k-19))`；(b) 写入判定用 `evicted_valid_frames`：==0（k≤2）跳过写；1-8（仅 k=3）做**带 frame mask 的部分写**（Enc 只消费有效帧 token）；==9 全写。**禁止**硬编码 `choice_idx<=1` 边界。无需改 `tools/offload_data`。*弃选：离线预算写入特征——写源是模型 hidden state，随权重演化，离线缓存必然陈旧。弃选：只允许完整写（要求 k≥4）——白白丢弃 k=3 的首次写监督，且推理期首次驱逐同样是错位的。*

**D5 写源前向（需新增模型 API）**：TF 下没有去噪循环。以驱逐 9 帧为 X_Noisy、timestep=0、携带其前置 19 帧 history 做**一次 `no_grad` 干净前向**，取**最后一层、`norm_out`/`proj_out` 之前的目标 token hidden**。现有 forward 在 :1548 切掉 history token、:1550 投影后只返回 `(output, logits)`（`transformer_helios.py:1575-1578`）——**必须新增 `return_final_hidden=True` 接口**（在 :1548 处捕获 `hidden_states[:, -original_context_length:, :]` 返回），这是显式实现任务，不是现成集成点。形状：368×640 全分辨率桶下为 `[B, 8280, 5120]`（8280 = 9 帧 × 920 tokens/帧，920 = 23×40，`docs/STAGE1_DATALOADER_FRAMES.md:184-185`）[^1]；但 `single_res` 同时保留全/半/四分之一三种分辨率桶（dataloader:107-113），故 **Enc 必须接受可变长写源序列**，H/W 按桶推导；memory 状态 M = N_Q×920 为固定 token 数（N_Q=3 → 2760），与桶无关。梯度只经"写→状态→读→flow loss"进入 Enc/gate。此处与 Echo 的关系是**梯度阻断上的类比而非机制同构**：Echo 的 recache 前向确在 no_grad 下（`pipeline/self_forcing_training.py:171-176`），但其 update 消费的是持久 KV cache 中被驱逐的末 block K/V（`causal_model.py:822-828`），并非返回的 hidden——Helios 的 last-hidden→Enc 写路径是**新设计**。以 `memory_single_write_prob≈0.5` 混合"纯静态读"与"单写后读"样本。*弃选：patchify/layer-0 嵌入作写源——太浅（先验 #1）；弃选：当前 step 噪声前向中 history token 的深层 hidden——联合双向注意力污染（先验 #1）。*

## 3. Stage B：trainer 内多 section TF 展开

**D6 数据改造**：`memory_tf_unroll=true` 时：(a) trainer 向 dataset 构造传 `num_rollout_sections=memory_unroll_sections`——当前 `dataset_kwargs` 完全不传该参（`train_helios.py:875-890`），构造默认 3（dataloader:20），U=4 需要的 all-latent 切片为 19+9U=55 帧（:184-186），默认值下只有 46 帧、必然错配；(b) 输出 dict 增加 `start_section_idx` 与逐 section `prompt_embeds` 列表（复用 `_pick_prompt_embed(feature_data, start_section_idx+u)`，:282-303）；(c) 起始索引改用与 `choice_idx` 同纪律的 seeded generator（:169-172），并同时返回之（修复 F3 的对齐缺陷）；(d) 索引期过滤 `total_sections < U` 的样本，且 metadata cache（每 folder 的 `dataset_cache.pkl`，:52-64）**按 U 失效重建**；(e) 单元测试 `clean_all_latents` 形状恰为 `[B,16,19+9U,H,W]`。U=4 需 ≥4×33=132 可用 RGB 帧（离线编码器按 33 帧/chunk 切，`tools/offload_data/get_short-latents.py:214-232`），但 metadata 只保证 `num_frame>=121`（dataloader:102-104），**语料保留率必须先跑离线直方图扫描再做预算**（"绝大多数满足"不可从代码验证）。

**D7 展开循环、backward 重构与 detach 语义（三处修正）**：

```python
# NOTE: _flow_loss currently backwards internally (utils_helios_base.py:107).
# Refactor: add a section-level primitive that RETURNS the scalar loss
# (no internal backward); the outer accelerator.accumulate scope
# (train_helios.py:1484-1486) keeps ownership of sync boundaries.
state = mem.init_state(B)                       # from query_init, per-sample
for u in range(U):                              # forward in time
    hist = slice_tiers(clean_all, u)            # 16/2/1 + anchor
    hist = corrupt_history_latents(hist)        # existing anti-drift
    sync_this = accelerator.sync_gradients and (u == U - 1)
    ctx = nullcontext() if sync_this else accelerator.no_sync(transformer)
    with ctx:
        loss_u = flow_loss_section(..., memory_state=state) / U
        accelerator.backward(loss_u)            # frees activations per section
    # BPTT truncation happens BEFORE the next write, AFTER this read's
    # backward, so update(u) stays in graph until read(u+1) consumes it.
    if (u + 1) % memory_bptt_sections == 0:
        state = state.detach()
    if u + 1 < U:
        ev, n_valid = evicted(u + 1)            # slice [9j, 9j+9), D4 semantics
        if n_valid > 0:
            with torch.no_grad():
                h = transformer(ev, t=0, ..., return_final_hidden=True)
            state = mem.update(state, h, frame_mask(n_valid))  # grad kept
optimizer.step()
```

- **detach 次序（关键修正）**：detach 施加在**下一次写之前、本次读的 backward 之后**。若照初稿"update 后立即 detach"，`bptt_sections=1` 时 read(u+1) 只见 detach 后的状态，Enc/gate **永远收不到梯度**（写路径无学习信号）。修正后 `bptt_sections=1` 精确对应"Enc 梯度仅来自紧邻下一 section 的读"，语义对齐 Echo 的 `detach_state()`（`streaming_training.py:212-217`）。**验收：U=2、bptt=1 下单测 Enc/gate 梯度非零。**
- **backward/DDP 语义与显存均标记为实现假设**：现 trainer 无内嵌 `no_sync` 多次 backward 先例；祖先配置 `gradient_accumulation_steps: 2`（yaml:95-97）意味着内层 sync 判定必须与外层 `accelerator.accumulate` 的 `sync_gradients` 合取（如伪码）。显存推断锚点是 bs4 单 section ~90 GiB（`docs/STAGE1_SPEED_OPTIMIZATION.md:83-98`），bs2×U=4 在 H200 141GB 可行是**未测假设**——落地前先跑 1 节点 smoke profile。*弃选：U 段 loss 求和后一次 backward——激活 ×U 存留，必 OOM。*
- **DDP unused params**：写路径条件触发，保持 `HELIOS_DDP_FIND_UNUSED` 默认 true，或按 zero-touch 技巧（`utils_helios_base.py:100-105`）补零梯度。

## 4. Stage C：SF/DMD 末端校准

**D8 集成点（按实际控制流重写）**：入口是 `consistency_backward_simulation`（`helios/utils/utils_helios_post.py:1403-1498`），在 `is_enable_stage2=true`（父配置 :206）时分派到 `inference_with_trajectory_stage2`（:947-1400）；Stage-1 分支是死代码（:679 无条件 `raise NotImplementedError`）。**本设计将 Stage C 显式绑定 Stage-2 金字塔路径，不新开 Stage-1 rollout 实现任务。**具体机制：

1. **写源不能是"被选中的 exit 前向"**：exit timestep 是随机采样的 `denoising_step_list[init_exit_flag]`（每个金字塔 stage 各有独立 flag，:991-995、:1257），并非 t≈0；且所有 transformer 调用只取 `[0]`（flow pred，:1264-1324），不存在 hidden 返回。**改为**：在最终金字塔 stage 的 `pred_x0` 拼入 `history_latents`（:1350）之后，对该全分辨率 `pred_x0` 追加**一次 `no_grad` timestep=0 recache 前向**（携带该 section 当时的三层 history，`return_final_hidden=True`），输出喂 Enc——与 Stage A/B 写源分布一致（t=0、干净输入）。代价：每 section +1 前向。
2. **梯度结构**：前缀 section（`should_compute_grad=False`，:1160；exit 前向在 `torch.set_grad_enabled` 下随之关闭，:1300）的 `mem.update` 也置于 `no_grad` 且状态保持 detach；梯度后缀 section 的 update 留在图内，按 `memory_bptt_sections` 以 D7 的"写前 detach"次序截断。读侧梯度来自后缀 section 带 grad 的 exit 前向：DMD 伪 loss 经 `original_latent` 反传（:2044-2053）→ transformer → memory K/V → update 图 → Enc/gate。**验收：两 section 后缀下单测 Enc/gate 梯度非零。**
3. **状态所有权（修正为显式局部对象）**：memory state 作为 rollout 函数的**显式入参/返回值张量**，严禁持久 module 属性——generator-loss rollout（`train_helios.py:1653-1750`）与 critic-loss rollout（:1842-1916，其内部 generator unroll 全程 `torch.no_grad`，`utils_helios_post.py:2978-3023`）各自从 `query_init` 独立 reset，互不污染。`dmd_is_low_vram_mode: true`（yaml:222）下 state 张量随 generator 的设备搬移同步 `.to()`，加回归测试。
4. **score 模型保持 memory-blind（显式设计决策）**：DMD real/fake score 前向在非 GT 模式下 history 全传 `None`（:1657-1679；critic 只给裁剪出的生成片段打分，:3124-3136），且 `real_score_model` 是独立构造的冻结模型（`train_helios.py:330-353`）——它从未见过 memory 条件，强喂 memory K/V 反而破坏 real score 分布。第一期 critic/real score **均不建 memory 模块**；DMD 梯度仍会校准"memory 条件下的生成分布"。若后期 fake score 欠拟合再评估"冻结快照喂 critic"方案。*弃选：critic 同步维护 state——需 critic 侧构造模块与设备/EMA 全链路配套，现无证据必要。*
5. **critic LoRA 目标排除（新增必做）**：critic 的 target 列表复制自 generator all-linear 扫描、仅排除四个 patch 名（:517-527）；generator 的 all-linear 扫描本身发生在模块构造之后（:378-393），会把 memory 的 Linear 一并收进目标。两侧都必须过滤 `"evolving_memory" not in name` 并加入 `exclude_modules`，注入后断言不存在 `evolving_memory.*lora_*` 参数（D10/D12 的直拷语义依赖此点）。
6. **配置**：沿 `stage_3_post_self-forcing_version.yaml`，初期 `dmd_num_latent_sections_max` 44→**8**（:290-291）、`is_use_gan: false`（父配置 true，:256）。保留 `is_dmd_vae_decode: true`（:251）：其 decode 只对随机首/尾 21 帧做直通估计重建（:2424-2485）——因 score 是 memory-blind（第 4 点），无需为被评分裁剪窗维护 memory 快照。*弃选：stock `stage_3_post.yaml`（GT-history）——GT 路径被 `assert num_rollout_sections == 1` 锁死（`utils_helios_post.py:2349`、:2966），递归写入根本不被执行。*

**D9 为何 Stage C 不可省**：(a) gate 决定写幅度，其输入在 TF 下只见过 GT 统计的 t=0 hidden；推理时写源是自生成漂移分布，误差经 `g·s+(1-g)·p` 逐 section 复利——正是 Echo 批评的手写规则等价物（先验 #5）；(b) 本仓已有同类实证：I2V 训练/推理格式错配曾致真实退化（`docs/HELIOS_CORRECT_RECIPE.md:11-31`）；(c) Echo 的记忆写入在训练中始终消费自生成上下文（`causal_model.py:785-828`）——省略 Stage C 意味着移植版比原版少一个其第一步就具备的性质。

## 5. 参数组、LoRA 并行训练与持久化

**D10 模块归属与 Stage A 冻结（修正）**：`evolving_memory` 注册为 `HeliosTransformer3DModel` 顶层子模块（构造 kwargs 见 D2）。两处必改：(i) **LoRA 排除**——all-linear 目标发现会捕获 memory Linear（:378-393），按 D8-5 过滤+exclude；(ii) **Stage A 显式冻结通道**——`add_adapter` 先把 LoRA adapter 参数置为可训练（:456），其后的子串循环**只开不关**（:463-477），单靠 `trainable_modules.append("evolving_memory")` 实现不了"只训 memory"。新增：`memory_freeze_backbone=true` 时在 adapter 注入后把所有 `lora_` 与 patch_* 参数 `requires_grad=False`，再解冻 `evolving_memory`；optimizer 构造前**断言可训练参数名集合**恰为预期前缀集并落日志。*弃选：Echo 式 FSDP 外挂独立模块+手动 all-reduce（Echo `distillation.py:209-250`）——违反本 trainer 单模型 save-hook 类型断言（`train_helios.py:667-668`）与 EMA 位置序假设。*

**D11 optimizer param groups（修正）**：按**参数名**构组、名集互斥断言、**空组不构造**：

```python
mem_names = {n for n, p in transformer.named_parameters()
             if "evolving_memory" in n and p.requires_grad}
base = [p for n, p in transformer.named_parameters()
        if p.requires_grad and n not in mem_names]
mem = [p for n, p in transformer.named_parameters() if n in mem_names]
params_to_optimize = []
if base: params_to_optimize.append({"params": base, "lr": lr, "role": "base"})
if mem:  params_to_optimize.append({"params": mem, "lr": memory_lr, "role": "memory"})
```

（现源列表本就是"所有 requires_grad 参数"而非仅 LoRA，:815-818，命名构组同样覆盖。）配套：(a) 日志按 role 记录每组 LR——现 `_flow_loss` 只记 `get_last_lr()[0]`（`utils_helios_base.py:152-155`），Stage A 下组 0 就是 memory 组，索引假设必须去掉；(b) 守卫收窄（替代初稿"禁 full-FT"）：设定 `memory_learning_rate` ⇒ 运行时断言 `not use_deepspeed_optimizer and not use_deepspeed_scheduler`（:820-827）**且** `HELIOS_FORCE_LR != "1"`（:1086-1099 会压平组间比例；若必须 FORCE_LR，需先扩展该覆写为按 role 恢复各组 LR）；(c) `accelerator.prepare` 后冒烟断言两组 lr 不同。

**D12 持久化/推理/合并/EMA（从"自动覆盖"改为显式清单）**：`save_extra_components` 增第 5 节（`utils_base.py:259` `torch.save` 前）收取前缀 `evolving_memory.`，gate 条件用 **enable** 而非 train（Stage C 冻结变体也持久化）；`load_extra_components`（:260+）对称加载，且**当 checkpoint 含 memory 键而模块未构造时必须 fail loudly**。但仅此**不足**，以下为必做清单：
- **推理**：`infer_helios.py` 在 `from_pretrained` 时不传任何 additional kwargs（:238-242），`partial_path` 只人造四个既有 flag（:283-292）——需新增 CLI/配置项并**在构造期**传 memory kwargs；`pipeline_helios.py` 的整个 chunk 循环（:1205-1454）没有任何 state 传递/重置/写入调用——**推理侧状态机是独立实现任务**（每次生成 reset、去噪后每 section 恰好一次写、state 进每次 forward），归属推理设计文档。
- **merge 工具**：`tools/merge_lora_partial_for_helios.py` 构造 kwargs 硬编码无 memory 项（:58-74），`load_extra_components` 的 Namespace 同样缺 flag（:126-137）——两处补齐后，`evolving_memory.*` 直拷（全秩、无 LoRA 算术）才成立，且依赖 D8-5 的 LoRA 排除（否则 memory 的 adapter 键会被当作 base Linear delta 合并，:78-124）。
- **EMA**：覆盖**以构造同构为条件**，非"天然"。EMA `step` 是位置 zip 且长度不匹配时**静默截断**（`helios/utils/create_ema_zero3.py:245-250`）；resume/export 均从 `transformer_additional_kwargs` 重建模型（`create_ema_zero3_lora.py:37-44, 116-124`）。必做：EMA 初始化/resume 前比对双方有序 `(name, shape)` 列表、不匹配即 fail；memory kwargs 完整传入；模块设计约定**不引入需持久化的 buffer**（EMA 导出只收 named_parameters，:96-105）。deep-copy 起点（`train_helios.py:509-513`）在同构前提下成立。
- **DS resume**：从 kwargs 重建 generator/critic（:718-730），使用 D2 的同一份字典。

## 6. main() 断言与配置校验

**D13 断言块新增**（插入 :2656 "For Wan" 节附近）：`is_train_memory_module ⇒ is_enable_evolving_memory`；`is_enable_evolving_memory ⇒ is_enable_stage1 and has_multi_term_memory_patch`；`memory_tf_unroll ⇒ use_stage1_dataset and NOT use_stage3_dataset and memory_unroll_sections>=2 and not is_train_dmd and not use_error_recycling`——dataset 分派优先检查 `use_stage3_dataset`（`train_helios.py:112-129`），双真时仅断言 stage1 会静默放行错误数据路径，故**并加 dataset 选择 one-hot 断言**；`is_train_dmd and is_enable_evolving_memory ⇒ is_enable_stage2 and not is_use_gt_history`（Stage-1 rollout 死代码 + GT 单 section 断言）；`memory_bptt_sections>=1`；D11 的 DS/FORCE_LR 守卫；`validation_config.use_kv_cache ⇒ not is_enable_evolving_memory`（记忆 KV 第三组与 kv-cache 路径未验证兼容前封死）。**配置校验流程（修正表述）**：`compare_yaml.py` 的递归 diff 算法零改动（:39-58），但更新 `__main__` 默认对（:62-66，路径相对 `scripts/training`）本身就是代码改动，且**一次 A/B diff 不能保证全谱系显式含新 flag**——校验程序改为：枚举谱系内全部 YAML，逐一 OmegaConf merge 到 `Args` schema 并断言 memory 键显式存在；A/B diff 仅作人工复核辅助。

## 7. gate bias 定量换算（INFERRED 启发式，非代码推导的时间常数匹配）

名义换算：Echo causal block 3 帧（`echo_infinity.yaml:43-57`），单次保留率 `σ(2)=0.8808` → 每帧对数保留 `ln(0.8808)/3=−0.0423`；Helios 每写覆盖 9 帧 → `r=e^{−0.0423×9}=0.683` → `bias=logit(0.683)≈0.77`，取 **0.75**（状态半衰期 ≈1.8 次写 ≈16 latent 帧）。照抄 2.0 则每帧等效保留 `0.8808^{1/9}≈0.986`，周转慢约 3 倍，撞先验 #6"双稳定器冻结动态"风险。**必须标注的近似**：(a) Echo 实际按 `num_exited_frames>0` 触发更新、驱逐帧数可变（`causal_model.py:792-828`），训练 chunk 长度/overlap 也可变（`streaming_training.py:153-185`），"每 3 帧一写"只是名义节奏；(b) gate 是逐 token 逐 channel 的（`gate_linear` 输出 hidden_dim，`query_memory.py:85-87, 152-155`），`σ(bias)` 只是初始化均值。落地时实测 Helios rollout 的写节奏与初始 gate 分布，sweep `{0.75, 2.0}` 裁决。

## 8. 算力预算（全部为 INFERRED 预测，无一来自可执行基准）

可引用的实测锚点仅有：Stage-1 1 节点 bs2 throttled 13.1 s/it（`docs/STAGE1_SPEED_OPTIMIZATION.md:22`）、4 节点(32 GPU) gb64 ~28 s/it、gb128 ~45 s/it（:62-69）。Echo "+10.6% 记忆开销"**只出现在上游研究笔记，Echo 仓库内无基准佐证**，仅作数量级参考。
- **Stage A**：+1 次 no_grad 写前向 + 记忆读 ≈1.5× → ~20 s/it，1 节点×8，4k 步 ≈ 22h / ~180 GPU·h。
- **Stage B**（U=4）：≈4×(grad 前向+反传)+4×no_grad 前向 → 1 节点 bs2 ~70 s/it；2 节点 ≈40 s/it，3k 步 ≈ 33h / ~530 GPU·h。
- **Stage C**：rollout（3-8 段，后缀带 grad）+ 14B real score 双 pass + 14B critic + 每 section 1 次 recache 写前向，估 3-4× Stage-1 单步；4 节点×8，10k trainer 迭代 ≈ 3-4 天 / ~2.5k GPU·h。
- 含 N_Q{3,6}×bias{0.75,2.0} 缩减 sweep，总额 ≈ **4.5-5.5k H200·h**。参照系（同为投影）：一次 lora368_correct 续训 5000 步、gb160=5 节点×40 GPU（yaml:95-97），以 4 节点/32 GPU 实测 ~45 s/it 外推 ≈2-2.5k GPU·h——本课程约为其 2 倍。**所有乘数在 Stage A/B smoke profile（D7）后修订。**

---
[^1]: 验证器称 "9×920 与 8,280 不符"——算术上二者恒等：9×920=8280，920=23×40 为 368×640 全分辨率桶每 latent 帧 token 数（`docs/STAGE1_DATALOADER_FRAMES.md:184-185`；patchify (1,2,2)，`transformer_helios.py:982-1024`）。该 issue 的实质部分（现 forward 无 hidden 返回、多分辨率桶导致形状可变）成立，均已纳入 D5。

---

# 第三章 推理集成与评测协议


我已对全部审查意见逐条回到磁盘代码复核:blocker 5 项中 4 项属实并已改设计,1 项经实测证伪(脚注 2);major 全部属实并已修正,minor 均已修复。以下所有行号均实测。

路径约定(均在 `/mnt/beegfs/siyuan/workspace/helios-team/` 下):**P** = `helios/diffusers_version/pipeline_helios_diffusers.py`,**T** = `helios/diffusers_version/transformer_helios_diffusers.py`,**S** = `helios/diffusers_version/scheduling_helios_diffusers.py`,**I** = `infer_helios.py`,**T_m** = `helios/modules/transformer_helios.py`,**P_m** = `helios/pipelines/pipeline_helios.py`,**U** = `helios/utils/utils_helios_post.py`,**B** = `helios/utils/utils_base.py`;Echo 仓库在 `/mnt/beegfs/siyuan/workspace/Echo-Infinity/`。张量尺寸按默认推理分辨率 384×640(I:80-81):latent 48×80,patch (1,2,2) → 每 latent 帧 960 token;一个 section = 9 latent 帧 = 8640 noisy token;history = short 1920(anchor+1 帧,T:699-713)+ mid 240(T:717-734)+ long 240(T:737-754)= 2400;全分辨率全序列 11040。N_mem = 3 帧当量 = 2880 token(sweep 6 帧 = 5760)。

## 与先验承诺及初稿的偏差(实测代码为准,置顶)

- **偏差 0(前提修正)**:"Helios 无 KV cache" 仅对本设计选定的 `diffusers_version` 推理路径成立。主代码库的 attention processor 持有逐层 `kv_cache`(T_m:182-194)、消费缓存的 history K/V(T_m:208-225)、在首个去噪步存入(T_m:383-388);P_m:928 暴露 `use_kv_cache`(默认 False),P_m:1429 每段清理。本移植的写入源设计不依赖该 cache,但训练侧实现(第〇节)必须与其共存(memory 捕获与 `restrict_self_attn`+cache 路径的相容性需在实现时显式断言)。
- **偏差 1(承诺 #1 "t≈0",数值修正)**:distilled(DMD)最后一次 transformer 调用发生在 **stage-local σ≈0.5**,而非初稿的 0.17。推导:每 stage 存储 sigmas `linspace(0.999,0,1001)[:-1]`(S:165-166);`[2,2,2]` 下 set_timesteps 先 +1 步(S:207-211),按 stage 端点 3 点插值得 `[0.999, 0.5, 0.001]`(S:234-236),再裁掉末点(S:244-246)→ 两次调用位于 σ≈0.999 与 σ≈0.5,随后 `step_dmd` 以 `x0 = xt − σ·flow_pred` 解析回推(S:862-868, S:883)。Mid(非 DMD)末段末步 σ≈0.001(S:227-236),最干净。Base 名义 σ≈0.02(P:1175 `linspace(0.999,0,51)[:-1]`);若发布 scheduler config 启用 `use_dynamic_shifting`(S:248-256)会被抬升——用 in-code 默认 `calculate_shift`(P:110-121)与 8640 token 估算约 0.038,实际端点取决于随 checkpoint 发布的 scheduler config(仓库内未包含,标注为外部依赖)。**结论:"t≈0" 严格成立于 Mid,近似成立于 Base,distilled 在 σ≈0.5 —— 三模式写入源噪声差异远大于初稿估计,σ_last 条件化(D3)与训推对齐(D13)升级为必选。**
- **偏差 2(承诺 #4)**:窗口 19 = 2×9+1,一个 section 分两个边界离窗:生成 section k 后 section k−2 尚余 1 个 latent 帧在窗内,k−3 才完全离窗(由 P:1218-1220 / P:1237-1239 的 `[:, :, -num_history_latent_frames:]`(=−19)切片直接推出)。写入时序按 D4 处理跨界帧。
- **偏差 3(措辞修正)**:Echo 把"被逐 KV→Enc.update"延迟到**下一次模型 forward**(memory KV 在进入所有 blocks 前计算,`wan/modules/causal_model_infinity_memory.py:219-232`;捕获与 update 在所有 blocks 之后,同文件 261-289),而非初稿的"下一 block 首次调用"。Helios 无 KV cache,无需该延迟。
- **偏差 4(新增,冷启动)**:Echo 冷启动为"首写之前无 memory":`reset` 置 `has_history=False`(`model/query_memory.py:118`),`get_kv` 在此期间返回 `None`(同文件 194-196),仅 `update` 置 True(:190)。D7 的"M₀ 恒在"是本设计对 Echo 语义的**主动偏离**(动机见 D7),非 Echo Stage-A 不变量。
- **偏差 5(新增,prompt 切换)**:Echo 发布的 long/interactive 配置均启用 `memory_recache: true`(`configs/echo_infinity-long_interactive.yaml:52`、`configs/echo_infinity-long_inference.yaml:50`),切换时重置并从窗内 KV 重建 memory(`pipeline/interactive_causal_inference.py:82-96`,并以新 prompt 重前向保留帧,:43-65)。初稿 D8 "Echo recache 无 Helios 对应必要"不成立,D8 已改写。

## 〇、实现目标澄清与训练侧前置条件(修订新增)

- **D0(双实现目标)**:训练入口导入的是 T_m 与 P_m,不是 diffusers_version(train_helios.py:31-38)。因此 memory 模块、capture、注入逻辑**首先落在 T_m / U 的训练路径**(该 forward 另含 NaviT、restrict_self_attn、kv_cache、GAN hook 分支且返回 `(output, logits)`),完成后再镜像到 T / P 推理文件。拒绝方案:只改 diffusers_version——Stage A/B/C 根本无法训练所提模块。
- **D13(专用 memory rollout,替代初稿"现有 Stage-C 出口点已对齐"的错误说法)**:现有 DMD rollout 不满足两个必要条件:① 出口步随机——`generate_and_sync_flag` 在非 `last_step_only` 时均匀采样出口(U:575-593),rollout 在 `index == exit_flag` 处带梯度前向后 break(U:1259-1324),且两份发布配置均 `dmd_last_step_only: false`(`scripts/training/configs/stage_3_post.yaml:239`、`stage_3_post_self-forcing_version.yaml:241`);② 默认 post 配置只 roll 1 个 section 且用 GT history(stage_3_post.yaml:246-247, 288-289),延迟 k−2 的写入在其中一次也不会发生;只有 self-forcing 变体(3–44 sections、`is_use_gt_history: false`,self-forcing yaml:248, 290-291)满足段数要求。设计:memory 训练基于 self-forcing 变体新增专用 rollout:`dmd_num_latent_sections_min ≥ 4`;每 section 在**本模式推理的 σ_last**(distilled 为 stage-local σ≈0.5)执行捕获——若采样出口恰为末步则复用该前向,否则补一次 `no_grad` 的 σ_last 前向。保证训推同分布捕获。
- **D14(trainer/保存/优化器集成)**:现有训练把 transformer 整体冻结后仅按硬编码模块名单解冻(train_helios.py:357-364, 463-477),LoRA save hook 只序列化 PEFT 状态 + `save_extra_components`(train_helios.py:648-694),而后者只认 patch 模块、restricted LoRA、history_key_scale、GAN 头(B:157-256;loader B:260-363 同样无 memory 分支)。需:① 名单追加 `mem_` 前缀模块;② save/load_extra_components 各加 memory 分支(`mem_encoder`、`mem_init_queries`、逐层 `to_k_mem/to_v_mem/memory_key_scale`);③ memory 参数独立 param group(独立 lr),DeepSpeed zero2(stage_3_post.yaml:228-229)下将模块注册在 transformer 内统一分片并以 resume 测试验证 save hook 覆盖——Echo 的教训是 encoder 置于 FSDP 之外、显式梯度同步、单独优化器与保存(`model/query_memory.py:106` 的 FSDP 展平报错信息;`trainer/distillation.py:194-250, 290-299`);④ critic transformer 由独立 kwargs 字典构建(train_helios.py:334-347),**保持 critic 无 memory**。
- **D15(BPTT/detach 契约,初稿缺失)**:队列 push 时 `capture.detach()`(否则每个入队 capture 挂 40 层反向图,84 MiB 激活 × 最多 44 sections 不可承受);写入 `mem.M = Enc(mem.M.detach(), capture_detached)`——TBPTT 视野 = 1 次写入:Enc 参数梯度来自"本次写入算子 + 后续 sections 读出损失经 M 的路径",读通路梯度照常进主干与 `to_k_mem/to_v_mem`。对齐 Echo 的 `detach_state`(query_memory.py:214-219)与按 clip detach(`model/streaming_training.py:212-217`)。可 sweep 视野 W∈{1,2}(W=2 保留上一次写入的图)。拒绝方案:隐式不 detach——梯度图与显存跨段无界。

## 一、`infer_helios.py` / pipeline section 循环的确切改动

`infer_helios.py` 无 section 循环(I:343/404/476/525 为单次 `pipe(...)` 调用,透传新增参数 `memory_enabled/memory_state/perturb_at_chunk/perturb_strength`);改动落在 **P** 的 `__call__`(P:832 起)与两个采样函数,并同步镜像到 P_m。

**D1(捕获挂钩,修订)**:capture 取自 block 循环整个 if/else 之后、`norm_out` 之前(T:791-809 为 checkpoint / 非 checkpoint 两分支,T:812 为 norm_out;外层缩进插入对两分支均生效)[^1]:`capture = hidden_states[:, -original_context_length:, :]`(shape `[B, 8640, 5120]`,bf16 ≈ 84 MiB;`HeliosOutputNorm` 本身即按此切片,T:91-95)。transformer 新增参数 `return_capture: bool = False`,为 True 时额外返回 `capture.clone()`;现有调用全以 `[0]` 取主输出(P:551-564, 566-581),追加返回向后兼容。compile 影响:布尔参数使每分辨率产生 capture/no-capture 两张图——这是**实现假设**,必须用带 graph-count 统计的 compiled smoke test 验证(仓库无现成测试)。拒绝方案 a:每次前向恒返回 capture——延长中间激活生命周期且无谓改变单输出契约;拒绝方案 b:`blocks[-1]` forward hook——compile 与 CP gather(T:571-573)下顺序脆弱。

**D2(捕获选择)**:仅在**最终去噪步的 cond 前向**传 `return_capture=True`:stage1 为 `i == len(timesteps)−1` 的 P:551-564 调用(scheduler.step 在 P:599 之后);stage2 为末段末步的 P:723-737 调用。拒绝方案:transformer 属性存"最近一次前向"——CFG 下同一步 cond 后还有 uncond 调用(P:566-581),会被覆盖。`stage1_sample`/`stage2_sample` 改为 `return latents, capture`(P:622 / P:804)。

**D4(写入时序)**:延迟 D=2——循环体 k 末尾、anchor 捕获(P:1322-1325)与 `history_latents = torch.cat`(P:1328)之后:`queue.push(capture_k.detach())`;若队首为 k−2 则 `M ← Enc(M.detach(), queue.pop())`(detach 契约见 D15)。此时 section k−2 已有 8/9 帧离窗、仅剩 1 帧与 long 层 4× 粗 patch 双重表示,代价可忽略(类比 Echo 中 sink 内容恒可见)。拒绝方案:生成即写(D=0)——被写内容仍完整在窗内两个 section,背离 Echo"逐出即写"语义且加剧双稳器冻结风险。

**D5(段内只读)**:section k 全部去噪步读同一 M_k,更新只在 section 之间。拒绝方案:段内更新——破坏段内确定性与 compile;Echo 中新状态在触发 forward 内本就不可见(causal_model_infinity_memory.py:261-289 在 blocks 之后才 update)。

**D6(读注入,全面改写——初稿接口与代码不自洽)**:现有 `HeliosAttnProcessor` 无任何 KV 追加点:单一 `hidden_states` 做 QKV 投影(T:119)、单一 `rotary_emb` 同时旋转 Q/K(T:128-130)、直接 SDPA 分发(T:141-151);timestep 条件经 AdaLN 只作用于序列 token(T:450-459)。因此**纯 KV memory 不可能"走 zero_history_timestep 路径"**(该路径只为已拼入序列的 history token 构造 t=0 embedding,T:756-768, 773-782),初稿该说法作废。新接口:

- 每层 `HeliosAttention` 新增 `to_k_mem/to_v_mem: Linear(5120, 5120)`、`norm_k_mem`、`memory_key_scale`。**逐层投影**而非跨层共享:Echo 的单一投影是专门训练投到该模型 KV head 空间的(query_memory.py:88-91),Helios 40 层 `to_k/to_v` 相互独立(T:186-188),跨层共享无合法性依据;开销按 +10% 档预算(第五节)。
- processor 签名扩展:`memory_states: Tensor[B, N_mem, 5120] | None` 与 `rotary_emb_mem`。在序列 Q/K/V 投影、norm、RoPE **之后**执行:

```python
k_mem = attn.norm_k_mem(attn.to_k_mem(memory_states)).unflatten(2, (attn.heads, -1))
k_mem = apply_rotary_emb_transposed(k_mem, rotary_emb_mem)
v_mem = attn.to_v_mem(memory_states).unflatten(2, (attn.heads, -1))
scale_mem = 1.0 + torch.sigmoid(attn.memory_key_scale) * (attn.max_scale - 1.0)
key   = torch.cat([k_mem * scale_mem, key], dim=1)   # [K_mem, K_hist*amp, K_noisy]
value = torch.cat([v_mem, value], dim=1)
```

- "timestep=0 条件"重新定义:memory 不经任何 per-step AdaLN 调制,其"干净态"语义由 Enc 产出保证;噪声水平条件放在 Enc 侧(FiLM(σ_last),第二节)。拒绝方案:memory 作为序列 token 进 FFN/AdaLN——FLOPs 约 +26%,且需侵入 temb 拼接(T:770-782)。

**D6b(位置,修订——补齐空间网格并消除 ID 漂移)**:定义 `indices_memory = [−N_mem_frames, …, −1]`,每"帧"按 24×40 token 网格经 `self.rope(frame_indices=indices_memory, height=24, width=40)` 生成 `rotary_emb_mem`(与 noisy 的 post-patch 网格同构;T:354-372 对任意 float 帧索引成立)。**负 ID 保持既有全部 ID 不动**(P:1139-1147:prefix=0, long=1-16, mid=17-18, 1x=19, noisy=20-28),不扰动已发布 checkpoint 学到的相对位置分布,语义为"比窗内一切更老"。这与承诺 #3 的"anchor 之后、long 之前"槽位差一格,属主动偏离:插入式槽位会把既有 ID 整体 +N_mem_frames,改变已训练的位置分布(拒绝理由)。

**D7(恒在 M₀,重新标注)**:memory KV 自 section 0 恒在(M₀ = 学习的 `mem_init_queries`,Stage-A 静态 query)。显式标注为对 Echo 冷启动语义(偏差 4)的**主动偏离**,动机:compile 形状恒定 + 与 Stage-A 训练分布一致——前提是 Stage A 起就以 M₀ 恒在方式训练,使契约自洽。备选(若 M₀ 读出干扰前 3 段):固定形状 + attention mask 屏蔽 memory 列(形状不变,仍 compile 友好)。冷启动:k=0,1,2 用 M₀,首个写入发生在 k=2 末(写 section 0)。

**伪代码(P.__call__ 主循环改动)**:

```python
# P:~1110 处:状态初始化(见第三节)
mem = memory_state or MemState(M=transformer.mem_init_queries.expand(B, N_mem, d).float(), queue=deque())

for k in range(num_latent_chunk):                     # P:1184
    if perturb_at_chunk == k:                          # 评测用持久扰动,见第六节
        history_latents[:, :, -19:] = anti_drift.apply_frame_aware_corruption(
            history_latents[:, :, -19:], perturb_strength)
    ...tier split (P:1215-1239)...
    latents, capture = self.stage{1,2}_sample(..., memory_states=mem.M,
                                              return_capture_at_last_cond=True)  # P:1265-1320
    ...anchor capture (P:1322-1325)...
    history_latents = torch.cat([history_latents, latents], dim=2)               # P:1328
    mem.queue.append((k, capture.detach()))            # D4/D15
    if mem.queue[0][0] == k - 2:
        _, evicted = mem.queue.popleft()
        mem.M = transformer.mem_encoder(mem.M.detach(), evicted, sigma_last)     # 经 forward 调用,见第四节
```

## 二、与 pyramid(Mid)/distilled few-step 的交互

- σ_last 修订表:**Mid ≈0.001;Base ≈0.02–0.04(依外部 scheduler config);distilled stage-local σ≈0.5**(偏差 1)。
- **D3(修订)**:Enc 输入的 FiLM(σ_last) 标量条件由可选升级为**必选**(σ 跨度 0.001→0.5);Stage-C 经 D13 专用 rollout 在相同 σ_last 捕获,保证训推一致——不再宣称现有出口点已对齐。可选项 `memory_capture_extra_t0`(额外一次 t=0 干净前向)保留:Base/Mid 增量约 +1~3%(token 线性模型约 1.5%,含二次注意力项的整块模型约 3%,以实测定夺),distilled 约 +30%(2 调用 ×3 段 ≈ 3.36 个全分辨率 token 当量,token 线性口径),默认关闭;若 distilled 下 gate 均值系统性 <0.5× Mid 值(写入源退化信号)则启用。
- Mid(`[20,20,20]`,CFG=5):金字塔各 stage noisy token 数为 540/2160/8640,history 全程全分辨率传入(P:734-736),memory N_mem 恒定——各 stage 形状本已不同(compile 本就 3 张图),memory 不增加图数;capture 仅在末段末步(上采样加噪只在段首,P:695-718),σ≈0.001,写入源质量最佳。
- Distilled(`[2,2,2]`,CFG=1;`is_amplify_first_chunk` 使首 chunk 步数翻倍,P:1259-1263 + S:208-209,不改变"最后一次调用"判定)。

## 三、memory state 生命周期

- **初始化**:每次 `pipe.__call__` 入口(P:~1110,history buffer 分配处)重置为 M₀;M 用 fp32 存储(`[B, 2880, 5120]` ≈ 59 MB),Enc 内部 bf16 计算、fp32 累积,防千段递推漂移。i2v/v2v 预填充(P:1125-1136)的真实视频 latents 无 hidden-state 捕获即入窗,离窗时无法入 memory——v1 接受此缺失(OBSERVED gap);v2 可选对每个预填充 chunk 跑 capture-only 的 t=0 前向。
- **D9(修订:更名为 memory-state export,而非断点续跑)**:暴露 `pipe.get_memory_state()/set_memory_state()`,状态字典 `{"M": fp32, "k_written": int, "queue": [(k, capture_bf16)]}`,供流式服务与消融注入。**明确它不是完整生成 checkpoint**:精确续跑还需 `history_latents` 缓冲、anchor `image_latents`(P:1322-1325)、`total_generated_latent_frames`(P:1327)、prompt 插值位置(P:1188-1211)与 generator RNG 状态(P:1106-1136),v1 不承诺;Echo 侧同样无中途续跑(整段一次调用,`inference/inference.py:167-175`;query_state 为普通属性,query_memory.py:95-99)。拒绝方案:落盘全量状态——queue 含 84 MiB/段 capture,收益/复杂度不成比例。
- **D8(prompt-switch,改写为待检验契约 + 消融)**:默认假设 no-reset(训练契约:chunk 对齐 caption、视觉历史跨 prompt 连续),但**不再宣称这是代码推论**,理由:(a) Echo 发布配置在切换时 `memory_recache=True` 重置并重建(偏差 5);(b) Helios capture **非提示词中立**——每个 block 都含文本 cross-attn(T:461-487),末层 hidden states 携带旧 prompt 语义,memory 有拖拽新 event 的真实风险。切换策略消融矩阵:{no-reset,reset-to-M₀,recache(对窗内 19 帧跑 capture-only 前向重建 M),decay(M←αM₀+(1−α)M)},按第七节 C-⑥ 指标裁决。interactive 路径(I:448-519,`interpolation_steps=3`)与 multievent 硬切(`infer_multievent.py:170-187`,`interpolation_steps=0`)都只改 `prompt_embeds`(P:1185-1213),机制上兼容全部四种策略。

## 四、兼容性逐项分析

- **enable_parallelism(I:312-324)**:(a) 既有隐患照旧成立:`_cp_plan` 只声明 `blocks.*.attn1` 的 `hidden_states/rotary_emb` 按 seq 维切分与逐层输出 gather(T:558-574),CP 分片下 processor 拿到的是本 rank 分片而 `original_context_length` 是全长(T:696),T:133 得负 `history_seq_len`、T:135 使 amplify-history **静默失效**——移植 memory 放大参数前必须先修(传入全局 seq 元信息)。(b) **D10(改写)**:`_cp_plan` 无 memory 概念,Ulysses/Ring 对带外 KV 参数的分片/复制/重组语义未在代码中定义(dispatch 仅内部接收 `parallel_config`,T:141-151),初稿的 per-rank 分片方案**降级为未验证假设**;v1 声明 memory 不支持 CP 路径(吞吐评测本走按 rank 分 prompt 的非 CP 路径,I:329-331),CP 支持须逐 backend 与单卡 attention 数值对照测试通过后开启。拒绝方案:每 rank 复制全量 memory KV 直接上线——正确性取决于 backend 的 softmax 重组语义,未经测试不可用。(c) capture 在逐层 gather 之后的 forward 级切片,天然全序列。(d) Enc 更新在所有 rank 以确定性算子重复执行 + debug all-reduce 校验。
- **enable_compile(I:294-298,`dynamic=False`)**:D7 恒在 memory KV 保持形状恒定;D1 的 `return_capture` 双图假设须 smoke test(覆盖 Base/Mid/Distilled × CFG 分支,统计 graph 数)。Enc 每段一次,保持 eager;compile 分支跳过 kernel 替换(I:243-246),Enc 用标准 `nn.LayerNorm` 兼容两路径。`enable_low_vram_mode` 与 compile 互斥(I:189-191)。
- **low_vram(I:300-308 group offload,修订)**:不依赖"hooks 自动覆盖直接方法调用"——Enc 更新统一经 `mem_encoder.forward(...)` 调用(offload pre-forward hook 语义针对 forward;直接调 `.update()` 是否触发 hook 无代码依据),并纳入 offload 集成测试(leaf 级与 block 级各一)。M 与 capture 为普通张量不受 offload 影响;capture 队列峰值 ≤3×84 MiB ≈ 253 MiB;low_vram 下 `queue_on_cpu=True` v1 用同步拷贝(每段一次 84 MiB,数十 ms 可接受),真异步需 pinned buffer + copy stream,列为 v2。CP 与 low_vram 互斥(I:211)。

## 五、开销估计(重做;口径:MACs,FLOPs = 2×MACs)

d=5120、40 层、全分辨率全序列 L=11040(T:576-587),文本 cross-attn 的 K/V 按 512 text token 摊销忽略。每 token 每层:attn1 QKVO 4d² ≈ 104.9M;attn2 QO 2d² ≈ 52.4M;FFN 2·d·13824 ≈ 141.6M;SDPA 2·L·d ≈ 113.0M;合计 ≈ 412M MACs(SDPA 占 27%)。单次全分辨率前向 ≈ 11040×40×412M ≈ 182 TMAC(≈ 364 TFLOPs);Base 一段 50 步 × CFG2 = 100 次 ≈ 18.2 PMAC(≈ 36.4 PFLOPs)。

- memory(N_mem=2880,逐层投影,按 D6):ΔSDPA = 2·N_mem·d ≈ 29.5M/token;逐层 K/V 投影 2d²·N_mem 摊到 L 个 token ≈ 13.7M/token → 全分辨率序列合计 ≈ **+10.5%**;Mid 低分辨率段序列短、memory 相对占比更高,整 chunk 加权约 +9~12%;N_mem=5760 约 +19%(SDPA 与投影近似同比放大)。
- Enc 更新(2 层 cross-attn,query 2880 × KV 8640,FFN×4):≈ 3 TMAC/段(≈ 6 TFLOPs),对任意模式 <0.1%,忽略。
- 显存:M 59 MB + 队列 ≤253 MiB + 每层 KV 拼接临时量;extra_t0 增量见第二节。

以上均为**解析粗界**,不含 FlashAttention 下 KV 拼接、激活物化、队列拷贝等实现开销;墙钟与峰值显存以基准实测裁决。初稿引用的"Echo 实测 10.6% 吞吐损失"在 Echo 仓库中不存在(README 无该数字;`pipeline/causal_inference.py:49-59, 95-127` 仅通用分块计时)——该数字源自团队研究文档对 Echo 论文的转述,撤销其"代码事实"地位。**验收线:端到端吞吐损失 ≤ 12%,作为提议目标由基准实测确认。**

## 六、评测协议

**复用配置**:`tools/long_video_eval/configs/vs24_long.json`(2178 帧 ≈ 90.75 s,主战场)、`vs24_eventswitch.json`(1386 帧,prompt-switch 持久性)、新增 `vs24_ultralong.json`(4356 帧 ≈ 181.5 s,132 段 ≈ 130 次写入,检验递推稳定)。**D11**:斜率指标做独立脚本 `run_helios_long_timeseries_metric.py`,以 `external_command` step 注册——runner 只渲染并子进程执行(`tools/long_video_eval/long_video_eval/runner.py:55-82, 97-106`;初稿路径少写一级目录,已更正),零核心改动。eventswitch 的 event 信息**不改 manifest schema**(现 schema 仅 index/model/video_path/prompt,`tools/long_video_eval/long_video_eval/manifest.py:12-18`,由纯文本 prompt 文件构建 :20-34):以独立 event-spec JSON 直接传给 external 脚本。拒绝方案:改造 manifest/core——违背零核心改动。

**新增指标(缺失定位修正)**:(1) motion-amplitude 斜率——Farneback 光流与打分**已有实现**(`eval/1_get_motion_amplitude.py:34-59, 62-69`),OBSERVED ABSENCE 仅限"33 帧 chunk 对齐时间序列 + OLS/Theil-Sen 双斜率聚合",新脚本复用既有实现;(2) saturation/chroma 曲线(HSV-S 与 CIELAB chroma,报斜率与 AUC);(3) 边界 LPIPS-excess:LPIPS(chunk i 末帧, chunk i+1 首帧) 减 chunk 内相邻帧对照值;(4) event 感知 CLIP-over-time(每 chunk 中点帧对所属 event prompt;现有 eventswitch 文本指标只对首 prompt 打分,`docs/VS24_EVAL_ANALYSIS.md:88-99, 142-147`)。(2)(3)(4) 为真 OBSERVED ABSENCE。

**扰动后再漂移测试(语义修订)**:pipeline debug 参数 `perturb_at_chunk=30, perturb_strength∈{0.1,0.2}`。定义为**持久型状态扰动**:tier split(P:1215-1239)前就地写回缓冲:`history_latents[:, :, -19:] = anti_drift.apply_frame_aware_corruption(history_latents[:, :, -19:], s)`(B:752-761 返回新张量而不就地修改,必须显式赋回;被扰帧随窗口滚动自然退出)。瞬态变体(只改当段三个 tier 张量、不写回缓冲)作为对照实验。度量:chunk 30-60 的 CLIP/saturation/subject-consistency 序列;**恢复时间** = 回到扰动前 chunk 10-29 均值 ±1σ 带内所需 chunk 数;**再漂移斜率** = chunk 35-55 的 OLS 斜率。假设:memory-on 恢复更快且恢复后 dynamic-degree 不塌陷(检验双稳器冻结风险,承诺 #6)。multievent 硬切路径(`interpolation_steps=0`)可直接用于 eventswitch 评测[^2]。

**VS24 对比方案(D12,消融组修订)**:同 seed 同 prompt 五组:(a) memory-on;(b) **no-KV**(同 checkpoint,推理时完全不拼 memory 列);(c) **frozen-M₀**(恒用初始 query,不写入);(d) **shuffled-M**(打乱 section 对应关系,检验内容特异性);(e) 历史锚点 step-0 与 step-5000(`docs/VS24_EVAL_ANALYSIS.md` 认定的最佳折中,DOVER drift 0.069 / dynamic degree 0.75)。注意 **M=0 不是"无记忆"**:`to_k/to_v` 带 bias(T:186-188)且过 norm_k,零输入仍产生非零 KV,不用作对照。全部指标以斜率优先、端点均值为辅报告。

## 七、分阶段 go/no-go 判据(量化)

- **Stage A(TF + 静态 query,按 D7 契约以 M₀ 恒在方式训练)**:① 验证集 flow loss 相对无 memory 基线劣化 ≤ +1%;② memory token 平均注意力质量 > 1%(否则读通路死,no-go);③ 91 s rollout 端点 DOVER-overall 回退 ≤ 0.02。任一不满足 → 修读通路,不进 B。
- **Stage B(TF unroll 写语义,按 D15 TBPTT=1 契约)**:① gate sigmoid 均值 ∈ [0.05, 0.6](双端饱和均 no-go);② 91 s memory-on vs no-KV:VBench subject-consistency ≥ +0.005 且 boundary-LPIPS-excess ≤ −5%;③ 冻结检测:dynamic-degree 均值下降 ≤ 0.1 且 motion-amplitude 斜率不比 no-KV 更负超 30%,违反 → 按承诺 #4 重调 gate bias(稀疏大写入),仍违反 → no-go。
- **Stage C(self-forcing/DMD 校准,必须使用 D13 专用 rollout 训练的 checkpoint)**:91 s 与 181.5 s 自生成 rollout 上:① DOVER drift ≤ 0.069(不劣于 step-5000);② saturation |斜率| ≤ 0.7× no-KV;③ 扰动恢复时间 ≤ 0.7× no-KV;④ dynamic degree ∈ [0.55, 0.90];⑤ 吞吐损失 ≤ 12%(实测);⑥ eventswitch 切换后 1 个 chunk 内对新 prompt 的 CLIP 不低于 no-KV(memory 不得拖拽语义;此项同时裁决 D8 的切换策略)。若校准后 memory-on 在 ≥2 项主斜率指标上仍劣于 no-KV → 按先验结论判定"仅 TF 训练的写 gate 为净负",no-go 并回退 no-KV 发布。

**范围提醒**(承诺 #6):~2880 token 抽象状态不承担精确长程回忆,eventswitch 只考核一致性/边界平滑,不设 recall 指标;stationarity 由 Helios 相对位置 + 固定窗口负责,不计入 memory 的功劳或过失。

---

[^1]: 验证者称插入点"仅在非 checkpoint 分支之后":不准确——T:791-809 的 if/else 结束后、T:812 之前的外层缩进语句对 gradient-checkpointing 分支同样执行;但采纳其表述建议("块循环整体之后、norm_out 之前")以消歧,并采纳"compile 行为属未测假设"的定性。

[^2]: 验证者(blocker)称 `interpolation_steps=0` 会触发 `interpolate_prompt_embeds` 内 `torch.linspace(steps=0)` / `chunk(0)` 崩溃:经实测证伪——P:1198-1200 中 `position_in_interval = k − interval_start ≥ 0`,判断 `position_in_interval < interpolation_steps`(即 `< 0`)恒假,插值函数(调用点 P:1202-1206,定义 P:459-471)在 steps=0 时不可达,硬切换直接走 P:1211 选取新 prompt embeds;`infer_multievent.py:184-186` 的现行路径可用,无需修 pipeline。

---

# 第四章 风险矩阵、消融计划与主线协调


前置核验声明：本节全部代码事实已在磁盘逐项复核（helios-team @ mid_training_xiangbo、Echo-Infinity）；本次修订另对 19 条 verifier issue 全量回查，均获代码证实并已吸收，无需反驳性脚注。两项关键 OBSERVED ABSENCE 置顶：
(1) **仓库内不存在任何 history 离散化/VQ 实现**——宽口径 grep `vq|vector quant|codebook|fsq` 的命中全部无关：VideoAlign 奖励维度 "VQ"（`helios/videoalign/prompt_template.py:12`、`helios/utils/train_config.py:383`、`helios/utils/utils_helios_post.py:2736-2744`、`helios/videoalign/data.py:20-43`）及调度器 docstring 中泛义 "discrete"（`helios/scheduler/scheduling_helios.py:305`）；针对 history 量化器/codebook/FSQ 的窄口径检索零实现命中，R6 按"纯规划态实验"处理；
(2) **评测侧无任何 within-video 时序斜率后端**——长评测只有 full/start15/end15 端点聚合（报表消费 `full_mean`/`drift_abs_mean`：`scripts/evaluation/make_eval_report.py:132-147`；逐视频 start/end 差值计算：`scripts/evaluation/summarize_helios_long_ratio_eval.py:92-110`），全仓库无 OLS/Theil-Sen 实现。本文所有标 [NEW] 的检测信号须先按设计节 3 §6 的 `external_command` 方案落地（`tools/long_video_eval/long_video_eval/runner.py:55-107`）才可执行——**检测手段本身是交付物，不是现货**。

## 0. R0：跨设计节机制冲突与两项 blocker 级仲裁（先于一切实验）

三份已验证设计节对同一机制给出互斥参数化：**读机制**（节 1 = token 插入、memory 获得 t=0 AdaLN；节 3 = 逐层 `to_k_mem/to_v_mem` 纯 KV 注入、AdaLN 不作用）；**M 定义**（节 1 mid-tier 390 token/帧、M=1170 @480×832；节 2 全分辨率 920 token/帧、M=2760 @368×640；节 3 960 token/帧、N_mem=2880 @384×640）；**RoPE 槽位**（节 1 (0,1) 分数 id vs 节 3 负 id）；**命名/API**（`helios_memory`/`evolving_memory`/`mem_encoder`；`capture_last_hidden`/`return_final_hidden`/`return_capture`）。

**D1（读机制/M 仲裁，写第一行实现代码前锁定）**：合成基线 = 节 1 的 token 插入读机制 + 节 2 的配置键/课程/trainer 集成 + 节 3 的 FiLM(σ_last) 写源条件化与评测协议；M 语义统一为"帧当量 N_Q × 主训练桶 mid-tier 网格"（368×640：full-res 23×40=920/帧，mid-tier 网格 12×20=240/帧，N_Q=3 → M=720），M 为固定形状（状态跨 section 持有必须定形），节 1 的 1170 与节 3 的 2880 是各自分辨率/密度下的换算，消融自变量统一为 N_Q 帧当量；RoPE 槽位交由消融 A9 裁决（分数 id 为默认）。*弃选：两种读机制并行实现再实验裁决——双份 diffusers 镜像移植成本翻倍，且 t=0 条件语义不可比；若 token 插入的 +FFN 开销实测不可接受，KV 注入是唯一回退项，由 A0 仲裁实验兜底。*

**D8（谱系仲裁，blocker——课程三阶段的父模型不同源）**：OBSERVED：Stage A/B 父系 = Helios-Base `transformer_init` + LoRA rank 128（`scripts/training/configs/stage1_lora_cfr_368_correct.yaml:34-48`），而节 2 提议的 Stage C 父配置 = Helios-Distilled `transformer_ode` + LoRA rank 256（`scripts/training/configs/stage_3_post_self-forcing_version.yaml:35-53`）；trainer 从配置路径重建 generator（`train_helios.py:310-325`）并按配置 rank 新建 adapter（`train_helios.py:397-405`），checkpoint hook 将 PEFT adapter 与 extra 全秩组件分开保存（`train_helios.py:662-694`）。rank-128 adapter 无法载入 rank-256 构造，"A/B 直通 C"不存在形状兼容的交接。裁决四条：(a) **memory 全部可训练参数（queries M、Enc、gate、读路径新增投影）一律实现为 PEFT adapter 之外的全秩 named modules**，走 `save/load_extra_components` 通道——形状只依赖 config dim（Base 与 Distilled 同为 dim 5120/40 层，同构可互载）；(b) Stage A/B 仍在 Base@ckpt-19500 rank-128 上训（保 C2 可比性），结束后仅将 memory 组件移植进 Stage C 的 distilled rank-256 generator；(c) 承认移植后 backbone 失配：Stage C 前置强制 **C0 warm-up 子阶段**（distilled generator + memory 加载、backbone 冻结或低 lr 的短程 TF 回归 ~500-1000 步），预算并入 C；(d) 落地 checkpoint-conversion 测试：把完整 A/B memory+读路径 state 加载进 Stage C 构造器，任何 missing/mismatched key 即 fail——列入 K0。*弃选一：A/B 直接在 distilled rank-256 上训——丢失与主线 `_correct` 谱系的横比与 C2 冻结基线，且 few-step 生成器上 TF flow-matching 的 σ 采样语义未经验证；弃选二：merge rank-128 adapter 进全模型再重跑 ODE 蒸馏过渡——重蒸馏成本与风险远超 C0 warm-up。*

**D9（CP 兼容，v1 硬约束）**：v1 **禁止 memory 与 `--enable_parallelism` 同时开启**（入口 `infer_helios.py:312-324`）。依据：diffusers CP plan 仅按序列维切分 `hidden_states`/`rotary_emb`（`helios/diffusers_version/transformer_helios_diffusers.py:558-573`），不携带全局偏移标注；processor 用 rank-local 序列长减全局 `original_context_length` 推 history 长度（`transformer_helios_diffusers.py:132-139`）——任何"前 M 个 rank-local key 属于 memory"的切片在 CP 下会在不同 rank 命中不同的 [memory|history|current] 片段。启用前必须扩展 CP plan 携带全局偏移并逐 backend 通过单卡 vs CP 数值等价测试。*弃选：v1 直接改 CP plan——上游 diffusers 侵入面大，且 memory 单卡收益未证前不值。*

## 1. 风险矩阵

### R1 双稳定器过稳 / 动态度冻结
- **机制（修订）**：x0/image anchor 在默认 rollout 中持续存在——T2V 于第一个生成 chunk 后由 `image_latents = latents[:, :, 0:1]` 建立（`helios/pipelines/pipeline_helios.py:1447-1450`）并在后续 section 前缀拼接（`pipeline_helios.py:1294-1298`），I2V 自始存在；但它**并非不可拆**：`is_keep_x0` 是 API 参数、默认 True（`pipeline_helios.py:906`），False 时走无 anchor 分支（`pipeline_helios.py:1175-1177,1299-1309`），故 anchor×memory 交互可单变量消融（已并入 A6）。history key 放大则是**可用但当前关闭**的机制：冻结基线显式 `is_amplify_history: false`（`stage1_lora_cfr_368_correct.yaml:173-174`），参数仅在构造期按 flag 分配（`helios/modules/transformer_helios.py:518-528`，构造默认 False：`transformer_helios.py:1006-1009`）；`infer_helios.py:290` 的 `is_amplify_history = True` 只是 `lora_path+partial_path` 分支里给 `load_extra_components` 的 loader shim（`infer_helios.py:279-292`），transformer 早已构造完毕（`infer_helios.py:238-242`），不会激活运行时放大。因此本风险的静态前缀组合是 **anchor + t=0 memory（含 amp_mem）**，不含 history 放大；若实验要启用 history 放大，须构造期开启并配对训练，作为独立 paired baseline。注意力质量向静态内容集中的风险不变。基线已有静动权衡实证：step-0 dynamic_degree 0.30 vs 续训 0.55–0.90（`docs/VS24_EVAL_ANALYSIS.md:35-59`）——向冻结方向的回退空间只有约 0.25。
- **检测信号**：[NEW] dynamic-degree / motion-amplitude 逐 chunk（33 帧对齐）斜率，OLS+Theil-Sen 双报；阈值 = memory-on 相对 no-KV 均值降 >0.1 或斜率负向劣化 >30%（对齐节 3 Stage B ③）；扰动实验（`perturb_at_chunk=30`）恢复达标但恢复后 dynamic-degree 掉出 [0.55, 0.90]，判"以冻结换稳定"，同样触发。
- **缓解**：gate bias 下调（0.75 → 更低，加快状态周转，节 2 §7 换算）；`memory_key_scale` 从中性 init −4.0 起步且可冻结（节 1 §1）；**D2（修订）：训练期 memory-dropout p≈0.1，实现为 attention 列 mask 或 gated bypass**（阻止 current query attend memory K/V，保持 compile-stable 形状），**禁止用零值 token 冒充关断**——AdaLN 有加性 shift（`transformer_helios.py:792-820`）且 `to_q/to_k/to_v` 均 bias=True（`transformer_helios.py:488-490`），全零 token 经投影后仍是非零 Q/K/V 参与无 mask 注意力（`transformer_helios.py:410-415` 的 `attn_varlen_func` 调用），no-memory 对照必须是真移除前向并做数值等价验证。*弃选：不加 dropout 全靠 Stage C 校准——C 最贵且只校准写分布，不治读依赖。*
- **残余风险**：斜率指标未经校准，首个 campaign 须先测 no-KV 的自然斜率分布定 ±1σ 带；anchor 默认开启下双前缀交互仍需靠 A6 的 `is_keep_x0` arm 分离。

### R2 TF-only 写门净负资产（含 Stage C 可训练性前提）
- **机制**：gate 输入在 TF 下只见 GT 统计的 t=0 hidden；推理写源是自生成漂移分布，`g·old+(1−g)·proj`（Echo 更新式 `Echo-Infinity/model/query_memory.py:152-155`）逐 section 复利放大失配——先验 #5 判定无末端校准则净负。
- **Stage C 可训练性前提（新增，源自代码事实）**：拟用父配置 `dmd_num_latent_sections_min: 3`（`stage_3_post_self-forcing_version.yaml:289-295`）且 cold-start 强制取该最小值（`train_helios.py:1571-1581`）；[16,2,1]+x0 窗口 19 latent 帧、每 section 追加 9 帧（`pipeline_helios.py:1294-1298,1452-1454`），首次含真实帧的驱逐写最早发生在第 2 个 section 末、其读收益落在下一 section——3-section rollout 在任何 loss 读到写入状态前即结束，Enc/gate 得不到读回报梯度。**强制项：memory 开启时断言 `dmd_num_latent_sections_min >= 4`（含 cold-start 分支），并附最短 rollout 上 Enc/gate 梯度非零的单元测试（列入 K0）。**
- **检测信号**：(a) 训练日志逐步记录 gate sigmoid 均值/分位数，TF 与自 rollout 双通道对比；(b) capture 特征统计（norm、与 TF 均值 cos）在 rollout 下漂移 >2σ；(c) 裁决性信号 = Stage B checkpoint 直接跑 91s 自生成 rollout：若 memory-on 在 ≥2 项斜率指标劣于 no-KV 而 Stage C 后翻正 → 风险实锤且缓解有效（这同时就是消融 A5a）。
- **缓解**：Stage C 必做（节 2 D9），谱系交接按 D8 执行（C0 warm-up 吸收 backbone 失配）；Stage B 混入低比例自生成 history / 对 capture 加噪模拟漂移（廉价前置缓解）；D13 同 σ_last 捕获保证训推同分布（节 3，σ 取值见 R4 修订）。
- **残余风险**：Stage C 自身脆弱——critic lr 4e-7、`dfake_gen_update_ratio: 5`、GAN 初期需关（`stage_3_post_self-forcing_version.yaml:232-233,256`）——"校准失败"与"写门本质不可救"不可区分；C 占全课程预算约一半。

### R3 粒度失配（Enc 容量 / 写稀疏）
- **机制（修订）**：Echo 的写是**驱逐驱动**而非逐 block 无条件——block 大小 3 帧（`Echo-Infinity/configs/echo_infinity-long.yaml:55`），但 `query_memory_encoder.update` 仅在局部窗推进产生 `num_exited_frames > 0` 时触发（`Echo-Infinity/wan/modules/causal_model.py:799-828`），窗满前与不推进全局 cache 的调用不写；稳态下仍是高频小写（每次 ~3 帧）。Helios 每次驱逐写 E≤9×920=8280 token（368×640，多分辨率桶可变，`helios/dataset/dataloader_history_latents_dist.py:107-113`）压进 M≈720–2760 query，压缩率 3–11×。**写频修订**：dataloader 从 [0, total_sections−U] 均匀采样 `start_section_idx`（`dataloader_history_latents_dist.py:176-186`），晚起点的 U=4 展开有 3 次全真实驱逐，仅绝对起点附近的样本是 ≤2 次真实写——写频/梯度稀疏度是 start-index 分布的函数，非常数；且当前 output dict 不返回 `start_section_idx`（`dataloader_history_latents_dist.py:365-379`），驱逐 mask 无法从绝对 section 位置推导——**数据契约必须新增该字段（并入节 2 的 dataloader 改造），写稀疏与成本假设在 smoke profile 后按实测起点分布重算**。
- **检测信号**：每次写的 ‖ΔM‖/‖M‖ 时序（塌向 0 = 死写，发散 = 不稳）；Enc cross-attn 对写源的注意力熵；Stage B ② 一致性增益缺失但 gate 未饱和 → 指向容量而非门。
- **缓解**：N_Q 上调（A1）；Enc `ffn_mult=4` 保留（降 2 只作显存备选）；9 帧写拆 3×3 帧子写（回退向 Echo 粒度，写开销 ×3 仍 <0.5% 单次前向，节 1 §8）；U 4→6（须先跑语料帧数直方图，节 2 D6）。
- **残余风险**：`bptt_sections=1` 使 Enc 只从紧邻 section 的读回报学习，>1 section 后才兑现的写价值在梯度上不可见；sweep 只覆盖 {1,2}。

### R4 蒸馏 few-step 写入源质量
- **机制（修订）**："distilled σ_last≈0.5 vs Mid σ≈0.001、固定 500× 差距"**不成立为常数**——0.5/0.001 是 pre-shift 排布值（per-stage 原始 sigmas `np.linspace(0.999,0,...)`：`helios/diffusers_version/scheduling_helios_diffusers.py:165-166`；dmd 截断：`scheduling_helios_diffusers.py:244-246`），而 `set_timesteps` 末尾会对 `self.sigmas` 施加 dynamic shifting（`scheduling_helios_diffusers.py:248-256`），拟用 Stage C 父配置正开启 `use_dynamic_shifting: true`（`stage_3_post_self-forcing_version.yaml:128-134`）——transformer 实际消费的是 post-shift σ。定性结论保留：同一 Enc 面对跨数量级的写源噪声差异。
- **检测信号**：分模式 gate 均值，distilled <0.5× Mid 判写源退化（节 3 D3）；capture 逐通道方差对 σ 的标定曲线——**标定与 FiLM 条件一律采用 `set_timesteps` 后调度器实际给出的 σ_last 并逐 forward 落日志，按模式分别建曲线，不用 linspace 端点**。
- **缓解**：FiLM(σ_last) 必选；`memory_capture_extra_t0`（显式 t=0 commit 前向）仅在检测触发后启用——**该路径在两仓库均不存在，其 "+30% 成本" 与"必然击穿 ≤12% 吞吐线"均降级为 INFERRED、待端到端 profile**（现行 distilled launcher 为 3 stage × 2 步：`scripts/inference/helios-distilled_t2v.sh:14-16`；推理循环：`helios/diffusers_version/pipeline_helios_diffusers.py:674-720`），不作为确定性 kill 条件；Stage C 直接在 distilled σ_last 捕获训练（D13）。
- **残余风险**：若 profile 证实 extra_t0 开销不可接受，才讨论 Mid-memory / distilled-no-KV 分裂配置（同为 INFERRED 备案）。

### R5 与 correct.yaml Easy Anti-Drifting：共存，非替代
- **机制**：`corrupt_history_latents` 作用于输入侧 latent history——先以 `noise_corrupt_clean_prob`（=0.1）掷 clean，未中才按 `noise_mode_prob`（=0.8889）在 noise/downsample 间二选（`helios/utils/utils_helios_base.py:310-318`；配置及其自注释 "P(noise)≈0.8/P(blur)≈0.1/P(clean)=0.1"：`stage1_lora_cfr_368_correct.yaml:156-166`，上游依据 `scripts/training/configs/correct.yaml:18-26`）；memory 作用于状态侧驱逐抽象——目标变量不同。真实交互点：Stage B 展开的 loss 前向保留 corrupt（节 2 D7 伪码），若 capture 前向也吃 corrupted history，Enc 学的是"被人为损坏的场景摘要"，而推理 history 无 corruption → 写源训推错位。
- **D3**：capture 前向一律用未 corrupt 的 history（写源干净化），loss 前向保留 corrupt。*弃选：capture 也 corrupt——主动制造写源训推不一致，与 D13 同分布原则矛盾。*
- **检测信号（修订）**：消融 A7 的自变量改为**总损坏概率** `noise_corrupt_clean_prob_history` ∈ {0.1, 0.5}（损坏率 90%→50%，条件 noise/blur 比例 0.8889 保持不动）——原稿把 0.8889 当损坏概率是语义错误，改它只会把 noise/blur 配比从 ~80/10 变 ~45/45 而总损坏仍 90%。显著更优 = 双机制打架；无差异 = 共存成立。
- **缓解/残余**：correct.yaml 三项是主线已付学费的修正（`docs/HELIOS_CORRECT_RECIPE.md:11-31`），v1 绝不移除；"memory 替代 anti-drifting"仅作 Tier-2 实验。残余：corrupt 随机性使 Stage B 各 section 损坏水平独立采样，写-读两通路见到的 history 不一致是常态，需分层统计。

### R6 与 history 离散化（VQ）的先后与交互
- **现状**：OBSERVED ABSENCE（见前置声明），纯路线图排序问题。
- **机制/交互（修订，INFERRED）**：静态代码上 memory 不改写 history latents 张量，但 **rollout 级耦合是双向的**：memory 改变去噪输出，生成 chunk 直接 `torch.cat` 进 `history_latents`（`pipeline_helios.py:1452-1454`）并被切片为后续 section 的 history 输入（`pipeline_helios.py:1294-1298`）——未来 VQ 若作用于该递归 history，其输入分布随 memory 开关而变；反向亦然（VQ 改 patchify 输入分布 → 写源分布 → Enc/gate 全重校准）。
- **D4（排序，理由修正）**：memory Stage A/B 先行；VQ 排在 memory M1 之后、独立分支单独验证；禁止同一 run 首次引入两个未验证机制。排序依据不再是"零副作用"（该说法撤销），而是重校准成本不对称：memory 侧重校准已知且较小（一次 Stage A ≈7.5 GPU·天），VQ 先行则立刻废掉 memory 全部 Stage A 校准。**无论何序，预算里都预留一次联合重校准。***弃选：VQ 先行——重校准代价方向相反且更大。*
- **残余风险**：若主线强推 VQ 优先，memory 预算需预留一次 Stage A 重训（≈7.5 GPU·天，可接受）。

### R7 Wan2.1-14B vs Wan2.2 目标底座
- **现状**：Wan2.2-TI2V-5B 移植已 smoke 验证（load/48ch encode/30 步训练/推理视频，`docs/WAN22_PORT.md:98-103`），但无任何"替换 14B"政策文件（OBSERVED ABSENCE）；主线 lora368 全谱系仍锚定 Wan2.1/Helios-Base（`stage1_lora_cfr_368_correct.yaml:32-48`）。
- **D5**：memory v1 锚定 Wan2.1-14B——VS24 全部基线与 step-5000 锚点（DOVER drift 0.069、dynamic 0.75，`docs/VS24_EVAL_ANALYSIS.md:61-71`）都在 14B 上；Wan2.2-5B 仅作机制级 smoke 沙盒（梯度连通/形状/BPTT 正确性，单节点、约便宜 3×）。*弃选：在 5B 上跑全课程再迁回——dim 5120→3072、40→30 层、VAE 16×（`docs/WAN22_PORT.md:6-29`），gate/amp 校准不可迁移，等于白做。*
- **缓解**：模块全部维度从 config 推导（节 2 D2 单一构造字典，与 D8(a) 全秩组件原则同构），未来切底座代码零改、只重训。**残余**：主线若在 Stage C 前切底座，约 104 GPU·天的 C 段投入重做——由协调条款 C5 冻结该决策。

## 2. 消融矩阵

成本口径：单位 = H200 单卡·天。实测锚点：1 节点 bs2 throttled 13.06 s/it、4 节点 gb64 ≈28 s/it、bs4 峰值 ≈90 GiB（`docs/STAGE1_SPEED_OPTIMIZATION.md` §3 杠杆表）。课程基价（节 2 §8，全 INFERRED，smoke profile 后修订）：Stage A ≈180 GPU·h ≈7.5 天；B ≈530 ≈22 天；C ≈2500 ≈104 天（含 D8 的 C0 warm-up）。评测 arm（VS24-long 20 视频生成+全指标）≈2–4 天/组 [INFERRED]。注："Echo +10.6% 开销"在 Echo 仓库无佐证（节 3 §5 已撤销其代码事实地位），吞吐验收以实测为准。

| # | 变量与组 | 依赖 | 成本(GPU·天) | 判读方式 | 优先级 |
|---|---|---|---|---|---|
| A0 | 读机制 {token 插入, KV 注入} | 2×Stage A | 15 | Stage A ①②③ + FLOPs 实测；并列取 token 插入（t=0 语义完整）。仅在 D1 被质疑时启动 | 备用 |
| A1 | N_Q ∈ {1,3,6} 帧当量 | 3×(A+B) | 90（含默认组） | Stage B ② 增益-开销曲线拐点；full-res 上界仅在胜出组 +1 arm(+30) | **Tier-0** |
| A2 | gate bias {0.75, 2.0} | 2×B（共享 A ckpt，gate bias 重 init） | 44 | gate 均值轨迹 + R1 动态度斜率；0.75 若动态更好且一致性不降 → 确认稀疏大写重标定 | **Tier-0** |
| A3 | 写入源层深 {40 末层, 30, 20} | 3×A 短程(2k 步) | 12 | flow loss 劣化 + 读注意力质量(D10 诊断路径) + 线性 probe(capture→驱逐帧 latent 的 R²) | Tier-1 |
| A4 | amp_mem {on(init −4), off(无参数)} | 2×B | 44 | memory 注意力质量分布 + ②；off 无损则删参简化 | Tier-1 |
| A5a | 去 Stage C（B ckpt 直接 rollout 评测） | 评测 only | 3 | 先验 #5 的裁决实验：B 劣于 no-KV 而 C 翻正 = 课程论成立 | **Tier-0** |
| A5b | 去 Stage A（直接 B 联训） | 1×B | 22 | 对照 A+B：验证 warm-start 必要性 | Tier-2 |
| A5c | Stage C 后 extra_t0 开关（distilled，先 profile 后启用，R4） | 评测 only | 3 | distilled gate 均值 + C-⑤ 吞吐实测 | Tier-1 |
| A6 | 推理消融 {on, no-KV, frozen-M₀, shuffled-M, **is_keep_x0=False×{on,no-KV}**} + eventswitch 切换策略 {no-reset, reset, recache, decay} | 评测 only（共享 C ckpt） | 12–17 | 节 3 §6/§7 全套斜率 + C-⑥；x0 arm 分离 anchor×memory 交互（R1，走 `pipeline_helios.py:906` 现成参数） | **Tier-0** |
| A7 | `noise_corrupt_clean_prob_history` {0.1, 0.5}（R5，总损坏率 90%→50%，mode 配比不动） | 1×B 加组 | 22 | 双机制交互判定（见 R5） | Tier-2 |
| A8 | bptt_sections {1,2} | A1 胜出组 +1 arm | 22 | ② + ‖ΔM‖/‖M‖ 稳定性 | Tier-1 |
| A9 | RoPE 槽位 {分数 id, mid-tier 池化频谱, id=0} | 3×A 短程 | 12 | Stage A 即可判读（读注意力质量 + flow loss），胜者进 B | **Tier-0** |

Tier-0 合计 ≈ 主课程(134) + A9(12) + A1 增量(≈60) + A2 增量(22) + A5a(3) + A6(15) ≈ **246 GPU·天 ≈ 5.9k GPU·h**，与节 2 预算（4.5–5.5k）同量级、略超——A2 可与 A1 胜出组串行裁剪至 ~5.3k。
**D6（消融纪律）**：全部 arm 单变量、共享 seed/prompt 集、判读一律斜率优先；v1 评测一律单卡或数据并行，禁用 CP（D9）。*弃选：全因子 sweep——A1×A2×A8 即 12 arm ≈260 天，超总预算。*

## 3. 分阶段 Kill Criteria

- **K0（实现前，第 0–1 周）**：D1/D8 仲裁未签字；或 **D8(d) checkpoint-conversion 测试未通过**（完整 A/B memory+读路径 state 载入 Stage C 构造器，missing/mismatched key 即 fail）；或 **memory 开启下 `dmd_num_latent_sections_min>=4` 断言与最短 rollout Enc/gate 梯度非零测试未落地**（R2，cold-start 分支 `train_helios.py:1571-1581` 一并覆盖）；或 Stage B smoke（1 节点 bs2×U=4，显存参照 bs4 单 section ≈90 GiB，`docs/STAGE1_SPEED_OPTIMIZATION.md`）OOM 且降 U=2 仍不可行 → 不进入训练，回炉模块设计。凡 kill 均产出验尸报告归档 `logs/`。
- **K1（Stage A 末，预算 ≤15 GPU·天含一次返工）**：节 3 Stage A ①②③（flow loss 劣化 ≤+1%、memory 读注意力质量 >1%、91s DOVER 回退 ≤0.02）。**D10（前置交付物）**：现行两条 attention 路径都只返回 attended tensor、不暴露注意力概率（主训练 dispatch：`helios/modules/helios_kernels/attention_dispatch.py:132-167`；diffusers processor：`helios/diffusers_version/transformer_helios_diffusers.py:141-157`）——须实现诊断专用采样层路径（采样 3–5 层，fp32 下由 Q/K 显式 softmax 计算 current-query→memory 质量，明确 query/layer/head 聚合定义，不替换生产注意力），与 [NEW] 时序脚本同列 Stage A 期间交付；**未就绪则 1% 门从 K1 移除，K1 仅以 flow loss + DOVER 判读**。一次修复迭代后读通路仍死 → kill，保留 no-KV。*弃选：全局换 eager attention 取概率——训练吞吐不可接受。*
- **K2（Stage B 末，预算累计 ≤70 天含 A2 返工）**：gate 均值 ∈[0.05,0.6]；subject-consistency ≥+0.005 且 boundary-LPIPS-excess ≤−5%（vs no-KV）；R1 冻结检测不触发。gate 双端饱和且 bias 重调无效 → kill；冻结触发且 memory-dropout+降 bias 无效 → kill。
- **K3（Stage C 末，预算累计 ≤130 天；C 不重跑，一次即裁决）**：节 3 C-①..⑥，其中 **drift 判据改为配对相对口径**：与同 `checkpoint-19500`、同 seed/prompt 的 no-KV 基线（C2 重跑产物）对比，memory-on 的 DOVER drift 不劣于该基线置信区间；**0.069 仅作历史参照**——它属另一 campaign 的 step-5000（`docs/VS24_EVAL_ANALYSIS.md:61-71`），而 memory 分支冻结基座是 `_correct/checkpoint-19500`（`stage1_lora_reweight_368.yaml:100-107`；`train_stage1_lora_rwtag.sbatch:29-35`），该谱系无 0.069 实测，绝对阈值会仅因祖先不同而误杀改进模型。其余：saturation |斜率| ≤0.7× no-KV、扰动恢复 ≤0.7×、dynamic ∈[0.55,0.90]、吞吐 ≤12%、eventswitch CLIP 不劣。≥2 项主斜率仍劣于配对 no-KV → 判"TF-only 写门净负"成立（先验 #5 被证实），发布 no-KV，项目转验尸。
- **K4（全局）**：累计 >6k GPU·h，或占用主线节点 >2 周，或 C5 冻结决策被推翻 → 无条件暂停、重新立项。
- **D7（判读前置条件）**：每个 K 点的数据必须来自 [NEW] 时序指标脚本与 D10 诊断路径——Stage A 期间并行落地、Stage B 判读前完成 no-KV 基线校准；未就绪则 K2 顺延而非放行。*弃选：用现有端点均值凑合判读——端点均值正是先验 #6 明令禁止的口径。*

## 4. 与 lora368 主线的资源/检查点协调

- **C1（节点，修订）**：主线 `_correct` 占 5 节点（mc-node01,c-node03–06，`train_stage1_lora_cfr_368_correct.sbatch:3-5`）至 step 22000（monitor 上限 `MAXSTEPS=22000`：`scripts/training/monitor_lora368c.sh:13`）；descendants 中 reweight 复用同一批 5 节点（`train_stage1_lora_reweight.sbatch:3-5`），而 **rwtag 已把 c-node07 整节点独占 pin 死**（`train_stage1_lora_rwtag.sbatch:2-8`，理由即该节点跨节点 NCCL wedge、节点内 NVLink 可用：`train_stage1_lora_rwtag.sbatch:18-20`）——c-node07 **不是**结构性无竞争资源。调度规则改为条件式：memory 单节点作业仅在 rwtag 窗口外使用 c-node07，或以 Slurm dependency 与 rwtag 显式串行；同时提名一个备选单节点（使用前必须先过单节点 NVLink smoke），预算中不得把 c-node07 记为常态空闲。Stage C 需 4 节点 ×3–4 天，与主线排程窗口协商，建议卡在主线两个 5000 步区段之间。*弃选：Stage C 抢占 5 节点主线窗口——主线 monitor 会按名字自动检测并重提（`monitor_lora368c.sh:30-47`），排队冲突双输。*
- **C2（基座 checkpoint 冻结）**：Stage A 分叉点钉死 `_correct/checkpoint-19500`——与 reweight/rwtag 同源（`stage1_lora_reweight_368.yaml:100-107`），memory 结果可与主线 descendants 横比；并在**同一 ckpt** 上重跑一次 no-KV VS24 基线——该基线同时就是 K3 的配对对照组（step-5000 的 0.069 属另一 campaign 谱系，只作锚不作对照组）。*弃选：追最新 ckpt——移动基线毁掉全部消融可比性。*
- **C3（checkpoint 卫生）**：memory run 独立 `output_dir` + symlink 播种模式复用（`train_stage1_lora_rwtag.sbatch:29-35`）；500 步/limit 30 沿用（`stage1_lora_cfr_368_correct.yaml:98-100`）；恢复前必须先落地 D11 的 FORCE_LR 按 param-group 扩展——主线 `HELIOS_FORCE_LR=1` 会把 memory lr×5 压平为单一标量（`train_helios.py:1090-1099`），这是复用主线脚本的第一颗雷。
- **C4（工具链先行合并）**：`save/load_extra_components` 第 5 节、merge 工具与 EMA 的 memory 分支（节 2 D12）、**D8 的全秩组件跨谱系载入路径与 conversion 测试**须在 Stage B 前合入——对主线是纯加法（默认 `is_enable_evolving_memory: false` 门控），可先行合并不阻塞主线；否则 Stage C 与评测无 checkpoint 可消费。
- **C5（决策冻结）**：请主线在 memory M1（Stage B 通过）前书面冻结两项：底座不切 Wan2.2（R7）、VQ 不先行（R6/D4）；任一提前变更触发 K4。
- **C6（评测共建）**：[NEW] 时序指标脚本与 D10 注意力诊断按 `external_command` 注册为共同资产——主线 checkpoint 同样受益（现 campaign 的 "drift" 也只是端点差：`make_eval_report.py:132-147`、`summarize_helios_long_ratio_eval.py:92-110`），落地成本由两条线分摊。

（全文已同步写入 `/mnt/beegfs/siyuan/workspace/helios-echo/logs/research/design-risks.md`）
