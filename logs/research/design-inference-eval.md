# 设计任务 3/4:推理集成与评测协议(修订版)

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
