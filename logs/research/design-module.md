# HeliosMemory 模块精细架构设计(Design Task 1/4,修订版)

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
