# Memory, drift and stability in video generation models

**Model:** claude-fable-5

### User

我正在撰写文档和笔记，基于之前我们的讨论和全部资料深入详细研究下面的全部内容并梳理成完整的经验结论(有什么需要讨论的可以先讨论，再给结论/知识)：


Memory and Drift

Memory forgetting and visual drifting 是两个问题； 同一问题的两端
[为什么强化 memory 会加速 drift? 反过来也是？]


FramePack 提到一个处理 Drift 很有趣的做法，通过离散化来防止单帧 drift， 让生成的单帧更加稳定(对输入也更加robust)从而防止 error accumulate
[history discretization 具体的论文中的做法以及 motivation，以及如何适用于 helios model?]

Echo 的记忆机制简言之是什么 insight? 怎么实现的超长视频 rollout?
事实上 echo 是 memory 而不是 drift; 为什么他这个 memory 机制/算法 优化能够在超长视频 rollout 中不出现 drift?
简言之如果我要迁移到 helios 上有哪些挑战？chunk-ar vs. frame-ar, echo query on kv cache / helios history on latent frame? 等等

此外 MoC 的意义是什么，应该如何理解？与前面这些工作有什么不同，如何放在坐标轴上一起理解这些工作？

distill 会引入哪些问题，比如更差的 prompt following,更容易 mode seeking?；通过各种算法比如 rCM 能修复哪些问题？

---

### Assistant

资料我已经全部过了一遍：FramePack 论文原文（含 §4.2 History Discretization 与消融表）、Echo-Infinity 全文、MoC 全文、Helios 全文、rCM grounding、小红书那篇误差累积长文、以及你 6 月那份 teacher 可行性评审文档（里面 challenge 1483/1575/1582 等恰好和今天的问题正面相关）。先把几个需要校准的争议点摆出来讨论，再给逐题的结论。

---

## 〇、先讨论：三个需要校准的点

**a) 「强记忆必然加速 drift」是定理还是经验假设？** FramePack 原文的表述其实有明确限定域："any method that mitigates forgetting **by enhancing memory** may also increase error accumulation；any method that reduces drifting **by interrupting error propagation** needs to weaken temporal dependencies"。注意两个限定：它约束的是**通道增益型**记忆增强和**通道衰减型**抗漂移，即把历史→预测的敏感度当作一个标量旋钮来拧的方法。你评审文档里的 challenge 1575 用 Echo 24h 结果反驳「passive memory 无法改变收缩性」的二分——这个反驳成立，但不构成对 FramePack 命题的证伪，因为 Echo 恰恰**不在**该命题的否定域内：Echo 的记忆在保真度意义上是极弱的（4680 token 抽象态 + 门控 EMA），它没有增大通道增益，而是把「遗忘」本身学成了最优。所以正确结论是：**FramePack 权衡在「标量敏感度」假设下是真的，逃逸路径全部来自打破这个标量假设**（下详）。

**b) 「Echo 是 memory 而不是 drift」需要拆成两半。** Echo 的 24h 不崩，我的判断是主要由**平稳化**（unified relative RoPE + O(1) state 结构）交付的，memory queries 交付的是质量/一致性的增量（5s 消融 +0.78 total，240s user preference 从 14% 拉到 72%）。把这两件事混在一起会误判迁移收益：迁移 memory queries 到 Helios 拿到的是一致性/质量项，而 Helios 的 Relative RoPE + 固定 T_Hist 已经把平稳化那一半做掉了。

**c) 离散化与 Frame-Aware Corrupt 机制上是两类东西。** 你们已实测 Frame-Aware Corrupt 在 10min+ 失效，且 Self-Forcing++ §8.2 证伪了「向历史注噪声」一族。但离散化不在被证伪的族里：注噪/腐蚀是**随机增广**（要求模型对腐蚀分布泛化），离散化是**确定性投影**（训练和推理的历史被映到同一个离散格点上，亚阈值偏差在表示层面**字面上不存在**）。这是一个未被你们现有否定证据覆盖、且实施接近零成本的方向。

---

## 一、Forgetting 与 Drifting：两个问题，共享一条通道

首先严格区分：这**不是同一个问题**。Forgetting 是信息量问题——即使历史全是 ground truth，窗口截断也会丢信息，无任何 drift 参与；Drifting 是分布偏移问题——即使记忆无限完美，逐帧生成误差 ε 依然存在，是生成器本身的不完美。说它们是「两端」，准确的说法是：**二者被同一条信息通道（历史条件化路径）耦合，而朴素方法只有一个标量旋钮去调这条通道的增益。**

把 rollout 写成迭代映射 x_t = f(x_{t-k:t-1}) + ε_t，扰动动力学近似为 δ_t ≈ Σᵢ Jᵢ δ_{t-i} + ε_t，其中 Jᵢ = ∂f/∂x_{t-i} 就是「记忆强度」的数学化身。强记忆 = 大 ‖J‖、长依赖跨度 → 伴随系统谱半径 ≥ 1 → 误差指数增长；抗漂移的朴素手段（renoise、mask、压缩、截断）= 压 ‖J‖ → 谱半径 < 1 保证误差有界，但**信号和误差以同一速率衰减** → forgetting。在通道各向同性的假设下，这个权衡是精确的。

但 FramePack 引言里还有一句常被忽略的话，让关系变成非单调："a stronger memorization mechanism can lead to better temporal consistency and **reduce the occurrence of initial errors**; on the other hand, it also memorizes more errors and thus accelerates error propagation **when errors do occur**"。即总漂移 ≈ 初错率 × 传播增益，弱记忆会抬高初错率（历史信息不足 → 生成更不一致），强记忆抬高传播增益。**所以存在内部最优**，而 FramePack 的 K 消融正是这个最优的实证：K=1 无漂移无记忆，K→∞ 全漂移，K=128 是甜点。

「反过来也是」的适用范围要收窄：**只有通道衰减型抗漂移必然加速遗忘**。有三类逃逸不受此约束——①**纠错型**（FramePack endpoint planning / inverted sampling、Resampling Forcing、Self-Forcing）：不衰减通道，改的是训练目标分布或采样拓扑；②**各向异性敏感型**：让模型对语义维度敏感、对误差维度不敏感——前提是能把误差与信号在表示上分开（离散化、学习型写门、稀疏检索都是这条路的不同实现）；③**规模鲁棒型**（小红书文章的核心论点）：LLM 式的、靠训练域覆盖坏分布获得的泛化——prefilling 塞乱码也能接着写，不是因为通道被衰减，而是因为坏历史在训练分布内。这三条路和「标量旋钮」的本质区别，就是后面所有工作的坐标原点。

---

## 二、FramePack 的 History Discretization

**论文原始做法（§4.2，精确复述）**：对预计算的数据集 latent（按 latent 像素展开，每个像素是 C 维通道向量）做一次全局 K-Means，得到 codebook Ω ∈ R^{K×C}。任意历史 latent 帧 F 逐像素量化 Q(F)_p = argmin_k ‖F_p − Ω_k‖₂，然后用 Ω_{Q(F)} **替换所有历史帧**送入模型——训练和推理一致地做。K=1 时历史退化为单色（漂移消灭、记忆归零，"errors do not accumulate but sections become unrelated"）；K→∞ 等价于不量化。实测 K=128 是甜点（对比表里用 K=256），且 K 越大抗漂移越强但段间平滑过渡越难学。

**Motivation 原文逻辑**：drift 的一个成因是训练/推理在**历史帧上的分布差异**；如果历史表示"not sensitive enough to distinguish between the training and inference frames"，drift 就无处发力。离散整数 token 恰好压缩这个 mode gap——旁证是离散 AR 系统（LLM）明显比连续 AR 视频扩散更抗漂移。这和小红书文章第三节的「离散 vs 连续」论证是同一件事的两个表述。

**我的深层解读**：量化是一个带死区（deadzone）的投影算子——在每个 Voronoi 胞内局部 Lipschitz 为 0（亚阈值误差每步被**硬清零**），跨胞则保持码字恒等（内容信息无衰减）。这正是「各向异性敏感」最纯粹的实现：它不调标量旋钮，而是假设**误差是小的连续微偏、信号是大的离散结构**，然后用几何把两者切开。误差累积从连续扩散过程变成离散马尔可夫链——要么清零，要么跳码；跳码是大事件，不会以 0.1% 的速率悄悄积累。消融结果也验证了它逃出了权衡曲线：vanilla+D 拿到与 inverted anti-drifting 并列的 rank-1 ELO（1224 vs 1220），但 **dynamic 保在 91–93**，而 inverted 一族被明确指出 dynamic range 偏小（约 75–89）——即它抗住了漂移却没付出「冻结运动」的经典代价。

**如何适用于 Helios 系统（Wan2.2-TI2V-5B 底座）**：

作用点很自然：在 Multi-Term Memory Patchification **之前**对 X_Hist 做量化，训练推理一致。与你们现有结构的关系值得注意——Multi-Term 的粗 patchify 核本身就是一种**软离散/低通**（老帧 (4,8,8) 核抹掉高频，误差的高频分量随之被抹），FramePack 的渐进压缩哲学与之同源；VQ 补的是 patchify 补不了的那一段：**低频通道上的连续微偏**（色偏、曝光漂、整体统计漂移——恰是 Helios 定义的 Color Shift 和 Restoration Shift 的主要载体）。两者是互补而非重复。

主要风险在 VAE：Wan2.2 是 48 通道、32×32/token 的高压缩 latent，每个 latent 像素承载的信息量远高于 FramePack 实验的 HunyuanVideo（16ch、8×8）。这意味着 (i) K 的甜点必须重标定，大概率要上到数千甚至更高；(ii) 量化残差在解码端被 32×32 放大，人脸/文字这些你们已知的 VAE 薄弱项会最先受损。合理的对冲是**分层 K**：短期槽（(1,2,2) 核那 2 帧）不量化或用极大 K 保运动连续性，中长期槽用小 K 狠截误差通道——误差主通道虽然经短期槽最强，但短期槽的误差存活时间只有一个 section，真正需要截断的是经中长期槽的**慢速累积**分量。这与 FramePack「重要的帧高保真、久远的帧狠压」是同一设计语言。

实施成本几乎为零：预计算一次 K-Means，推理时一个 argmin lookup（可写成小 matmul），符合你们 zero-latency 原则；训练侧只是数据管线改动。建议的 ablation：K ∈ {512, 2K, 8K, 32K} sweep + 你们的 per-chunk motion amplitude 曲线 + LPIPS identity cost + 长 rollout 下 drift 指标的**斜率**（不是终点值）。诚实边界：FramePack 的验证是 30s 尺度的 start-end contrast，10min+ 未验证；且它的底座是 next-section 预测而非 Helios 的 continuation + Guidance Attention，效果需重验，不能直接搬结论。

---

## 三、Echo 的记忆机制与超长 rollout

**一句话 insight**：把 KV cache 的 memory curation 从「手工规则」（窗口截断/离线调度/固定压缩比）变成**端到端可学**——在 AR 生成目标下、在带生成噪声的自产分布上，让模型自己学「驱逐时该往长期记忆里写什么」；同时用统一相对 RoPE 把训练与推理的位置分布彻底对齐。论文对既有方法的核心批评就一句："ignorance of autoregressive generation noise"——手工规则不知道推理时的 KV 是脏的。

机制回顾（细节你已熟，只列迁移相关项）：三层 KV = sink 3 帧（id 恒 0 起）+ memory queries Q（N_Q=3 帧等效 × 1560 token，跨层共享，占位于 local window 之前的 id 区段）+ local window 9 帧；驱逐发生时，**取最后一层的 evicted KV**（"closest to the pixel output, suitable for layer-shared memory"）过 2 层 cross-attention Enc 刷新 Q，再经 sigmoid 门残差写入：Q ← g⊙Q + (1−g)⊙Q̃，gate bias 2.0 → 初始保留 ≈0.88、单步写入 ≈12%；注入时经跨层共享的 W_k^Q/W_v^Q 投成 KV 拼进因果注意力。Stage-1（5s）梯度穿透全部记忆更新，Stage-2（60s streaming）跨 5s sub-clip detach。开销恒定，实测 10.6% 吞吐损失。

**为什么这个 memory 优化能让 24h rollout 不 drift**——拆成三个必要条件：

其一，**平稳性（最关键、最被低估）**。进入 mature phase 之后，每一步的 RoPE id 布局、三层 cache 的尺寸、更新规则全部与绝对时间无关：模型可见的输入特征里**不存在「当前生成到第几帧」这个变量**。长度外推之所以致命（LongLive 4.25 分钟即崩+溢出；∞-RoPE 推理时打补丁仍因 train-test mismatch 掉一致性），是因为 rollout 长度以绝对位置的形式泄露给了模型；Echo 把这个泄露通道从训练到推理一致地删掉后，「第 100 步」和「第 130 万步」对模型是同分布输入——60s 训练已完整覆盖平稳态，24h 就不再是外推，只是同一个时不变映射的更多次迭代。

其二，**state 有界且收缩**。唯一随时间演化的对象是三层 state：local window 每 9 帧自动冲刷（误差若不被写入 Q，存活期 ≤9 帧）；Q 是门控 EMA——单次污染以 ~0.88ⁿ 几何衰减；且 Q 本身是 4680 token 的强压缩**抽象**（身份、布局、风格这类慢统计量），高频误差分量在压缩中天然被丢弃。误差传播算子按构造收缩。

其三，**写门在自产分布上训练**。causal-forcing DMD + streaming tuning 让 Enc/gate 见过带生成噪声的 evicted KV，学会「过滤生成噪声、保留生成相关信息」——这是它与 handcrafted 方案的机制性差异，也是为什么它的记忆增强没有落入 FramePack 权衡的原因：**它增强的不是通道增益，而是通道的选择性**。

诚实边界必须写进笔记：24h 是定性展示（论文自认现有量化协议撑不起小时级评测）；高动态场景在小时级仍降稳（Limitation (2)）；底座是 1.3B/480p/16fps，「质量」的 bar 低；最重要的——**4680 token 的抽象态做不了精确长程回忆**（场景 revisit、遮挡后精确身份、叙事状态）。在 FramePack 的语义里 Echo 依然站在 forgetting 一侧，它只是把 forgetting 做成了最优学习问题。它解决的是「视觉统计不漂」，不是「内容记得住」——后者是 MoC 那条路线的地盘。

**迁移到 Helios 的挑战（逐项）**：

第一，**写入源缺失**，这是最实质的架构错位。Helios 无 KV cache，历史以 latent 帧形式每步重新 patchify 编码，「驱逐」在 Helios 里只是「下一步不再编码」，不存在 Echo 依赖的 last-layer evicted KV。候选方案：(a) 用 patchify 后的 layer-0 embedding 做写入源——干净、便宜，但特征太浅，丢掉了 Echo 特意选 last-layer 的抽象性；(b) 直接用当前步历史 token 的深层 hidden states——**不可行**，因为 Helios 的注意力是 [Q_Noisy, Q_Hist] 对 [K_Noisy, K_Hist] 的全双向，历史 token 的深层表示被当前去噪状态污染；(c) 我推荐的方案：**缓存每个 section 最终去噪步（t≈0）时 X_Noisy 的 last-layer hidden states**——这些帧即将滚入历史、最终被驱逐，其 t≈0 深层特征是 Echo「last-layer clean KV」在 Helios 语境下的天然对应物，只需一段逐 section 的 transient cache，成本可忽略。

第二，**注入点与位置编码**。注入走 Guidance Attention 的拼接处：[K_Noisy, K_Hist·amp, K_Q·amp_Q]，给记忆层配独立的 amp token 让模型学会对抽象记忆的逐头加权；Q 固定 timestep=0 当 clean history 处理。Helios 的 Relative RoPE 已提供平稳性，**不需要搬 Echo 的 rotate 机制**，只需给 Q 留固定 id 槽位——建议放在 first-frame anchor 之后、long-term 之前，语义上是「比 long-term 更久远的抽象层」。

第三，**粒度错位**。Echo 是 frame/chunk 级驱逐、高频小步写入；Helios 是 section 级滑窗（约 20 latent 帧整体前移），写入事件稀疏、每次驱逐帧数多。gate 的初始 bias 需要重调（更新更稀疏 → 单次允许更大写入比例），Enc 的 cross-attention 要吃更长的驱逐序列。BPTT 沿用 Echo 策略：section 内穿透、跨 section detach。

第四，**训练课程是成败关键**。Helios Stage 1 是 teacher forcing、真数据历史、无 rollout——在这个分布上训出来的写门恰是 Echo 批评的「不知生成噪声为何物」的手工规则等价物，推理时会把漂移内容当真写进记忆，重新打开 memory→drift 通道。你们既有的三阶段课程（Stage 1 静态可学 token → Stage 1.5 TF unroll 学语义 → 末端 SF 校准分布）是对的，这里给出它为什么必要的机制理由：**没有末端 SF 校准的 memory queries 是净负资产**。

第五，**双稳定器干扰风险**。Helios 已有 first-frame anchor（即 sink），再加 Q 是两个稳定器叠加——VideoSSM 的教训（fixed sink over-stabilize、压制 evolving context）和 Echo 自己的消融（memory 同时影响 consistency 和 dynamic degree）都指向同一风险：一致性指标上涨、动态度被冻结。验收指标必须绑定 dynamic-degree 随时间的斜率与扰动后再漂移斜率（用你们的 `eval/1_get_motion_amplitude.py` 曲线），不能只看 VBench 终点均值——这也是你评审文档 challenge 1517 的结论，我完全同意。

第六，容量与成本是好消息：Wan2.2 高压缩 VAE 下每帧 token 数远少于 Wan2.1 的 1560，N_Q=3 的 Q 极便宜，可以直接 sweep N_Q ∈ {3, 6}。

---

## 四、MoC 的意义与统一坐标系

MoC 做的事：把长上下文视频生成重述为**内部信息检索**。每个 query 通过与 mean-pooled chunk 描述子的点积做无参 top-k 路由，只对选中的 chunk（+ 两类强制锚：全部文本 token 作为跨模态 sink、intra-shot 局部窗）计算注意力；chunk 按内容边界（帧/shot/模态）切分而非等长切分；路由阶段施加**因果 DAG mask** 防两节点闭环（消融图显示无因果时 shot 9↔11 互路由形成孤立回路，表现为运动停滞和帧重复）；drop-in/out 正则防死路由；逐层逐头独立路由构成 L×H 个路由器的集成。底座是 LCT（3B MMDiT、8-shot 真实长数据、约 180k token），85% 稀疏、7× FLOPs 削减、2.2× 实测加速，且 dynamic degree 反超 dense（0.56 vs 0.46）。

**它与前面所有工作的本质差异在存储制式**：FramePack、Helios Multi-Term、Echo queries、LaCT fast weight、VideoSSM，全部是「**有损压缩进常数 state**」——即在 forgetting 一侧寻找最优安放点，记忆上限被 state 容量硬约束。MoC 是「**无损全量保存 + 学习稀疏访问**」：所有历史的 KV 都在，被压缩的是访问而非存储。代价从「信息丢失」换成了「存储/计算 O(n) 增长 + 检索正确性」。这就是它能跳出 fidelity×sensitivity 权衡曲线的原因——那条曲线只约束有损通道；同时稀疏访问顺手细化了误差通道（每 query 只触 k 个相关 chunk，误差传播路径由相关性路由而非全连接，因果 DAG 又砍掉了反馈环这一漂移放大器）。MoC 的结论句值得原样记下："removing the quadratic attention bottleneck is not just an efficiency gain but a direct path to unlocking **emergent, long-term memory**"——记忆是数据规模 + 检索机制下涌现的，不是模块堆出来的。

同样要记录它的适用域：MoC 是 bidirectional 多 shot joint denoising（LCT 语境）+ AR 采样靠 outer-loop 路由外推（约多撑 2–3× shot 数），**不是 streaming real-time**。它和小红书文章是同一作者（Shengqu Cai）的同一纲领：不在小模型短 rollout 上修补误差累积，而是用真实 GT 长数据 + 底层表示机制赋予基模真正的 long-context（MMM 管分布目标、MoC 管注意力机制）。

统一坐标系（三个维度：历史存储制式 × 抗漂移机制类型 × 推理制式）：

| 工作 | 历史存储 | 抗漂移机制 | 推理制式 |
|---|---|---|---|
| FramePack | 有损·时间衰减压缩（O(1)） | 采样拓扑（endpoint/inverted）+ 表示投影（离散化） | next-section，段内双向 |
| Helios | 有损·三层压缩（≈20 latent 帧 ≈5s） | 分布对齐（Frame-Aware Corrupt）+ 平稳化（Relative RoPE）+ anchor | section 续写，段内双向 |
| Self-Forcing 系 | 窗口/cache | 分布对齐（rollout train-as-infer）+ mode seeking（DMD） | streaming causal，蒸馏绑定 |
| RF / SVI | 各异 | **纠错**（脏历史→干净目标 / error recycling） | streaming causal |
| Echo-Infinity | 有损·学习型常数 state | 平稳化（relative RoPE）+ 收缩 state + 写门在自产分布上学 | streaming causal，real-time |
| LaCT / TTT / VideoSSM | 有损·weight/SSM state | 状态压缩即遗忘 | causal（LaCT 换掉双向注意力） |
| MoC | **无损全量 + 稀疏检索** | 因果路由 DAG + 强制文本锚（防 prompt 漂）+ 真实长数据 | 长上下文 joint，AR 为外推模式 |
| 基模规模（小红书结论） | — | 规模鲁棒：坏分布进训练域 | — |

Helios 在这张图上的坐标与缺口一目了然：它是「有损常数 state + 分布对齐 + 平稳化」的组合，历史只有约 5 秒——缺的一是**超过 5s 的内容记忆**（Echo tier-4 补抽象层是近期解；MoC 式无损检索是远期路线，与你们代码库里闲置的 `restrict_self_attn`/KV cache 基建正好有交点），二是**纠错型机制**（当前完全缺位，Frame-Aware Corrupt 属于对齐型且已被你们实测证明不够）。

---

## 五、蒸馏引入的问题，以及 rCM 能修什么

蒸馏引入的问题按层拆开：

**分布层（mode seeking）**：DMD/GAN 的 reverse-KL 是 mode-seeking 的——rCM 论文实测 DMD2 出现位置/朝向坍缩式的多样性丢失；Helios 观察到蒸馏模型普遍过饱和；SF 的 ODE 初始化带来 dynamic degree 下降（Echo 消融原话 "the expected dynamic-degree drop due to its ODE limitation"）。**Prompt following 掉点**的主要来源有两个：一是 mode seeking 砍掉低密度模式，而罕见 prompt 组合恰是低密度尾部；二是 CFG 蒸馏——Helios Stage 3 明确 "eliminating the need for CFG"，把 guidance 以固定尺度内化进权重，语义对齐的可控裕量随之消失。Helios 数据佐证：Distilled 的 Semantic 从 Base 的 6 掉到 5，换 Self-Forcing 训练进一步掉到 4。

**步数层**：sCM 这类前向散度方法自带另一种误差累积——一致性函数学的是教师速度场的积分，误差沿 t 从小到大传播并被 sin(t) 加权的 JVP 自反馈放大，表现为小字渲染崩、纹理糊、跨帧几何不稳（这是 clip 内的「drift」，与 rollout drift 结构同源）。few-step 还意味着语义决策在首步冻结、无迭代纠偏机会；1-step 视频至今是糊的（rCM 也需要 ≥2 步）。

**训练动力学层**：fake score 联合训练的估计偏差会把梯度带偏——Helios 消融里 "w Decouple DMD" 直接崩到 5.21。

**AR 特有层**：SF 的抗漂移与训练 rollout 长度强耦合（小红书第八节：训练只 rollout 5s，超过即外推，误差分布完全不同）；rollout 上做 mode seeking 会偏向「安全延续」，prompt 遵循随时间单调衰减——Echo 论文 Table 2 里所有方法的 CLIP score 从 0-10s 到 50-60s 无一例外下降，是行业性现象。

一个必须写进笔记的**反直觉点**：mode seeking 本身是天然抗漂移的。漂移的本质是离开数据流形，reverse-KL 把样本拉回教师高密度区内部，恰好远离漂移所在的 OOD 区——这正是 SF+DMD 在显存约束下还能撑住长 rollout 的深层原因（小红书也点到 "reverse KL 有很多 benefit"）。所以蒸馏的净效应是「**drift 变好、coverage 变差**」：Helios 大表里蒸馏系的 Drifting 分数普遍不差（7–9），受压的是 Semantic 和 Naturalness。抗漂移与保多样，在蒸馏目标层面又是同一条权衡的两端。

**rCM 修复的与修不了的**：rCM 的构造 L = L_sCM + λ·L_DMD（λ=0.01）是把 mode-covering 的前向散度当主干（保多样、保语义覆盖、贴教师轨迹），把 mode-seeking 的反向散度降格为弱正则（只负责修 sCM 的细节质量问题）。实效：修掉了 sCM 的纹理糊/小字崩/几何不稳，同时保住多样性——4-step 下 GenEval 0.83 逼近教师、VBench 超 480p Wan 教师，对比 DMD2 有明确多样性优势。**这直接回答了「prompt following 掉点是否蒸馏必然」：不是必然，是纯 reverse-KL 路线的必然**；λ 是显式的质量-多样性旋钮。rCM 修不了的：1–2 step（视频 1-step 仍糊）；多样性与物理一致性不严格回到教师水平；以及 rollout drift——rCM 是 clip 内蒸馏，它对 AR 的贡献停留在 future work 那句话："forward-divergence distillation by teacher forcing with a causal teacher could complement Self-Forcing's reverse-KL approach to enhance diversity and motion dynamics"——这正是你们「causal CD 初始化 → joint rCM（sCM+DMD）→ 短 SF 校准」管线的理论出处：rCM 管 clip 内的散度平衡，SF 尾段管 rollout 分布校准，教师端的 Easy Anti-Drifting/记忆机制管漂移根源，三层各司其职。

---

## 六、总括：一个统一的心智模型

把五个问题放在一起，会发现同一个结构在三个尺度上重复出现：

clip 内尺度上，是前向散度（mode covering，保多样保语义，但误差沿 t 累积致糊）与反向散度（mode seeking，锐利抗漂但坍缩）的混合——rCM 用 λ=0.01 找到甜点；rollout 尺度上，是 mean seeking 与 mode seeking 的混合——Cai 的 MMM 直接以此命名；记忆尺度上，是保真（记住一切，含误差）与收缩（遗忘一切，含信号）的混合——FramePack 用 K、Echo 用 gate 找甜点。**三个尺度上的经验结论完全同构：两个端点都不可用，最优解都是结构化混合；而一流工作与二流工作的分野，在于混合是「调一个标量」还是「做成各向异性/可学习/有分工的」**——离散化靠几何切开误差与信号，Echo 靠在自产分布上学写门，MoC 靠无损存储 + 稀疏检索干脆跳出有损权衡，规模路线靠把坏分布纳入训练域。Helios 下一步的每个候选（离散化 ablation、Echo tier-4、joint rCM）都可以在这个框架里说清自己动的是哪个尺度的哪个旋钮、以及为什么不是在拧标量。

行动上的三个具体建议：离散化是零训练成本的先手 ablation（分层 K，绑 motion amplitude 曲线和 LPIPS）；Echo 迁移采用「section 末次去噪步 last-layer hidden states」作写入源、且末端 SF 校准不可省；所有记忆/抗漂移实验的验收指标统一换成斜率三件套（error/consistency/dynamic-degree 随时间的斜率 + 扰动后再漂移斜率），终点均值一律只作参考。

有哪一块想展开成独立笔记（比如离散化的 Wan2.2 适配实验设计，或 Echo tier-4 的完整训练配方），我可以继续往实现层面细化。

---

### User

我现在已经有 helios-distilled 版本会在 1min 尺度内就进行 drift; 你认为这个 context/history 离散化是 training-free 还是需要轻量化 train 比如 lora training? 具体应该怎么做，能否先在这个 drift 明显的 model上验证能否减缓 drift

---

### Assistant

这个问题机制上可以拆得很干净：**离散化的抗漂移效果由两个独立成分构成，一半天然 training-free，另一半才需要训练**。先把这个拆解说清楚，结论和实验设计就自然出来了。

## 机制拆解：哪一半不需要训练

回到递推式。加了量化后，系统变成 x_t = f(Q(x_{t-1:t-k})) + ε_t，条件输入永远活在码本格点上。

**成分一：deadzone 截断（投影算子的性质，与模型是否见过量化输入无关）。** 只要某帧的漂移偏差 δ 没有越过 Voronoi 胞边界，Q(x+δ) = Q(x)——模型收到的条件输入和无漂移情形**逐 bit 相同**，历史通道上的误差传播被物理切断。更细一点：如果模型每步有一个系统性偏置 δ（比如输出总比历史略饱和），无量化时它会被历史通道积分成线性增长；有量化时，只要 δ < 半胞尺寸，下一步的条件仍是原码字，偏置就**饱和在 δ 而不再增长**——积分器被拆掉了，剩下的只有「偏置大到跳码」的棘轮式跳变，而跳码阈值由 K 控制。这一整套性质在推理端外挂就成立。

**成分二：分布匹配（这一半才需要训练）。** 模型没在量化历史上训过，Q(x) 带有它没见过的量化伪影（posterize 式的表示偏移）。这不会引入新的**累积**项（伪影分布是平稳的、不随 rollout 长度增长），但会引入一个**常数级的每步条件质量代价**：段间衔接变糙、细节延续性变差、identity 微抖。训练要消除的就是这个常数项，顺带允许你把 K 压得更低（更强的 deadzone）。

所以直接回答你的问题：**training-free 版本理论上能拿到「斜率变平」，代价是可能付出一个常数级的质量偏移；轻量训练（LoRA）负责消掉常数项并解锁更小的 K。** 用你手上这个 1 分钟内就明显 drift 的 Helios-Distilled 先做 training-free 验证，不仅可行，而且是信息量最大的第一步——drift 明显意味着效应量大、信号快，且这个实验单独隔离了成分一。另外有两个先例支持 training-free 的可行性：Helios 论文自己的 adaptive sampling 就是「推理时对历史做 training-free 扰动」（RGB 统计触发的 targeted Frame-Aware Corrupt），证明这个模型架构吃得下推理端历史扰动；而且模型训练时见过 Frame-Aware Corrupt 的加噪/降采样腐蚀，量化残差在结构上与这族腐蚀部分重叠，模型对它有先天鲁棒性。需要诚实标注的是：**FramePack 论文里所有 +D 变体都是训练版，training-free 一臂文献中没人测过**——所以这个实验本身就有独立的信息价值，无论结果正负。

一个对你有利的事实：如果你说的是官方放出的 Helios-Distilled（Wan2.1-14B 底座），那它用的是 Wan2.1 VAE——16 通道、每 latent 像素对应 8×8 像素——和 FramePack 实验的 HunyuanVideo latent 几乎同规格，**K=128–1024 这个区间可以直接参考**，不存在我上一条里说的 Wan2.2 48ch/32×32 重标定问题（那是你未来 5B 系统才要面对的）。

## Phase 0：training-free 验证的具体做法

**第一步，建码本（几小时）。** 从训练分布采 1–2k 条视频，用 Wan2.1 VAE 在 384×640 编码，把 latent 摊平成 16 维像素向量，采 ~5M 个样本，minibatch K-Means（k-means++ 初始化）分别拟合 K ∈ {64, 128, 256, 1024, 4096}。关键：**码本必须拟合在 GT latent 上，绝不能用生成 latent**——整个机制的语义就是「把推理历史投影回训练流形的格点」。先用原始 L2 空间，不做 per-channel 标准化（FramePack 就是裸 L2，先对齐它）。

**第二步，推理端挂钩（一天）。** 在生成的 section latent 进入历史 buffer 之后、Multi-Term patchify 之前插入量化：flatten 成 (N,16)，对码本做 chunked matmul 求 argmin（‖x‖² − 2xΩᵀ + ‖Ω‖²），替换为质心。每帧入 buffer 时量化一次即可（之后它已在格点上，重复量化是恒等）。注意 patchify 在量化之后跑：粗核对格点值做卷积后输出仍是连续值，但输入级的「同码字→同 token」性质保住了，这就是 deadzone 需要的全部。

**第三步，实验臂设计。** 主臂：量化全部**自生成**历史（短/中/长三槽都量），first-frame anchor 保持连续不量化——anchor 是干净的全局色彩/身份参照，量化它只有损失没有收益。消融臂：只量短期槽、只量中长期槽、anchor 也量化的完整版、以及软量化 x′ = x + α(Q(x)−x)（α=0.5，把硬 deadzone 换成向格点的收缩，伪影更轻但收缩率打折）。另外强烈建议加一个**廉价竞争基线**：对历史 latent 做 per-channel 统计重归一化（AdaIN 式，把近期历史的通道均值/方差拉回 anchor 帧统计）。如果 AdaIN 单独就能压平你的 1 分钟 drift，说明漂移是纯全局统计量级的，VQ 是杀鸡用牛刀；如果 VQ 显著优于 AdaIN，说明它截住了结构化漂移分量——这个对比决定后续投入方向。还有一个必须控制的变量：如果推理代码里有 adaptive sampling（RGB 统计触发的推理时腐蚀），实验臂里先关掉它，否则两个机制混在一起归因不清。

**第四步，度量。** 全部用轨迹和斜率，不用终点：Helios Fig.6 那套 saturation / RGB mean / RGB std / aesthetic 随时间曲线；你们的 per-chunk motion amplitude 曲线（盯 motion 冻结这个经典代价）；段边界相邻帧 LPIPS（盯量化伪影导致的 stitching flicker——这是成分二的常数代价最可能显形的地方）；start-end contrast Δ_drift。rollout 拉到 3–5 分钟而不只 1 分钟，因为你要看的是斜率是否变平，1 分钟刚过拐点可能分不出来。Prompt 集用 HeliosBench 子集 32 条 × 2 seed 足够出结论。

**判定规则。** 三种结局：(a) 某个 K 下色偏/restoration 类曲线斜率显著变平，且 motion amplitude 和边界 LPIPS 代价可接受 → deadzone 成立，进 Phase 1；(b) 曲线变平但边界闪烁/细节延续明显变差 → 成分一有效但成分二的常数代价太大，**这恰是「需要轻量训练」的判据**，同样进 Phase 1，且预期训练收益明确；(c) 所有 K 都不改变斜率 → 说明你这个 1 分钟 drift 的主通道不经过历史 latent 的连续微偏（可能是位置编码、蒸馏本身的 exposure、或跳码级的大偏差），VQ 方向降优先级。

## Phase 1：如果要训，怎么训

训练目标是让模型把量化历史当母语读。配方直接镜像 Helios 自己的做法：**patchify 输入投影（patch_short/mid/long）全参数 + DiT LoRA（rank 128）**，continuation 格式 teacher forcing，dataloader 里对历史槽施加同一个 Q，数据预算不用大——这是输入分布的小幅适配，不是能力训练。

但有一个重要的路线选择：**不要直接用 flow matching loss 去 tune 蒸馏后的 student**。DMD 蒸出来的 few-step 学生分布很脆，拿普通 FM loss 短 tune 会破坏它的 few-step 行为，你会分不清质量变化是量化造成还是 tune 本身造成。两条干净路径：要么在 **Helios-Base 上做这个 LoRA 验证**（50 步采样评 drift，慢但归因干净）；要么——对你的真实系统而言更本质——**把 Q 直接烧进 Wan2.2-5B 新系统的 Stage 1 dataloader**，让后续蒸馏免费继承（real/fake score 的历史条件也同步量化，训练推理全程一致，这才是 FramePack 论文验证过的完整形态）。Phase 0 的 training-free 结果就是决定要不要走这一步的 go/no-go 信号。

## 两个必须提前想清楚的交互

**与 Frame-Aware Corrupt 的关系是替代而非叠加。** 两者关闭的是同一个 gap（训练/推理历史分布差），机制不同（增广 vs 投影）。如果训练时先腐蚀再量化，小幅腐蚀会被 deadzone 抹掉、白做；先量化再腐蚀，则训练出对格点外历史的鲁棒性，而推理时历史永远在格点上、用不上。所以引入 Q 后应该**大幅削减或移除 Frame-Aware Corrupt**，最多留极小概率作正则——考虑到你们已实测 Corrupt 在 10min+ 无效，这个替换没有沉没成本。

**与你的 I2V 解冻修复有直接冲突风险。** 你的 convergent zoom 方案靠的是在历史里制造**小幅**帧间差分来传递运动信号——而 deadzone 专门吃掉的就是小幅差分。如果 zoom 的逐帧位移量级落在量化胞内，量化后的历史会退回「逐码字全同」，解冻问题原地复发。对策：验证时确认 zoom 幅度跨码字（直接检查 Q(zoom 序列) 的码字是否逐帧变化），必要时首个 section 的历史不量化、或短期槽用大 K。这个检查一行代码，但漏掉会白跑一轮。

总时间预算：码本几小时、挂钩一天、sweep 加评测 3–5 个 GPU 天。如果 Phase 0 出正信号，我可以帮你把 Stage 1 的 Q 集成方案（含 real/fake score 条件一致性和与 rCM 管线的衔接）细化成配方。

---

