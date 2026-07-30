# History Projection:训练无关的抗漂移干预(设计契约,2026-07-29)

**结论:8 分钟尺度的漂移是历史 latent 误差的逐段复利,演进记忆治不了它——记忆提供的是 onset 之后的回复力,不是预防。本设计在 history patchify 之前插入一个可调强度的投影算子,把条件历史每段拉回固定支撑集,从源头切断复利。三个算子分别检验三个不同假设,其中"逐通道仿射拉回冻结参照(renorm)"是唯一与实测病症(饱和度单调上冲 = 逐通道统计量漂移)直接对症的候选;FramePack 的码本离散化是论文原式但需训练,免训练使用属于新的未验证变体。**

用户动机(2026-07-29):"8min 推理在接近 3min 左右开始都会出现比较明显的 drift,而且很显然 echo memory 对于 anti-drift 无作用……可以尝试采用类似于 framepack-p1 中对 history 进行离散化的操作,首先 training free 直接离散化;如果有点作用,可以通过 training 的方式把质量救回来"。

## 一 病症与因果定位(证据)

| 事实 | 证据 |
|---|---|
| off 臂 8min 饱和度四分段 67→80→101→**124**,2/5 单调失控至 ~200 | `logs/research/p2-longhorizon-verdict-r5-2026-07-28.md` |
| onset 中位数 209s(定义 a)/168s(定义 b),单 case 方差 109–404s | `logs/research/read-r5-drift-onset-analysis.md` |
| **记忆不推迟 onset**:pilot@4000 onset 173s(5/5 触发)早于 off 209s(3/5);但 0/5 失控、5/5 末段回收 | 同上 |
| full@8000 在 8min 上 +0.297,3/5 失控(劣于基线),而 58s 切换协议优于基线 | r5 判定附录 |

⇒ 记忆是**事后回复力**,与"防止误差进入历史"是两个互补机制。漂移在 latent 侧表现为逐通道统计量偏移(RGB 饱和度单调上升),这决定了算子选型。

**继承的 `AdaptiveAntiDrifting` 为何无用**(`helios/utils/utils_base.py:743-815`,默认 `use_adaptive_anti_drifting=False`,评测从未开启):参照系是 ρ=0.9 的 EMA(记忆约 10 chunk),**慢速蠕变会被参照系一起带走,按构造检测不到**;`and` 双阈值再降灵敏度;"修正"是加白噪声(只抬方差、不复位均值),且作用于 `latents` 而非条件历史 → 污染可见输出(`pipeline_helios.py:1520-1541`)。本设计不复用它。

## 二 插桩点与不变量

插入 `helios/pipelines/pipeline_helios.py:1412-1427`——三个 history tier 切出之后、`prepare_latents` 与 sampler 之前,fp32。

| 性质 | 保证 | 依据 |
|---|---|---|
| 覆盖三个 tier-split 分支(首节 keep-x0 / 后续 keep-x0 / else) | 单点位于 if/else 汇合处 | `pipeline_helios.py:1339-1412` |
| 投影后的历史即 KV cache 缓存内容 | 投影在首个 denoising step 之前 | `transformer_helios.py:224-235,392-397` |
| x0 anchor 不被投影 | 调用方剥离 `short[:, :, :1]`(keep-x0 时 short = [anchor, 1x]) | `pipeline_helios.py:1374-1383` |
| history buffer / 输出 latent / RoPE ids 不被改写 | 只改当段三个 tier 张量 | 审查确认 |
| memory 读写路径不改 | encoder 读 `capture_last_hidden` 而非 latent | `helios/modules/helios_memory.py:139-182` |
| A/B 种子配对不破 | 算子不消费 generator、不改变抽样次数 | 审查确认 |
| 默认严格 no-op | `mode="none"` → `enabled=False`,分支不执行 | `HistoryProjector.enabled` |
| padding 判定是结构性的 | pipeline 用 `torch.zeros` 造 padding ⇒ 只有精确零帧被跳过;统计量判定会误伤合法近常量帧(审查缺陷 ②) | `utils_base.py` `__call__` |
| 效应量可验证 | `num_changed` / `max_abs_delta` 入 manifest;驱动在"跑了但没改变任何值"时报错(审查缺陷 ①) | `run_p2_interim_drift_ab.py` applied-check |
| 实验身份可核对 | dry-run 与真实运行共用 `config_digest_payload`;基线 digest 与投影前一致(实测 `6d72246b…` 不变) | 同上 |

无需改 `train_config.py` 或 50 个 stage YAML(纯推理期);评测驱动确认走 training-side pipeline(`run_p2_interim_drift_ab.py:457-505`),`pipeline_helios_ode.py` 与 `helios/diffusers_version` 无需同步。

## 三 三个算子(同一 α 旋钮:`z' = (1−α)z + α·op(z)`,α=0 严格 no-op)

| 算子 | 操作 | 假设 | 先验强弱 |
|---|---|---|---|
| **renorm** | 逐通道仿射,把当段有效历史的均值/std 映射回**首个全真实历史窗口**的冻结参照 | 漂移 = 逐通道统计量偏移,复位统计量即止漂 | **强**(直接对症;零拟合、零新分布) |
| **codebook** | FramePack 原式(arXiv 2504.12626 v3 Eq.6):逐 latent pixel 替换为 K-means 最近质心 | 有限支撑集把训练/推理历史压到同一分布 | 中(论文有消融,但**论文是训练期替换**;免训练 = 新变体) |
| **quantize** | 均匀网格 `round(z/s)·s` | 死区抹掉每段小增量 | **弱(已证伪其机制)**:按 dither 理论,偏置 b<s/2 会让约 b/s 比例像素跳整格,**均值位移原样保留**;实测 19.9% 元素跳一整格(`tests/test_history_projector.py::test_quantize_does_not_remove_a_sub_step_bias`)⇒ 降级为**对照臂**:若它也有效,说明机制不是"统计量复原"而是"任何历史扰动都有效" |

renorm 时序(默认几何):k=0/1/2 历史含零 padding;**k=3 首次全真实 → 只冻结参照**;**k=4 起真正生效**。

## 四 码本事实(已拟合)

`scripts/evaluation/fit_history_codebook.py` → `results/history_codebooks/k{128,256}.pt`。

- 语料 `/mnt/beegfs/siyuan/dataset/human_single_368x640_slice512g/latents`(168,431 `.pt`),随机 200 文件 × 2,000 像素 = 360k 拟合 + 40k held-out,seed 44,文件列表 sha256 `58e979de…`。
- **空间对齐(最大正确性风险,已解决)**:离线编码器存盘前已做 `(z_raw−mean)/std`(`tools/offload_data/get_short-latents.py:104-107,229-232`),loader 原样传递 ⇒ 磁盘上就是模型空间,**不得二次归一化**(误做一次评分 0.5304→0.8853)。
- 语料模型空间逐通道 std 0.65–0.86(聚合 0.823),均值 −0.70…+0.59 —— 不严格等于 0/1 先验(VAE 常量全局、本语料是单人子集)。
- k128:held-out L2 1.237,保留方差 81.9%,128/128 质心被使用;k256:L2 1.135,保留方差 84.7%,256/256 全用。
- **量级警告**:16 维向量典型模长 ≈3.1,α=1 的码本投影平均改动 1.14 ⇒ **约 37% 的扰动**。连续历史上训练的骨干面对此为明显 OOD,必须跑 α sweep 而非只跑 α=1。

## 五 实验矩阵与判据

**判据(取自 onset 分析,替代全程斜率)**
- 主:onset 定义 (a) = 60s 滚动均值持续超 baseline+2σ 达 60s 的首个时刻。待超越:off 209s(3/5 触发)/ pilot@4000 173s / full@8000 204s。
- 副:末段回收比 = 末 10 chunk 均值 / 全程峰值(越低越好)。待超越:off 0.801 / **pilot@4000 0.511(抗漂移参照下限)** / full@8000 0.809。
- 一票否决:motion 塌陷(冻结)、`num_changed=0`、人眼 QC 明显伪影/糊化。

**wave-1 筛选(9 支,8min 协议,off 骨干臂 = 隔离历史干预与记忆)**
| 臂 | case | 目的 |
|---|---|---|
| renorm α=1 | 2, 3, 0 | 主候选:失控 case 能否被压住 + 稳定 case 是否被破坏 |
| codebook K=256 α=1 | 2, 3, 0 | 论文原式免训练上限 |
| quantize s=0.25 α=1 | 2, 3, 0 | 机制对照 |

case 2/3 = off 臂两个单调失控 case;case 0 = off 臂稳定 case(质量护栏)。种子沿用既有配对。约 45 min 单波(9 卡)。

**门槛(wave-1 → wave-2)**:任一配置须同时满足 ① case 2/3 末窗饱和度显著低于 off 同 case;② case 0 的 onset 不早于 off 同 case;③ motion 未塌陷;④ 站点人眼 QC 无明显画质损伤。过门者进 wave-2:α ∈ {0.25, 0.5} sweep + 5 case 全量 + 与 pilot@4000 记忆臂的交互测试(检验"防误差进入 + 事后回拉"是否叠加)。

**训练路径(仅当 wave-1/2 出现正向信号)**:复用 Stage A 冻结骨干配方,把训练期 history 替换为同一算子(这才是 FramePack 原式的用法),用最小步数验证"质量救回"。

## 六 风险与已知未测

- 免训练码本投影是**论文未验证的用法**;若 wave-1 崩画质,只能说明"免训练不成立",不能推翻论文的训练期结论。
- n=3(wave-1)不做统计主张,只做门槛筛选;wave-2 才回到 n=5。
- renorm 的参照取自视频自身早期窗口:若某 case 早期本身已异常,参照会锁死一个坏统计量(现象上表现为全程贴合早期而非改善)。
- 审查未覆盖(缺失测试):pipeline 调用点级集成测试(spy transformer 断言"transformer 收到的历史已投影 + anchor/buffer/RoPE/generator 未被改写")。当前由 CPU 单测 + 驱动效应量护栏 + 人工 file:line 追踪替代。

## 七 与论文的精确对应,以及论文留白处的选择(2026-07-30 补)

论文给定、可精确复现(已实现):① Ω ∈ ℝ^{K×C} 由 K-means 在数据集 latent 上离线拟合;② `Q(F)_p = argmin_k ‖F_p − Ω_k‖₂`,p = 逐 latent pixel 的 C 维通道向量;③ VAE encode 之后、patchify 之前,只作用条件帧,绝不碰加噪目标;④ 论文用法是**训练期**替换。

实现对应:`flat = latents.permute(0,2,3,4,1).reshape(-1, C)` 每行一个 latent pixel;`argmin(‖m_k‖² − 2·x·m_k)` 与 `argmin‖x−m_k‖²` 等价(省略项对每像素为常数,单测钉住);`out = book[index]` 精确取质心行,且 α=1 短路返回投影本身(避免 `x+1.0*(y−x)` 的浮点回舍——码本模式的意义就是历史严格落在有限支撑集上)。

论文留白 → 本设计的选择:

| 留白 | 选择 | 理由 |
|---|---|---|
| K | 128 与 256 均拟合,当扫描维度 | 正文推荐 128、Table 2 用 256,自相矛盾且无 K-sweep 公开 |
| 拟合空间 | 骨干实际消费的模型空间 | 语料存盘前已归一化;二次归一化评分 0.5304→0.8853 |
| 拟合抽样 | 200 文件 × 2000 像素 = 360k 向量 | 算力;质心使用率 128/128、256/256,无死码 |
| x0 anchor | 排除,不投影 | Helios 特有固定锚帧,FramePack 无对应物;它是参照而非累积状态 |
| 零填充帧 | 精确零判定跳过 | 质心是真实 latent 向量,投影零帧等于往空历史注入内容 |

不可复现:其权重与结果(从未发布)、Δ_M/ELO 数字(骨干与数据不同,且其模型带该替换训练)。

## 八 P1 两机制的可移植性边界(2026-07-30)

官方结果页原文:P1 = **Planned Anti-Drifting**("predicts sections that are far away from the next section before generating nearby sections",减少端点**之间**漂移)+ **History Discretization**(减少**跨端点**漂移)。

**Planned Anti-Drifting 不可移植**:它是非因果的(先远后近),与 Helios 流式自回归 rollout 及记忆 k−2 逐出写入语义根本冲突(记忆状态机建立在"只见过去")。本设计只借离散化那一半。发布状态见 `logs/findings.md` 2026-07-30 条(仓库逐文件核实:76 KB、无训练代码、离散化关键词零命中、无 P1 仓库、无第三方实现)。

## 九 实测验证(GPU 冒烟,数字对账)

8-section 冒烟(off 臂 case 2),计数器与结构推算逐项一致:

| 指标 | renorm | codebook k256 | 结构推算 |
|---|---|---|---|
| 应用次数 | 4/8 | 7/8 | renorm k=3 冻结/k=4-7 生效;codebook k=0 全零跳过 |
| padding 帧 | 30/152 | 30/152 | 8×19=152;19+10+1=30 |
| 改变元素 | 4,669,440 | 7,495,680 | 4×(19×16×48×80);122 有效帧×61,440 |
| max_abs_delta | 0.329 | 3.166 | latent std ≈0.94;后者印证 37% 模长扰动 |

两支 `EXIT_CODE=0`;`reference_call=4` 入 manifest。

## 十 判读工具

`scripts/evaluation/drift_onset_report.py`(`--campaign` 可重复、`--baseline` 按 prompt_index 对位出 delta、`--json`、`--bin-seconds`、`--slope-kind theil_sen|ols`)。验收 = 精确复现 `logs/research/read-r5-drift-onset-analysis.md` 的全部数字:onset a 209/173.2/203.5s(触发 3/5、5/5、5/5)、onset b 168.4/178.8/123.8s、回收比 0.8007/0.511/0.8094、失控 2/0/3、零冻结、30s 轨迹逐箱一致;短跑(57.8s < 60s 基线窗)返回 `None` 带原因串。

## 十一 战役清单与"训练救回"配方

已发(2026-07-30):wave-1 筛选 5888-5896 + metrics 5897;wave-1b 强度/K 扫描 5901-5908 + 5909(renorm α=0.5/0.25、k256 α=0.5、k128 α=1 × case 2/3);held-out 泛化 5911-5934 + 5935(rep50 case 8-15 × {off, pilot@4000, full@12000},切换协议)。种子沿用 `base_seed 7 + case`,与既有 `p2-rep50-r5long-off` 逐 case 配对,不重跑基线。

**训练救回配方(wave 出正向信号后的第一发)**:量化历史的 OOD 集中在 history patchify 边界,而读历史的就是 `patch_long/mid/short` 三个 Conv3d(6 个张量),它们已在 `save/load_extra_components` 的可训练额外件机制内(Stage A 冻结之)。⇒ 只解冻这 3 个 conv(±记忆)在量化历史上微调,参数量极小、正对 OOD 位置,且这才是 FramePack 原式的用法。
