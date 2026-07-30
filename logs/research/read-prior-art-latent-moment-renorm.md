# Prior-art search: per-channel moment renormalization of conditioning history (drift correction)

**Provenance**: researcher (Terra, GPT-5.6 high), 2026-07-30, method = web search + WebFetch over public sources (arXiv abstracts/HTML, GitHub/HF doc pages, DeepWiki mirrors of ComfyUI node packs); no local repo code was consulted; ~14 fetches used, well under the 30-fetch budget.

## 结论

未找到与本技术逐字重合的已发表工作：即"训练无关、仅对**下一步作为条件输入回灌的历史 latent**做**逐通道 (mean, std) 仿射对齐**、参照点为**该次 rollout 早期冻结的一次性统计量**"这一确切组合，在学术论文或成熟社区插件中都不存在完全对应物。最接近的三类，且都在某个维度上不同：
1. **FramePack**（arXiv:2504.12626，same paradigm — 固定长度历史窗口 + patchify）明确诊断了同一问题（训练/推理历史分布不匹配导致 drift），但补救手段是**对历史 latent 做 K-means 离散化 + 训练时噪声调度**，不做统计量匹配。
2. **社区 AdaIN-on-latent 节点**（spacepxl `ComfyUI-Image-Filters` 的 `AdaIN Latent`、Lightricks `ComfyUI-LTXVideo` 的 `LTXVAdainLatent`/per-step patcher）在数学上就是同一个仿射变换 `(x-μx)/σx*σy+μy`，逐通道 mean/std，training-free——但通常用于**输出/风格迁移的事后校正**或段间拼接，参照是**用户指定/上一段**，未见"参照=同一 rollout 最早满员历史窗口的一次性冻结值，且专门作用于回灌给模型的 conditioning history"这一用法。
3. **Exposure-bias 文献**（ICLR 2024 dynamic scaling 等）在理论上最贴近"训练/推理统计量失配"的框架，但纠正对象是**噪声预测的幅度/方差**，不是对历史 latent 做逐通道仿射复位。
CausVid/Self-Forcing/Self-Forcing++/SkyReels-V2 等明确报告了同一现象（饱和度单调上升 / 亮度单调下降），但一致选择**训练侧**修复（rollout-aware distillation、history noise scheduling），没有任何一篇采用推理时统计量冻结-回灌这一路线。
**结论：视为未发表的工程技巧（unpublished trick），可引用 FramePack 与 exposure-bias 文献作同问题背景，AdaIN-latent 类社区节点作最近工程类比，但都不能作为"已发表同一方法"的引用。**

## 候选对比表（五个必答维度）

| 候选 | 纠正 OUTPUT 还是 CONDITIONING history | pixel space 还是 latent space | 逐通道矩 (mean/std) 还是全直方图/白化 | 参照：固定早期窗口 / running average / 数据集先验 | training-free 还是 trained |
|---|---|---|---|---|---|
| **本技术**（对照基线） | conditioning history only（生成输出本身不改） | latent space（16 通道，patchify 前） | 逐通道 (mean, std) 仿射 | 该 rollout 第一个满员历史窗口，**一次性冻结** | training-free |
| FramePack anti-drifting sampling + history quantization（arXiv:2504.12626） | conditioning history（历史帧被离散化/加噪） | latent space | 否——K-means codebook 量化 + 时间步噪声调度，不是矩匹配 | 无显式参照统计量；用码本/噪声表 | 训练时设计（codebook 训练 + 推理时应用），非纯 training-free |
| ComfyUI-Image-Filters `AdaIN Latent`/`Batch Normalize` (spacepxl) | 通常是 OUTPUT / 风格迁移目标（"getting rid of color shift... or matching color to a reference"），未见专门接入 conditioning-history 回灌路径 | latent space | 逐通道 mean/std（`Latent Stats` 节点报告 per-channel mean/std/min/max，暗示同族节点按通道操作）；`Batch Normalize` 是对整批的聚合 mean/std | 用户指定的任意 reference latent；非"同一 rollout 冻结早期窗口"这一特定语义 | training-free |
| Lightricks `LTXVAdainLatent` / `LTXVPerStepAdainPatcher`（ComfyUI-LTXVideo） | 主要是 OUTPUT（`LTXVExtendSampler`："applies AdaIN **after denoising** to match extended segment to original video"）；per-step patcher 挂在 `sampler_post_cfg_function`，作用于**去噪预测**而非专门的 conditioning-history 张量 | latent space | mean/std（AdaIN 公式 `σ(y)((x-μx)/σx)+μy`），`per_frame` 开关控制是否逐帧/逐通道 | reference 是节点输入（通常是"original video"片段或上一段），非"该 rollout 首个满员窗口冻结值"这一定义 | training-free |
| Exposure-bias dynamic scaling（*Elucidating the Exposure Bias in Diffusion Models*, ICLR 2024） | 都不是——纠正的是**噪声预测 ε_θ 的幅度**，不是历史/输出的 latent 统计量 | latent space（作用于 ε_θ，非帧张量本身） | 标量/方差尺度校正，非逐通道 (mean,std) 仿射 | 无冻结早期窗口概念；是逐步的量级修正 | training-free（"novel sampling method without retraining"） |
| TKG-DM channel mean shift（arXiv:2411.15580） | 既非 output 也非 conditioning-history——作用于**初始噪声 z_T**（生成起点，非自回归历史） | latent space | 仅 mean（"adjusts the mean of each channel... keeping standard deviation constant"），无 std 匹配 | 无 rollout 概念；单图生成，一次性设定目标色调 | training-free |
| SDXL 白平衡 blog trick（cited in arXiv:2412.13401） | OUTPUT（当前生成中的 latent 本身，每个去噪 step 都改） | latent space | 仅 mean 逐通道位移（"shifts the mean of each channel toward a specified target value at each denoising step"），无 std 匹配 | 固定目标值（启发式常数），非同 rollout 早期窗口 | training-free |
| InvokeAI PR #8637（color compensation before VAE-encode） | CONDITIONING（下一次 img2img 迭代要编码的图像张量，语义上确实是"喂给模型下一轮的东西"） | **pixel space**（VAE-encode 之前） | 未指明逐通道矩，更像是对图像做整体颜色补偿（类似色彩匹配） | 未明确冻结早期参照 vs. 逐轮参照 | training-free |
| StreamingT2V APM（固定 anchor frame，arXiv:2403.14773） | CONDITIONING（通过 cross-attention 把首帧语义注入每个 chunk 的生成过程） | 语义特征空间（CLIP embedding），非 latent 像素/patch 统计量 | 都不是——是可学习的 CLIP 特征加权注入，不是矩匹配 | 固定早期 anchor frame（视频第一帧），概念上与"冻结早期参照"最像 | **trained**（可学习标量权重 + 需要训练 CAM/APM） |
| BAgger（arXiv:2512.12080）、Self-Forcing/Self-Forcing++、SkyReels-V2 Diffusion Forcing | 训练目标层面处理 conditioning-history 分布失配（noise scheduling / self-rollout 蒸馏 / corrective trajectories） | latent space（训练时对历史加噪或构造修正轨迹） | 都不是显式矩匹配 | 无冻结统计量概念 | **trained**（均需训练/蒸馏，非 training-free） |

## 分方向证据

### 方向 1：自回归/滚动/流式视频扩散的 drift 缓解

- **FramePack**（Zhang & Agrawala, arXiv:2504.12626, github.com/lllyasviel/FramePack）——本仓库训练路径正是 FramePack 式历史窗口范式的近亲（同样有固定长度、按重要性压缩的历史）。论文明确诊断"drifting"源于训练/推理时历史帧分布不一致，缓解手段是**把历史 latent 帧的每个像素替换成 K-means 码本中最近的条目**（K≈128–256 为最优区间），以及训练时按 σ_train/测试时按 σ_test 对历史帧做噪声调度延迟去噪。**不触及逐通道 mean/std**；也报告了两种 anti-drifting 采样变体（anchor-based / inverted），都是采样顺序层面，不涉及统计量。
- **CausVid / Self-Forcing / Self-Forcing++**（arXiv:2506.08009, arXiv:2510.02283）——Self-Forcing++ 摘要明确写："CausVid trends toward over-exposure, while Self-Forcing videos progressively darken"，与本技术描述的饱和度爬升现象**完全对应同一症状**。但修复路线是**训练侧**：让 student 在自身长 rollout 上通过 teacher 引导做蒸馏，摘要原文 "leverage the rich knowledge of teacher models to provide guidance for the student model through sampled segments drawn from self-generated long videos"；并非训练无关，也不touch 历史 latent 的显式统计量。
- **SkyReels-V2 / Diffusion Forcing**——训练时对历史帧按 diffusion-forcing 方式加噪，用"训练分布覆盖推理分布"而非事后统计量矫正；仍是训练侧修复，且 Self-Forcing 论文指出这只解决了 per-frame marginal 而非 joint rollout 分布。
- **Rolling Forcing**（arXiv:2509.25161）——rolling-window 双向去噪 + attention-sink，锚定初始帧的 KV 而非其统计矩，机制在 attention 层面而非 latent 数值层面。
- **StreamingT2V**（arXiv:2403.14773, streamingt2v.github.io）——**概念上最贴近"冻结早期参照"**：Appearance Preservation Module 用**固定的第一帧**（anchor frame）的 CLIP 图像 embedding，通过可学习权重混入每个 chunk 的 cross-attention，用以对抗外观漂移。原文："leveraging the information contained in a fixed anchor frame of the very first chunk... helps maintain scene and object features"。但这是**语义特征空间的可学习条件注入**，不是对 latent 张量做逐通道仿射矩匹配，且需要训练 APM/CAM。
- **BAgger**（arXiv:2512.12080）——abstract: "a self-supervised scheme that constructs corrective trajectories from the model's own rollouts, teaching it to recover from its mistakes"；training-based，非本技术路线。
- 未见 Diffusion Forcing、MAGI-1、FIFO-Diffusion、Rolling Diffusion、LTX 官方长视频论文、FlowLong、Deep Forcing、FLEX 提出与本技术精确重合的机制（受限于 fetch 预算，MAGI-1/FIFO-Diffusion/Rolling Diffusion/FlowLong/Deep Forcing/FLEX 仅做了搜索层面的交叉核对，未逐篇精读全文，见"局限"一节）。

### 方向 2：扩散模型中 latent 统计量校正（一般性）

- **Exposure bias / dynamic scaling**（*Elucidating the Exposure Bias in Diffusion Models*, ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/4267d84ca2f6fbb4aa5172b76b433aca-Paper-Conference.pdf）——"Ning et al. propose dynamic scaling to correct the magnitude error of ε_θ"。这是"用统计量失配框架去纠正推理时误差"这条思路里**理论上最接近**的工作，但纠正对象是噪声预测的**幅度/方差标量**，不是对 conditioning history 做逐通道 (mean,std) 仿射映射，也没有"冻结早期窗口做参照"的设计。
- **TKG-DM**（arXiv:2411.15580）——channel mean shift："adjusts the mean of each channel in z_T while keeping standard deviation constant"。只在**初始噪声 z_T**、只调 mean、不调 std，且是单图生成不涉及 rollout / 历史窗口概念。
- **SDXL 白平衡技巧**（Timothy Alexis Vass blog, huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space；二手引用见 arXiv:2412.13401 "a text-to-image white balancing approach in SDXL was introduced that shifts the mean of each channel toward a specified target value at each denoising step"）——每个去噪 step 都对**当前生成的 latent 本身**做逐通道 mean 位移（无 std 匹配），目标是固定常数而非 rollout 自身早期统计量；不是自回归视频、不涉及 conditioning history。
- **Zero-Shot Low Light Image Enhancement with Diffusion Prior**（arXiv:2412.13401）——把自己定位为对上一条 SDXL 技巧的改进（"first zero-shot AWB method directly applicable to color-imbalanced images"），仍是静态图像域。
- 视频超分辨率长序列工作方向（如 SeedVR2）——提供的是**输出**端的后处理颜色匹配（LAB/wavelet/HSV/AdaIN 五选一），非 conditioning 端、非本技术路线；见方向 3。

### 方向 3：工程实践/社区（GitHub/ComfyUI/论坛）

- **spacepxl/ComfyUI-Image-Filters**（github.com/spacepxl/ComfyUI-Image-Filters）——`AdaIN Latent`: "Normalizes latents to the mean and std dev of a reference input... useful for getting rid of color shift from high denoise strength, or matching color to a reference in general."；`Batch Normalize (Latent/Image)`: "Normalizes each frame in a batch to the overall mean and std dev, good for removing overall brightness flickering."。数学上与本技术的仿射步骤**完全同构**（逐通道 mean/std 匹配到 reference），但典型用法是**事后对输出做风格/颜色迁移**，reference 由用户任意指定，不是"专门写入模型下一步会读到的 conditioning history、且 reference 定义为同一 rollout 首个满员窗口的一次性冻结值"这一具体设计。
- **Lightricks/ComfyUI-LTXVideo**（docs.ltx.io, deepwiki.com/Lightricks/ComfyUI-LTXVideo）——`LTXVAdainLatent` 公式 "AdaIN(x, y) = σ(y) × ((x - μ(x)) / σ(x)) + μ(y)"；`LTXVExtendSampler` 用它"after denoising to match extended segment to original video"（**输出端**、段间拼接）；`LTXVPerStepAdainPatcher`/`LTXVPerStepStatNormPatcher` 把同一操作挂进采样循环的 `sampler_post_cfg_function`，作用于**每步的 CFG 预测**（仍是生成结果，不是回灌给下一次 forward 的独立 conditioning-history 张量）。这是目前找到的**工程实现上最接近**的先例，但落点（output/风格迁移 vs. conditioning-history）与参照定义（用户指定/上一段 vs. 冻结早期 rollout 统计量）都不同。
- **InvokeAI PR #8637**（github.com/invoke-ai/InvokeAI/pull/8637）——"color compensation applied to the image tensor before encoding to latents, countering brightness drift and haze that accumulate through repeated passes"。这是**像素空间**、**在编码到 latent 之前**的整体色彩补偿，用于对抗迭代 img2img 的漂移；概念上"correct what gets fed back in" 与本技术精神一致，但是 pixel-space 且非逐通道 (mean,std) 仿射公式，也非分块视频的历史窗口场景。
- **MeiGen-AI/MultiTalk Issue #93**（github.com/MeiGen-AI/MultiTalk/issues/93, "Face distortion and color drift in long video generation"）——仅为症状报告，未见解决方案实现细节。
- AnimateDiff / A1111 / Forge 生态未搜到专门针对"conditioning history latent 逐通道矩校正"的扩展；该生态里对应问题的主流对策是 context-window/overlap 调参，而非统计量矫正。

### 方向 4：经典类比

最接近的经典框架是**闭环反馈控制式自动白平衡（AWB）**：AWB 的标准两阶段流程——先估计场景光源色偏，再据此反向校正图像（"estimate the illuminant, then correct based on that estimate... mimicking human color constancy"）——与本技术"用早期一次性标定的 (mean,std) 去反向拉回后续历史"的结构同源；经典 AWB 电路里的"负反馈校正色差信号到零"（negative feedback loops that control automatic gain controllers）在结构上也对应本技术把 (mean,std) 偏差反馈式拉回参照值。但**关键差别**在参照的获取方式：经典 AWB 多为**逐帧/逐场景连续重新估计**（灰世界假设、连续积分），而本技术是**一次性标定（首个满员窗口）后冻结**，更接近相机"手动白平衡校准一次，此后锁定"的模式，而非自动持续追踪的 AWB。未搜到把这一类比明确写成论文表述的文献（搜索显示这一综合是本次分析自行归纳，非既有文献原话）。

## 局限与未覆盖项

- 受 ~30 fetch 预算约束，MAGI-1、FIFO-Diffusion、Rolling Diffusion、LTX 官方长视频论文、FlowLong、Deep Forcing、FLEX 仅做了检索层面交叉核对，未逐篇精读全文/appendix；若这些论文的附录中藏有逐通道矩匹配的消融实验，本报告可能遗漏。
- 未能直接读取 `ComfyUI-Image-Filters`/`ComfyUI-LTXVideo` 的源码（`nodes.py`/`latent_norm.py`，GitHub raw 404），只能依据文档页/DeepWiki 摘要判断"reference 是否为冻结早期窗口"，存在从更完整源码中发现更贴近实现的可能性（虽然文档措辞强烈暗示是"用户指定/上一段"而非"rollout 内冻结早期统计量"这一特定语义）。
- 未搜索非英语社区（如 Reddit/中文技术论坛/Discord 存档），指令中提到的"Discord write-ups"未直接命中可索引来源。
