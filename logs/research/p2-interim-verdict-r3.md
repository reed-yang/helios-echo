# P2 r3 判定:rep50 结构化 prompt 三臂重建(2026-07-28)

**结论三条:① 分布内(结构化)prompt 下基线漂移远比 raw prompt 温和——作废的 r1/r2 绝对量级确系 prompt-OOD 伪影;② 未训练记忆在合规协议下依然有害(3/8 case 静态冻结 + 1 case 饱和爆冲)——Stage A 必要性在合规口径下复证;③ pilot@1000 训练后记忆完全消除未训练病理(8/8 motion 存活),漂移指标与 off 基线基本持平——90s 时域上"无害化"已达成,超越基线的收益需更长时域/更多训练步数检验。**

## 协议

rep50 标准结构化 case(id 3000-3007)各取段 0(`<header>/<event>/<role>/<Background>/<style>/<scene>` 全结构体);66 sections = 2178 帧 = 90.75s @384×640;DMD [2,2,2]×3;base-seed 7,三臂同 case 同 seed 配对。24 作业(5725-5748,5739 OOM 由 5749 补发)@ c-node08 + mc-node02 叠加(峰值 12-14 卡并行,用户授权)。

## 臂均值(8 case)

| 臂 | motion 首→末 | 末段冻结 case(<0.05) | sat 首→末 | sat 斜率均值 |
|---|---|---|---|---|
| off | 0.192 → 0.311 | 0/8 | 82 → 76 | −0.137 |
| on-untrained | 0.144 → **0.109** | **3/8**(+1 临界) | 77 → 88 | +0.150(含 case7 爆冲 +2.61) |
| **on-pilot1000** | 0.182 → **0.291** | **0/8** | 75 → 74 | **+0.045** |

逐 case 数据:`results/p2_interim/p2-rep50-r3-*/metrics/*.json`;显著个案——untrained case2/3/5 末段 motion 0.003/0.008/0.001(冻结),case7 sat 97→254(爆冲);pilot1000 全部 case 末段 motion ≥0.147。

## 解读

1. **Prompt 合规性的重要性实证**:off 臂在结构化 prompt 下 90s 漂移温和(sat 82→76 vs raw prompt 的 175→60 崩塌)。backbone 在训练分布内表现良好,此前的剧烈漂移大半是 OOD 伪影——r1/r2 作废正确。
2. **未训练记忆仍是净负**:冻结与爆冲两种失稳并存,与 r2(旧口径)结论一致且更细粒度——记忆读入未训练先验会把生成拖向病态吸引子。
3. **训练 1000 步 = 无害化达成**:训练后记忆把病理全部消除、指标回到基线水位(sat 斜率 +0.045 居中优于 off 的 −0.137,但幅度小,n=8 不宜过度解读)。**净收益的判定要件**:更长时域(>90s,窗口外遗忘真正发生的区间)、更多训练步数(4000 终态)、一致性/身份类指标(当前 motion/sat 无法捕捉记忆的核心价值)。
4. 早段 sat 差异(off 82 vs pilot 75)延续 M₀ 自 section 1 参与读取的已知行为。

## 保留意见

- n=8 case × 1 seed;无视觉 QC;chunk 级 motion/saturation 无法度量身份/场景一致性——记忆的主要预期收益不在本指标组内。
- 训练基座(merged Stage-1 @19500)≠ 评测基座(Distilled)迁移混杂仍在。
- untrained case7 爆冲与 pilot1000 case0 斜率 +0.98 值得人眼复核。

## 产物与过程

- 24 视频 + manifest + metrics:`results/p2_interim/p2-rep50-r3-{off,on-untrained,on-pilot1000}/`
- 预览站(仅 r3 合规结果):`results/p2_site/`,公网 URL 见 `.current_url`
- 过程事故:5739 OOM(mc-node02 xiangbo 进程 63→116GB 动态膨胀)→ 重映射升级为"选最大空余 Default 卡 + 60GB 门槛"后补发全绿;c-node08 第三次 "Kill task failed" drain → resume。
- 相关提交:rep50 支持、余量守卫、站点 r3 化(见 git log 2026-07-28)。

## 下一步

1. 4000 步终态 ckpt 重跑本协议(off/untrained 臂可复用本轮工件,种子配对不变)。
2. 长时域扩展(如 132 sections = 181.5s)与一致性指标(身份/背景 CLIP 相似度轨迹)进入协议 v3 讨论。
3. 每 1000 步 ckpt 例行 8-case on 臂追踪(增量 ~2 GPU·h/次)。
