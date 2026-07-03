# 按作者建议配置后续训练(correct.yaml / issue #38 完整落地指南)

> 写于 2026-07-03。背景:与 Helios 作者(SHYuanBest)沟通确认,官方训 Helios-Base/Mid 时
> Easy Anti-Drifting 没有完全开启、I2V 格式从未参与训练。上游 `scripts/training/configs/correct.yaml`
> 是作者给出的补救推荐(随 commit `e7b1f92` 加入,回应 issue #38)。本文档给出在**本仓库
> (helios-team)** 落地这些建议的精确步骤,供后续 agent 直接执行。所有 file:line 以当前
> `mid_training_xiangbo` 分支为准;行号可能漂移,请以文中给出的代码锚点(函数名+相邻代码)为准。

---

## 0. 先决事实(已核实,不要重复验证成本)

1. **本仓库代码已包含上游两个关键修复,无需打补丁:**
   - `e7b1f92`(2026-03-20):i2v drop 分支 + t2v/i2v/v2v 互斥控制流。已确认存在于
     `helios/utils/utils_helios_base.py` 的 `prepare_stage1_clean_input_from_latents`
     (i2v 分支在 ~line 697),`train_config.py:234` 有 `random_drop_i2v_ratio` 字段,
     `train_helios.py:1351` 已传参。
   - `156c7a8`(2026-04-09, issue #63):`downsample_corrupt` 的 5D reshape 修复
     (permute 后再 reshape)。已确认本地是修复版(~line 195-196, 209)。
2. **所有历史 run 的现状**(不需要回溯重训,但新 run 应改):
   - `corrupt_history: true` + noise 腐蚀:所有 run 都开着 ✅
   - Blur(downsample 分支):仅 `stage1_init_cfr368(_gb240)` 和 wan22 系列开了
     (`corrupt_mode_history: "random"`);LoRA / full-SFT / repro_wan / video_single /
     continue_ohv 是 noise-only ❌
   - Saturation:**没有任何 run 开过,且 Stage-1 代码路径不通**(见 §3)❌
   - `random_drop_i2v_ratio`:**没有任何 run 设过,默认 0,I2V 格式从未训过** ❌
3. **不要动的东西**(已经是正确状态):
   - `validation_config.use_dynamic_shifting: true` + `time_shift_type: "exponential"`
     (所有 config 都对;training 侧保持 `false` 也是对的)。
   - `downsample_min/max_corrupt_ratio_history: 0.9 / 1.0`(所有 config 已写,值即推荐值)。
   - `is_random_drop: true` + `random_drop_v2v_ratio: 0.4` + `random_drop_t2v_ratio: 0.4`。

---

## 1. 改动一:开启 Blur(noise+downsample 混合腐蚀)——纯 config

**适用:** 所有仍为 noise-only 的 config 及由它们派生的新 config:
`stage1_lora_cfr.yaml`、`stage1_lora_cfr_368.yaml`、`stage1_fullft_cfr.yaml`、
`stage1_fullbase_cfr368.yaml`、`stage1_repro_wan.yaml`、`stage1_video_single_24fps.yaml`、
`stage1_continue_ohv*.yaml`(以及各自 `_smoke`/`_dbg` 变体)。

在 `training_config` 中把:

```yaml
  corrupt_mode_history: "noise"
  corrupt_mode_prob_history: 0.9
```

改为:

```yaml
  corrupt_mode_history: "random"     # 开启 noise+downsample(Blur)混合
  corrupt_mode_prob_history: 0.8889  # P(noise)=0.9*0.8889≈0.8, P(blur)=0.9*0.1111≈0.1, P(clean)=0.1
```

说明:`0.8889` 与本仓库 `stage1_init_cfr368.yaml` / wan22 系列已用的 SPEC 一致
(p_noise=0.8 / p_blur=0.1 / p_clean=0.1)。若想严格跟 correct.yaml(它不设 prob,落到
`train_config.py` 默认 0.9),得到 0.81/0.09/0.10,差异可忽略——**统一用 0.8889** 以保持
仓库内一致。

生效路径(无需改代码):`train_helios.py:1456 → prepare_stage1_noise_input`
(`utils_helios_base.py:740`)→ `corrupt_history_latents`(`:281`),模式选择在 `:313-315`。

---

## 2. 改动二:开启 I2V 格式训练(`random_drop_i2v_ratio`)——纯 config

**适用:所有后续 Stage-1 类 run(全部 config)。**

在 `training_config` 的 `is_random_drop` 区块加一行:

```yaml
  random_drop_i2v_ratio: 0.1   # correct.yaml 推荐值;修 I2V 开头慢动作(issue #38)
```

要点:
- 生效路径已接通:`train_helios.py:1351 → prepare_stage1_clean_input_from_latents`
  (`utils_helios_base.py:697`)。前提 `is_random_drop: true`(所有 config 已满足)。
- **嵌套概率**:i2v 检查在 t2v drop 的 `else` 里,且优先于 v2v。在 t2v=0.4、i2v=0.1、v2v=0.4 下,
  实际样本占比:t2v 40% / **i2v 6%** / v2v-drop 21.6% / 完整 history 32.4%。
  作者的 0.1 也是在 t2v=0.4 语境下给的(即他预期的就是 6% 有效占比),**保持 0.1 即可**;
  若明确想要 10% 有效占比,设 `0.1667`。
- 训练格式与推理侧已核对一致:训练 i2v 分支保留 x0 锚点 + 最后 1 帧 short history、其余清零;
  推理 I2V 为 x0=图像 latent + history 末格=`fake_image_latents`、其余全零
  (`helios/pipelines/pipeline_helios.py:1186-1188, 1280-1294`)。

---

## 3. 改动三:开启 Saturation——**必须先改代码,再改 config**

### 3.1 为什么必须改代码

`add_saturation_to_history_latents`(`utils_helios_base.py:485`)目前**只**被
`utils_helios_post.py:810, 1147`(DMD/Stage-3 蒸馏路径)调用。Stage-1 flow-matching 路径
(`prepare_stage1_noise_input`)没有这条支路——只在 yaml 里加 `is_add_saturation: true`
对 Stage-1 是 **no-op**。

### 3.2 代码补丁(镜像 `utils_helios_post.py:809-818` 的写法)

**位置 A(必做):** `helios/utils/utils_helios_base.py`,函数 `prepare_stage1_noise_input` 内,
`if args.training_config.corrupt_history and latents_history_short is not None:` 的
`corrupt_history_latents(...)` 调用块结束之后、`if args.training_config.corrupt_model_input:`
之前(当前约 line 844),插入:

```python
    if args.training_config.is_add_saturation and latents_history_short is not None:
        latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            saturation_ratio_min=args.training_config.saturation_ratio_min,
            saturation_ratio_max=args.training_config.saturation_ratio_max,
            saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
        )
```

注意:顺序必须是 **先 corrupt 再 saturation**(与 post 路径一致);`latent_window_size`、
`args` 均在函数作用域内,`is_keep_x0=True` 与同函数内 corrupt 调用保持一致。
`add_saturation_to_history_latents` 定义在同文件内,无需新增 import。

**位置 B(仅当以后跑 Stage-2 pyramid 训练时):** 同文件 `prepare_stage2_noise_input` 内,
corrupt 块之后、`if is_navit_pyramid:` 之前,插入同样的代码块。

**Stage-3/DMD 无需任何补丁**:`is_add_saturation` 已在 `train_helios.py:1673, 1862` 接线。

### 3.3 config 增补(打完补丁后,加到**每一个** stage config)

```yaml
  is_add_saturation: true
  saturation_ratio_clean_prob: 0.1   # 每个 history window 有 10% 概率跳过
  saturation_ratio_min: 0.3          # correct.yaml 推荐;train_config 默认值相同
  saturation_ratio_max: 1.7
```

### 3.4 风险提示

作者的 saturation 收益证据来自 **Helios-Distilled(Stage-3)的消融**;在 Stage-1 上开属于
外推(作者在 issue #38 表示相信对所有 stage 有益,理由是 Base 生成 1452 帧后仍会过曝,
saturation 腐蚀应能缓解)。**若算力允许,建议留一个不开 saturation 的对照 run**;若只能跑
一个,按作者建议开。

---

## 4. 汇总:新 run 的 config diff 模板

以任一现有 stage-1 config 为基,`training_config` 内的净改动:

```yaml
  # --- 改 ---
  corrupt_mode_history: "random"        # 原 "noise" 的 config 才需要改;init_cfr368/wan22 系已是 "random"
  corrupt_mode_prob_history: 0.8889     # 同上
  # --- 加 ---
  random_drop_i2v_ratio: 0.1
  is_add_saturation: true               # 需 §3.2 代码补丁先落地,否则是 no-op
  saturation_ratio_clean_prob: 0.1
  saturation_ratio_min: 0.3
  saturation_ratio_max: 1.7
```

其余全部保持不变(特别是 `corrupt_history: true`、downsample ratio 0.9/1.0、
`is_random_drop`/v2v/t2v 0.4、validation 的 dynamic_shifting)。

---

## 5. 落地后的验证清单(执行 agent 必做)

1. **config 一致性**:`python scripts/training/compare_yaml.py` 过一遍;新键要加到
   **每一个**会用到的 stage config(含 smoke 变体),不要只改一个(仓库惯例,见 CLAUDE.md)。
2. **断言冲突**:确认 `use_error_recycling: false`(它与 `corrupt_history`/`corrupt_model_input`
   在 `train_helios.py` 的启动断言里互斥;我们所有 config 本来就是 false,别打开)。
3. **分支冒烟验证**(强烈建议,~10 分钟):复制一份 `_smoke` config,临时设
   `random_drop_i2v_ratio: 1.0`、`random_drop_t2v_ratio: 0.0`,在
   `utils_helios_base.py` 的 i2v 分支(`total_drop = max(0, hist_seq_len - 1)` 处)和
   saturation 补丁块里各加一行一次性 `print`,跑 10 step 确认两条分支都真实命中、无 shape
   报错,然后**移除 print、恢复 ratio**。downsample 分支同理可用
   `corrupt_mode_history: "downsample"` 强制命中一次。
4. **full-FT 注意**:若是 `is_full_finetune` + ZeRO-2 的 run,本仓库 `_flow_loss` 里的
   zero-touch 逻辑(`utils_helios_base.py:~101-113`)已保证条件分支不会造成 None-grad,
   新增的 i2v drop / saturation 分支不影响它,无需额外处理。
5. **训练中途开启是安全的**:这些都是输入侧数据增广(history 腐蚀/格式模拟),不改 loss 目标,
   从已有 checkpoint 继续训练时直接启用即可,作者亦如此建议。

---

## 6. 术语对照(防误解)

| correct.yaml 注释 | 对应机制 | 开关 |
|---|---|---|
| Easy Anti-Drifting: Noise | history 加噪腐蚀 | `corrupt_history: true`(一直开着) |
| Easy Anti-Drifting: Blur | downsample 再上采样(空间低通) | `corrupt_mode_history: "random"` |
| Easy Anti-Drifting: Saturation | latent 通道均值对比度缩放 | `is_add_saturation`(需 §3 补丁) |
| I2V train/inference mismatch | 训练时模拟"仅首帧锚点+末帧"起步 | `random_drop_i2v_ratio: 0.1` |
