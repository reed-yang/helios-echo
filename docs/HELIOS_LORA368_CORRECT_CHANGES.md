# `stage1_lora_cfr_368_correct` — 训练时做了哪些调整

> 写于 2026-07-07。本文只回答一个问题:相比基线 run `stage1_lora_cfr_368`,
> `_correct` 版本在**训练时改了什么、为什么改**。改动动机来自与 Helios 作者
> (SHYuanBest)确认的事实:官方训 Base/Mid 时 **Easy Anti-Drifting 腐蚀没开满、
> I2V 格式从未训过**;上游 `correct.yaml`(随 commit `e7b1f92`,回应 issue #38)
> 是作者给的补救推荐。完整落地指南见 [`HELIOS_CORRECT_RECIPE.md`](./HELIOS_CORRECT_RECIPE.md);
> 本文是那份 recipe 在这个具体 run 上的**实际应用记录**。

- **Run**:`helios_runs/stage1_lora_cfr_368_correct`,从基线的 `checkpoint-17000` 续训。
- **配置**:`scripts/training/configs/stage1_lora_cfr_368_correct.yaml`
- **代码提交**:`9370669`(改动前的备份)→ `a613974`(启用 recipe)→ `127d5bd`(monitor)
- **作业**:job 4276(5 节点 mc-node01,c-node03–06,gb160)+ 在线 eval job 4289。

核心是**三项"腐蚀/anti-drifting"调整**(Blur、I2V-drop、Saturation),外加为了让它们生效
所必需的**两处代码改动**(saturation 需要新代码路径,改 lr 需要强制覆盖)。

---

## 一句话总结

基线 run 只开了最基础的 **noise 腐蚀**;`_correct` run 把作者推荐的**三种 history 腐蚀
全部打开**——Blur(降采样)、I2V 格式训练、Saturation(饱和度抖动)——以更强地对抗
自回归长视频的 drifting / 掉色 / I2V 慢动作问题。

---

## 1. Blur 腐蚀:`corrupt_mode_history "noise" → "random"`

| 项 | 基线 `stage1_lora_cfr_368` | `_correct` |
|---|---|---|
| `corrupt_mode_history` | `"noise"` | `"random"` |
| `corrupt_mode_prob_history` | `0.9` | `0.8889` |

- **改了什么**:`"noise"` 模式只往 history latents 加噪;`"random"` 模式在每步随机
  在 **noise / downsample(Blur)/ clean** 之间选一种。概率:P(noise)=0.9×0.8889≈0.8,
  P(blur)≈0.1,P(clean)≈0.1。
- **为什么**:downsample(Blur)分支模拟推理时 history 越传越糊的退化,是 Easy
  Anti-Drifting 的关键一环。基线的 `"noise"` 模式**永远不会触发 Blur 分支**,
  `downsample_min/max_corrupt_ratio_history`(0.9/1.0)在基线里是死配置。
- **注**:本仓库已含上游 `156c7a8`(issue #63)对 downsample 的 5D reshape 修复
  (permute 后再 reshape),所以 `"random"` 模式的 Blur 腐蚀执行是正确的。

## 2. I2V 格式训练:`random_drop_i2v_ratio 0 → 0.1`(新增)

- **改了什么**:基线里此项没设(默认 0),I2V 格式从未参与训练。`_correct` 设为 `0.1`。
- **含义**:以一定概率把 history 丢到只剩"首帧 + 最后 1 帧,其余置零",结构上等价于
  I2V 推理输入(x0=图像 latent + 末尾 history 槽)。这是上游 `e7b1f92` 为修复 issue #38
  的"I2V 慢动作"问题引入的。
- **有效采样率**:i2v 分支嵌在 t2v(0.4)的 else 分支里,所以实际 ≈ (1−0.4)×0.1 = **6%**。
- **注**:本仓库已含 `e7b1f92` 的 i2v-drop 分支与互斥控制流,所以这一项是**纯配置开关**,
  无需改代码。

## 3. Saturation(饱和度抖动):`is_add_saturation false → true`(新增,且需改代码)

| key | 值 |
|---|---|
| `is_add_saturation` | `true`(基线 false) |
| `saturation_ratio_clean_prob` | `0.1` |
| `saturation_ratio_min` | `0.3` |
| `saturation_ratio_max` | `1.7` |

- **改了什么**:对 history latents 施加随机饱和度缩放(min/max 之间),对抗长视频逐渐
  掉色/褪色。
- **为什么需要改代码**:上游 `add_saturation_to_history_latents` **只**接在 DMD/Stage-3
  路径上(`utils_helios_post.py`),Stage-1 flow-matching 路径根本不调用它。
  仅在 config 里开 `is_add_saturation` 在 Stage-1 是**无效的死开关**。
- **代码补丁**(commit `a613974`,`helios/utils/utils_helios_base.py`,
  `prepare_stage1_noise_input` 内,history 腐蚀之后):镜像 post 路径,新增
  ```python
  if args.training_config.is_add_saturation and latents_history_short is not None:
      latents_history_short, latents_history_mid, latents_history_long = \
          add_saturation_to_history_latents(..., is_keep_x0=True, ...)
  ```
  这样 Stage-1 flow 路径才真正应用饱和度腐蚀。

---

## 4. 为让 recipe 生效而必需的两处工程改动

### 4.1 `HELIOS_FORCE_LR=1` —— 否则改 lr 在 resume 时被静默还原

- **背景**:`_correct` 把学习率从 **3e-5 降到 1e-5**(续训精修)。
- **坑**:`accelerator.load_state` 会从 checkpoint 恢复 optimizer/scheduler 状态,
  包括旧的 `base_lrs`。不处理的话,config 里写的新 lr 会被**静默覆盖回 3e-5**。
- **修复**(commit `a613974`,`train_helios.py:1086–1099`):`HELIOS_FORCE_LR=1` 时,
  在 `load_state` 之后把 config 的 `learning_rate` 重新写回 optimizer 的
  `param_groups` 和 scheduler 的 `base_lrs`。sbatch 里已导出该环境变量。
  (第一步 tqdm 显示的 lr 可能还是旧值,但 `optimizer.step` 已用新值。)

### 4.2 `[DBG-BRANCH]` 调试打印(env 门控)

- commit `a613974` 在三个腐蚀分支(i2v-drop / downsample / saturation)加了
  `HELIOS_DEBUG_BRANCH=1` 门控的打印,用于确认三条 recipe 分支确实被触发。
  默认关闭,长期保留,不影响正常训练。

---

## 5. 附带的训练规模/续训参数(非 recipe,但一并改了)

| 项 | 基线 | `_correct` | 说明 |
|---|---|---|---|
| 续训起点 | checkpoint-9000 | **checkpoint-17000** | 从更成熟的 LoRA 续 |
| `learning_rate` | 3e-5 | **1e-5** | 精修(配合 §4.1) |
| `max_train_steps` | 17000 | **22000** | 17000 + ~2 epoch(388k 语料 / gb160) |
| global batch | gb128(8节点×bs2×accum1) | **gb160**(5节点×bs2×**accum2**) | |
| `gradient_accumulation_steps` | 1 | **2** | |

LoRA 结构(r128/a128)、语料(合并 cfr_int + cfr_int_bprime ~388k clips ≥121f @368×640)、
`corrupt_history: true`、分辨率 368×640 **保持不变**——LoRA adapter 的 state-dict 必须与
基线完全一致才能从 checkpoint-17000 续训。

---

## 6. 冒烟验证结论(`*_correct_smoke`)

- 三条 recipe 分支(Blur / I2V-drop / Saturation)在 `HELIOS_DEBUG_BRANCH=1` 下都确认触发;
- resume + `HELIOS_FORCE_LR` 生效、loss 有限;
- smoke 在 200-clip 小数据集跑约 3 步后会 hang(32/step vs 200 样本的跨 rank 尾部对齐),
  这是**小数据集 artifact,不是代码 bug**——看到分支打印后即可 kill;全量语料跨 epoch 正常。

---

### 相关记忆 / 文档
- 落地指南:[`HELIOS_CORRECT_RECIPE.md`](./HELIOS_CORRECT_RECIPE.md)
- 审计(哪些 run 开了哪些腐蚀):记忆 `helios-correct-yaml-audit`
- 本 run 运行记录(job 4276、monitor、eval):记忆 `helios-lora368-correct-run`
