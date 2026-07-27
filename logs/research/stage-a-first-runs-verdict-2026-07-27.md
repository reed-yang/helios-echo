# Stage A 首批真训练 run 判定(2026-07-27,窗口内快照)

**结论:两条 Stage A 真训练 run 全部健康在跑,首个 checkpoint 结构验证通过,损失面正常。**
Gate 4(用户放行)由 2026-07-27 的 10h 自主窗口指令给出("推进得到实验训练结果")。

## 运行矩阵(快照 ~15:55 UTC)

| Run | Job | 节点 | 语料 | 配置 | 快照步数 | 步速 | Checkpoint |
|---|---|---|---|---|---|---|---|
| pilot | 5666 | c-node06 8×H200 | 40GB 子集 2,869 clips(≈44.6 ep/4000 步) | `stage1_lora_mem368_A_pilot.yaml` | 854+ | 29-35 s/it | **checkpoint-500 已验证** |
| slice512 | 5685 | c-node04 8×H200 | 512GiB 切片 36,673 clips(≈3.5 ep/4000 步) | `stage1_lora_mem368_A_slice512.yaml` | 320+ | 29-36 s/it | 首个将于 step 500 |

批次几何:bs2 × accum2 × 8 GPU = 全局 32;merged 基座(stage1_lora368_correct@19500 全量合并);memory_freeze_backbone;memory_single_write_prob 0.5。

## 证据

1. **checkpoint-500 结构(pilot)**:9.0G;`pytorch_lora_weights.safetensors` 814 tensors、**零 memory 键**(exclude_modules 生效);`transformer_partial.pth` 84 键 = evolving_memory 38(query_init/Enc 投影/gate)+ patch_long/mid/short 6 + blocks 40(per-head memory_key_scale);**query_state 缺席**(fp32 态不入 state_dict 的不变量在生产保存路径首次实证);optimizer.bin/scheduler.bin/random_states 齐全。
2. **损失面**(`results/stageA_{pilot,slice512}_loss.csv`):pilot 窗口均值 0.0933(1-100)→0.0950(101-300)→0.0845(301-500)→0.0864(501-854);slice512 ~0.090 同量级。flow-matching 逐批 σ 方差主导,冻结骨干下缓降符合预期。grad_norm 全程有限(0.004-0.6 区间)。
3. **lr 语义验证**:tqdm 显示的是 memory 参数组 lr,轨迹精确 = step/500 × 5e-5,step 500 达标后恒定 —— warmup 语义正确。**已排除**:早期"warmup 提前完成"怀疑(step=100 时 1e-5 为巧合,与基座 lr 相等)。
4. **v2 cache 36k 生产首演**(slice512 日志):恰 1 次构建("Processing 36673 files"+ 原子发布)+ 7 rank 缓存加载,零冗余扫描 —— 388k 扫描修复的设计行为在生产规模成立。
5. **GPU 健康**:pilot 起跑时 8 卡 100% util / ~120G 显存(compute-bound,非 dataloader 停顿)。

## 运行事实与后续

- 两 run `--time=24:00:00`:pilot 到限 ~明日 08:15(约 step 2800),slice512 ~明日 13:05(约 step 2600)。`resume_from_checkpoint: "latest"` + 每 500 步 checkpoint ⇒ 到限后重发同命令即续训;或管理员 `scontrol update jobid=<id> TimeLimit=` 延长。
- wandb offline:`wandb/offline-run-20260727_082300-jg1fwbk4`(pilot)、`offline-run-20260727_1305*`(slice512);`wandb sync` 可回传。
- 真值通道:`results/stageA_pilot_run1.log` / `results/stageA_slice512_run1.log`(含 EXIT_CODE 尾行约定)。
- 本判定为窗口内快照;终态判定(4000 步完成、ckpt 链完整性、跨 ckpt loss 面)待续训完成后补。
