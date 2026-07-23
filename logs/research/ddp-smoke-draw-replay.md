# Stage A 2-Rank DDP Smoke Draw Replay

## Verdict

The epoch-0 draw stream is deterministically replayable offline on CPU. In optimizer steps 1–5, concurrently aligned rank-local micro-batches mix zero and nonzero `evicted_valid_frames` in **steps 3, 4, and 5** (micro-draw ordinals 5, 7, 8, 9, and 10). Therefore, **at least one mixed concurrent pair exists**; in fact, five aligned pairs are mixed.

No trainer process or GPU was used.

## Construction and deterministic-order evidence

Config replayed: `scripts/training/configs/stage1_lora_mem368_A_smoke_ddp.yaml` at epoch 0, with seed 44, world size 2, batch size 1, gradient accumulation 2, shuffle enabled, and 5 optimizer steps (10 micro-draws per rank).

- The stage-1 branch imports `BucketedFeatureDataset` and `BucketedSampler` from `helios/dataset/dataloader_history_latents_dist.py` (`train_helios.py:120-125`).
- The trainer's exact stage-1 dataset kwargs are assembled at `train_helios.py:918-941`: feature roots, single-resolution settings, reward/GAN/TF-derived return flags, history sizes, `is_keep_x0=True`, cache rebuild flag, seed, and `return_evicted_latent=is_train_memory_module`. Because this config has `memory_tf_unroll=false`, the conditional rollout kwargs at lines 939-941 are not supplied. The replay mirrored this exactly.
- The trainer constructs the sampler with batch size, `drop_last=True`, shuffle, seed, `num_sp_groups=accelerator.num_processes`, `sp_world_size=1`, and `global_rank=accelerator.process_index` (`train_helios.py:961-971`). Thus this two-rank replay used `num_sp_groups=2`, `sp_world_size=1`, and rank 0/1 respectively.
- At every epoch the trainer sets both sampler and dataset epoch before iteration (`train_helios.py:1304-1311`). This replay set both to epoch 0.
- Sampler order is seeded by `seed + epoch` (`helios/dataset/dataloader_history_latents_dist.py:768-771`). Within each bucket, indices are globally shuffled before sharding (`helios/dataset/dataloader_history_latents_dist.py:779-788`). Sharding is **round-robin/strided, not bucket-contiguous**: each rank receives `indices[rank::num_sp_groups]` (`helios/dataset/dataloader_history_latents_dist.py:678-707`). The sampler then randomly interleaves available buckets using the same seeded generator (`helios/dataset/dataloader_history_latents_dist.py:800-809`).
- For each sampled dataset index, the actual dataset code seeds a private generator with `base_seed + epoch * 1,000,000 + idx` and uses its first `torch.randint` draw as `choice_idx` (`helios/dataset/dataloader_history_latents_dist.py:339-345`).
- The real eviction helper computes `evicted_valid_frames = min(latent_window_size, max(0, choice_idx * latent_window_size - history_window_size))` (`helios/dataset/dataloader_history_latents_dist.py:248-280`). The real `__getitem__` calls `prepare_stage1_latent` (`helios/dataset/dataloader_history_latents_dist.py:505-556`) and returns the helper's `evicted_valid_frames` when enabled (`helios/dataset/dataloader_history_latents_dist.py:590-594`).

The standalone replay called the real sampler and the real `dataset[idx]` for all 20 loads. A temporary wrapper around the real `prepare_stage1_latent` captured the actual `choice_idx` and source tensor section count; eviction validity was read directly from the returned item rather than re-derived. Dataset length was 2869, matching an independent count of 2869 `.pt` files. There were no load retries or sampler-index substitutions.

Replay script: `/home/siyuan/.claude/jobs/47442cbe/tmp/replay_ddp_draws.py`.

Execution command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python \
  /home/siyuan/.claude/jobs/47442cbe/tmp/replay_ddp_draws.py \
  --config /mnt/beegfs/siyuan/workspace/helios-echo/scripts/training/configs/stage1_lora_mem368_A_smoke_ddp.yaml \
  --world-size 2 --optimizer-steps 5
```

A second fresh-process execution produced byte-identical JSON output.

## Full draw table

Optimizer step is `floor((micro_draw - 1) / 2) + 1` because gradient accumulation is 2.

### Rank 0

| Micro draw | Optimizer step | Dataset idx | Filename | Total sections | Chosen k | Evicted valid frames |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 2504 | `f2729346932e22942d51065b80694e47_0-158_141_368_640.pt` | 4 | 2 | 0 |
| 2 | 1 | 233 | `be626be7b45fe619f5ba7200b19ea8af_0-191_181_368_640.pt` | 5 | 3 | 8 |
| 3 | 2 | 2251 | `c2c18bfef7b720296f618a4eb1ae2846_0-139_121_368_640.pt` | 3 | 0 | 0 |
| 4 | 2 | 73 | `c25566c81b5ec62f3ca3b74396fc5647_0-137_121_368_640.pt` | 3 | 0 | 0 |
| 5 | 3 | 2437 | `1394d62c03e9cf3b2d46512618d707a6_0-227_221_368_640.pt` | 6 | 1 | 0 |
| 6 | 3 | 551 | `63c2f4bce05ef7be65da8f30b622ffe8_0-439_421_368_640.pt` | 12 | 7 | 9 |
| 7 | 4 | 2624 | `f7acc8684ce6e71ec256989cd0d19368_0-180_161_368_640.pt` | 4 | 3 | 8 |
| 8 | 4 | 625 | `6e5df30046ec7df3258000e78b759c58_0-446_441_368_640.pt` | 13 | 0 | 0 |
| 9 | 5 | 1491 | `ec423a98ae8613f1e0147f78b5a7e23e_0-308_301_368_640.pt` | 9 | 4 | 9 |
| 10 | 5 | 2017 | `52aa8189cb813100a0dac2c61d9e138b_0-234_221_368_640.pt` | 6 | 5 | 9 |

### Rank 1

| Micro draw | Optimizer step | Dataset idx | Filename | Total sections | Chosen k | Evicted valid frames |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 1903 | `159d1ea36ae3af5bb522f282392cade5_0-158_141_368_640.pt` | 4 | 0 | 0 |
| 2 | 1 | 2619 | `790c73e33b8c69850c3bd3bc73ec202b_0-188_181_368_640.pt` | 5 | 4 | 9 |
| 3 | 2 | 2751 | `9a4cfc02eb8c7d4f221be5fb10461f85_0-133_121_368_640.pt` | 3 | 1 | 0 |
| 4 | 2 | 631 | `d9a8d227bb3d94e0d5972d36283d9637_0-122_121_368_640.pt` | 3 | 1 | 0 |
| 5 | 3 | 1500 | `fe200e18d74f3350d1b2c8b52a4b8766_0-223_221_368_640.pt` | 6 | 4 | 9 |
| 6 | 3 | 816 | `5f92deaed76375e26bd5246448b38e8e_0-424_421_368_640.pt` | 12 | 9 | 9 |
| 7 | 4 | 2257 | `a4feb0a355810a7a6826a1d33be3ba6d_0-176_161_368_640.pt` | 4 | 2 | 0 |
| 8 | 4 | 2440 | `f9739e1601506926b69d783b7ae59102_0-454_441_368_640.pt` | 13 | 10 | 9 |
| 9 | 5 | 684 | `dc57c28541aa70d743f92daded727299_0-312_301_368_640.pt` | 9 | 0 | 0 |
| 10 | 5 | 1837 | `a405c608c69e63ddb9259cce2603b09d_0-239_221_368_640.pt` | 6 | 2 | 0 |

## Concurrent eviction-validity classes

| Optimizer step | Micro draw | Rank 0 valid frames | Rank 1 valid frames | Mixed zero/nonzero? |
|---:|---:|---:|---:|:---:|
| 1 | 1 | 0 | 0 | No |
| 1 | 2 | 8 | 9 | No |
| 2 | 3 | 0 | 0 | No |
| 2 | 4 | 0 | 0 | No |
| 3 | 5 | 0 | 9 | **Yes** |
| 3 | 6 | 9 | 9 | No |
| 4 | 7 | 8 | 0 | **Yes** |
| 4 | 8 | 0 | 9 | **Yes** |
| 5 | 9 | 9 | 0 | **Yes** |
| 5 | 10 | 9 | 0 | **Yes** |

Mixed optimizer steps: **3, 4, 5**.
