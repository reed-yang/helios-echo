# Rollout smoke verdict — 2026-07-22

## Overall verdict: FAILED

The five-section GPU rollout completed on Slurm job `5551` (`c-node06`) and exited with code 1. The memory state-machine checks passed, but both the memory-enabled and no-memory outputs contained non-finite latent values. No Python traceback was emitted.

## Checks

```text
[PASS] memory state exported 
[PASS] state evolved from M0 (3 writes over 5 sections) 
[PASS] queue holds 2 pending sections len=2
[PASS] pending sections are 3 and 4 
[PASS] per-capture sigma recorded {0.2813178405761719}
[FAIL] memory-on latents finite 
[FAIL] no-memory latents finite 
```

## Timing and memory

- Memory-on wall time: **148.6 s**
- No-memory wall time: **140.7 s**
- Computed overhead: **+5.6%** (`(148.6 - 140.7) / 140.7 * 100`, rounded)
- Peak GPU memory: **43.3 GiB**

## Exact failure

```text
SMOKE ROLLOUT RESULT: FAILED ['memory-on latents finite', 'no-memory latents finite']
srun: error: c-node06: task 0: Exited with exit code 1
```

## Diagnosis

The evolving-memory state machine itself appears to execute correctly: exported state changed from M0, three eviction writes occurred, the final queue held sections 3 and 4, and capture sigma was recorded. However, non-finite latents occur in both the memory-on rollout and the independently rerun no-memory reference under the same real Helios-Base weights, shape, seed, and 8-step scheduler. This makes the most likely root cause an existing numerical-instability/regression in the shared base pipeline/model sampling path rather than the new eviction/write queue state machine. The log also contains an RMSNorm bf16-input/fp32-weight fallback warning during the memory-on pass; it merits follow-up, but cannot by itself explain the no-memory failure and is therefore not established as the root cause.

## Last 40 log lines (verbatim)

> 100%|██████████| 8/8 [00:26<00:00,  3.37s/it]
> 100%|██████████| 8/8 [00:27<00:00,  3.47s/it]
> 
>   0%|          | 0/8 [00:00<?, ?it/s]
>  12%|█▎        | 1/8 [00:03<00:23,  3.35s/it]
>  25%|██▌       | 2/8 [00:06<00:20,  3.36s/it]
>  38%|███▊      | 3/8 [00:10<00:16,  3.37s/it]
>  50%|█████     | 4/8 [00:13<00:13,  3.38s/it]
>  62%|██████▎   | 5/8 [00:16<00:10,  3.38s/it]
>  75%|███████▌  | 6/8 [00:20<00:06,  3.39s/it]
>  88%|████████▊ | 7/8 [00:23<00:03,  3.39s/it]
> 100%|██████████| 8/8 [00:27<00:00,  3.38s/it]
> 100%|██████████| 8/8 [00:27<00:00,  3.48s/it]
> 
>   0%|          | 0/8 [00:00<?, ?it/s]
>  12%|█▎        | 1/8 [00:03<00:23,  3.37s/it]
>  25%|██▌       | 2/8 [00:06<00:20,  3.38s/it]
>  38%|███▊      | 3/8 [00:10<00:16,  3.38s/it]
>  50%|█████     | 4/8 [00:13<00:13,  3.38s/it]
>  62%|██████▎   | 5/8 [00:16<00:10,  3.37s/it]
>  75%|███████▌  | 6/8 [00:20<00:06,  3.37s/it]
>  88%|████████▊ | 7/8 [00:23<00:03,  3.37s/it]
> 100%|██████████| 8/8 [00:26<00:00,  3.37s/it]
> 100%|██████████| 8/8 [00:27<00:00,  3.48s/it]
> 
>   0%|          | 0/8 [00:00<?, ?it/s]
>  12%|█▎        | 1/8 [00:03<00:23,  3.36s/it]
>  25%|██▌       | 2/8 [00:06<00:20,  3.36s/it]
>  38%|███▊      | 3/8 [00:10<00:16,  3.37s/it]
>  50%|█████     | 4/8 [00:13<00:13,  3.36s/it]
>  62%|██████▎   | 5/8 [00:16<00:10,  3.36s/it]
>  75%|███████▌  | 6/8 [00:20<00:06,  3.36s/it]
>  88%|████████▊ | 7/8 [00:23<00:03,  3.37s/it]
> 100%|██████████| 8/8 [00:26<00:00,  3.38s/it]
> 100%|██████████| 8/8 [00:27<00:00,  3.47s/it]
> [FAIL] no-memory latents finite 
> wall time: memory-on 148.6s, no-memory 140.7s, overhead +5.6%
> peak GPU memory: 43.3 GiB
> SMOKE ROLLOUT RESULT: FAILED ['memory-on latents finite', 'no-memory latents finite']
> srun: error: c-node06: task 0: Exited with exit code 1
