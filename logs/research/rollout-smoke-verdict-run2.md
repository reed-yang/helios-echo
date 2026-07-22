# Rollout Smoke Verdict — Run 2 (final, ALL PASS)

**Conclusion: the pipeline memory state machine passes its GPU gate end-to-end.** All 8 checks green on real Helios-Base weights, 5 sections (165 frames, 384×640), with the validated sampling regime. The run-1 NaN is confirmed as a smoke-script calling-regime artifact, not a product defect: the only change between run 1 and run 2 was the sampling configuration (50 steps + resolution-dependent exponential dynamic shifting + standard negative prompt, replacing 8 steps + fixed mu=1 + empty negative prompt), and both arms went from NaN to finite.

## Facts

| Item | Value | Gate |
|---|---|---|
| State machine | 3 eviction writes over 5 sections; queue drains to pending [3, 4]; state export works | run-1 parity, PASS |
| Per-capture sigma | σ_last = 0.12199 recorded on every queue entry | non-zero ⇒ FiLM(σ_last) conditioning is live, PASS |
| memory-on latents | finite | PASS (run 1: NaN) |
| no-memory latents | finite | PASS (run 1: NaN) |
| Wall time | memory-on 943.4 s vs no-memory 894.6 s | **+5.5 %** ≤ 12 % limit, PASS |
| Peak GPU memory | 43.3 GiB | H200 headroom ample |

σ_last ≈ 0.122 (≠ 0) on Base at 50 steps directly confirms the design premise that the last-scheduled-step capture is not at σ≈0, i.e. the FiLM(σ_last) write conditioning is mandatory rather than decorative (design 总纲 §二, divergence on σ_last).

## Provenance

- Job: Slurm 5552, c-node06, srun interactive, started 2026-07-22T22:33:22, ~48 min total.
- Script: `tests/smoke_rollout.py` @ commit a02c365 lineage (regime fixed after run-1 root cause).
- Log: `results/rollout_smoke_run2.log` (full, unfiltered; copied from the srun's original redirect target by the terminal-state sentinel).
- Run-1 reference: `logs/research/rollout-smoke-verdict-run1.md` (state machine all-pass, +5.6 %, NaN both arms).
- Root-cause report: Sol worker analysis 2026-07-22 — smoke regime deviated from the two known-good entry points; product code paths unchanged by memory edits (verified by diff).

## Operational postscript

The run-2 log was invisible to all round-2 sentinels for ~30 min because the srun stdout redirect had been armed at the pre-convention path (`agent-research/`) while every sentinel watched the post-convention path (`results/`). BeeGFS refuses to rename an open-for-write file, so the fix was a fresh sentinel on the real path that copies the log into `results/` at terminal state. Lesson recorded in progress ledger: when a long job outlives a path-convention change, re-verify the truth channel's actual target (`/proc/<pid>/fd`) instead of trusting the intended path.
