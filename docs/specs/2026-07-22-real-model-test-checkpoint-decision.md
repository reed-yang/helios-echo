# Decision record: environment & checkpoint candidates for real-model memory testing

Status: v1 decided (revisit triggers below) | Date: 2026-07-22 | Owner: echo-memory branch

## Context

Unit/integration tests for the evolving-memory read path are green on CPU (SDPA) and H200 (flash-attn3, bf16) with tiny random models. The next verification tiers need real weights:
- **P1 integration smoke**: off-path bitwise equivalence + capture API on real 14B weights (loading path, shapes, dtype casting, `_keep_in_fp32_modules`).
- **P2 anti-drift effect A/B** (after the pipeline state machine and Stage A/B/C training land): rollout evals where memory-on vs no-KV slopes are compared.

## Environment (decided)

`/mnt/beegfs/yuheng/miniconda3/envs/helios` — this IS the team/xiangbo environment: `PY_ENV` in `scripts/training/train_stage1_lora_cfr_368_correct.sbatch:26`. All tests to date ran on it (torch 2.10.0+cu128, flash-attn3 kernels verified on c-node03). No alternative considered; a second env would fork kernel/dtype behavior from the training reality.

## Checkpoint candidates (verified on disk 2026-07-22)

| # | Checkpoint | Location | Form | Pros | Cons |
|---|---|---|---|---|---|
| A1 | lora368 `_correct` latest (≥21000; 19500 = design C2 training-fork freeze) | `/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-*` | LoRA adapter + partial components | training-lineage-exact; the weights Stage A will actually fork | needs lora+partial assembly to load; mid-run artifact |
| A2 | reweight lineage 27500 | `/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368/checkpoint-27500` | LoRA adapter + partial | freshest descendant the team runs ("27500") | different descendant recipe; same assembly cost |
| A3 | neworg lineage 27500 | `/mnt/beegfs/xiangbo/helios_runs/stage1_lora_neworg_368/checkpoint-27500` | LoRA adapter + partial | — | recipe provenance not studied in our deep-reads |
| B | Helios-Base (released) | HF `BestWishYsh/Helios-Base` (`scripts/inference/helios-base_t2v.sh:7`) | full transformer dir | simplest `from_pretrained`; exercises production load path; stable reference | 50-step sampling → slow rollouts; weakest drift signal |
| C | Helios-Distilled (released) | HF `BestWishYsh/Helios-Distilled` (`helios-distilled_t2v.sh:7`) | full transformer dir | drifts within ~1 min → strongest/fastest anti-drift signal (research doc); 4-step = cheap rollouts | distilled inference is the diffusers_version path — needs the M2 mirror port before memory participates end-to-end |

## Decisions

- **P1 integration smoke → B (Helios-Base full weights)**. Criteria: minimal load complexity (no LoRA/partial assembly), production `from_pretrained` path coverage, and independence from moving mid-run artifacts. One additional arm on **A1@19500** (the design C2 freeze) to validate the lora+partial assembly path our training will use.
- **P2 anti-drift A/B → C (Helios-Distilled) as the primary effect testbed** (fast drift = large effect size, exactly the research doc's Phase-0 logic), with **A-lineage checkpoints as the training-consistency arm**. Precondition: M2 diffusers mirror, or driving distilled weights through the training-side `pipeline_helios.py` (log_validation route) as an interim.
- The design ch.4 **C2 training-fork freeze stays `_correct/checkpoint-19500`** — a separate decision, NOT changed by this record.

## Revisit triggers

- HF download unavailable from cluster → P1 falls back to A1 assembly.
- Team promotes a new canonical checkpoint (e.g. `_correct` passes 22000-step gate) → re-pin A1 and note here.
- M2 mirror lands → move P2 from interim training-side driving to the release inference path.
