# Pipeline Memory State-Machine Plan (training-side pipeline_helios.py)

> Executed inline by the plan author (same deviation note as prior plans).

**Spec:** design doc ch.3 D1/D2/D4/D5/D7/D9 + ch.1 §6 eviction semantics + ch.1 D10 (pyramid capture at final stage only). Predecessors: transformer-memory-integration, trainer-wiring.

**Goal:** The section-loop state machine on the training-side pipeline (the one `log_validation` and the interim P2 route drive): per-section memory read, last-scheduled-step capture with its own sigma, section FIFO with k−2 eviction writes, M₀-always-readable init, state export.

## Edits (`helios/pipelines/pipeline_helios.py`)

1. `stage1_sample` / `stage2_sample`: `memory_tokens=None, capture_last_step=False`; cond forward gets `memory_tokens` + `capture_last_hidden` at the last scheduled step (stage2: final pyramid stage only, ch.1 D10); uncond forward also sees `memory_tokens` (memory is content-state conditioning; CFG contrasts text only — documented decision); conditional return `(latents, capture, sigma_last)` — external callers (`pipeline_helios_ode.py` has its own copies; only `__call__` consumes these) unaffected.
2. `__call__`: `enable_evolving_memory=False, memory_state=None`; init before the denoising loop (assert transformer has the module; `reset(B)` or state injection); per-section `get_tokens(transformer_dtype)`; unpack conditional returns; after the history append: push `(k, capture, sigma_last)` and pop-write when the queue head is `k−2` (`no_grad`, per-capture sigma — the amplified first chunk changes sigma_last per section); `self._memory_queue` maintained for export.
3. `get_memory_state()`: D9-lite export ({M fp32 cpu, queue}) — explicitly NOT a full generation checkpoint.

## Tests

- `tests/test_pipeline_memory.py`: stage1 capture contract via stub self (triple/shape/sigma value; plain return unchanged without capture), `get_memory_state` None-cases + round trip. Queue discipline runs inside `__call__` → covered by the GPU rollout smoke (next plan), not unit-testable without VAE/text encoders.
- SDPA monkeypatch moved to import time in `test_transformer_memory` (unittest-discover module order made `setUpModule` patching order-dependent).

## Smoke gates

- CPU suite green (39 tests) — met.
- GPU rollout smoke (real weights, `enable_evolving_memory=True`, ≥5 sections: verifies first write at k=2, queue drains, memory state export) — the NEXT plan's deliverable together with the P2-interim drift A/B scaffold.
