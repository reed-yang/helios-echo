# A1 checkpoint-assembly smoke plan

Spec: `docs/specs/2026-07-22-real-model-test-checkpoint-decision.md`

1. [x] Inspect A1 checkpoint layout and its source training config.
2. [x] Load `Helios-Base/transformer_init` through the training-side transformer with evolving memory enabled.
3. [x] Apply the saved adapter through the training-side pipeline LoRA mixin, then restore `transformer_partial.pth` with the source config flags.
4. [x] Check adapter tensors, matched partial keys, fresh-only memory state, and finite paired real-geometry forwards.
5. [x] Run AST parsing and CPU import-only verification. Do not launch a GPU job.
