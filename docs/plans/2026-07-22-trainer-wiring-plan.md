# Trainer Wiring Plan (memory params through PEFT/optimizer/checkpoint paths)

> Executed inline by the plan author (same deviation note as the transformer-integration plan).

**Spec:** design doc ch.2 D10/D11/D12, ch.4 D8(a). Predecessors: k0-memory-foundations, transformer-memory-integration.

**Goal:** Make `evolving_memory.*` + `memory_key_scale` trainable full-rank alongside LoRA, persisted through `save/load_extra_components` (section 5), with their own optimizer param group (`memory_learning_rate`, role-tagged), a Stage-A freeze-backbone channel, DS-optimizer guard, and a role-aware `HELIOS_FORCE_LR` resume path. Zero behavior change with the feature off.

## Verified integration facts

- all-linear scan (`train_helios.py:379-384`) collects every `nn.Linear` incl. the memory encoder's → target filter + post-adapter assertion required.
- `trainable_modules` substring loop (`train_helios.py:463-477`) only ENABLES → `memory_freeze_backbone` needs an explicit freeze pass after it.
- Optimizer params single group (`train_helios.py:815-818`, `transformer_lora_parameters` has no other consumers; clipping uses `transformer.parameters()` at :1754).
- DeepSpeed `DummyOptim` receives only the global lr (`utils_base.py:81-87`) → memory group + DS optimizer/scheduler must assert-out (design F4).
- `HELIOS_FORCE_LR=1` block (`train_helios.py:1090-1099`) flattens every group to one lr → made role-aware.
- `save_extra_components` sections 1-4 + `torch.save` tail (`utils_base.py:167-259`); `load_extra_components` mirrored, called on resume from `load_model_hook` (`train_helios.py:760`) → adding section 5 covers save AND resume.
- EMA wiring deferred (Stage A/B lineage configs don't enable EMA); noted for the Stage-C prep plan.

## Edits

1. `helios/utils/utils_base.py`:
   - `save_extra_components` section 5 (gate: `is_enable_evolving_memory` — ENABLE not train, so Stage-C frozen variants persist): collect `evolving_memory.*` (from submodule or state_dict) + per-block `memory_key_scale`.
   - `load_extra_components`: mirrored section; `evolving_memory` loads `strict=True`; **fail loudly** (assert) if the checkpoint has memory keys but the model was built without the module (design D12).
   - New helper `build_transformer_param_groups(transformer, base_lr, memory_lr)`: splits trainable params into role-tagged groups (unit-testable pure function).
2. `train_helios.py`:
   - After the `"norm" not in t` filter: drop `evolving_memory` entries from `target_modules`.
   - After adapter injection: assert no `evolving_memory.*lora_*` params exist.
   - `trainable_modules`: `is_train_memory_module` → `evolving_memory`; `+ is_amplify_memory` → `memory_key_scale`.
   - After the enable loop: `memory_freeze_backbone` channel (freeze everything trainable except the memory stack; assert `is_train_memory_module`).
   - Replace :815-818 with `build_transformer_param_groups(...)`; assert memory group ⇒ not DS optimizer/scheduler.
   - FORCE_LR block: per-role lr restore (`role` tag), scheduler `base_lrs` per group when lengths match.
3. `tests/test_trainer_wiring.py`: tiny-model (reused from test_transformer_memory) round-trip of save/load_extra_components (keys, equality, fail-loud on memory-less model), param-group construction (roles/lrs/order, memory-only case).

## Smoke gates

- CPU: full `tests/` suite green; `train_helios.py` ast-parses.
- Feature-off invariance: `save/load_extra_components` with memory flag False produces byte-identical behavior (no new keys section executes); param-group helper on a LoRA-only model returns the single base group.
