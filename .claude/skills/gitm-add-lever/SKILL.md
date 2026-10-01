---
name: gitm-add-lever
description: Add or modify an optimization lever (intervention) in the gitm catalogue — library.yaml entry, vLLM knob registration, applicability/safety gates, and tests. Use when asked to add a new tuning knob, env var, NCCL/RCCL setting, or engine arg for the loop to try.
---

# Adding a lever

A lever is an `InterventionSpec` (`gitm/kernels/spec.py`, pydantic, `extra="forbid"`, so
a typo'd field fails to load). Levers are listed in `gitm/kernels/library.yaml` and loaded
by `gitm/kernels/library.py:load_library`.

## 1. The catalogue entry (`gitm/kernels/library.yaml`)

Required fields:

- `name`: snake_case and unique. It becomes the key in history and diff, so don't rename
  a lever that has already run.
- `summary`
- `knob`
- One of these:
  - `value`
  - `value_multiplier` (with optional `value_multiplier_grid`, `value_min`, `value_max`)
    for relative levers
  - `knobs: {k: v}` for multi-knob levers
- `expected_delta_mean`, `expected_delta_lo`, `expected_delta_hi`: signed fractions,
  where 0.08 means +8%.
- `source`: required. A citation URL for where the expected delta comes from.
- `applicability`: workloads, dtypes, GPU/collective requirements. Copy from a neighbour.
- `safety`: tier.
- `review: null`, until the lever is signed off.

Scope it with one of these:

- `applies_to_kernels: [...]` using canonical op names (the vocabulary `predict_graph` /
  `classify_op` share, e.g. `attn_score_value`), **not raw kernel substrings**.
- `whole_step: true` for levers that act on the whole step: batch shape, scheduling,
  graph capture, sharding degree.

An empty `applies_to_kernels` without `whole_step` means **zero coverage**, and the lever
will never rank.

## 2. If it is a vLLM engine knob (`gitm/optimizer/vllm_knobs.py`)

- Register a `KnobSpec(name, "<config.path>", kind)` in `_KNOBS`. The kind decides how the
  knob is applied:
  - `"scheduling"`: hot-swapped on the live engine.
  - `"structural"`: fixed at construction, so it needs an engine restart
    (`StructuralKnobRequiresRestart`).
- An unknown knob is always treated as needing a restart. That's safe, but slow.
- If the knob only means something when another flag is on, add a
  `(knob_substring, required_flag)` pair to `KNOB_PREREQUISITES`.

## 3. Gates and ranking

- Preconditions live in `gitm/optimizer/preconditions.py`.
- Ranking lives in `gitm/agents/policy.py`. It uses the expected delta times coverage, and
  past measured results via `history.record_for` when `--use-history` is on.

## 4. Tests

```bash
pytest tests/test_catalog_unify.py tests/test_gate_wiring.py tests/test_whole_step_scope.py \
       tests/test_vllm_knobs_and_restart.py tests/test_policy_history.py
```

`test_full_library_loads_and_validates` catches schema errors. Add a targeted test when the
lever has a non-obvious gate, for example one that must *not* apply to MoE or to fp8 KV.

## 5. How it gets measured

Only a live engine run (`gitm run` with vLLM on a GPU) produces a measured result. It lands
in `runs/<id>/verification.json` with `kept` taken from the rollback gate. The catalogue's
expected deltas stay estimates until then. Use the `gitm-run-diff` skill to compare runs
once it has been measured.
