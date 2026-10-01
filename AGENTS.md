# AGENTS.md — how to work in this repo

For any coding agent (Claude Code, Codex, Cursor, Copilot, Gemini). `CLAUDE.md` imports this
file. Task-specific playbooks live in `.claude/skills/` (the same folder is exposed as
`.agents/skills/` for Codex and other tools that read that path).

## What this is

`gitm` (PyPI: `gitm-labs`; import name `gitm`; version in `gitm/__init__.py`) is a
job-level GPU runtime optimizer. `gitm run` works through five phases:

1. Capture a kernel trace.
2. Predict the execution graph (roofline).
3. Attribute the gap between prediction and trace to a cause.
4. Rank levers (runtime config changes) and A/B each one behind a rollback gate.
5. Write a report and `runs/<run_id>/verification.json`.

It never rewrites user code. It changes env vars, engine args, and library knobs.

## Setup and the commands you will need

```bash
uv venv && source .venv/bin/activate    # or python -m venv; Python >=3.10
pip install -e ".[dev,bench]"           # dev includes pytest, ruff, hypothesis, mcp
pytest                                  # full suite, no GPU needed, ~1-2 min
ruff check .                            # the only enforced linter (CI pins ruff 0.12.11)
```

- **Activate the venv.** Some tests shell out to `python`. With it off PATH,
  `tests/test_bench.py::test_run_seed_end_to_end_with_echo_harness` fails with
  `FileNotFoundError`.
- **pytest dirties the tree.** An autouse fixture in `tests/test_importers.py` regenerates
  `tests/fixtures/importers/*.sqlite` and `*.json.gz` on every run. Never commit those
  changes unless you meant to change the importer fixtures.
- **Don't run `ruff format`.** The repo is deliberately not ruff-formatted.
- **mypy is not enforced.**

## Map

| Path | What lives there |
|---|---|
| `gitm/cli.py` | Every `gitm` subcommand; argparse with lazy imports per command |
| `gitm/api.py` | `optimize()`, the embeddable entry, which calls `scheduler/loop.run_loop` |
| `gitm/scheduler/loop.py` | The loop itself (~1.7k lines), phases 1–5 |
| `gitm/tracer/` | CUPTI/ROCm capture (`capture.py`, C shims in `_cupti/`, `_rocm/`), kernel taxonomy |
| `gitm/planner/` | Predicted graph and roofline (`graph.py`, `roofline.py`, `moe_graph.py`, `models/*.yaml`), `gitm plan` |
| `gitm/optimizer/` | Deviation, attribution, apply and rollback gate (`apply.py`), report, `verification_export.py`, `history.py`, `run_diff.py`, `vllm_knobs.py` |
| `gitm/kernels/` | Lever catalogue: `library.yaml` plus the pydantic schema in `spec.py` |
| `gitm/agents/` | `policy.py` (rank or filter levers) and `autoresearch.py` (levers outside the catalogue) |
| `gitm/serve/` | vLLM launch, attach, and capture (`gitm capture serve\|attach`) |
| `gitm/importers/` | Nsight/Kineto customer dumps for `gitm analyze` |
| `gitm/safety/`, `gitm/telemetry/`, `gitm/traffic/`, `gitm/playbook/` | Audit and revert; NVML/AMD sampling; traffic replay; playbook schema |
| `gitm/mcp_server.py` | `gitm mcp`: `list_runs` / `history` / `diff` as MCP tools |
| `gitm/_paths.py` | `$GITM_SCRATCH` (default `~/.cache/gitm`, with `runs/`, `traces/`, …) and `$GITM_S3_ROOT` |
| `tests/` | Flat `test_*.py` files; `conftest.py` has `make_kernel` / `make_trace` builders; `fixtures/`; `golden/` |
| `benchmarks/`, `harness/`, `scripts/` | Per-track benches (hft, edge, kitti, …), pod glue, one-off drivers (`scripts/kimi_loop/` is the MI355X loop) |
| `docs/` | Design notes and runbooks (see below) |
| `deploy/k8s/` | Operator and MI355X manifests |

## CLI: what needs a GPU

**Laptop-safe:** `history`, `diff`, `mcp`, `plan`, `deviate`, `replay`, `apply`, `analyze`,
`doctor`.

**GPU, vLLM, or a cluster:** `run` (on a laptop it returns `status: no_data` and exits 3),
`capture serve|attach`, `install`, `attach`.

Exit codes are part of the contract:

- `gitm run`: 3 means no data.
- `gitm diff`: 0 means compared; 1 means a regression under `--check`; 2 means the runs are
  not comparable or unreadable.

## What a run leaves in `$GITM_SCRATCH/runs/<run_id>/`

`qualification.json`, `predicted_graph.json`, `residuals.json`, `deviations.json`,
`ranked_candidates.json`, `history_read.json`, `autoresearch.json`, `report.md`, and:

- `verification.json`, **only when a live engine ran an A/B**.
- `measurement.json` for runs that are not vLLM, or that had no data.

The trace goes to `$GITM_SCRATCH/traces/<run_id>.jsonl`. Both `gitm history` and `gitm diff`
read `verification.json` through `gitm/optimizer/history.py` (`read_run`, `aggregate`).
Don't write a second parser for it.

## House style (match it)

- **Write "why" docstrings.** Module and function docstrings explain the failure a
  decision prevents and what was rejected. Use `#:` comments for dataclass fields. Keep
  `from __future__ import annotations`.
- **`None` means "no number", never `0.0`.** "Never measured" and "measured zero" must
  look different. The same goes for `record_for` returning `None`.
- **Never fabricate.** An empty trace becomes `no_data`, not invented claims. A damaged
  input is *skipped with a reason* (`History.skipped`), kept separate from inputs that
  were *filtered* on purpose.
- **Key by context.** A lever result is keyed by `(lever, gpu_sku, fingerprint)`. Results
  from different GPUs or workloads never merge.
- **Degrade when deps are missing.** Optional deps are imported lazily and fail with an
  install hint (see `mcp_server.build_server`).
- **Tests are named as claims** (`test_a_lever_absent_from_b_is_not_a_regression`), with a
  docstring saying why when it isn't obvious. Tests avoid GPUs by monkeypatching
  `gitm.scheduler.loop.capture` and `sync_device`; see `tests/test_run_loop_workload.py`.

## Env vars you will meet

| Area | Vars |
|---|---|
| Paths and box | `GITM_SCRATCH`, `GITM_S3_ROOT`, `GITM_GPU_SKU` (set it on ROCm, where NVML can't name the part; otherwise the SKU is `None`) |
| Loop | `GITM_AB_REPS`, `GITM_RESTART_MODE`, `GITM_KNOBS_VIA_RESTART=1`, `GITM_DEVIATION_ONLY=1` |
| Tracer | `GITM_TRACE_NVTX`, `GITM_AUTOBUILD_CUPTI=0` |
| vLLM | `GITM_VLLM_SYNTHETIC=1` (a CPU stand-in for vLLM) |
| Goldens | `UPDATE_GOLDENS=1` (`tests/golden/report_basic.md`) and `GITM_UPDATE_GOLDEN=1` (customer report). The two names really do differ. |

## Docs worth reading before touching an area

- `ROADMAP.md`: what is real versus sequenced.
- `docs/safety.md`: detect, revert, page.
- `docs/invariants.md`: the three residual invariants.
- `docs/autoresearch.md`
- `docs/kernel_identity.md`
- `docs/playbook-schema.md`
- `docs/customer-intake.md`: `gitm analyze`.
- `docs/rocm.md`, `docs/mi355x_experiment_plan.md`, `docs/kimi_mi355x_loop_runbook.md`: AMD.
- `docs/glm52_runtime_validation.md`
- `RUN_DIFF.md`: `gitm diff`, the MCP server, and the CI diff gate.
- `IMPLEMENTATION.md`: how the diff, MCP server, and gate are built.
- `IMPLEMENTATION_SIMPLE.md`: a plain-language overview of the same.
- `MANUAL_TESTING.md`: the manual test pass for them.

## Skills (in `.claude/skills/`, mirrored at `.agents/skills/`)

- `gitm-run-diff`: compare runs, read history, use the MCP tools, work on the CI diff gate.
- `gitm-testing`: run, extend, and debug the GPU-less test suite and fixtures.
- `gitm-add-lever`: add or change an intervention in the catalogue, end to end.
- `gitm-run-triage`: read a run folder and explain what happened (no_data, rollbacks, claims).

## Git and CI

- Branch from `main`. PR titles follow `area: what changed` (e.g. `history: …`, `loop: …`).
- CI (`.github/workflows/`):
  - `tests.yml` runs pytest on Python 3.10–3.12 plus ruff.
  - `diff-gate.yml` runs `gitm diff` of the golden run against a CI run.
  - `claude-review.yml` / `gemini-pr-review.yml` review PRs.
  - `workflow.yml` publishes to PyPI on release.
- Pre-commit: `pre-commit install -t pre-commit -t pre-push` gives ruff on commit and the
  full pytest on push.
