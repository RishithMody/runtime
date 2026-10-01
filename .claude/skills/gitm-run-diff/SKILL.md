---
name: gitm-run-diff
description: Compare two gitm optimization runs, read per-lever history across runs, drive the gitm MCP server (list_runs/history/diff), or change the CI diff gate. Use when asked what moved between runs, whether a lever regressed, which levers win on a GPU, or when touching gitm/optimizer/history.py, run_diff.py, gitm/mcp_server.py, or .github/workflows/diff-gate.yml.
---

# Comparing runs: `gitm history`, `gitm diff`, `gitm mcp`

All three are read-only and GPU-free. They read `runs/<run_id>/verification.json` through
`gitm/optimizer/history.py` (`read_run` → `aggregate`). Never parse that file by hand.

## Answering questions

- **"What moved between the last two X runs?"**
  1. Call the MCP `list_runs(match="x")`. It lists runs newest first and matches on the
     model name, fingerprint, workload id, and run id.
  2. Call `diff(run_a=<older run_dir>, run_b=<newer run_dir>)`.

  Without MCP, use the CLI:
  ```bash
  gitm diff <run_a> <run_b>     # accepts a path, a run id, or a unique id prefix under $GITM_SCRATCH/runs
  gitm diff A B --json          # machine-readable form, same dict as the MCP tool returns
  gitm diff A B --all           # also list unchanged levers
  ```
- **"Which levers work on this box?"** Run `gitm history [--gpu "<exact SKU>"] [--json]`,
  or call the MCP `history`.

## Reading a diff correctly

- **Preconditions.** Both `gpu_sku` and `fingerprint` must be `match`.
  - On a `mismatch`, the diff has **no lever rows** and exits 2. Report that the runs are
    not comparable. Do not work around it by diffing the raw JSON.
  - `unverified` means a side didn't record the value, as on ROCm boxes without
    `GITM_GPU_SKU`. The diff refuses with no rows and exit 2, including with `--allow-empty`.
- **`change`** is one of `verdict_changed`, `delta_moved`, `only_in_a`, `only_in_b`,
  `unchanged`.
- **`move`** is `b.mean_delta − a.mean_delta` in delta points (0.02 = 2 pts). The default
  threshold is `MIN_NOISE_BAND` (0.02) from `verification_export.py`.
- **`regressed`** is true only when A kept the lever **and** B measured it more than the
  threshold lower.
  - A lever missing from B is not re-measured, which is *not* a regression.
  - A lever rolled back in A had no gain to lose.
- **`measured: false`** means the run has no `verification.json`. Every run with no live
  engine is like this, including every CI run. `--allow-empty` accepts a skipped comparison
  (`comparable=false`, no lever rows); a damaged
  export is always exit 2.
- **Exit codes:** 0 compared; 1 a regression under `--check`; 2 not comparable, unreadable,
  or empty without `--allow-empty`.

## Changing this code

- Validation of what counts as a readable run lives only in `history.read_run`. Both
  `load_history` and `diff_runs` depend on it, so a rule added there applies to both.
- JSON shapes come from `history_as_dict` and `diff_as_dict`. The CLI `--json` and the MCP
  tools share them, so change them in one place.
- Rules for interpreting results belong in the MCP tool descriptions (`DIFF_DOC` etc. in
  `gitm/mcp_server.py`), not in each result.
- Fixtures: `tests/fixtures/diff/runs/*` and the golden run `tests/golden/diff_gate/kimi-mi355x`
  come from `python tests/fixtures/diff/generate_diff_runs.py`. That script writes through
  the real `build_record` / `write_verification`. Regenerate; never hand-edit the JSON.
- Tests: `pytest tests/test_run_diff.py tests/test_mcp_server.py tests/test_history.py`.
- CI gate (`.github/workflows/diff-gate.yml`):
  - It first proves `--check` exits 1 on the `regressed` fixture.
  - Then it diffs the golden run against a GPU-less `gitm run`, which returns no_data, so
    the job passes `--allow-empty`.
  - Drop `--allow-empty` when the job moves to a GPU runner.
- MCP SDK is v2 (`mcp>=2.2,<3`). The class is `from mcp.server.mcpserver import MCPServer`,
  not `FastMCP`. Test in-process with `mcp.Client(server)`.
