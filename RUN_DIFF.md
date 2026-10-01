# `gitm diff`, the MCP server, and the diff gate

`gitm diff <run_a> <run_b>` puts two runs side by side, per lever. It covers four cases:

- tried in one run and not the other (`only_in_a` / `only_in_b`);
- kept in one and rolled back in the other (`verdict_changed`);
- a mean delta that moved by more than `--threshold` (`delta_moved`);
- a `REGRESSED` flag on any lever run A kept that run B measured more than the threshold
  lower.

**Arguments.** Each run can be a folder, a run id, or a unique id prefix under
`$GITM_SCRATCH/runs`. `--json` emits the same dict the MCP tool returns. `--check` exits 1
on a regression. Exit 2 means the runs can't be compared.
`--threshold` must be finite and nonnegative; `diff_runs` is the one place that enforces
this, so CLI and MCP share it. On the **CLI**, NaN, infinities, and negative values exit 2
without producing a diff, even with `--allow-empty`. Over **MCP** the rule is the same for
every value JSON can carry: a negative number, or a non-finite sent as a *string*
(`"nan"`, `"inf"`), is coerced to a float and returns a tool error. The one case the
transport cannot deliver is a non-finite *number*: standard JSON has no NaN/Infinity, so the
SDK serializes `nan`/`inf` to `null` before the request leaves the client. The server then
cannot distinguish it from an omitted threshold and falls back to the default — it is never
an error. This does not weaken the gate: the default still detects regressions, so a
non-finite number can only make the gate stricter-or-equal, never silently pass. Moves must
strictly exceed the threshold. Decimal boundary ties are allowed a
machine-roundoff tolerance of one ULP from each input mean and the threshold;
this is not an additional measurement noise band. Zero thresholds still detect
any nonzero move, and reported moves are not rounded by this check.

**Preconditions.** The runs must be on the same GPU SKU and the same workload fingerprint.
For two runs with exports, both identities must be nonempty and match.
Missing, blank, or mismatched identity
returns `comparable: false`, no lever rows, and CLI exit 2, even with `--allow-empty`.

**MCP tools.** `gitm mcp` serves `list_runs`, `history`, and `diff` over stdio. It is
registered for Claude Code in `.mcp.json` and for Codex in `.codex/config.toml`.
`list_runs(match="kimi")` matches the model name recorded in the engine config, so "the last
two Kimi runs" resolves even though real fingerprints are `<vendor>:<hash>`.

**CI gate.** `.github/workflows/diff-gate.yml` runs in three steps:

1. It checks the gate can fail: the `regressed` fixture must make `--check` exit 1.
2. It runs the loop with no engine attached.
3. It diffs the checked-in golden run (`tests/golden/diff_gate/kimi-mi355x`) against that
   run, with `--check --allow-empty`.

With no engine, the loop writes no `verification.json` (`scheduler/loop.py` only writes one
after a live A/B), so the CI side reports "measured nothing" and a skipped comparison. With
`--allow-empty`, the gate exits 0 while JSON still says `comparable: false` and has no
lever rows. This exception only applies when an export is absent; damaged exports
always fail. Remove `--allow-empty` once the job has a GPU.

**`history.py` changes, and why.** `load_history` did its per-run validation inline. I
extracted it into `read_run()` and the fold into `aggregate()`, so a single-run diff and
`gitm history` share one definition of "readable run" and one averaging rule. I also added
`NoExport`, which separates "measured nothing" from "export is damaged": the gate may accept
the first but must never accept the second. The existing history tests pass unchanged.

## Running it

```bash
pip install -e ".[dev]"          # or ".[mcp]" for just the server
pytest tests/test_run_diff.py tests/test_mcp_server.py tests/test_history.py
gitm diff tests/golden/diff_gate/kimi-mi355x tests/fixtures/diff/runs/regressed --check
gitm mcp                         # stdio MCP server
python tests/fixtures/diff/generate_diff_runs.py   # regenerate fixtures and golden
```

**Versions.** Built against commit `737f0ca`, with Python 3.12.14 and the MCP Python SDK
`mcp==2.2.0` (v2 API: `MCPServer`; the extra pins `mcp>=2.2,<3`). CI runs 3.10–3.12.

**Fixtures.** `tests/fixtures/diff/runs/`, synthesized through the real `build_record` /
`write_verification`:

- `rerun_noise`
- `regressed`
- `improved`
- `other_gpu`
- `other_workload`
- `no_sku`
- `no_export`
- `truncated`

Plus the golden run. `tests/fixtures/history/` is unchanged.

## Judgment calls

The threshold defaults to 2 delta points: the `MIN_NOISE_BAND` each export already states as
its own re-measurement agreement band. It is set per invocation with `--threshold` (the
workflow passes it explicitly, so every gate log shows its bar), because a move inside the
band a run declares for itself is not evidence of anything. "Regressed" requires two
numbers: A kept the lever and B measured it lower. A lever rolled back in A had no gain to
lose, and a lever absent from B was never re-measured. Both are reported, but neither fails
the gate, so a run that simply tried less can't block a PR. The interpretation rules (units,
what `regressed` means, why a mismatched pair has no rows) live in the MCP tool
descriptions, which an agent reads once. Results carry only data.
