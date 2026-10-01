"""``gitm history`` and ``gitm diff`` as MCP tools, so an agent can ask instead of read JSON.

    gitm mcp [--scratch DIR]          # stdio; what Claude Code / Codex launch
    python -m gitm.mcp_server         # same thing

Built on the official MCP Python SDK, v2 (``mcp>=2.2,<3``: ``MCPServer``, which
was ``FastMCP`` in v1). It is an optional extra (``pip install 'gitm-labs[mcp]'``)
so the core package's dependency set does not grow for a read-only side door.

Three tools, all read-only and GPU-free:

* ``list_runs`` exists because the question people actually ask is "what moved
  between the last two Kimi runs", and neither run id is known up front. Real
  fingerprints are ``<vendor>:<hash>``, so "Kimi" is matched against the model
  name the engine config recorded as well as the fingerprint and workload id.
* ``history`` and ``diff`` return exactly what ``--json`` prints, from the same
  functions. The rules for reading them (what "regressed" means, the threshold's
  unit, why a mismatched pair has no rows) live in the tool descriptions, which
  an agent reads once, rather than being repeated inside every result.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from gitm.optimizer.history import (
    NoExport,
    UnreadableRun,
    history_as_dict,
    load_history,
    read_run,
)
from gitm.optimizer.run_diff import (
    DEFAULT_THRESHOLD,
    diff_as_dict,
    diff_runs,
    resolve_run,
)

SDK_REQUIREMENT = "mcp>=2.2,<3"

INSTRUCTIONS = (
    "Read-only access to Git.M optimization runs on this machine. Each run folder "
    "records the levers (runtime config changes) the loop tried, whether its rollback "
    "gate kept each one, and the measured throughput delta. Start with list_runs to "
    "find run ids (newest first), then diff two of them or read history across all."
)

LIST_RUNS_DOC = """List run folders, newest first.

`match` is a case-insensitive substring tested against the run id, workload fingerprint,
workload id, and the model name in the engine config (e.g. "kimi"). `gpu` is a substring
of the GPU SKU. Runs that measured nothing (no verification.json; any run with no engine
attached) are listed with measured=false so a missing run is visible, but they never match
a `match` or `gpu` filter because they record neither. Pass a row's `run_dir` to diff;
`run_id` is the id the run recorded, which is not always its folder name.
"""

HISTORY_DOC = """Aggregate every run into one record per (lever, GPU SKU, workload fingerprint).

Per record: runs, attempts (A/Bs), wins (kept and significant), losses (rolled back),
inconclusive (kept but inside the noise band), mean/best/worst delta as fractions
(0.05 = +5% throughput; null means no number, never zero), and conflicted=true when a lever
both won and lost. `skipped` lists unreadable run folders and why; `filtered` counts sound
runs excluded by the `gpu` / `fingerprint` filters (both exact matches).
"""

DIFF_DOC = f"""Compare two runs lever by lever. run_a is the baseline and run_b the run under test.
Each can be a run folder path, a run id, or a unique run-id prefix.

`preconditions` checks gpu_sku and fingerprint: "match", "mismatch", or "unverified" (a
side did not record a nonempty value). Both checks must be "match"; otherwise
`comparable` is false and `levers` is empty,
because a delta on one box or model says nothing about another. Say that; do not diff the
raw files instead.

Each lever has a `change`: verdict_changed (kept <-> rolled_back/mixed), delta_moved (mean
delta moved by more than `threshold`), only_in_a / only_in_b (tried in one run only), or
unchanged. `move` = b.mean_delta - a.mean_delta, in delta points. `regressed` is true only
when run A kept the lever AND run B measured it more than `threshold` lower. A lever
missing from B was not re-measured; that is not a regression. `threshold` defaults to
{DEFAULT_THRESHOLD} (2 pts), the export's own noise floor. A side with measured=false has
no verification.json: comparison is skipped, comparable=false, with no lever rows.
An empty regressions list in a refused/skipped comparison is not evidence of no regression.
The `regressions` list for a comparable pair is what a CI
gate fails on.
"""


def _model_of(results: list[dict[str, Any]]) -> str | None:
    """The model the engine was configured with, if the export recorded it."""
    for r in results:
        for side in ("baseline_config", "candidate_config"):
            cfg = r.get(side)
            if isinstance(cfg, dict) and isinstance(cfg.get("model"), str):
                return cfg["model"]
    return None


def list_runs_data(runs_dir: Path, *, match: str | None = None, gpu: str | None = None,
                   limit: int = 20) -> dict[str, Any]:
    """What ``list_runs`` returns. Plain function so tests need no MCP client."""
    rows: list[dict[str, Any]] = []
    if runs_dir.is_dir():
        for d in (p for p in runs_dir.iterdir() if p.is_dir()):
            try:
                e = read_run(d)
            except UnreadableRun as exc:
                rows.append({
                    "run_dir": str(d), "run_id": d.name, "measured": False,
                    "reason": str(exc), "damaged": not isinstance(exc, NoExport),
                    "mtime": d.stat().st_mtime,
                })
                continue
            rows.append({
                "run_dir": str(d), "run_id": e.run_id, "measured": True,
                "gpu_sku": e.gpu_sku, "fingerprint": e.fingerprint,
                "workload_id": e.workload_id, "model": _model_of(e.results),
                "n_results": len(e.results), "mtime": e.mtime,
            })

    def keep(row: dict[str, Any]) -> bool:
        if match:
            hay = [row.get(k) for k in ("run_id", "fingerprint", "workload_id", "model")]
            if not row["measured"] or not any(
                    isinstance(h, str) and match.lower() in h.lower() for h in hay):
                return False
        if gpu:
            sku = row.get("gpu_sku")
            if not (isinstance(sku, str) and gpu.lower() in sku.lower()):
                return False
        return True

    kept = sorted((r for r in rows if keep(r)), key=lambda r: r["mtime"], reverse=True)
    return {"runs_dir": str(runs_dir), "total": len(kept), "runs": kept[:max(limit, 0)]}


def build_server(scratch: str | None = None) -> Any:
    """The configured server. ``scratch`` pins the runs dir; else $GITM_SCRATCH."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise SystemExit(
            f"gitm mcp needs the MCP SDK ({SDK_REQUIREMENT}): "
            f"pip install 'gitm-labs[mcp]'  ({exc})"
        ) from exc

    from gitm import __version__
    from gitm._paths import runs_dir

    server = MCPServer("gitm", instructions=INSTRUCTIONS, version=__version__)
    ro = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

    @server.tool(description=LIST_RUNS_DOC, annotations=ro)
    def list_runs(match: str | None = None, gpu: str | None = None,
                  limit: int = 20) -> dict[str, Any]:
        return list_runs_data(runs_dir(scratch), match=match, gpu=gpu, limit=limit)

    @server.tool(description=HISTORY_DOC, annotations=ro)
    def history(gpu: str | None = None, fingerprint: str | None = None) -> dict[str, Any]:
        return history_as_dict(load_history(runs_dir(scratch), gpu_sku=gpu,
                                            fingerprint=fingerprint))

    @server.tool(description=DIFF_DOC, annotations=ro)
    def diff(run_a: str, run_b: str, threshold: float | None = None) -> dict[str, Any]:
        rd = runs_dir(scratch)
        try:
            d = diff_runs(resolve_run(run_a, rd), resolve_run(run_b, rd),
                          threshold=DEFAULT_THRESHOLD if threshold is None else threshold)
        except (UnreadableRun, ValueError) as exc:
            # A ToolError reaches the model as an error result it can act on
            # ("give more of the id"), not a crashed call.
            raise ToolError(str(exc)) from exc
        return diff_as_dict(d)

    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gitm mcp", description=__doc__.splitlines()[0])
    p.add_argument("--scratch", default=None, help="Override $GITM_SCRATCH.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    build_server(args.scratch or os.environ.get("GITM_SCRATCH")).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
