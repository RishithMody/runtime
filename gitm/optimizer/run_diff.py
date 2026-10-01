"""Put two runs side by side, per lever.

    diff_runs(run_a, run_b, threshold=0.02) -> RunDiff
    render_diff(diff) -> str
    diff_as_dict(diff) -> dict

``gitm history`` answers "what has this lever done, across every run". This
answers the narrower question two people ask after running the same model on the
same box a week apart: what moved between *these two*. Each side is read through
:func:`gitm.optimizer.history.read_run` and folded by
:func:`~gitm.optimizer.history.aggregate`, so a run this reports on is exactly a
run ``gitm history`` would have counted, and a lever measured twice inside one
run is averaged the same way in both places.

Three decisions are made here and nowhere else:

* **Same GPU and same workload is a precondition.** Both identities must be
  populated and equal before any lever rows are produced. Unknown identity
  cannot establish comparability. The CLI may explicitly accept an absent export
  as a skipped comparison for GPU-less CI, without claiming the runs match.
* **The threshold is in delta points and defaults to the export's own noise
  floor** (:data:`~gitm.optimizer.verification_export.MIN_NOISE_BAND`). A move
  smaller than the band each run already states for itself is, by the export's
  own definition, a re-measurement agreeing with the first.
* **"Regressed" needs two numbers.** A lever regressed only if run A kept it and
  run B measured it at a delta more than ``threshold`` below A's. A lever absent
  from B was not measured, so it is reported (``only_in_a``) but never called a
  regression; a lever rolled back in A had no gain to lose. Treating absence as a
  regression would fail every gate against a run that simply tried less.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gitm.optimizer.history import (
    LeverRecord,
    NoExport,
    RunExport,
    UnreadableRun,
    aggregate,
    read_run,
)
from gitm.optimizer.verification_export import MIN_NOISE_BAND

__all__ = [
    "DEFAULT_THRESHOLD",
    "LeverDiff",
    "LeverState",
    "RunDiff",
    "RunSide",
    "diff_as_dict",
    "diff_runs",
    "render_diff",
]

#: Default move, in delta points (0.02 == 2 percentage points of speedup), below
#: which a lever counts as unchanged. See the module docstring for why.
DEFAULT_THRESHOLD = MIN_NOISE_BAND

# Change kinds, in the order the table shows them.
VERDICT_CHANGED = "verdict_changed"
DELTA_MOVED = "delta_moved"
ONLY_IN_A = "only_in_a"
ONLY_IN_B = "only_in_b"
UNCHANGED = "unchanged"
_ORDER = (VERDICT_CHANGED, DELTA_MOVED, ONLY_IN_A, ONLY_IN_B, UNCHANGED)

# Precondition states.
MATCH = "match"
MISMATCH = "mismatch"
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class RunSide:
    """Who one side of the diff is. ``measured`` is False when it has no export."""

    run_dir: str
    run_id: str | None
    gpu_sku: str | None
    fingerprint: str | None
    workload_id: str | None
    measured: bool
    #: Why the run has no numbers, when ``measured`` is False.
    reason: str | None = None


@dataclass(frozen=True)
class LeverState:
    """One lever inside one run.

    ``verdict`` is ``mixed`` when the same run kept one attempt and rolled back
    another — the in-run version of history's ``conflicted``, kept distinct for
    the same reason: it is a disagreement, not an average.
    """

    verdict: str  # "kept" | "rolled_back" | "mixed"
    attempts: int
    mean_delta: float | None

    @classmethod
    def of(cls, r: LeverRecord) -> LeverState:
        kept = r.wins + r.inconclusive
        verdict = "kept" if r.losses == 0 else "rolled_back" if kept == 0 else "mixed"
        return cls(verdict=verdict, attempts=r.attempts, mean_delta=r.mean_delta)


@dataclass(frozen=True)
class LeverDiff:
    lever: str
    change: str
    a: LeverState | None
    b: LeverState | None
    #: ``b.mean_delta - a.mean_delta`` in delta points, or ``None`` when either
    #: side has no number. Never 0.0 for "unknown", same as history.
    move: float | None
    regressed: bool


@dataclass(frozen=True)
class RunDiff:
    a: RunSide
    b: RunSide
    threshold: float
    #: ``{"gpu_sku": ..., "fingerprint": ...}``, each match / mismatch / unverified.
    preconditions: dict[str, str]
    levers: list[LeverDiff] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return all(self.preconditions.get(k) == MATCH for k in ("gpu_sku", "fingerprint"))

    @property
    def regressions(self) -> list[LeverDiff]:
        return [d for d in self.levers if d.regressed]


def _side(run_dir: Path) -> tuple[RunSide, RunExport | None]:
    """Read one side. A run with no export is a side with no numbers; any other
    unreadable export raises, because "damaged" must not diff as "tried nothing"."""
    try:
        e = read_run(run_dir)
    except NoExport as exc:
        return RunSide(str(run_dir), None, None, None, None, measured=False,
                       reason=str(exc)), None
    except UnreadableRun as exc:
        raise UnreadableRun(f"{run_dir}: {exc}") from exc
    return RunSide(str(run_dir), e.run_id, e.gpu_sku, e.fingerprint, e.workload_id,
                   measured=True), e


def _precondition(a: RunSide, b: RunSide, attr: str) -> str:
    va, vb = getattr(a, attr), getattr(b, attr)
    if not va or not vb or not va.strip() or not vb.strip():
        return UNVERIFIED
    return MATCH if va == vb else MISMATCH


def _levers(e: RunExport | None) -> dict[str, LeverState]:
    if e is None:
        return {}
    return {name: LeverState.of(r) for (name, _, _), r in aggregate([e]).items()}


def _compare(name: str, a: LeverState | None, b: LeverState | None,
             threshold: float) -> LeverDiff:
    if a is None or b is None:
        change = ONLY_IN_B if a is None else ONLY_IN_A
        return LeverDiff(name, change, a, b, move=None, regressed=False)
    move = (
        b.mean_delta - a.mean_delta
        if a.mean_delta is not None and b.mean_delta is not None else None
    )
    if a.verdict != b.verdict:
        change = VERDICT_CHANGED
    elif move is not None and abs(move) > threshold:
        change = DELTA_MOVED
    else:
        change = UNCHANGED
    regressed = a.verdict != "rolled_back" and move is not None and move < -threshold
    return LeverDiff(name, change, a, b, move=move, regressed=regressed)


def diff_runs(run_a: str | Path, run_b: str | Path, *,
              threshold: float = DEFAULT_THRESHOLD) -> RunDiff:
    """Compare two run folders, lever by lever.

    Raises :class:`~gitm.optimizer.history.UnreadableRun` if either folder is
    missing or its export is damaged. A folder that simply has no export comes
    back as a side with ``measured=False``. Unknown or mismatched identity
    refuses the comparison with no lever rows, including when an export is absent.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    a, ea = _side(Path(run_a))
    b, eb = _side(Path(run_b))
    pre = {attr: _precondition(a, b, attr) for attr in ("gpu_sku", "fingerprint")}
    result = RunDiff(a=a, b=b, threshold=threshold, preconditions=pre)
    if not result.comparable:
        return result
    la, lb = _levers(ea), _levers(eb)
    rows = [_compare(n, la.get(n), lb.get(n), threshold) for n in sorted(la.keys() | lb.keys())]
    rows.sort(key=lambda d: (not d.regressed, _ORDER.index(d.change), d.lever))
    return RunDiff(a=a, b=b, threshold=threshold, preconditions=pre, levers=rows)


def diff_as_dict(diff: RunDiff) -> dict[str, Any]:
    """The JSON shape of ``gitm diff --json``, shared with the MCP tool."""
    return {
        "a": asdict(diff.a),
        "b": asdict(diff.b),
        "threshold": diff.threshold,
        "preconditions": diff.preconditions,
        "comparable": diff.comparable,
        "regressions": [d.lever for d in diff.regressions],
        "levers": [asdict(d) for d in diff.levers],
    }


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.1%}"


def _state(s: LeverState | None) -> str:
    if s is None:
        return "-"
    return f"{s.verdict:11s} {_pct(s.mean_delta):>7s}"


def _who(label: str, s: RunSide) -> str:
    if not s.measured:
        return f"run {label}  {Path(s.run_dir).name}  measured nothing ({s.reason})"
    return (f"run {label}  {(s.run_id or '-')[:8]}  {s.gpu_sku or '-'}  "
            f"{s.fingerprint or '-'}  ({Path(s.run_dir).name})")


def render_diff(diff: RunDiff, *, show_unchanged: bool = False) -> str:
    """The diff as a terminal table: what moved first, what did not collapsed."""
    out = [_who("a", diff.a), _who("b", diff.b)]
    pre = ", ".join(f"{k} {v}" for k, v in diff.preconditions.items())
    out.append(f"preconditions: {pre}; threshold ±{diff.threshold * 100:.1f} pts")

    if not diff.comparable:
        if not (diff.a.measured and diff.b.measured):
            out.append("  Comparison skipped: an export is absent; no levers compared.")
            return "\n".join(out)
        bad = [k for k, v in diff.preconditions.items() if v != MATCH]
        for k in bad:
            reason = "differs" if diff.preconditions[k] == MISMATCH else "is unverified"
            out.append(f"  NOT COMPARABLE: {k} {reason} "
                       f"(a: {getattr(diff.a, k)!r}, b: {getattr(diff.b, k)!r})")
        out.append("  A lever's delta on one box or workload says nothing about another; "
                   "no levers compared.")
        return "\n".join(out)

    shown = [d for d in diff.levers if show_unchanged or d.change != UNCHANGED]
    same = [d.lever for d in diff.levers if d.change == UNCHANGED]
    if not diff.levers:
        out.append("\nno levers in either run")
        return "\n".join(out)
    if shown:
        w = max(len("lever"), *(len(d.lever) for d in shown))
        out.append("")
        out.append(f"  {'lever':{w}s}  {'a':19s}  {'b':19s}  {'move':>9s}  change")
        for d in shown:
            move = "n/a" if d.move is None else f"{d.move * 100:+.1f} pts"
            flag = "  REGRESSED" if d.regressed else ""
            out.append(f"  {d.lever:{w}s}  {_state(d.a):19s}  {_state(d.b):19s}  "
                       f"{move:>9s}  {d.change}{flag}")
    if same and not show_unchanged:
        out.append(f"\n  unchanged within threshold: {len(same)} ({', '.join(same)})")
    n = len(diff.regressions)
    out.append(f"\n{n} regression{'s' if n != 1 else ''}"
               + (": " + ", ".join(d.lever for d in diff.regressions) if n else ""))
    return "\n".join(out)


def resolve_run(ref: str, runs_dir: Path) -> Path:
    """A run folder from a path, a run id, or a unique run-id prefix.

    Prefixes because the table prints eight characters of the id, and that is
    what a person will paste back.
    """
    p = Path(ref).expanduser()
    if p.is_dir():
        return p
    if runs_dir.is_dir():
        hits = sorted(d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(ref))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise UnreadableRun(
                f"{ref!r} matches {len(hits)} runs under {runs_dir}; give more of the id")
    raise UnreadableRun(f"no run folder {ref!r} (not a path, nor a run id under {runs_dir})")
