"""Read back what previous runs measured.

    load_history(runs_dir) -> History
    record_for(history, "kv_cache_dtype_fp8", gpu_sku=sku) -> LeverRecord | None

Every run writes ``runs/<run_id>/verification.json`` — each lever it tried, the
A/B behind it, and whether the rollback gate kept it. Nothing has ever opened one
again, so the loop ranks candidates from the same hand-authored constants on every
run no matter what it measured last time. This module is the missing direction.

No new writer is needed: the export already carries the knob, the speedup, the rep
count, whether the gain cleared the noise band, the gate's keep/rollback verdict,
and the GPU it ran on.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LeverRecord",
    "NoExport",
    "RunExport",
    "UnreadableRun",
    "aggregate",
    "runs_with_results",
    "History",
    "history_as_dict",
    "load_history",
    "read_run",
    "record_for",
    "render_history",
]

EXPORT_NAME = "verification.json"


@dataclass(frozen=True)
class LeverRecord:
    """What every past run, taken together, says about one lever on one GPU.

    ``conflicted`` rather than a single verdict: a lever that won twice and lost
    twice has not "averaged out to neutral", it has behaved differently under
    conditions this record does not capture. Collapsing that into a mean would
    present a disagreement as a measurement.
    """

    intervention_name: str
    gpu_sku: str | None
    #: The workload fingerprint the result was measured under — model plus
    #: config, from :func:`gitm.optimizer.qualification.fingerprint`. Part of the
    #: key for the same reason ``gpu_sku`` is: a lever that helps a sparse-MoE
    #: checkpoint says nothing about a dense one, and averaging a +40% on one
    #: model with a -20% on another reports a confident +10% describing neither.
    fingerprint: str | None
    #: Distinct run folders this lever appears in. Kept apart from ``attempts``
    #: because five A/Bs inside one run is far weaker evidence than five across
    #: five runs, and a single count cannot tell those apart.
    runs: int
    #: Individual A/Bs. ``wins + losses + inconclusive`` sums to this, not to
    #: ``runs`` — one run can measure the same lever more than once.
    attempts: int
    wins: int
    losses: int
    inconclusive: int
    #: ``None`` when no attempt recorded a usable delta, for the same reason
    #: :func:`record_for` returns ``None`` for a lever never tried: "we have no
    #: number" and "the number was zero" point opposite ways, and 0.0 for both
    #: would read as a measured no-op.
    mean_delta: float | None
    best_delta: float | None
    worst_delta: float | None
    last_run_id: str | None

    @property
    def conflicted(self) -> bool:
        return self.wins > 0 and self.losses > 0


@dataclass(frozen=True)
class History:
    """Every lever seen across the runs that were readable.

    ``skipped`` is part of the result, not a log line. A history assembled from
    three of twenty runs looks exactly like a history of three runs, and a caller
    that cannot tell those apart will read a thin record as a weak lever.

    ``filtered`` is counted separately and deliberately: a run excluded because it
    ran on a different GPU is the filter working, not history going missing. Adding
    the two together would make a clean read of one box look like a damaged one.
    """

    records: dict[tuple[str, str | None], LeverRecord] = field(default_factory=dict)
    runs_read: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    filtered: int = 0

    def __len__(self) -> int:
        return len(self.records)


def _named(value: Any) -> bool:
    """True when ``value`` is usable as an identity field: a string, or absent.

    These become part of the record key and of the run set, so a list or a dict
    here raises ``unhashable type`` out of the whole read — one damaged export
    taking every sound run with it, which is the failure this module exists to
    contain rather than cause. A number is hashable and would survive the load,
    then break the renderer on ``len()`` instead.
    """
    return value is None or isinstance(value, str)


def _finite(value: Any) -> float | None:
    """A real, finite number, or ``None``.

    ``bool`` is excluded deliberately: it is an ``int`` subclass, so ``True``
    would otherwise be read as a measured delta of 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _delta_of(result: dict[str, Any]) -> float | None:
    """The signed delta this A/B recorded, or ``None`` when it has none usable.

    A value that is not a real finite number is treated as absent rather than
    coerced, and the attempt still counts as won or lost. Two failures this
    prevents, both of which land in Phase 3 *after* the capture has been paid
    for: a string ``speedup`` raised ``TypeError`` out of the whole read, and a
    NaN propagated through the mean into ``json.dumps``, which emits the literal
    ``NaN`` \u2014 not valid JSON, in an artifact meant to be machine-read.
    """
    delta = _finite(result.get("delta"))
    if delta is not None:
        return delta
    speedup = _finite(result.get("speedup"))
    return None if speedup is None else speedup - 1.0


def _verdict(result: dict[str, Any]) -> str:
    """win / loss / inconclusive for one A/B.

    ``kept`` is the rollback gate's decision and is authoritative — the export is
    explicit that the gate decides, not the raw delta. ``significant`` only says
    the gain cleared the measured noise band, so a kept-but-insignificant result
    is a lever that was allowed to stay without having proven anything.
    """
    if not result.get("kept"):
        return "loss"
    return "win" if result.get("significant") else "inconclusive"


def runs_with_results(runs_dir: str | Path) -> int:
    """How many run folders under ``runs_dir`` left a verification export.

    Cheap enough to call before deciding whether there is anything worth asking
    the operator about: it stats each directory rather than parsing any of them.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return 0
    return sum(1 for p in runs_dir.iterdir() if p.is_dir() and (p / EXPORT_NAME).exists())


class UnreadableRun(ValueError):
    """A run folder whose export could not be read; ``str(exc)`` is the reason.

    The reason strings are the ones :attr:`History.skipped` has always carried,
    so a caller reading one run (``gitm diff``) and a caller reading all of them
    (``gitm history``) report the same failure in the same words.
    """


class NoExport(UnreadableRun):
    """The run folder exists but never wrote an export.

    Split out because it is the one unreadable case that is not damage: a run
    with no live engine attached measures nothing and writes no export by design
    (``scheduler/loop.py`` writes one only when an A/B produced results). A caller
    may reasonably accept "measured nothing" while still refusing "truncated".
    """


@dataclass(frozen=True)
class RunExport:
    """One run's export, validated and with its identity fields pulled out."""

    run_dir: Path
    run_id: str
    gpu_sku: str | None
    fingerprint: str | None
    workload_id: str | None
    mtime: float
    results: list[dict[str, Any]]


def read_run(run_dir: str | Path) -> RunExport:
    """Read and validate one run folder's ``verification.json``.

    Raises :class:`NoExport` when the folder has no export and
    :class:`UnreadableRun` when the export is there but cannot be trusted. Every
    check that decides whether a run counts lives here, so :func:`load_history`
    and a single-run reader cannot drift apart on what "readable" means.
    """
    d = Path(run_dir)
    if not d.is_dir():
        raise UnreadableRun("run dir does not exist")
    path = d / EXPORT_NAME
    if not path.exists():
        raise NoExport(f"no {EXPORT_NAME}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnreadableRun(f"unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise UnreadableRun("not a JSON object")
    results = doc.get("results")
    if not isinstance(results, list):
        raise UnreadableRun("no results array")
    env = doc.get("environment")
    if env is not None and not isinstance(env, dict):
        # Not tolerated the way a bad provenance is: run_id has a documented
        # fallback and the SKU has none, and a record keyed under the wrong
        # GPU is the one mistake this key exists to prevent.
        raise UnreadableRun("malformed environment")
    if not all(isinstance(r, dict) for r in results):
        # Reading the sound entries and dropping the rest would under-count
        # attempts with nothing saying so, which is what skipped is for.
        raise UnreadableRun("malformed result entry")
    if not all(_named(r.get("intervention_name")) for r in results):
        raise UnreadableRun("malformed intervention name")
    prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    sku = (env or {}).get("gpu_sku")
    fp = prov.get("fingerprint")
    # Both are keys and neither has a fallback: a record filed under the wrong
    # box or the wrong model is the mistake the key exists to prevent, so a
    # malformed one skips the run rather than keying under None.
    if not _named(sku):
        raise UnreadableRun("malformed gpu_sku")
    if not _named(fp):
        raise UnreadableRun("malformed fingerprint")
    # run_id is the one identity field with a documented fallback, so a
    # malformed one costs the run nothing. workload_id is display-only.
    raw_run_id = prov.get("run_id")
    run_id = raw_run_id if isinstance(raw_run_id, str) and raw_run_id else d.name
    workload_id = prov.get("workload_id")
    return RunExport(
        run_dir=d,
        run_id=run_id,
        gpu_sku=sku,
        fingerprint=fp,
        workload_id=workload_id if isinstance(workload_id, str) else None,
        mtime=path.stat().st_mtime,
        results=results,
    )


def aggregate(
    exports: list[RunExport],
) -> dict[tuple[str, str | None, str | None], LeverRecord]:
    """Fold exports into one :class:`LeverRecord` per (lever, gpu, workload).

    ``exports`` must already be in run order: ``last_run_id`` is whichever run
    came last in the list. Given a single export this is that run's own per-lever
    record, which is what ``gitm diff`` compares.
    """
    acc: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
    for e in exports:
        for r in e.results:
            name = r.get("intervention_name")
            if not name:
                continue
            key = (name, e.gpu_sku, e.fingerprint)
            a = acc.setdefault(
                key,
                {"runs": set(), "attempts": 0, "win": 0, "loss": 0,
                 "inconclusive": 0, "deltas": [], "last_run_id": None},
            )
            a["runs"].add(e.run_id)
            a["attempts"] += 1
            a[_verdict(r)] += 1
            delta = _delta_of(r)
            if delta is not None:
                a["deltas"].append(delta)
            a["last_run_id"] = e.run_id

    records = {}
    for (name, sku, fp), a in acc.items():
        deltas = a["deltas"]
        records[(name, sku, fp)] = LeverRecord(
            intervention_name=name,
            gpu_sku=sku,
            fingerprint=fp,
            runs=len(a["runs"]),
            attempts=a["attempts"],
            wins=a["win"],
            losses=a["loss"],
            inconclusive=a["inconclusive"],
            mean_delta=(sum(deltas) / len(deltas)) if deltas else None,
            best_delta=max(deltas) if deltas else None,
            worst_delta=min(deltas) if deltas else None,
            last_run_id=a["last_run_id"],
        )
    return records


def load_history(
    runs_dir: str | Path, *, gpu_sku: str | None = None,
    fingerprint: str | None = None,
) -> History:
    """Aggregate every readable ``verification.json`` under ``runs_dir``.

    ``gpu_sku`` filters to one GPU and ``fingerprint`` to one workload: a result
    measured on an H100 says nothing about an MI355X, and one measured on a
    sparse-MoE checkpoint says nothing about a dense one. A caller ranking for a
    particular box and model should see neither of the others' records. Runs are ordered by the export's mtime — the export carries a
    ``run_id`` but no timestamp, and the run directories are UUIDs, so the file
    is the only ordering available for ``last_run_id``.
    """
    runs_dir = Path(runs_dir)
    skipped: dict[str, str] = {}
    filtered = 0
    if not runs_dir.is_dir():
        return History(skipped={str(runs_dir): "runs dir does not exist"})

    exports: list[RunExport] = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        # Shape is checked (in read_run) before the GPU filter so that
        # "filtered" only ever means a sound export from another box, never a
        # damaged one that was never really read.
        try:
            export = read_run(d)
        except UnreadableRun as exc:
            skipped[d.name] = str(exc)
            continue
        if (gpu_sku is not None and export.gpu_sku != gpu_sku) or (
            fingerprint is not None and export.fingerprint != fingerprint
        ):
            filtered += 1
            continue
        exports.append(export)

    exports.sort(key=lambda e: e.mtime)
    return History(records=aggregate(exports), runs_read=len(exports),
                   skipped=skipped, filtered=filtered)


def history_as_dict(history: History) -> dict[str, Any]:
    """The JSON shape of ``gitm history --json``, shared with the MCP tool."""
    return {
        "runs_read": history.runs_read,
        "filtered": history.filtered,
        "skipped": history.skipped,
        "records": [
            {**asdict(r), "conflicted": r.conflicted}
            for r in history.records.values()
        ],
    }


def record_for(
    history: History, intervention_name: str, *, gpu_sku: str | None = None,
    fingerprint: str | None = None,
) -> LeverRecord | None:
    """The record for one lever on one box under one workload, or ``None``.

    ``None`` rather than a zeroed record on purpose: "never measured" and
    "measured, and it did nothing" must not look alike to a caller deciding
    whether this lever is worth an experiment. They point opposite ways.

    Both ``gpu_sku`` and ``fingerprint`` are part of the key, so asking with the
    wrong one returns nothing rather than another context's answer. A caller
    that has no fingerprint gets records measured without one, not all of them.
    """
    return history.records.get((intervention_name, gpu_sku, fingerprint))


def _fit(value: str, width: int) -> str:
    """``value`` inside ``width``, marked when it did not fit.

    A silent cut reads as the whole value, which is how a truncated SKU came to
    name a board it was not measured on.
    """
    return value if len(value) <= width else value[: width - 1] + "\u2026"


def _column(rows: list[LeverRecord], of, header: str, *, cap: int) -> int:
    """Width of a column: the longest value shown, bounded by ``cap``."""
    return min(max([len(header)] + [len(of(r)) for r in rows]), cap)


def render_history(history: History, *, top: int = 20) -> str:
    """The record as a table, most-tried first."""
    head = f"read {history.runs_read} runs"
    if history.filtered:
        head += f", {history.filtered} filtered out by gpu"
    if history.skipped:
        head += f", skipped {len(history.skipped)}"
    out = [head]
    if history.skipped:
        reasons: dict[str, int] = {}
        for reason in history.skipped.values():
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        out.append("  skipped: " + ", ".join(f"{n} {r}" for r, n in sorted(reasons.items())))
    if not history.records:
        out.append("\nno levers recorded")
        return "\n".join(out)

    rows = sorted(history.records.values(), key=lambda r: (-r.runs, r.intervention_name))
    unknown = sum(1 for r in rows if r.gpu_sku is None)
    if unknown:
        out.append(
            f"  WARNING: {unknown} record(s) have no GPU. Runs that did not report a SKU "
            "all key together, so results measured on DIFFERENT boxes are merged here. "
            "Set GITM_GPU_SKU on boxes where NVML cannot name the device (any ROCm/AMD "
            "part) so these separate correctly."
        )
    shown = rows[:top]
    # Sized to what is actually on screen, because a fixed width silently ate the
    # part of a SKU that identifies the board: "AMD Instinct MI355X" rendered as
    # "MI355", and an H100 HBM3 and HBM3e came out byte-identical. Two boxes that
    # look like one box is the exact confusion keying on gpu_sku exists to stop.
    name_w = _column(shown, lambda r: r.intervention_name, "lever", cap=40)
    gpu_w = _column(shown, lambda r: r.gpu_sku or "-", "gpu", cap=30)
    fp_w = _column(shown, lambda r: r.fingerprint or "-", "workload", cap=24)

    out.append("")
    out.append(f"  {'lever':{name_w}s} {'gpu':{gpu_w}s} {'workload':{fp_w}s} "
               f"{'runs':>5s} {'a/b':>4s} "
               f"{'won':>4s} {'lost':>5s} {'incon':>6s} {'mean':>8s}  last")
    for r in shown:
        flag = "  CONFLICTED" if r.conflicted else ""
        mean = f"{r.mean_delta:+.1%}" if r.mean_delta is not None else "n/a"
        out.append(
            f"  {_fit(r.intervention_name, name_w):{name_w}s} "
            f"{_fit(r.gpu_sku or '-', gpu_w):{gpu_w}s} "
            f"{_fit(r.fingerprint or '-', fp_w):{fp_w}s} "
            f"{r.runs:5d} {r.attempts:4d} {r.wins:4d} {r.losses:5d} {r.inconclusive:6d} "
            f"{mean:>7s}  {(r.last_run_id or '-')[:8]}{flag}"
        )
    if len(rows) > top:
        out.append(f"  ... {len(rows) - top} more")
    return "\n".join(out)
