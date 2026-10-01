"""`gitm diff`: two runs side by side, per lever, and the exit codes the CI gate reads.

Fixtures are checked in under tests/fixtures/diff/runs/ and tests/golden/diff_gate/,
written by tests/fixtures/diff/generate_diff_runs.py; see its docstring for what each
folder is for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gitm.cli import main
from gitm.optimizer.history import UnreadableRun, load_history, read_run
from gitm.optimizer.report import Provenance
from gitm.optimizer.run_diff import (
    DEFAULT_THRESHOLD,
    diff_as_dict,
    diff_runs,
    render_diff,
    resolve_run,
)
from gitm.optimizer.verification_export import (
    MIN_NOISE_BAND,
    VerificationRecord,
    write_verification,
)

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "diff_gate" / "kimi-mi355x"
RUNS = REPO / "tests" / "fixtures" / "diff" / "runs"


def _by_lever(d):
    return {row.lever: row for row in d.levers}


def test_the_checked_in_fixtures_are_readable_exports():
    """If the export format moves, regenerate rather than let these rot."""
    assert read_run(GOLDEN).fingerprint == "rocm:3f9c1a7be2d40c55"
    for name in ("rerun_noise", "regressed", "improved", "other_gpu", "no_sku"):
        read_run(RUNS / name)


def test_a_rerun_inside_the_noise_band_moves_nothing():
    d = diff_runs(GOLDEN, RUNS / "rerun_noise")

    assert d.comparable
    assert d.preconditions == {"gpu_sku": "match", "fingerprint": "match"}
    assert {row.change for row in d.levers} == {"unchanged"}
    assert d.regressions == []


def test_each_kind_of_change_is_named():
    rows = _by_lever(diff_runs(GOLDEN, RUNS / "regressed"))

    assert rows["enable_expert_parallel"].change == "delta_moved"
    assert rows["kv_cache_dtype_fp8"].change == "verdict_changed"
    assert rows["enforce_eager"].change == "verdict_changed"
    assert rows["mla_triton_backend"].change == "only_in_a"
    assert rows["rccl_algo_ring"].change == "only_in_b"
    assert rows["max_num_seqs_512"].change == "unchanged"


def test_regressed_means_kept_in_a_and_measured_lower_in_b():
    d = diff_runs(GOLDEN, RUNS / "regressed")
    rows = _by_lever(d)

    assert {r.lever for r in d.regressions} == {"enable_expert_parallel", "kv_cache_dtype_fp8"}
    assert rows["enable_expert_parallel"].move == pytest.approx(-0.18)
    # Two attempts in one run fold into one mean, as history folds them.
    assert rows["kv_cache_dtype_fp8"].a.mean_delta == pytest.approx(0.095)
    assert rows["kv_cache_dtype_fp8"].a.attempts == 2


def test_a_lever_absent_from_b_is_not_a_regression():
    """Absence is 'not re-measured', not 'got worse': a run that tried less must not
    fail a gate for levers it never measured."""
    rows = _by_lever(diff_runs(GOLDEN, RUNS / "regressed"))

    assert rows["mla_triton_backend"].b is None
    assert rows["mla_triton_backend"].move is None
    assert not rows["mla_triton_backend"].regressed


def test_a_lever_rolled_back_in_a_cannot_regress():
    """enforce_eager was rolled back in the golden run: there was no kept gain to lose.
    Swap the sides and it is kept (+1%) in A and rolled back (-9%) in B, which is
    exactly a regression; the rule is about A's verdict, not about which run is older."""
    assert not _by_lever(diff_runs(GOLDEN, RUNS / "regressed"))["enforce_eager"].regressed
    reverse = _by_lever(diff_runs(RUNS / "regressed", GOLDEN))
    assert reverse["enforce_eager"].regressed
    assert not reverse["rccl_algo_ring"].regressed  # only in A, and rolled back there


def test_an_improvement_moves_but_does_not_regress():
    d = diff_runs(GOLDEN, RUNS / "improved")

    assert _by_lever(d)["enable_expert_parallel"].move == pytest.approx(0.06)
    assert d.regressions == []


def test_the_threshold_decides_what_counts():
    assert DEFAULT_THRESHOLD == MIN_NOISE_BAND
    loose = diff_runs(GOLDEN, RUNS / "regressed", threshold=0.20)
    assert loose.regressions == []
    tight = diff_runs(GOLDEN, RUNS / "rerun_noise", threshold=0.001)
    assert any(r.change == "delta_moved" for r in tight.levers)
    with pytest.raises(ValueError):
        diff_runs(GOLDEN, RUNS / "rerun_noise", threshold=-0.1)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_invalid_threshold_is_rejected_before_reading_runs(tmp_path, threshold):
    """Invalid configuration must fail even before an empty-run exception can apply."""
    with pytest.raises(ValueError, match="threshold must be >= 0 and finite"):
        diff_runs(tmp_path / "missing-a", tmp_path / "missing-b", threshold=threshold)


@pytest.mark.parametrize("threshold,regressions", [(0.0, 2), (0.02, 2), (0.5, 0)])
def test_finite_nonnegative_thresholds_still_gate_measured_regressions(threshold, regressions):
    assert len(diff_runs(GOLDEN, RUNS / "regressed", threshold=threshold).regressions) == regressions


@pytest.mark.parametrize("other,field", [("other_gpu", "gpu_sku"),
                                         ("other_workload", "fingerprint")])
def test_different_box_or_workload_is_refused_not_footnoted(other, field):
    d = diff_runs(GOLDEN, RUNS / other)

    assert not d.comparable
    assert d.preconditions[field] == "mismatch"
    assert d.levers == []  # no rows to misread as regressions
    assert "NOT COMPARABLE" in render_diff(d)


def _with_identity(dest, field, value):
    """Use the export writer so identity tests exercise the real reader contract."""
    original = read_run(GOLDEN)
    identity = {"gpu_sku": original.gpu_sku, "fingerprint": original.fingerprint}
    identity[field] = value
    provenance = Provenance(
        workload_id=original.workload_id, fingerprint=identity["fingerprint"],
        run_id=dest.name, git_sha="test", gitm_version="test",
        started_at_ns=0, ended_at_ns=1,
    )
    write_verification([VerificationRecord(**r) for r in original.results], provenance,
                       dest / "verification.json", gpu_sku=identity["gpu_sku"])
    return dest


@pytest.mark.parametrize("field", ["gpu_sku", "fingerprint"])
@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("side", ["a", "b", "both"])
def test_unknown_identity_cannot_establish_comparability(tmp_path, field, value, side):
    unknown = _with_identity(tmp_path / "unknown", field, value)
    a = unknown if side in ("a", "both") else GOLDEN
    b = unknown if side in ("b", "both") else GOLDEN
    d = diff_runs(a, b)

    assert not d.comparable
    assert d.preconditions[field] == "unverified"
    assert d.levers == []
    assert d.regressions == []
    assert f"NOT COMPARABLE: {field} is unverified" in render_diff(d)


def test_a_run_with_no_export_is_a_side_that_measured_nothing():
    d = diff_runs(GOLDEN, RUNS / "no_export")

    assert not d.b.measured
    assert d.b.reason == "no verification.json"
    assert not d.comparable
    assert d.levers == []
    assert d.regressions == []


def test_a_damaged_export_is_refused_never_read_as_empty():
    with pytest.raises(UnreadableRun, match="truncated.*unreadable"):
        diff_runs(GOLDEN, RUNS / "truncated")


def test_the_diff_reads_what_history_reads():
    """Same reader, same numbers: a lever's mean in a one-run diff equals its mean in
    a one-run history."""
    runs = REPO / "tests" / "golden" / "diff_gate"
    rec = next(r for r in load_history(runs).records.values()
               if r.intervention_name == "kv_cache_dtype_fp8")
    row = _by_lever(diff_runs(GOLDEN, RUNS / "rerun_noise"))["kv_cache_dtype_fp8"]
    assert row.a.mean_delta == rec.mean_delta


def test_json_shape_round_trips():
    doc = json.loads(json.dumps(diff_as_dict(diff_runs(GOLDEN, RUNS / "regressed"))))

    assert doc["comparable"] is True
    assert doc["regressions"] == ["kv_cache_dtype_fp8", "enable_expert_parallel"]
    assert doc["a"]["gpu_sku"] == "AMD Instinct MI355X"
    assert {row["change"] for row in doc["levers"]} >= {"only_in_a", "only_in_b"}


def test_resolve_run_takes_a_path_an_id_or_a_unique_prefix(tmp_path):
    for name in ("abc123", "abd456"):
        (tmp_path / name).mkdir()

    assert resolve_run(str(tmp_path / "abc123"), tmp_path) == tmp_path / "abc123"
    assert resolve_run("abc", tmp_path) == tmp_path / "abc123"
    with pytest.raises(UnreadableRun, match="matches 2 runs"):
        resolve_run("ab", tmp_path)
    with pytest.raises(UnreadableRun, match="no run folder"):
        resolve_run("zzz", tmp_path)


# --- CLI: the exit codes are the gate's contract ---------------------------------


def _cli(capsys, *argv):
    rc = main(["diff", *map(str, argv)])
    return rc, capsys.readouterr()


def test_cli_exits_0_when_nothing_regressed(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "rerun_noise", "--check")
    assert rc == 0
    assert "0 regressions" in out.out


def test_cli_check_exits_1_on_a_regression_and_only_with_check(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "regressed", "--check")
    assert rc == 1
    assert "REGRESSED" in out.out
    rc, _ = _cli(capsys, GOLDEN, RUNS / "regressed")
    assert rc == 0


def test_cli_threshold_flag_reaches_the_diff(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "regressed", "--check", "--threshold", "0.5")
    assert rc == 0
    assert "±50.0 pts" in out.out


def test_cli_exits_2_on_a_mismatch_even_with_check(capsys):
    rc, _ = _cli(capsys, GOLDEN, RUNS / "other_gpu", "--check")
    assert rc == 2


def test_cli_empty_run_needs_allow_empty(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "no_export", "--check")
    assert rc == 2
    assert "--allow-empty" in out.err
    rc, _ = _cli(capsys, GOLDEN, RUNS / "no_export", "--check", "--allow-empty")
    assert rc == 0


def test_cli_damaged_export_exits_2_even_with_allow_empty(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "truncated", "--allow-empty")
    assert rc == 2
    assert "unreadable" in out.err


def test_cli_json(capsys):
    rc, out = _cli(capsys, GOLDEN, RUNS / "regressed", "--json")
    assert rc == 0
    assert json.loads(out.out)["regressions"] == ["kv_cache_dtype_fp8", "enable_expert_parallel"]


def test_cli_resolves_run_ids_under_scratch(tmp_path, capsys):
    runs = tmp_path / "runs"
    for name, src in (("aaaa1111", GOLDEN), ("bbbb2222", RUNS / "regressed")):
        shutil.copytree(src, runs / name)

    rc, out = _cli(capsys, "aaaa", "bbbb", "--scratch", tmp_path, "--check")
    assert rc == 1
    assert "(aaaa1111)" in out.out


def test_cli_diff_against_what_a_no_engine_run_leaves(tmp_path, capsys):
    """The CI gate's exact shape: the loop with no engine attached, then a diff."""
    assert main(["run", "--workload", "vllm-decode", "--budget", "1s",
                 "--scratch", str(tmp_path)]) == 3  # no_data
    capsys.readouterr()
    (ci_run,) = (tmp_path / "runs").iterdir()

    rc, out = _cli(capsys, GOLDEN, ci_run, "--check", "--allow-empty")
    assert rc == 0
    assert "measured nothing" in out.out


@pytest.mark.parametrize("field", ["gpu_sku", "fingerprint"])
@pytest.mark.parametrize("flags", [[], ["--check"], ["--allow-empty"],
                                   ["--check", "--allow-empty"]])
def test_cli_cannot_override_unknown_measured_identity(tmp_path, capsys, field, flags):
    unknown = _with_identity(tmp_path / "unknown", field, None)
    rc, out = _cli(capsys, GOLDEN, unknown, "--scratch", tmp_path, "--json", *flags)
    assert rc == 2
    doc = json.loads(out.out)
    assert doc["comparable"] is False
    assert doc["levers"] == []


@pytest.mark.parametrize("other", ["other_gpu", "other_workload"])
def test_allow_empty_cannot_override_a_measured_mismatch(capsys, other):
    rc, _ = _cli(capsys, GOLDEN, RUNS / other, "--check", "--allow-empty")
    assert rc == 2


@pytest.mark.parametrize("side", ["a", "b", "both"])
def test_allow_empty_accepts_a_skip_without_claiming_comparison(capsys, side):
    a = RUNS / "no_export" if side in ("a", "both") else GOLDEN
    b = RUNS / "no_export" if side in ("b", "both") else GOLDEN
    rc, out = _cli(capsys, a, b, "--check", "--allow-empty", "--json")
    assert rc == 0
    doc = json.loads(out.out)
    assert doc["comparable"] is False
    assert doc["levers"] == []
    assert doc["regressions"] == []


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "1e309", "-0.1"])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("candidate", ["regressed", "no_export"])
def test_cli_invalid_threshold_cannot_pass_the_gate(capsys, threshold, json_output, candidate):
    flags = ["--json"] if json_output else []
    rc, out = _cli(capsys, GOLDEN, RUNS / candidate, "--check", "--allow-empty",
                   f"--threshold={threshold}", *flags)
    assert rc == 2
    assert out.out == ""
    assert "threshold must be >= 0 and finite" in out.err


@pytest.mark.parametrize("ref,expected", [("abc", "abc"), ("abcdef", "abcdef"),
                                         ("abcd", "abcdef")])
def test_exact_run_ids_take_precedence_and_unique_prefixes_still_work(tmp_path, ref, expected):
    for name in ("abc", "abcdef"):
        (tmp_path / name).mkdir()
    assert resolve_run(ref, tmp_path) == tmp_path / expected


def test_a_nonexact_shared_prefix_remains_ambiguous(tmp_path):
    for name in ("abc", "abcdef"):
        (tmp_path / name).mkdir()
    with pytest.raises(UnreadableRun, match="'ab' matches 2 runs.*give more of the id"):
        resolve_run("ab", tmp_path)


def test_an_unknown_run_id_preserves_the_not_found_error(tmp_path):
    (tmp_path / "abc").mkdir()
    with pytest.raises(UnreadableRun, match="no run folder 'unknown'"):
        resolve_run("unknown", tmp_path)


@pytest.mark.parametrize("absolute", [False, True])
def test_existing_paths_take_precedence_over_scratch_ids(tmp_path, monkeypatch, absolute):
    """A relative directory in cwd must still win over the same ID under scratch."""
    runs = tmp_path / "runs"
    (runs / "abc").mkdir(parents=True)
    (runs / "abcdef").mkdir()
    local = tmp_path / "abc"
    local.mkdir()
    monkeypatch.chdir(tmp_path)
    ref = str(local) if absolute else "abc"
    assert resolve_run(ref, runs) == (local if absolute else Path("abc"))


def test_exact_id_lookup_does_not_add_scratch_relative_path_resolution(tmp_path, monkeypatch):
    """A slash-containing ref that is not an existing path is not a child run ID."""
    runs = tmp_path / "runs"
    (runs / "group" / "abc").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UnreadableRun, match="no run folder"):
        resolve_run("group/abc", runs)


def test_cli_exact_id_wins_over_a_longer_run_id(tmp_path, capsys):
    for name, src in (("abc", GOLDEN), ("abcdef", RUNS / "regressed")):
        shutil.copytree(src, tmp_path / "runs" / name)
    rc, out = _cli(capsys, "abc", "abcdef", "--scratch", tmp_path, "--check", "--json")
    assert rc == 1
    doc = json.loads(out.out)
    assert Path(doc["a"]["run_dir"]).name == "abc"
    assert Path(doc["b"]["run_dir"]).name == "abcdef"
    assert len(doc["regressions"]) == 2
