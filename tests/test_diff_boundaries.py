"""A decimal threshold tie must not become a regression through binary roundoff."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from gitm.cli import main
from gitm.optimizer.history import read_run
from gitm.optimizer.report import Provenance
from gitm.optimizer.run_diff import LeverState, _compare, diff_runs
from gitm.optimizer.verification_export import VerificationRecord, write_verification

GOLDEN = Path(__file__).resolve().parent / "golden/diff_gate/kimi-mi355x"


@pytest.mark.parametrize("a,b", [(0.10, 0.08), (0.08, 0.10), (-0.08, -0.10),
                                 (-0.10, -0.08), (0.01, -0.01), (-0.01, 0.01),
                                 (0.49, 0.47)])
@pytest.mark.parametrize("offset", [-1e-15, 0.0, 1e-15])
def test_only_moves_above_the_decimal_boundary_count(a, b, offset):
    direction = 1 if b > a else -1
    candidate = b + direction * offset
    row = _compare("lever", LeverState("kept", 1, a),
                   LeverState("kept", 1, candidate), 0.02)
    exceeds = offset > 0
    assert row.change == ("delta_moved" if exceeds else "unchanged")
    assert row.regressed is (exceeds and direction < 0)
    assert row.move == candidate - a  # Output retains the original measured move.


@pytest.mark.parametrize("b", [math.nextafter(0.1, math.inf), math.nextafter(0.1, -math.inf),
                               0.1])
def test_zero_threshold_preserves_the_smallest_representable_moves(b):
    row = _compare("lever", LeverState("kept", 1, 0.1), LeverState("kept", 1, b), 0.0)
    assert row.change == ("unchanged" if b == 0.1 else "delta_moved")
    assert row.regressed is (b < 0.1)


def test_a_verdict_flip_at_the_boundary_is_visible_without_a_regression():
    row = _compare("lever", LeverState("kept", 1, 0.1),
                   LeverState("rolled_back", 1, 0.08), 0.02)
    assert row.change == "verdict_changed"
    assert not row.regressed


@pytest.fixture
def boundary_runs(tmp_path):
    """Write one real-shaped attempt per run, including speedup-to-delta roundoff."""
    original = read_run(GOLDEN)
    record = VerificationRecord(**original.results[0])

    def write(name, speedup):
        dest = tmp_path / name
        attempt = replace(record, speedup=speedup, delta=speedup - 1.0,
                          candidate_tps=record.baseline_tps * speedup)
        prov = Provenance(workload_id=original.workload_id, fingerprint=original.fingerprint,
                          run_id=name, git_sha="test", gitm_version="test",
                          started_at_ns=0, ended_at_ns=1)
        write_verification([attempt], prov, dest / "verification.json",
                           gpu_sku=original.gpu_sku)
        return dest

    return write


@pytest.mark.parametrize("speedup,rc", [(1.080000000001, 0), (1.08, 0), (1.079999999999, 1)])
@pytest.mark.parametrize("json_output", [False, True])
def test_cli_and_library_agree_on_exported_boundaries(boundary_runs, capsys, tmp_path,
                                                     speedup, rc, json_output):
    a, b = boundary_runs("a", 1.10), boundary_runs("b", speedup)
    diff = diff_runs(a, b)
    assert bool(diff.regressions) is bool(rc)
    flags = ["--json"] if json_output else []
    assert main(["diff", str(a), str(b), "--scratch", str(tmp_path),
                 "--check", "--threshold", "0.02", *flags]) == rc
    out = capsys.readouterr().out
    if json_output:
        doc = json.loads(out)
        assert bool(doc["regressions"]) is bool(rc)
        assert doc["levers"][0]["change"] == ("delta_moved" if rc else "unchanged")
    else:
        assert f"{rc} regression" in out


def test_stdio_mcp_preserves_strict_exported_boundaries(boundary_runs, tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    import anyio
    from mcp import Client, StdioServerParameters

    a = boundary_runs("baseline", 1.10)
    cases = [(boundary_runs("below", 1.080000000001), False),
             (boundary_runs("exact", 1.08), False),
             (boundary_runs("above", 1.079999999999), True)]

    async def check():
        params = StdioServerParameters(command="gitm", args=["mcp", "--scratch", str(tmp_path)])
        async with Client(params) as client:
            for b, regressed in cases:
                res = await client.call_tool("diff", {"run_a": str(a), "run_b": str(b),
                                                       "threshold": 0.02})
                assert not res.is_error
                doc = res.structured_content
                assert bool(doc["regressions"]) is regressed
                assert doc["levers"][0]["change"] == ("delta_moved" if regressed else "unchanged")

    anyio.run(check)


def test_tiny_positive_threshold_does_not_get_a_fixed_epsilon_dead_zone():
    row = _compare("lever", LeverState("kept", 1, 1e-20),
                   LeverState("kept", 1, 0.0), 5e-21)
    assert row.change == "delta_moved"
    assert row.regressed
