"""`gitm mcp`: history and diff as MCP tools, driven through the SDK's own client."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.mcpserver", reason="needs the MCP SDK v2 (gitm-labs[mcp])")

import anyio  # noqa: E402  (ships with the SDK)
from mcp import Client  # noqa: E402

from gitm.mcp_server import build_server, list_runs_data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "diff_gate" / "kimi-mi355x"
RUNS = REPO / "tests" / "fixtures" / "diff" / "runs"


@pytest.fixture
def scratch(tmp_path):
    """A scratch dir holding: an old Kimi run, a newer regressed Kimi run, a GLM run,
    and a folder with no export — in that mtime order."""
    runs = tmp_path / "runs"
    for i, (name, src) in enumerate([
        ("1111aaaa", GOLDEN), ("2222bbbb", RUNS / "regressed"),
        ("3333cccc", RUNS / "other_workload"), ("4444dddd", RUNS / "no_export"),
    ]):
        shutil.copytree(src, runs / name)
        t = 1_000_000 + i * 1000
        for p in (runs / name, *(runs / name).iterdir()):
            os.utime(p, (t, t))
    return tmp_path


def _call(server, tool, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(tool, args)
    return anyio.run(go)


def test_tools_are_listed_read_only_with_their_rules():
    async def go():
        async with Client(build_server()) as c:
            return await c.list_tools()
    tools = {t.name: t for t in anyio.run(go).tools}

    assert set(tools) == {"list_runs", "history", "diff"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    # The interpretation rules travel in the description, once, not in each result.
    assert "not a regression" in tools["diff"].description


def test_list_runs_finds_the_last_two_kimi_runs_by_model_name(scratch):
    doc = list_runs_data(scratch / "runs", match="kimi")

    assert [Path(r["run_dir"]).name for r in doc["runs"]] == ["2222bbbb", "1111aaaa"]
    assert doc["runs"][0]["model"] == "moonshotai/Kimi-K2.5"


def test_list_runs_shows_a_run_that_measured_nothing(scratch):
    rows = {Path(r["run_dir"]).name: r for r in list_runs_data(scratch / "runs")["runs"]}

    assert rows["4444dddd"]["measured"] is False
    assert rows["4444dddd"]["damaged"] is False


def test_diff_tool_answers_what_moved_between_the_last_two_kimi_runs(scratch):
    server = build_server(str(scratch))
    newest, older = list_runs_data(scratch / "runs", match="kimi")["runs"][:2]

    res = _call(server, "diff", {"run_a": older["run_dir"], "run_b": newest["run_dir"]})

    assert not res.is_error
    doc = res.structured_content
    assert doc["comparable"] is True
    assert doc["regressions"] == ["kv_cache_dtype_fp8", "enable_expert_parallel"]


def test_diff_tool_refuses_across_workloads(scratch):
    res = _call(build_server(str(scratch)), "diff", {"run_a": "1111", "run_b": "3333"})

    assert res.structured_content["comparable"] is False
    assert res.structured_content["levers"] == []


def test_diff_tool_reports_a_bad_run_as_a_tool_error(scratch):
    res = _call(build_server(str(scratch)), "diff", {"run_a": "1111", "run_b": "nope"})

    assert res.is_error
    assert "no run folder" in res.content[0].text


def test_history_tool_matches_gitm_history_json(scratch):
    res = _call(build_server(str(scratch)), "history", {"fingerprint": "rocm:3f9c1a7be2d40c55"})

    doc = res.structured_content
    assert doc["runs_read"] == 2
    assert doc["filtered"] == 1  # the GLM run
    assert doc["skipped"] == {"4444dddd": "no verification.json"}
    kv = next(r for r in doc["records"] if r["intervention_name"] == "kv_cache_dtype_fp8")
    assert kv["conflicted"] is True  # won in the golden run, lost in the regressed one


@pytest.mark.parametrize("other", ["no_sku", "no_export"])
def test_diff_tool_does_not_claim_comparison_without_identity(scratch, other):
    res = _call(build_server(str(scratch)), "diff",
                {"run_a": str(GOLDEN), "run_b": str(RUNS / other)})
    assert not res.is_error
    doc = res.structured_content
    assert doc["comparable"] is False
    assert doc["preconditions"]["gpu_sku"] == "unverified"
    assert doc["levers"] == []
    assert doc["regressions"] == []
