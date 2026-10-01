#!/usr/bin/env python3
"""The run folders `gitm diff` is tested against, and the golden run CI gates on.

    python tests/fixtures/diff/generate_diff_runs.py      # rewrites both, in place

Written through the real build_record / write_verification path, like
tests/fixtures/history/generate_demo_runs.py, so an export-format change breaks
this script instead of leaving fixtures the reader no longer parses. Run ids are
derived from the folder name rather than uuid4 so regenerating is a no-op diff.

The golden run (tests/golden/diff_gate/kimi-mi355x) is a Kimi K2.5 decode run on
MI355X with the levers named in scripts/kimi_loop/gen_pods.py. Every folder under
runs/ is one thing a diff against it has to get right:

  rerun_noise      same box and model; every lever within ±2 pts  -> nothing moved
  regressed        EP kept but 18 pts lower (REGRESSED), kv fp8 flipped to rolled
                   back (REGRESSED), eager flipped to kept (not a regression),
                   mla absent (only_in_a), rccl ring new (only_in_b)
  improved         EP up 6 pts, nothing lower -> moves but no regression
  other_gpu        same model on an H100               -> not comparable
  other_workload   GLM on the same MI355X              -> not comparable
  no_sku           the golden's levers, SKU unreported -> refused, unverified
  no_export        a run folder with no export (what a no-engine run leaves)
  truncated        an export cut off mid-write         -> refused, never "empty"
"""

import hashlib
import shutil
import sys
from pathlib import Path

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import ApplyResult, EngineABResult
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import build_record, write_verification

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE / "runs"
GOLDEN = REPO / "tests" / "golden" / "diff_gate"

MI = "AMD Instinct MI355X"
H100 = "NVIDIA H100 80GB HBM3"
KIMI = ("moonshotai/Kimi-K2.5", "rocm:3f9c1a7be2d40c55")
GLM = ("zai-org/GLM-5.2", "rocm:81d0e4c6a9b2f713")

KNOBS = {
    "enable_expert_parallel": ("enable_expert_parallel", True),
    "enforce_eager": ("enforce_eager", True),
    "max_num_seqs_512": ("max_num_seqs", 512),
    "kv_cache_dtype_fp8": ("kv_cache_dtype", "fp8"),
    "mla_triton_backend": ("attention_backend", "TRITON_MLA"),
    "rccl_algo_ring": ("NCCL_ALGO", "Ring"),
}


def run(dest: Path, gpu_sku, model, attempts):
    """One run folder. ``attempts`` is [(lever, speedup, kept, significant)]."""
    model_name, fp = model
    run_id = hashlib.sha256(dest.name.encode()).hexdigest()[:32]
    base_cfg = {"model": model_name, "tensor_parallel_size": 8, "max_num_seqs": 256}
    records = []
    for name, speedup, kept, significant in attempts:
        knob, value = KNOBS[name]
        spec = InterventionSpec(
            name=name, summary=name, knob=knob, value=value,
            expected_delta_mean=0.08, expected_delta_lo=0.02, expected_delta_hi=0.14,
            source="https://docs.vllm.ai/en/latest/configuration/engine_args.html",
        )
        ab = EngineABResult(
            knob=knob, value=value, baseline_tps=538.8, candidate_tps=538.8 * speedup,
            speedup=speedup, kept=kept, via="restart", baseline_std=4.1,
            candidate_std=4.4, reps=3, significant=significant,
        )
        records.append(build_record(
            spec, ab, ApplyResult(True, not kept, speedup - 1.0),
            baseline_config=base_cfg, candidate_config={**base_cfg, knob: value},
        ))
    prov = Provenance(workload_id="vllm-decode", fingerprint=fp, run_id=run_id,
                      git_sha="737f0ca", gitm_version="0.1.13",
                      started_at_ns=0, ended_at_ns=1)
    dest.mkdir(parents=True, exist_ok=True)
    write_verification(records, prov, dest / "verification.json", gpu_sku=gpu_sku)


GOLDEN_ATTEMPTS = [
    ("enable_expert_parallel", 1.49, True, True),
    ("enforce_eager", 0.91, False, True),
    ("max_num_seqs_512", 1.02, True, False),
    ("kv_cache_dtype_fp8", 1.08, True, True),
    ("kv_cache_dtype_fp8", 1.11, True, True),  # twice in one run: mean +9.5%
    ("mla_triton_backend", 1.06, True, True),
]


def main() -> int:
    for d in (RUNS, GOLDEN):
        if d.exists():
            shutil.rmtree(d)

    run(GOLDEN / "kimi-mi355x", MI, KIMI, GOLDEN_ATTEMPTS)

    run(RUNS / "rerun_noise", MI, KIMI, [
        ("enable_expert_parallel", 1.482, True, True),
        ("enforce_eager", 0.905, False, True),
        ("max_num_seqs_512", 1.025, True, False),
        ("kv_cache_dtype_fp8", 1.09, True, True),
        ("mla_triton_backend", 1.058, True, True),
    ])
    run(RUNS / "regressed", MI, KIMI, [
        ("enable_expert_parallel", 1.31, True, True),
        ("enforce_eager", 1.01, True, False),
        ("max_num_seqs_512", 1.021, True, False),
        ("kv_cache_dtype_fp8", 0.94, False, True),
        ("rccl_algo_ring", 0.97, False, True),
    ])
    run(RUNS / "improved", MI, KIMI, [
        ("enable_expert_parallel", 1.55, True, True),
        ("enforce_eager", 0.91, False, True),
        ("max_num_seqs_512", 1.02, True, False),
        ("kv_cache_dtype_fp8", 1.10, True, True),
        ("mla_triton_backend", 1.06, True, True),
    ])
    run(RUNS / "other_gpu", H100, KIMI, GOLDEN_ATTEMPTS)
    run(RUNS / "other_workload", MI, GLM, GOLDEN_ATTEMPTS)
    run(RUNS / "no_sku", None, KIMI, GOLDEN_ATTEMPTS)
    # The folder a no-engine run leaves (scheduler/loop.py writes no export
    # without a live A/B). .gitkeep only so git keeps the empty directory.
    (RUNS / "no_export").mkdir(parents=True)
    (RUNS / "no_export" / ".gitkeep").write_text("")
    (RUNS / "truncated").mkdir(parents=True)
    (RUNS / "truncated" / "verification.json").write_text(
        '{"schema": 1, "results": [{"intervention_name": "kv_cac')

    print(f"wrote {GOLDEN} and {len(list(RUNS.iterdir()))} run folders under {RUNS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
