---
name: gitm-testing
description: Run, write, or debug tests in the gitm repo without a GPU — pytest layout, faking trace capture, fixture/golden regeneration, lint, and known environment traps. Use before adding tests, when a test fails unexpectedly, or when git status shows fixture files changed after a test run.
---

# Testing gitm without a GPU

## Run

```bash
source .venv/bin/activate          # REQUIRED; some tests shell out to `python`
pip install -e ".[dev,bench]"      # dev = pytest, hypothesis, ruff, mcp, ...
pytest tests/test_<area>.py        # targeted
pytest                             # full suite (CI: Python 3.10, 3.11, 3.12)
ruff check .                       # lint; CI pins ruff 0.12.11. Never run `ruff format`
```

`addopts = "-q"` is set in pyproject, so `pytest -q` goes to `-qq` and hides the summary.
Use the exit code, or `-rf`.

## Traps

- **`test_bench.py::test_run_seed_end_to_end_with_echo_harness` → `FileNotFoundError`**:
  the venv isn't on PATH. Activate it.
- **After any run, `tests/fixtures/importers/*.sqlite` / `*.json.gz` show as modified or
  untracked.** The autouse fixture `_ensure_fixtures` in `tests/test_importers.py`
  regenerates them. This is expected; don't commit them.
- **Goldens use two different env var names:**
  - `UPDATE_GOLDENS=1` rewrites `tests/golden/report_basic.md`.
  - `GITM_UPDATE_GOLDEN=1` rewrites the customer report golden.
- **`gitm run` on a laptop returns `status: no_data` and exits 3.** Tests that need the
  loop to see kernels must fake the capture.

## Faking the GPU (the house pattern)

```python
import gitm.scheduler.loop as loop
from tests.conftest import make_trace, make_kernel   # builders; conftest has no autouse fixtures

def fake_capture(out_path, *, workload_id="w", fingerprint="f", run_id=None):
    ...  # a contextmanager that yields make_trace([...make_kernel("paged_attention", ...)])

monkeypatch.setattr(loop, "capture", fake_capture)
monkeypatch.setattr(loop, "sync_device", lambda: None)
optimize(workload="vllm-decode", budget="1s", scratch=str(tmp_path), workload_runner=lambda: {})
```

See `tests/test_run_loop_workload.py::_fake_capture_with_kernels` for a working version.
Always pass `scratch=str(tmp_path)` so nothing writes to `~/.cache/gitm`.

## Writing tests here

- Name tests as claims (`test_a_malformed_export_is_skipped_rather_than_raising`). Add a
  docstring when the *why* isn't obvious.
- Guard optional deps with `pytest.importorskip(...)` placed **before** the imports that
  need them.
- Build fixture data through the real writers, not hand-written JSON. For example,
  `tests/fixtures/history/generate_demo_runs.py` and `tests/fixtures/diff/generate_diff_runs.py`
  write through `verification_export.build_record` and `write_verification`, so a format
  change breaks the generator loudly.
- Test the honest-failure paths too: missing file, truncated JSON, a `None` SKU, `NaN`,
  or a wrong type. The codebase treats "skip with a reason" as a feature.
