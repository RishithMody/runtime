---
name: gitm-run-triage
description: Explain what a gitm run did from its output folder — status, why it produced no data, which levers were tried/kept/rolled back, deviations and attribution. Use when handed a run id or $GITM_SCRATCH/runs/<id> folder, a report.md, or asked "why did this run do nothing / roll everything back".
---

# Triage a run folder

Runs live in `$GITM_SCRATCH/runs/<run_id>/`, with `~/.cache/gitm` as the default scratch.
The trace is at `$GITM_SCRATCH/traces/<run_id>.jsonl`. `gitm/scheduler/loop.py` wrote
everything in the folder. Grep it for a filename to see exactly when it's written.

## Read in this order

1. **`report.md`**: the human summary. It covers claims, evidence, deltas, rolled-back
   levers, and provenance.
2. **`qualification.json`**: `fingerprint` is `<vendor>:<hash>`, a hash of the kernel mix.
   `commit` / `floor` say whether the 15% floor was committed. `diagnostic` explains
   refusals.
   - `No kernels in trace` means `status: no_data`, so capture failed or the workload
     never ran. On a laptop or CI this is expected. The CLI exits 3.
3. **`predicted_graph.json`, `residuals.json`, `deviations.json`**: prediction versus
   trace, and which invariant deviated (kernel-time, memory-traffic, stream-concurrency;
   see `docs/invariants.md`).
4. **`ranked_candidates.json`, `history_read.json`**: which levers were ranked and whether
   past runs informed the ranking (`--use-history`).
5. **`autoresearch.json`**: levers generated outside the catalogue, and why each was
   rejected.
6. **`verification.json`**: the measured A/B per lever (baseline/candidate tps, reps,
   `significant`, `kept` from the gate, full configs).
   - **Absent means no live engine was attached.** Phase 4 ran `DryRunApplicator`
     (predict-only), so nothing was measured or kept.
7. **`audit.jsonl`**: live mutations and reverts. It exists only when something was
   actually applied.

## Common answers

- **"Why did it do nothing?"** Check `qualification.json` `diagnostic` and the summary's
  `diagnostic`. A missing `vllm`, missing CUPTI shim, or no GPU gives `no_data`.
  `gitm doctor` shows what the box has.
- **"Why was a good lever rolled back?"** In `verification.json`, compare `delta` with the
  gate threshold (`min_keep_delta`), and check `significant` / `agreement_band`. A +1%
  inside a 2% band is noise.
- **"Did this regress since last time?"** Run `gitm diff <previous> <this>` (see the
  `gitm-run-diff` skill).
- **"Is this lever good in general?"** Run `gitm history --gpu "<SKU>"`.

Say plainly when a number doesn't exist. Don't infer a delta from `ranked_candidates`,
because those are predictions, not measurements.
