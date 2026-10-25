# gitm-submit — target interface

One script to run a traced job on the H200s over the Slurm REST API, from your
laptop, no SSH. Sebastian gets his baseline and noise-floor series without
learning Slurm; Pamela's run record keys on the run ID this script mints.

- **Spec:** [gitm-submit: H200 job submit script](https://linear.app/git-machine/document/gitm-submit-h200-job-submit-script-rishith-mody-c2704067f5bb) · **Owner:** Rishith · **Reviewer / merge gate:** Sebastian.
- **Umbrella:** ENG-74. This doc is **ENG-75 (S1)** — the target interface, what already works, what changes, and the in-container command sequence. It describes the **v1 target**; where behavior isn't built yet it's marked **[v1]**.
- Start point: `deploy/h200/gitm-submit` on `main` (PR #182). Slurm stays the substrate; keep what works.

---

## What already works today (on `main`)

The current `gitm-submit` already does all of this — v1 keeps it:

- **Submit over the Slurm REST API with a per-person token, no SSH** (`SLURM_JWT` + `SLURM_API`). `run`, `ls`, `status`, `cancel`.
- **One whole node per job** (all 8 H200s), in a **throwaway container** (Pyxis/Enroot) with the node's **HF cache** (`/hf`) and **results dir** (`/out`) mounted. Container is deleted when the job ends.
- Flags: `--name`, `--model`, `--node` (pin `gitmachine-1`/`-2`), `--vllm-args`, `--capture-args`, `--image`, `--max-hours`, `--gitm-ref`, `--env KEY=VALUE`, `--dry-run`, and a custom command after `--`.
- Node paths resolve at run time from each node's `/etc/gitm-gpu/config` (`RESULTS_DIR`, `HF_CACHE`, `DEFAULT_IMAGE`), so each node uses its own cache/results.
- **Results land on the status page** (`gitm-labs.github.io/gpu-status`) when the job ends.
- Today results go to **`$RESULTS_DIR/<slurm jobid>/`** (+ `<jobid>.log`) on the node that ran the job — `RESULTS_DIR` defaults to `/var/lib/gitm/results` (root disk).
- **What the job runs in the container today** (synthetic load):
  ```bash
  gitm capture serve --out /out <--requests N --concurrency C …> -- \
    vllm serve "$MODEL" --tensor-parallel-size 8 --trust-remote-code <vllm flags>
  ```
  `gitm capture serve` launches vLLM under the CUPTI collector and drives its **own synthetic load** (`--requests`, `--concurrency`, fixed token counts), capturing a kernel-trace window.

## What changes in v1 (this spec adds)

| Area | Today | v1 target |
|---|---|---|
| **Run ID** | the Slurm job id | our own ID, minted by the script (format below) |
| **Results root** | `/var/lib/gitm/results/<jobid>/`, per-node root disk | durable `<results root>/<run id>/` (see [Run ID & results](#run-id--results-layout)) |
| **Inputs record** | only the Slurm comment | an `inputs.json` capturing exactly what the script was called with |
| **Load** | synthetic (`--requests`/`--concurrency`) | a **traffic replay file** as input; synthetic is never generated when a replay file is given |
| **Completion** | fire-and-forget (watch the page) | **block or poll** until the job ends |
| **Model** | takes any `--model`, relies on cache/download | **resident-model check**; never downloads; fail + name the node that has it |
| **Health** | none | **pre/post-run hook points** for Nathan's health check (he wires the check; we leave the hooks) |

## Target interface

- **Inputs:** model; engine config (engine name + launch flags — vLLM now, SGLang when Collin's R1 lands); **traffic replay file**; trace output path; node (`gitmachine-1` | `gitmachine-2`).
- **Behavior:** write the Slurm job → submit → **block or poll until done** → copy results and logs to `<results root>/<run id>/`.
- **Output —** `<results root>/<run id>/`:
  - `trace/…` — the CUPTI kernel trace (multi-GB; may live on ephemeral scratch).
  - `engine.log` — the vLLM/SGLang server log.
  - `slurm.log` — the Slurm `%j.log`.
  - `inputs.json` — exact inputs: `run_id`, `slurm_job_id`, `submitter`, `node`, `model`, `hf_revision`, `engine`, `engine_version`, `engine_flags`, `replay_file`, timestamps, `status`.
- **Resident-model rule (enforced by the script, not policy):** one model stays resident per node — **Kimi K2.6 → `gitmachine-2`**, **DeepSeek V4 Pro → `gitmachine-1`**. Jobs change **config only**; the script **never downloads weights**. *Resident = weights in the node's HF cache, not a live server* — every job launches its own engine in its own container (launch flags can't change on a running process). The check is exactly: `models--<org>--<name>` exists under `HF_CACHE` (from `/etc/gitm-gpu/config`; currently `/mnt/disks/scratch/hf_hub` on both nodes) on the target node — else fail and name the node that has it. Weights are downloaded **outside this script** (the queue lane).
- **Run ID** is minted by the script (not the Slurm job id). It names the results directory and goes in the Slurm job comment; the Slurm job id is recorded in `inputs.json`.

## Run ID & results layout

Agreed in **ENG-76** with Pamela and Sebastian:

```
<UTC>-<model-slug>-<node>-<rand4>
example:  20261021T174233Z-kimi-k26-gm2-7f3a
```

UTC-first (chronological sort, FS/Slurm-comment safe); `model-slug` from the HF id lowercased (`moonshotai/Kimi-K2.6` → `kimi-k26`); `node` `gm1`/`gm2`; `rand4` for same-second uniqueness. Outputs split by durability:

```
/mnt/gitm-runs/<run id>/          # durable: plain folder over a central GCS bucket (gcsfuse)
    record.json  inputs.json  engine.log  slurm.log
/mnt/disks/scratch/runs/<run id>/
    trace/...                     # ephemeral: multi-GB CUPTI traces (scratch wiped on VM stop/move)
```

`/mnt/gitm-runs` is one config value (`RESULTS_ROOT`), repointable later without touching run ids or loop code. The `record.json` schema is **ENG-72**'s job. *The gcsfuse bucket/mount is a separate ops step (feeds ENG-77); until it exists, runs land on the per-node `/var/lib/gitm/results`.*

## In-container command sequence — ⚠️ PROPOSED, pending Sebastian

> The spec requires S1 to write "the exact in-container command sequence agreed
> with Sebastian." Below is the proposal to confirm with him — he owns the
> baseline/replay series. The building blocks exist on `main`; what needs his
> sign-off is the exact wiring (window boundaries, how serve is told not to
> self-drive, adapter + tokenizer choices).

**Today (synthetic):** `gitm capture serve` serves vLLM under the collector and drives its own load — see above.

**v1 (replay):** launch the engine under the collector, fire the replay file at it, capture that window. The replay machinery is `python -m gitm.traffic --replay <adapter> <file> --fire` (adapters on `main`: **`mooncake`**, **`burstgpt`**; it drives `vllm bench serve` against a running server — see [`docs/TrafficReplayDoc.md`](../../docs/TrafficReplayDoc.md)).

**Proposed sequence (Option A — attach + observe window; recommended):**
```bash
# 1. launch vLLM under the CUPTI collector, serving, in the background
CUDA_INJECTION64_PATH=<cupti shim> \
  vllm serve "$MODEL" --tensor-parallel-size 8 --trust-remote-code <engine flags> &
# 2. wait for http://127.0.0.1:8000/health
# 3. open a capture window in OBSERVE mode for the replay's duration
gitm capture attach --port 8000 --out /out --duration <window_s> &
# 4. fire the replay file at the running server
python -m gitm.traffic --replay <mooncake|burstgpt> <replay_file> --fire \
    --model "$MODEL" --base-url http://127.0.0.1:8000 --result-dir /out
# 5. window closes → trace + bench-serve result land in /out
```
`gitm capture attach` observe-mode "watches the server's own traffic," so it captures exactly the replay window rather than synthetic load — which matches the spec. (`--tokenizer` is needed if the served model name isn't HF-resolvable.)

**Option B (single command):** if `gitm capture serve` grows a replay input (serve drives the replay itself instead of `--requests`), the job is one command and no separate `attach` is needed. Not available on `main` today; flag for Sebastian whether to add it in S3 vs. use Option A.

**To confirm with Sebastian (S1):** Option A vs B; the capture window length vs. replay duration; exact flag to suppress synthetic load; adapter + tokenizer defaults for the Kimi baseline series.

## CLI reference (today → v1)

| Flag | Status | Notes |
|---|---|---|
| `--name`, `--model`, `--node`, `--max-hours` | works | kept |
| `--vllm-args`, `--capture-args`, `--image`, `--env`, `--gitm-ref`, `--dry-run` | works | kept |
| `ls` / `status` / `cancel` | works | kept |
| `--replay <adapter> <file>` | **[v1]** | traffic replay input; disables synthetic load |
| `--wait` / poll-to-completion | **[v1]** | block or poll until the job ends |
| run id + `inputs.json` under `<results root>/<run id>/` | **[v1]** | replaces bare `<jobid>/` |
| resident-model check | **[v1]** | fail + name the node if weights absent; never downloads |
| pre/post health hooks | **[v1]** | hook points only; Nathan wires the check |

## Not in scope

Multi-node jobs · HTTP job runner · queue policy/quotas · the health-check logic itself (hooks only) · model downloads (queue lane) · dashboard work beyond showing S4's results. Tokens and the cluster REST address are **never** written into the repo, this README, or Slack.
