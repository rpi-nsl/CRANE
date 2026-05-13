# swe-lite — SWE-bench Lite eval stack

End-to-end harness for running any local or HuggingFace model on SWE-bench Lite
(or Verified / Full) on a multi-GPU HPC node, using:

- **vLLM** (OpenAI-compatible server) for inference — reused from
  `${CRANE_REPO_ROOT}/.venv`. Launched with
  `--enable-prefix-caching --enable-prompt-tokens-details` so we capture
  real cache-hit counts in per-instance traj metrics (typically ~97% for the
  agent-loop context re-sends).
- **OpenHands SDK** as the default agent scaffold (custom adapter at
  [openhands_swebench.py](openhands_swebench.py) — `openhands-sdk` +
  `openhands-tools` in `.venv-openhands` with Python 3.12). Also supports
  **mini-swe-agent** and **SWE-agent** via `--agent mini|swe-agent`.
- **podman** as the container backend for SWE-bench eval images
  (the cluster has no Docker daemon and no subuid entries, hence some
  workarounds — see below)
- **swebench.harness.run_evaluation** for scoring

## Layout

```
swe_bench/
├── env.sh                       # shared HF cache / venv paths / ghcr mirror helpers
├── run_swebench.sh              # default batch driver
├── run_swebench_30b.sh          # 30B-class batch driver (MODELS: qwen3-30b-*, crane-v2, baselines)
├── run_swebench_80b.sh          # 80B-class batch driver (MODELS: qwen3-next-80b-*, crane-next, baselines)
├── run_swebench_30b_ablation.sh # Table 5 ablation driver, 30B
├── run_swebench_80b_ablation.sh # Table 5 ablation driver, 80B
├── run_swebench_rain.sh         # RAIN baseline driver
├── openhands_swebench.py        # OpenHands SDK adapter — per-instance container + deadline + liveness
├── harness_fast.py              # swebench harness wrapper (podman timeout / list_images patches)
├── swebench_lchown_patch.py     # one-shot rootless-podman lchown fix for swebench
├── runtime.py                   # runtime backend abstraction (podman / daytona)
├── prebuild_wrappers.sh         # one-off: pull all bases + build all wrappers
├── prebuild_verified_phased.sh  # phased prebuild for the Verified subset
├── archive_images.sh            # save/load wrappers to NFS (survives node change)
├── cleanup_podman.sh            # kill orphan oh-*/sweb.eval.*/build processes
├── aggregate_model_results.py   # per-model results.{md,json} (pass rate + tokens + ref cost)
└── update_ablation_md.py        # ablation results post-processor
```

All script outputs land under `eval_results/`:

```
eval_results/
├── sbatch_log/                                  # Slurm stdout/stderr (%x-%j)
├── vllm/<model>.log                             # vLLM per-model log
├── podman-svc.log                               # podman API socket log
└── <model>_<subset>_<split>_<RUN_TS>/           # one dir per (model, subset, split, launch time)
    ├── run.log                                  # agent stdout
    ├── preds.json                               # {instance_id → model_patch}
    ├── *.traj.json                              # one per instance (incl. metrics + response_latencies)
    ├── watchdog.log                             # side-car watchdog kills
    ├── harness.log                              # swebench harness stdout
    ├── <model>.<run_tag>.json                   # harness summary (resolved_instances, ...)
    ├── results.{json,md}                        # aggregate: pass rate + token usage + vLLM cache + ref cost
    └── logs/                                    # per-instance harness logs (patch.diff, test_output.txt, …)
```

`RUN_TS` (e.g. `20260422_030200`) is set per-invocation so re-runs don't
overwrite previous outputs. Override with `RUN_TS=<value>` to resume a run
into the same dir.

## Quick start

```bash
# smoke — first Verified instance, first model
bash run_swebench_30b.sh --limit 1

# single instance
bash run_swebench_30b.sh --instance pallets__flask-4045

# one model from the registry
bash run_swebench_30b.sh --model qwen3-30b-instruct

# pass a full tuple (no registry edit needed)
bash run_swebench_30b.sh --model 'my-ft|/path/to/local/ckpt|false'

# 80B batch
bash run_swebench_80b.sh --model qwen3-next-80b-instruct

# just vLLM for debugging
bash run_swebench_30b.sh --skip-agent --skip-eval

# full run via Slurm
sbatch run_swebench_30b.sh
```

### Adding a model

Edit the `MODELS` array near the top of either `run_swebench_30b.sh` or
`run_swebench_80b.sh`:

```bash
MODELS=(
    "qwen3-30b-instruct|Qwen/Qwen3-30B-A3B-Instruct-2507|true"
    "my-finetune|${CRANE_REPO_ROOT}/some/ckpt|false"
)
```

Columns: `shortname | HF repo id OR local dir | is_hf (true|false)`. The
shortname is what gets passed as `served-model-name` to vLLM and as the
litellm model for the agent; it's also what shows up in predictions /
reports / `results.json`.

`--model` also accepts the full tuple form `shortname|path|is_hf` to run a
model that isn't in the registry. Handy for one-off experiments.

## How the pieces talk

```
┌─────────────────────┐  OpenAI API  ┌───────────────────────┐  podman run/exec  ┌────────────────────┐
│ openhands_swebench  │─────────────▶│ vLLM @ 127.0.0.1:18020│                   │ per-instance sweb  │
│ (OpenHands SDK,     │              │ bf16 tp=4             │                   │ eval container     │
│  threadpool workers,│              │ prefix-caching +      │                   │ (`oh-<iid>-<rand>`)│
│  per-inst deadline) │              │ prompt-tokens-details │                   │                    │
└──────────┬──────────┘              └───────────────────────┘                   └────────────────────┘
           │  preds.json
           ▼
┌──────────────────┐  docker-py via DOCKER_HOST=unix:///tmp/.../podman.sock
│ swebench.harness │─────────────────────────────────────────────────────▶ podman (rootless)
│ run_evaluation   │
└──────────┬───────┘
           │  per-instance report.json
           ▼
┌──────────────────────────────┐
│ aggregate_model_results.py   │─► results.{json,md}  (pass rate + tokens + vLLM cache + ref cost)
└──────────────────────────────┘
```

## Base-image registry and prebuild

Docker Hub rate-limits anonymous pulls (100/6h per IP), which is trivial to
exhaust when cold-starting on a new node. We pull from **Epoch AI's ghcr.io
mirror** instead — same SWE-bench eval images, no anon rate limit, and
~10× smaller overall (~67 GB for all 2290 images).

Naming:
- ghcr:  `ghcr.io/epoch-research/swe-bench.eval.x86_64.<iid>:latest`
- canonical: `docker.io/swebench/sweb.eval.x86_64.<mangled_iid>:latest`
  (iid with `__` → `_1776_`, lowercased — the name the harness + sweagent look up)

`pull_base_via_ghcr_one` in [env.sh](env.sh) pulls from ghcr and retags to
the canonical name so all downstream code is unchanged. Called by:
- [prebuild_wrappers.sh](prebuild_wrappers.sh) phase 1
- [run_swebench.sh](run_swebench.sh) `warm_bases_for_run` before each eval

### One-off prebuild (≈500 × ~1 min per wrapper)

```bash
# pull all 500 Verified bases from ghcr mirror + retag, then build wrappers
bash prebuild_wrappers.sh --subset verified --pull-parallel 4 --build-parallel 12
```

Two phases, both skip-idempotent:
1. `pull_one` via ghcr mirror → canonical tag
2. `build_one`: `FROM <base>` + swerex glibc layer (Python 3.11 standalone,
   swerex install). Tags as `localhost/sweb-wrapper-<iid>:latest`.

**parallel=12 is the sweet spot.** At 20+ the podman rootless graph-driver lock
thrashes; at <10 you leave throughput on the table. Occasional single builds
can deadlock on the graph lock (0-byte log, 0% CPU over many minutes) — kill
them by PID and the driver moves on. Missing wrappers are harmless (eval
builds them on-demand at runtime, ~1-2 min for that one instance).

### Node portability (save/load)

`/tmp/${USER}-podman/storage` is node-local xfs (supports xattrs podman
needs) but wipes when you land on a different node. NFS doesn't support
xattrs so it can't hold podman's graphroot — but it can hold zstd-compressed
`podman save` tarballs.

```bash
# snapshot (≈50 min at parallel=8 for 500 wrappers + builder ≈ ~590 GB on NFS)
bash archive_images.sh save --wrappers-only --parallel 8

# restore on a fresh node (≈20-30 min at parallel=4)
bash archive_images.sh load --parallel 4
```

`--wrappers-only` is the recommended scope. Wrapper tars transitively contain
all their base layers already — saving bases separately would double the NFS
footprint (~500 GB wasted). On load, bases aren't tagged but their layers are
present; `warm_bases_for_run` sees the canonical tag is missing and pulls from
ghcr, but podman finds the layers locally and just creates the tag
(near-instant).

Archive stream: `podman save <ref> | zstd -T0 -3 -o <file>.tar.zst`. Avoids a
~2 GB intermediate uncompressed tar per image. Compression ~3× over the raw
tar (since docker-archive layer tars are uncompressed inside); ~1 GB per
wrapper archive in practice.

## Cluster-specific caveats

The cluster has **no `/etc/subuid` entries for the user**, so rootless podman
only has a single-UID namespace (host UID maps to container UID 0, nothing
else is valid). This breaks two things:

1. **swebench harness's `copy_to_container`** tars files with the host UID
   (e.g. 11140) and calls `put_archive`; podman then fails with
   `lchown ... invalid argument`. Fix: we patched
   `.venv/.../swebench/harness/docker_utils.py:copy_to_container` to force
   `uid=gid=0` in the tarinfo filter. Reapply if swebench is reinstalled.
2. **mini-swe-agent's docker env** avoids this because it shells out to
   `podman run/exec` (no `put_archive`), so we just pass
   `MSWEA_DOCKER_EXECUTABLE=podman` in `run_swebench.sh`.

Other things to know:

- GPU: 4× H100 80GB per node. The driver auto-detects `tp_size` from
  `nvidia-smi -L`. Qwen3-30B-A3B fits comfortably at tp=4 with
  `--max-model-len 131072`; Qwen3-Next-80B fits at tp=4 with the same
  context. Override via `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_UTIL` / `VLLM_TP_SIZE`.
- HF cache: `${CRANE_REPO_ROOT}/cache/huggingface` (NFS).
  `Qwen3-30B-A3B-Instruct-2507` snapshot is 57GB on disk; `Qwen3-Next-80B-A3B`
  is ~160GB.
- Instance eval images pulled from ghcr (see above) and retagged to
  `docker.io/swebench/sweb.eval.x86_64.*`. Graph root at
  `/tmp/${USER}-podman/storage` (node-local xfs, ~940GB). Wrapper archive at
  `${CRANE_REPO_ROOT}/cache/podman-archive/` (~590GB for Verified).
- Podman API: the driver starts `podman system service --time=0` on
  `unix:///tmp/${USER}-podman/podman.sock` for the harness only — OpenHands
  agent shells out to `podman run/exec` directly (no socket).

## OpenHands agent setup

The default agent is the custom OpenHands SDK adapter
[openhands_swebench.py](openhands_swebench.py). Key defaults set in
that file (overridable via CLI flags):

**Sampling (Qwen3 recommended values)** — keep all three, dropping any one
causes observable regressions on SWE-bench Verified:

| Param | Value | Why |
|---|---|---|
| `temperature` | **0.6** | Higher (0.7) caused more output divergence in Qwen3-30B thinking |
| `top_p` | 0.8 | Qwen3 recommendation |
| `top_k` | **20** | Without it, long hallucinated outputs that never emit `finish` (agent loop doesn't terminate) |

**LLM call timing** — shorter than OpenHands defaults so a wedged call can't
block the state lock for 30 min:

| Param | Value | Notes |
|---|---|---|
| `timeout` | **90 s** | OpenHands default is 300 s; empirical p99 per-call latency is 3 s, so 90 s is 30× p99 headroom |
| `num_retries` | 5 | OpenHands default (kept for transient network blips) |
| `retry_min_wait` / `max_wait` | 4 / 16 s | Tighter than defaults (8 / 64); caps retry backoff at ~20 s |

**Per-instance wall-clock deadline** — four layers of cancellation so a
single wedged instance never holds the batch hostage:

1. **Tool-executor detection** (`podman_exec` returncode 125 + "no such
   container" → `conv.pause()`) — fires at the next agent tool call after
   the container dies.
2. **Background liveness probe** (every 15 s, `podman container exists` →
   `conv.pause()`) — fires even when the agent has stopped calling tools
   and is just hallucinating via LLM.
3. **`enforce_deadline` timer** at **T + 60 min** — forcibly `podman rm -f`
   the container and call `conv.pause()`.
4. **Main-thread hard cap**: `threading.Thread.join(60min + 60 s)` on the
   worker — dispatcher abandons the orphan thread at T + 61 min. Thread may
   linger but its Future is resolved; `preds.json` can be written.

Override the deadline with `OPENHANDS_INSTANCE_DEADLINE_MIN=<minutes>`.

**Agent + harness parallelism**:

| Param | Value | Flag |
|---|---|---|
| agent workers | 24 | `--workers N` or `AGENT_WORKERS=N` |
| harness workers | 24 | `--harness-workers N` or `HARNESS_WORKERS=N` |
| max iter per instance | 100 | `OPENHANDS_MAX_ITER=N` |

### Between-model hygiene

Each `MODELS` iteration ends with:
- `stop_vllm` (frees the H100s)
- `aggregate_model_results.py` → writes `results.json` / `results.md` in
  the model's output dir (pass rate, token totals, vLLM cache stats, ref
  cost)
- `cleanup_podman.sh --all` — kills lingering `oh-*` / `sweb.eval.*` /
  `podman build` processes so the next model starts with a clean graph

### Reference cost

`aggregate_model_results.py` reports absolute token totals;
[RESULTS.md](RESULTS.md) converts those into an **equivalent closed-model
cost** using GPT-5.4-mini pricing for 30B rows ($0.75 / $0.075 / $4.50 per
1M for input / cached / output) and GPT-5.4 pricing for 80B rows ($2.50 /
$0.20 / $15.00). `Input tok` in the table is `prompt_tokens − cache_read`,
i.e. the portion that actually required a prefill. Prefix cache hit rate is
~97% for the agent-loop context re-sends, so the cached-input charge
dominates.

## Full-subset run playbook (post-prebuild)

Once `bash prebuild_wrappers.sh --subset verified` has populated
`localhost/sweb-wrapper-*:latest` for every Verified instance, each eval run is
fast to start — sweagent's per-instance `podman build` hits the layer cache and
returns the existing wrapper id in <1 s instead of ~1 min of glibc build (the
OpenHands path directly `podman run`s from the base image so skips this
entirely).

```bash
# 500 Verified × 24 agent workers × 24 harness workers on the registry models
bash run_swebench_30b.sh      # 5 × 30B models, ≈ 7-8 h total
bash run_swebench_80b.sh      # 6 × 80B models, ≈ 10-12 h total
```

Per-run token totals + ref cost land in each model's
`eval_results/<model>_..._<RUN_TS>/results.{md,json}` automatically
(aggregate_model_results.py is called at end of each model iteration).
