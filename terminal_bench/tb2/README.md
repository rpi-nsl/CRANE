# tb2 — Terminal-Bench 2.0 runner

End-to-end setup for running the **89-task Terminal-Bench 2.0** benchmark from this HPC node using **Harbor + Daytona cloud sandboxes**. Local rootless-podman runs are not viable on this cluster (single-uid mapping blocks every `apt` postinst `chown`), so the path we use is: **Harbor CLI on HPC → Daytona sandbox in the cloud → pre-built `xiangyangli/*` images on Docker Hub**.

## Layout

| Dir / file | What |
|---|---|
| [tb2-zai/](tb2-zai/) | **Default dataset.** Clone of `zai-org/terminal-bench-2-verified` — 89 tasks, zai-org's env + instruction fixes, task.toml already references `xiangyangli/<task>:20260204`. |
| [tb2-official/](tb2-official/) | Baseline `laude-institute/terminal-bench-2` — same 89 task names, laude solutions/tests. We swapped task.toml to xiangyangli too for an apples-to-apples comparison. |
| [tasks-v0.2/](tasks-v0.2/) | Unrelated — an older `terminal-bench` dev branch (94 tasks, old `task.yaml` format). Kept for reference; don't use for 2.0 eval. |
| [run_harbor_daytona.sh](run_harbor_daytona.sh) | Run one task through Harbor on Daytona. Default dataset = `tb2-zai`. |
| [sh/run_tb2_vllm_models.sh](sh/run_tb2_vllm_models.sh) | SLURM-aware wrapper: for each entry in the `MODELS=(...)` registry, spin up vLLM on the node's GPUs, point Harbor at it, run the full 89-task sweep, tear down, move to next model, print a pass-rate table. |
| [sh/start_qwen_server.sh](sh/start_qwen_server.sh) | Portable single-node "Qwen-as-a-service" starter: vLLM + Cloudflare **named** tunnel at `https://your-vllm.example.com/v1`. Foreground by default. Ships anywhere with just the tunnel credentials JSON (see "Serving a local model to Daytona" below). |
| [patch_test_sh.py](patch_test_sh.py) | Rewrite `tests/test.sh` to install `uv` via pypi instead of `curl https://astral.sh/...`. Needed because Daytona Tier 2 whitelists `astral.sh` but **not** its subdomain `releases.astral.sh` which the install script redirects to. |
| [patch_task_toml_xiangyangli.py](patch_task_toml_xiangyangli.py) | Swap `task.toml`'s `docker_image` from `alexgshaw/` (laude) → `xiangyangli/` (zai env fixes); both are upstream Docker Hub namespaces. |
| [patch_compose.py](patch_compose.py), [patch_dockerfile.py](patch_dockerfile.py) | Local rootless-podman compose / Dockerfile patchers. Not needed for the Daytona flow. |
| [cleanup_daytona.py](cleanup_daytona.py) | List / delete Daytona sandboxes. Supports `--age N --delete` for one-shot reap, `--watch S` for a background janitor. The sweep scripts auto-run this on EXIT/SIGINT/SIGTERM and also spawn a watchdog during long sweeps. |

## Prereqs

```bash
# Harbor CLI (Python 3.12)
uv tool install --python 3.12 'harbor[daytona]'

# Daytona API key (tier 2 required for the astral.sh whitelist)
export DAYTONA_API_KEY=dtn_xxxxxxxxxxxxxxx
# or hard-code at top of run_harbor_daytona.sh
```

That's it — no docker, no podman, no subuid hack. Harbor talks to Daytona's API; Daytona spins up a sandbox per task from the prebuilt image; Harbor's server inside runs your agent; results stream back.

## Quick start

```bash
# single task, oracle agent (sanity check)
./tb2/run_harbor_daytona.sh chess-best-move

# full 89-task sweep (~10-20 min wall clock at n=10, ~$0.4 on Daytona)
./tb2/sh/run_tb2_vllm_models.sh

# terminus-2 agent driving an OpenAI-compat endpoint you've already started
AGENT=terminus-2 MODEL=openai/qwen3-30b \
  OPENAI_API_BASE=http://host:8000/v1 OPENAI_API_KEY=placeholder \
  ./tb2/sh/run_tb2_vllm_models.sh

# Multi-model sweep under SLURM: start vLLM per model, run harbor over all 89,
# teardown, next model. Edit the MODELS=() registry at top of the script first.
sbatch tb2/sh/run_tb2_vllm_models.sh
bash   tb2/sh/run_tb2_vllm_models.sh --limit 3          # smoke
bash   tb2/sh/run_tb2_vllm_models.sh --task chess-best-move
```

Results land in `/tmp/$USER-harbor-jobs/<timestamp>/` — each task gets its own `<task>__<id>/` with `agent/`, `verifier/`, `trial.log`, `result.json`.

## Switching datasets

```bash
# use laude baseline instead of zai-verified
DATASET=tb2-official ./tb2/sh/run_tb2_vllm_models.sh
```

zai-verified is preferred for real agent runs (env + instruction fixes are worth ~5-10 pp on Claude Opus per zai's own numbers). For oracle sweeps the difference is cosmetic — solutions are identical across both datasets.

## If you re-clone either dataset

Re-apply the two patches (both idempotent):

```bash
python3 tb2/patch_test_sh.py tb2-zai          # pip-install uv
python3 tb2/patch_task_toml_xiangyangli.py    # only needed on tb2-official
```

Originals are backed up to `*.orig` alongside every patched file.

## Serving a local model to Daytona

Harbor's terminus agent runs **on this HPC** (not inside the Daytona sandbox) — but the sandbox *itself* is the thing that calls the LLM when our agent steps into it over HTTP. Either way, the LLM endpoint has to be reachable from the public internet, because Daytona sandboxes egress only to whitelisted hostnames.

We use **Cloudflare named tunnel** to expose a local vLLM at `https://your-vllm.example.com`:

```bash
./tb2/sh/start_qwen_server.sh    # vLLM + tunnel, foreground
```

Behind the scenes:

- vLLM serves `Qwen/Qwen3-30B-A3B-Instruct-2507` on `localhost:18016`, tensor-parallel over every local GPU.
- Cloudflared connects that port to a Cloudflare named-tunnel UUID configured in `~/.cloudflared/<tunnel>.json`.
- DNS `your-vllm.example.com` is CNAME'd to that UUID via `<uuid>.cfargotunnel.com`. Cloudflare signs a Universal SSL cert (first request on a fresh subdomain can take up to 15 min).
- Once the tunnel is up, anything with internet access — Daytona sandboxes included — can reach `https://your-vllm.example.com/v1` with no extra auth.

### Moving this to another machine

Everything is computed at runtime **except** the tunnel's private key. One file has to travel:

```bash
# from this machine
scp ~/.cloudflared/qwen-tb2.json other-host:~/.cloudflared/

# on the new machine
mkdir -p ~/bin
curl -sSL -o ~/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/bin/cloudflared

git clone ... /some/path/terminal-ben     # or rsync tb2/
cd /some/path/terminal-ben
./tb2/sh/start_qwen_server.sh             # same URL comes up
```

The script also accepts the credentials as env vars if you don't want a secrets file:

```bash
CF_ACCOUNT_ID=... TUNNEL_ID=... TUNNEL_SECRET=... ./tb2/sh/start_qwen_server.sh
```

Then elsewhere (same or different HPC node / different shell):

```bash
OPENAI_API_BASE=https://your-vllm.example.com/v1 OPENAI_API_KEY=placeholder \
  AGENT=terminus-2 MODEL=openai/qwen3-30b \
  ./tb2/sh/run_tb2_vllm_models.sh
```

### Quick tunnel fallback

If you don't have the creds file, you can fall back to Cloudflare's ephemeral "quick tunnel":

```bash
~/bin/cloudflared tunnel --url http://localhost:18016
# → prints a random https://*.trycloudflare.com URL; good for a few hours
```

## Daytona-unsupported tasks (4 — excluded by default)

The sweep scripts default-exclude these 4 via `--exclude-task-name`. Our **Daytona Tier 2 environment cannot run them**, no amount of local patching will change that. If you upgrade to Tier 3 (larger disk, broader egress whitelist) you can re-include them with `EXCLUDE=""`.

| Task | Why Daytona Tier 2 can't run it |
|---|---|
| `pytorch-model-cli` | Solution `pip install torch==2.7.0` pulls ~3.5 GB of `nvidia-*-cu12` wheels → **sandbox disk (10 GB cap) OOMs**. CPU-only torch wheels live on `download.pytorch.org` which is **not in the Daytona egress whitelist**, so `--index-url https://download.pytorch.org/whl/cpu` gets TLS-reset. |
| `count-dataset-tokens` | Tokenizing a 1 k-sample dataset with `Qwen2.5-1.5B-Instruct` on a **1-vCPU sandbox** takes longer than the 900 s agent timeout. Doubling the timeout (30 min) wasn't enough. Needs `--override-cpus 4` + `--agent-timeout-multiplier 4`, still risky. |
| `mcmc-sampling-stan` | Solution `install.packages('remotes', repos='https://cloud.r-project.org/')` — **CRAN is not on Daytona's whitelist**. First line fails; the rest of the RStan install cascade can't proceed. |
| `rstan-to-pystan` | Same CRAN block + compiling httpstan/pystan3 from source needs other blocked hosts. |

### openhands-local stuck-thread tasks (1 — excluded by default)

The custom `openhands-local` agent (`tb2/agents/openhands_local.py`) drives openhands-sdk in-process, forwarding commands via harbor's `environment.exec()`. When the Daytona sandbox dies mid-exec the SDK's worker thread is stuck in an **uncancellable wait** inside `env.exec()` — harbor's `agent_timeout` never fires, the watchdog can't SIGTERM a Python thread, and the sweep wrapper stalls on that one trial until `harbor` itself gets SIGTERM'd. Until openhands-sdk exposes cancel-aware exec, we just skip the offending task.

| Task | Trigger |
|---|---|
| `reshard-c4-data` | Long file-IO phase causes Daytona sandbox to disconnect; SDK exec thread blocks indefinitely. Observed in Run 4 (instruct, 80 min idle) and Run 5 (instruct + crane, ~30 min idle each). |

Env override:
```bash
EXCLUDE="" ./tb2/sh/run_tb2_vllm_models.sh                    # include everything
EXCLUDE=pytorch-model-cli ./tb2/sh/run_tb2_vllm_models.sh     # keep others out, try this one
```

## Known-bad tasks (oracle fails ≠ infra bug)

Tasks where the upstream `solution/solve.sh` doesn't produce what `tests/test.sh` expects — bug in the task definition, not in our setup. zai-verified did **not** fix these. Run them if you want, they'll always be `reward=0`:

- `bn-fit-modify` — solution doesn't write `/app/final_bn_sample.csv`
- `build-pov-ray` — solution doesn't create `/app/povray-2.2` source dir
- `compile-compcert` — solution doesn't produce `/tmp/CompCert/ccomp`

## Why not local rootless podman

TL;DR: no `/etc/subuid` entry for the user on this cluster → single-uid namespace → `dpkg` postinst `chown` on `sshd`/`_apt`/`fontconfig` always fails with `Operation not permitted`. Apptainer hits the same wall. Only fixes are (a) admin adds subuid or (b) run in a cloud sandbox. We picked (b).
