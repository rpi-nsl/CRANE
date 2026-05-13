#!/usr/bin/env bash
#SBATCH -A ${SBATCH_ACCOUNT:-default}
#SBATCH -p ${SBATCH_PARTITION:-gpu}
#SBATCH -q ${SBATCH_QOS:-default}
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3:2
#SBATCH --cpus-per-task=24
#SBATCH -J tb2-30b
#SBATCH -o ${CRANE_REPO_ROOT}/terminal_bench/tb2/sbatch-log/%x-%j.out
#SBATCH -e ${CRANE_REPO_ROOT}/terminal_bench/tb2/sbatch-log/%x-%j.err
#
# Terminal-Bench 2.0 eval driver for 30B-class models (fits on 2× H100 80GB).
#
# vLLM on this HPC → Cloudflare named tunnel (https://your-vllm.example.com/v1) →
# Harbor (terminus-* agent) → Daytona cloud sandboxes, one per task.
#
# Usage:
#   sbatch sh/run_tb2.sh                                    # full: every model × 85 tasks
#   bash   sh/run_tb2.sh --limit 10                          # smoke (10 tasks per model)
#   bash   sh/run_tb2.sh --task chess-best-move              # single task per model
#   bash   sh/run_tb2.sh --model qwen3-30b-instruct          # only one model from registry
#   bash   sh/run_tb2.sh --agent terminus-1                  # switch agent
#   bash   sh/run_tb2.sh --dataset tb2-official              # laude baseline dataset
#   bash   sh/run_tb2.sh --concurrency 10                    # lower Daytona concurrency
#   bash   sh/run_tb2.sh --keep-vllm --model qwen3-30b-instruct  # don't tear vLLM down
#
# Add models by editing the MODELS array below. Same format as run_swebench.sh.
#
set -euo pipefail
: "${DAYTONA_API_KEY:?Set DAYTONA_API_KEY (see tb2/README.md)}"
: "${CF_API_TOKEN:?Set CF_API_TOKEN (Cloudflare DNS edit token)}"
# Bypass other users' nvidia-cuda-mps daemon on shared GPU nodes —
# their MPS server's UNIX socket wedges every other tenant's CUDA init forever.
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Share the roo/swe-lite venv + cache setup (HF cache, CUDA, etc.)
ROO_ENV="${CRANE_REPO_ROOT}/roo_test/sh/env.sh"
[[ -f "$ROO_ENV" ]] && source "$ROO_ENV" || true
VLLM_VENV="${VLLM_VENV:-${CRANE_REPO_ROOT}/.venv}"
export HF_HOME="${HF_HOME:-${HF_HOME}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

RESULTS_DIR="${RESULTS_DIR:-$TB2_ROOT/sbatch-log}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$RESULTS_DIR/vllm}"
JOBS_DIR="${JOBS_DIR:-/tmp/${USER}-harbor-jobs}"
mkdir -p "$RESULTS_DIR" "$VLLM_LOG_DIR" "$JOBS_DIR"

# ── Model registry ──────────────────────────────────────────────────────────
# Format: "shortname|hf_id_or_local_dir|is_hf(true|false)"
# Each row reboots vLLM. Short name is what harbor sends as openai/<name>
# and what vLLM exposes on /v1/models.
MERGED_DIR="${CRANE_MERGED_DIR}"
BASELINE_DIR="${CRANE_REPO_ROOT}/baseline/baseline_model"
MODELS=(
    # "qwen3-30b-instruct|Qwen/Qwen3-30B-A3B-Instruct-2507|true"
    # "qwen3-30b-thinking|Qwen/Qwen3-30B-A3B-Thinking-2507|true"
    "crane-30b|${MERGED_DIR}/crane_30b|false"
    # "baseline-ta|${BASELINE_DIR}/task_arithmetic|false"
    # "baseline-slerp|${BASELINE_DIR}/slerp|false"
    # "baseline-ties|${BASELINE_DIR}/ties|false"
    # "baseline-aim-ta|${BASELINE_DIR}/aim_ta|false"
    "baseline-aim-ties|${BASELINE_DIR}/aim_ties|false"
    # "baseline-lewis|${BASELINE_DIR}/lewis|false"

)
# 4 GPUs → TP=4 default (faster throughput for 30B MoE). Override with VLLM_TP_SIZE=N.
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"

# ── Defaults ────────────────────────────────────────────────────────────────
DATASET_NAME="${DATASET:-tb2-zai}"
AGENT="${AGENT:-openhands}"
N_CONCURRENT="${N:-20}"
N_ATTEMPTS="${N_ATTEMPTS:-3}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-1}"
# Harbor retries on DaytonaError / DaytonaNotFoundError / transient network blips
# (but not on AgentTimeoutError / VerifierTimeoutError — those are in its
# default --retry-exclude list, correctly treated as genuine failures).
MAX_RETRIES="${MAX_RETRIES:-2}"
EXCLUDE="${EXCLUDE:-pytorch-model-cli,count-dataset-tokens,mcmc-sampling-stan,rstan-to-pystan,reshard-c4-data}"
LIMIT=""
TASK_ID=""
MODEL_FILTER=""
SKIP_SWEEP=false
KEEP_VLLM=false

VLLM_PORT="${VLLM_PORT:-18016}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_PID=""

# Cloudflare named tunnel (set up once via `cloudflared tunnel create` + DNS CNAME).
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$HOME/bin/cloudflared}"
CF_TUNNEL_CREDS="${CF_TUNNEL_CREDS:-$HOME/.cloudflared/qwen-tb2.json}"
PUBLIC_URL="${PUBLIC_URL:-your-vllm.example.com}"
PUBLIC_API_BASE="https://${PUBLIC_URL}/v1"
CF_PID=""

# Daytona — API key for harbor + cleanup watchdog age threshold (min).
: "${DAYTONA_API_KEY:?Set DAYTONA_API_KEY (see tb2/README.md)}"
CLEANUP_AGE_MIN="${CLEANUP_AGE_MIN:-75}"
CLEANUP_INTERVAL="${CLEANUP_INTERVAL:-300}"
HARBOR_PY="${HARBOR_PY:-${MZ_CACHE}/uv-tools/harbor/bin/python3}"

# Force-kill harbor if all expected trials wrote result.json AND no new
# result.json was written for AGG_WATCHDOG_MIN minutes. Defends against the
# observed harbor post-sweep aggregation hang. 0 disables.
AGG_WATCHDOG_MIN="${AGG_WATCHDOG_MIN:-0}"

# ── Parse args ──────────────────────────────────────────────────────────────
while (($#)); do
    case "$1" in
        --dataset)      DATASET_NAME="$2"; shift 2 ;;
        --agent)        AGENT="$2"; shift 2 ;;
        --concurrency)  N_CONCURRENT="$2"; shift 2 ;;
        --attempts|-k)  N_ATTEMPTS="$2"; shift 2 ;;
        --limit|-l)     LIMIT="$2"; shift 2 ;;
        --task)         TASK_ID="$2"; shift 2 ;;
        --model)        MODEL_FILTER="$2"; shift 2 ;;
        --exclude)      EXCLUDE="$2"; shift 2 ;;
        --skip-sweep)   SKIP_SWEEP=true; shift ;;
        --keep-vllm)    KEEP_VLLM=true; shift ;;
        --port)         VLLM_PORT="$2"; shift 2 ;;
        -h|--help)      sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

DATASET_DIR="$TB2_ROOT/$DATASET_NAME"
[[ -d "$DATASET_DIR" ]] || { echo "ERROR: dataset dir not found: $DATASET_DIR" >&2; exit 1; }

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
sep()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; echo; }

vllm_ready_port() { curl -sf "http://localhost:${VLLM_PORT}/v1/models" -o /dev/null 2>/dev/null; }
vllm_serves_model() {
    local want="$1"
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | grep -q "\"$want\""
}
tunnel_ready() { curl -sf -m 5 "$PUBLIC_API_BASE/models" -o /dev/null 2>/dev/null; }

resolve_hf_path() {
    local hf_id="$1"
    "${VLLM_VENV}/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${hf_id}', cache_dir='${HF_HOME}',
      allow_patterns=['*.safetensors','*.index.json','config*.json','*.json','*.model','tokenizer*','*.txt']))
"
}

start_vllm() {
    local model_path="$1" served_name="$2" log_file="$3"
    if vllm_serves_model "$served_name"; then
        log "vLLM already serving ${served_name} on :${VLLM_PORT}; reusing."
        VLLM_PID=""   # we don't own it
        return 0
    fi
    # If something else is on the port serving a different model, we need to kill it
    if vllm_ready_port; then
        log "  other vLLM on :${VLLM_PORT} — stopping it to load ${served_name}"
        stop_vllm
    fi
    source "${VLLM_VENV}/bin/activate"
    local tp="${VLLM_TP_SIZE:-}"
    if [[ -z "$tp" ]]; then
        tp=$(nvidia-smi -L 2>/dev/null | wc -l); (( tp < 1 )) && tp=1
    fi
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    log "Launching vLLM: $served_name @ :${VLLM_PORT}  tp=$tp  len=$VLLM_MAX_MODEL_LEN"
    vllm serve "$model_path" \
        --served-model-name "$served_name" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --dtype bfloat16 --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.95 --tensor-parallel-size "$tp" \
        --enable-expert-parallel \
        --trust-remote-code --enable-prefix-caching \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --generation-config vllm \
        --override-generation-config '{"temperature":0.6,"top_p":0.8,"top_k":20}' \
        --enable-prompt-tokens-details \
        >> "$log_file" 2>&1 &
    VLLM_PID=$!
    log "  vllm pid=$VLLM_PID, waiting for /v1/models ..."
    local t=0 budget=360
    until vllm_ready_port; do
        kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM died; see $log_file"
        (( t >= budget )) && budget=$(( budget + 120 )) && log "    extending budget to ${budget}s"
        sleep 5; t=$(( t + 5 ))
        (( t % 60 == 0 )) && log "    still loading (${t}s)"
    done
    log "vLLM ready: $served_name"
    deactivate 2>/dev/null || true
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        pkill -9 -P "$VLLM_PID" 2>/dev/null || true
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm serve"       2>/dev/null || true
    local tries=0
    while (( tries < 30 )); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | tr -d ' ' | sort -rn | head -1)
        if [[ -z "$used" ]] || (( used <= 200 )); then
            log "  GPU free (${used:-n/a} MiB)"; break
        fi
        sleep 5; tries=$(( tries + 1 ))
    done
}

ensure_tunnel() {
    # If a cloudflared with the right cred file is already running, reuse it.
    if pgrep -u "$USER" -f "cloudflared.*$(basename "$CF_TUNNEL_CREDS")" >/dev/null; then
        log "Cloudflare tunnel (existing cloudflared process) — reusing"
        CF_PID=""
        return 0
    fi
    [[ -x "$CLOUDFLARED_BIN" ]] || die "cloudflared missing at $CLOUDFLARED_BIN"
    [[ -f "$CF_TUNNEL_CREDS" ]] || die "tunnel creds missing at $CF_TUNNEL_CREDS"
    local tun_name
    tun_name=$(python3 -c "import json;print(json.load(open('$CF_TUNNEL_CREDS'))['TunnelName'])")
    local cflog="$RESULTS_DIR/cloudflared.log"
    : > "$cflog"
    log "Starting cloudflared (tunnel=$tun_name, cred=$CF_TUNNEL_CREDS)"
    "$CLOUDFLARED_BIN" tunnel --credentials-file "$CF_TUNNEL_CREDS" \
        --url "http://localhost:${VLLM_PORT}" run "$tun_name" \
        >> "$cflog" 2>&1 &
    CF_PID=$!
    # Wait for at least one Cloudflare edge connection to register (proves
    # cloudflared is healthy). We do NOT wait for the origin (vLLM) — that
    # comes up later in the per-model loop.
    local t=0
    until grep -q "Registered tunnel connection" "$cflog" 2>/dev/null; do
        kill -0 "$CF_PID" 2>/dev/null || die "cloudflared died; see $cflog"
        sleep 3; t=$(( t + 3 ))
        (( t >= 120 )) && die "cloudflared never registered; see $cflog"
    done
    log "Cloudflare tunnel registered: ${PUBLIC_API_BASE}"
}

start_daytona_watchdog() {
    (( CLEANUP_INTERVAL > 0 )) || return 0
    [[ -x "$HARBOR_PY" ]] || return 0
    # PYTHONUNBUFFERED=1 so the watchdog's per-check log lines flush to disk
    # in real time (otherwise Python block-buffers stdout when redirected).
    PYTHONUNBUFFERED=1 "$HARBOR_PY" "${TB2_ROOT}/cleanup_daytona.py" \
        --watch "${CLEANUP_INTERVAL}" --age "${CLEANUP_AGE_MIN}" --delete \
        > "$RESULTS_DIR/cleanup-daytona.log" 2>&1 &
    WATCHDOG_PID=$!
    log "Daytona cleanup watchdog pid=$WATCHDOG_PID (every ${CLEANUP_INTERVAL}s, age ${CLEANUP_AGE_MIN} min)"
}

stop_daytona_watchdog() {
    [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
}

cleanup() {
    stop_daytona_watchdog || true
    if ! $KEEP_VLLM; then stop_vllm || true; fi
    if [[ -n "$CF_PID" ]]; then kill "$CF_PID" 2>/dev/null || true; fi
    if [[ -x "$HARBOR_PY" ]]; then
        "$HARBOR_PY" "${TB2_ROOT}/cleanup_daytona.py" --age 0 --delete \
            >> "$RESULTS_DIR/cleanup-daytona.log" 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

run_harbor_sweep() {
    local model_name="$1" out_dir="$2"
    mkdir -p "$out_dir"
    # terminus / openhands-local agents run LOCALLY (on this HPC) and call LiteLLM →
    # need the api key in our env, not --agent-env.
    export OPENAI_API_BASE="$PUBLIC_API_BASE"
    export OPENAI_API_KEY="placeholder"
    export LLM_API_KEY="${LLM_API_KEY:-dummy-key-for-local-vllm}"
    # openhands-local lives in tb2/agents/, loaded via --agent-import-path.
    export PYTHONPATH="${TB2_ROOT}:${PYTHONPATH:-}"

    local harbor_args=(
        --path "$DATASET_DIR"
        --env daytona
        --n-concurrent "$N_CONCURRENT"
        --n-attempts "$N_ATTEMPTS"
        --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT"
        --max-retries "$MAX_RETRIES"
        --jobs-dir "$out_dir"
        --job-name "${model_name}_${DATASET_NAME}_${RUN_STAMP}"
    )

    case "$AGENT" in
        openhands-local)
            # Custom agent: openhands-sdk in-process; routes terminal commands
            # via harbor environment.exec(). LiteLLM model prefix differs.
            harbor_args+=(
                --agent-import-path "agents.openhands_local:OpenHandsLocal"
                --model "${model_name}"
                --agent-kwarg "api_base=${PUBLIC_API_BASE}"
            )
            ;;
        *)
            # Stock harbor agent (terminus-2 etc.)
            harbor_args+=(
                --agent "$AGENT"
                --model "openai/${model_name}"
                --agent-kwarg "api_base=${PUBLIC_API_BASE}"
                --agent-kwarg "temperature=0.6"
                --agent-kwarg 'llm_call_kwargs={"top_p":0.8,"extra_body":{"top_k":20}}'
            )
            ;;
    esac
    if [[ -n "$TASK_ID" ]]; then
        harbor_args=(--path "$DATASET_DIR/$TASK_ID" "${harbor_args[@]:2}")
    fi
    if [[ -n "$LIMIT" ]]; then
        harbor_args+=(-l "$LIMIT")
    fi
    # Excludes (glob ok). Only applied when running the full dataset, not a single task.
    if [[ -n "$EXCLUDE" && -z "$TASK_ID" ]]; then
        IFS=',' read -ra _excl <<< "$EXCLUDE"
        for t in "${_excl[@]}"; do
            [[ -n "$t" ]] && harbor_args+=(--exclude-task-name "$t")
        done
    fi

    local agg_expected=0
    if [[ -n "$TASK_ID" ]]; then
        agg_expected=1
    elif [[ -n "$LIMIT" ]]; then
        agg_expected="$LIMIT"
    else
        local total excluded=0
        total=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | wc -l)
        [[ -n "$EXCLUDE" ]] && excluded=$(echo "$EXCLUDE" | tr ',' '\n' | grep -c .)
        agg_expected=$(( total - excluded ))
    fi
    agg_expected=$(( agg_expected * N_ATTEMPTS ))

    harbor run "${harbor_args[@]}" > >(tee -a "${out_dir}/run.log") 2>&1 &
    local harbor_pid=$!
    log "  harbor pid=$harbor_pid (agg-watchdog: ${agg_expected} trials, ${AGG_WATCHDOG_MIN} min idle)"

    local agg_pid=""
    if (( AGG_WATCHDOG_MIN > 0 )); then
        harbor_aggregation_watchdog "$harbor_pid" "$out_dir" "$agg_expected" "$AGG_WATCHDOG_MIN" &
        agg_pid=$!
    fi

    wait "$harbor_pid"
    local rc=$?
    [[ -n "$agg_pid" ]] && kill "$agg_pid" 2>/dev/null && wait "$agg_pid" 2>/dev/null
    return "$rc"
}

write_model_summary() {
    local model_name="$1" out_dir="$2" wall_seconds="$3"
    local job_dir
    job_dir=$(ls -td "$out_dir"/${model_name}_${DATASET_NAME}_${RUN_STAMP}*/ 2>/dev/null | head -1)
    [[ -z "$job_dir" ]] && { log "  no job dir found for summary"; return; }
    python3 - "$job_dir" "$model_name" "$wall_seconds" "$out_dir/summary.json" <<'PY'
import json, pathlib, sys, datetime
from collections import defaultdict
job_dir = pathlib.Path(sys.argv[1])
model_name, wall_seconds, out_path = sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
p = f = e = in_tok = cache_tok = out_tok = n_episodes = 0
trials = []
by_task = defaultdict(list)  # task_name -> [True/False per attempt]
for rj in sorted(job_dir.glob('*/result.json')):
    try:
        r = json.loads(rj.read_text())
    except Exception:
        continue
    ag = r.get('agent_result') or {}
    ip, cp, op = ag.get('n_input_tokens') or 0, ag.get('n_cache_tokens') or 0, ag.get('n_output_tokens') or 0
    ep = (ag.get('metadata') or {}).get('n_episodes') or 0
    in_tok += ip; cache_tok += cp; out_tok += op; n_episodes += ep
    exc = r.get('exception_info')
    reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward', 0)
    outcome = 'exception' if exc else ('pass' if reward > 0.5 else 'fail')
    if outcome == 'pass': p += 1
    elif outcome == 'fail': f += 1
    else: e += 1
    by_task[r.get('task_name')].append(outcome == 'pass')
    trials.append({
        'task': r.get('task_name'), 'trial': r.get('trial_name'),
        'outcome': outcome, 'reward': reward, 'n_episodes': ep,
        'n_input_tokens': ip, 'n_cache_tokens': cp, 'n_output_tokens': op,
        'exception_type': (exc or {}).get('exception_type'),
    })
n_tasks = len(by_task)
n_trials_total = sum(len(v) for v in by_task.values())
k = max((len(v) for v in by_task.values()), default=0)
pass_at_1 = (p / n_trials_total) if n_trials_total else 0.0
pass_at_k = (sum(1 for v in by_task.values() if any(v)) / n_tasks) if n_tasks else 0.0
pass_all  = (sum(1 for v in by_task.values() if v and all(v)) / n_tasks) if n_tasks else 0.0
per_task = {t: {'attempts': len(v), 'passes': sum(v)} for t, v in sorted(by_task.items())}
summary = {
    'model': model_name,
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'job_dir': str(job_dir),
    'wall_seconds': wall_seconds,
    'counts': {'pass': p, 'fail': f, 'exception': e, 'total': p + f + e,
               'n_tasks': n_tasks, 'k': k},
    'metrics': {'pass@1': pass_at_1, f'pass@{k}': pass_at_k, 'pass_all': pass_all},
    'tokens': {'input': in_tok, 'cache': cache_tok, 'output': out_tok},
    'n_episodes': n_episodes,
    'per_task': per_task,
    'trials': trials,
}
out_path.write_text(json.dumps(summary, indent=2))
print(f'  summary: {out_path.name} (tasks={n_tasks} k={k} '
      f'pass@1={pass_at_1:.3f} pass@{k}={pass_at_k:.3f} pass_all={pass_all:.3f} '
      f'in={in_tok} cache={cache_tok} out={out_tok})')
PY
}

harbor_aggregation_watchdog() {
    local harbor_pid="$1" out_dir="$2" expected="$3" idle_min="$4"
    local idle_sec=$(( idle_min * 60 ))
    while kill -0 "$harbor_pid" 2>/dev/null; do
        sleep 60
        local trial_count
        trial_count=$(find "$out_dir" -mindepth 3 -maxdepth 3 -name result.json 2>/dev/null | wc -l)
        (( trial_count >= expected )) || continue
        local latest
        latest=$(find "$out_dir" -name result.json -printf "%T@\n" 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
        [[ -z "$latest" ]] && continue
        local idle_for=$(( $(date +%s) - latest ))
        if (( idle_for > idle_sec )); then
            echo "[$(date +%H:%M:%S)] agg-watchdog: ${trial_count}/${expected} done, idle ${idle_for}s → SIGTERM harbor (pid $harbor_pid)" >&2
            kill -TERM "$harbor_pid" 2>/dev/null || true
            return
        fi
    done
}

# ── Filter models ───────────────────────────────────────────────────────────
if [[ -n "$MODEL_FILTER" ]]; then
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name _p _h <<< "$entry"
        [[ "$name" == "$MODEL_FILTER" ]] && FILTERED+=("$entry")
    done
    (( ${#FILTERED[@]} > 0 )) || die "Model '$MODEL_FILTER' not in registry"
    MODELS=("${FILTERED[@]}")
fi

# ── Go ──────────────────────────────────────────────────────────────────────
OVERALL_START=$(date +%s)
# Timestamp that tags this entire invocation. Harbor refuses to reuse an
# existing job_dir with a different config, so we never reuse — always a
# fresh dir. Historical runs stay archived under sbatch-log/ for compare.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
sep "TB 2.0  |  ${#MODELS[@]} model(s)  |  dataset=${DATASET_NAME}  |  agent=${AGENT}  |  n=${N_CONCURRENT}  |  stamp=${RUN_STAMP}"
log "Public URL:  $PUBLIC_API_BASE"
log "Daytona key set (length=${#DAYTONA_API_KEY})"
log "Models:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path _ <<< "$entry"
    log "  • $name  ($path)"
done
[[ -n "$TASK_ID" ]] && log "Task:         $TASK_ID"
[[ -n "$LIMIT"   ]] && log "Limit:        $LIMIT tasks"
log "Excluding:   ${EXCLUDE:-<none>}"

ensure_tunnel
start_daytona_watchdog

declare -a MODEL_SUMMARIES
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path is_hf <<< "$entry"
    sep "Model: ${model_name}"

    if [[ "$is_hf" == "true" ]]; then
        log "Resolving HF snapshot: $model_path"
        model_path=$(resolve_hf_path "$model_path") || die "snapshot_download failed for $model_path"
        log "  → $model_path"
    fi
    [[ -d "$model_path" ]] || die "Model dir not found: $model_path"

    vllm_log="${VLLM_LOG_DIR}/${model_name}.log"
    start_vllm "$model_path" "$model_name" "$vllm_log"

    # Re-verify tunnel hit the current vLLM (paranoia after restart)
    if ! curl -sf -m 10 "$PUBLIC_API_BASE/models" | grep -q "\"${model_name}\""; then
        log "tunnel still sees old model — giving Cloudflare 20s to pick up the new backend"
        sleep 20
    fi

    if $SKIP_SWEEP; then
        log "Skipping harbor sweep (--skip-sweep)"
        continue
    fi

    run_tag="${model_name}_${DATASET_NAME}_${RUN_STAMP}"
    out_dir="${RESULTS_DIR}/${run_tag}"
    log "Harbor sweep → $out_dir"
    agent_start=$(date +%s)
    if run_harbor_sweep "$model_name" "$out_dir"; then
        log "  Sweep ok."
    else
        log "  WARNING: sweep exited $?, continuing."
    fi
    echo $(( $(date +%s) - agent_start )) > "$out_dir/.agent_wall_seconds"
    write_model_summary "$model_name" "$out_dir" "$(cat "$out_dir/.agent_wall_seconds")"

    # Per-task aggregation across N_ATTEMPTS (best-of-n, majority, all-of-n).
    if [[ -f "${TB2_ROOT}/aggregate.py" ]]; then
        log "Per-task aggregation (N_ATTEMPTS=$N_ATTEMPTS):"
        python3 "${TB2_ROOT}/aggregate.py" "$out_dir" 2>&1 | tee -a "${out_dir}/aggregate.log" | sed 's/^/  /'
    fi

    if ! $KEEP_VLLM; then
        log "Stopping vLLM before next model..."
        stop_vllm
    else
        log "Keeping vLLM alive (--keep-vllm)"
    fi

    # Reap any sandboxes this model's sweep left behind BEFORE the next model
    # starts, so a cascade of orphans doesn't accumulate across models.
    if [[ -x "$HARBOR_PY" ]]; then
        log "Between-model Daytona reap (age 0)..."
        "$HARBOR_PY" "${TB2_ROOT}/cleanup_daytona.py" --age 0 --delete \
            >> "$RESULTS_DIR/cleanup-daytona.log" 2>&1 || true
    fi
done

# Defensive: even if the EXIT trap is missed (e.g. we get SIGKILL'd during the
# summary), force one final Daytona reap here so orphans never outlive the run.
if [[ -x "$HARBOR_PY" ]]; then
    log "Final Daytona reap (age 0)..."
    "$HARBOR_PY" "${TB2_ROOT}/cleanup_daytona.py" --age 0 --delete \
        >> "$RESULTS_DIR/cleanup-daytona.log" 2>&1 || true
fi

ELAPSED=$(( $(date +%s) - OVERALL_START ))
sep "DONE (${ELAPSED}s ≈ $(( ELAPSED / 60 ))m)"

# ── Summary table ───────────────────────────────────────────────────────────
printf "\n  %-32s  %-6s  %-4s  %-8s  %-8s  %-10s  %-10s  %-12s  %-12s\n" \
    "Model" "Tasks" "k" "pass@1" "pass@k" "pass_all" "Wall(s)" "InTok" "OutTok"
printf "  %s\n" "$(printf '%.0s-' {1..120})"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name _ _ <<< "$entry"
    run_tag="${model_name}_${DATASET_NAME}_${RUN_STAMP}"
    out_dir="${RESULTS_DIR}/${run_tag}"
    job_dir=$(ls -td "$out_dir"/${run_tag}*/ 2>/dev/null | head -1)
    if [[ -z "$job_dir" ]]; then
        printf "  %-32s  %-6s  %-4s  %-8s  %-8s  %-10s  %-10s  %-12s  %-12s\n" \
            "$model_name" "—" "—" "—" "—" "—" "—" "—" "—"; continue
    fi
    read n_tasks k p1 pk pa in_tok out_tok < <(python3 -c "
import json, pathlib
from collections import defaultdict
d = pathlib.Path('$job_dir')
by_task = defaultdict(list)
in_tok = out_tok = 0
for rj in d.glob('*/result.json'):
    try: r = json.loads(rj.read_text())
    except Exception: continue
    ag = r.get('agent_result') or {}
    in_tok  += ag.get('n_input_tokens') or 0
    out_tok += ag.get('n_output_tokens') or 0
    exc = r.get('exception_info')
    reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward', 0)
    by_task[r.get('task_name')].append((not exc) and reward > 0.5)
n_tasks = len(by_task)
n_trials = sum(len(v) for v in by_task.values())
k = max((len(v) for v in by_task.values()), default=0)
p_total = sum(sum(v) for v in by_task.values())
p1 = (p_total / n_trials) if n_trials else 0.0
pk = (sum(1 for v in by_task.values() if any(v)) / n_tasks) if n_tasks else 0.0
pa = (sum(1 for v in by_task.values() if v and all(v)) / n_tasks) if n_tasks else 0.0
print(n_tasks, k, f'{p1:.3f}', f'{pk:.3f}', f'{pa:.3f}', in_tok, out_tok)
" 2>/dev/null)
    wall=$(cat "$out_dir/.agent_wall_seconds" 2>/dev/null || echo "—")
    printf "  %-32s  %-6s  %-4s  %-8s  %-8s  %-10s  %-10s  %-12s  %-12s\n" \
        "$model_name" "${n_tasks:-0}" "${k:-0}" "${p1:-0}" "${pk:-0}" "${pa:-0}" \
        "$wall" "${in_tok:-0}" "${out_tok:-0}"
done
echo
