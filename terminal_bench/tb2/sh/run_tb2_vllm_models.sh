#!/usr/bin/env bash
#SBATCH -A ${SBATCH_ACCOUNT:-default}
#SBATCH -p ${SBATCH_PARTITION:-gpu}
#SBATCH -q ${SBATCH_QOS:-default}
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=48
#SBATCH -J tb2-vllm
#SBATCH -o ${CRANE_REPO_ROOT}/terminal_bench/tb2/sbatch-log/%x-%j.out
#SBATCH -e ${CRANE_REPO_ROOT}/terminal_bench/tb2/sbatch-log/%x-%j.err

#
# Terminal-Bench 2.0 eval for locally-deployed models via vLLM + Harbor + Daytona.
#
# Flow:
#   for each (model_name, model_path) in MODELS:
#       start vLLM (port 18016) on $model_path, tensor-parallel = #GPUs
#       harbor run --env daytona --agent terminus-2 --model openai/<model_name>
#                  over tb2-zai (89 tasks, n=10 concurrent sandboxes)
#       stop vLLM, wait for GPU to go idle
#   print summary table of pass rates per model
#
# Usage:
#   sbatch sh/run_tb2_vllm_models.sh                     # full run
#   bash   sh/run_tb2_vllm_models.sh --limit 5           # smoke (5 tasks only)
#   bash   sh/run_tb2_vllm_models.sh --task chess-best-move  # single task
#   DATASET=tb2-official bash sh/run_tb2_vllm_models.sh  # laude baseline
#   N=4 AGENT=terminus-1 bash sh/run_tb2_vllm_models.sh  # different concurrency / agent
#
# Prereqs: Harbor CLI with daytona extra installed; DAYTONA_API_KEY in env
# or hard-coded in tb2/run_harbor_daytona.sh (harvested automatically).

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${TB2_ROOT}/sbatch-log"
JOBS_DIR="${JOBS_DIR:-/tmp/${USER}-harbor-jobs}"
mkdir -p "$RESULTS_DIR" "$JOBS_DIR"

# Share the same vllm/cuda/venv env as the roo scripts
ROO_ENV="${CRANE_REPO_ROOT}/roo_test/sh/env.sh"
[[ -f "$ROO_ENV" ]] && source "$ROO_ENV" || true

# ── Config ──────────────────────────────────────────────────────────────────
VLLM_PORT="${VLLM_PORT:-18016}"
VLLM_PID=""
DATASET_NAME="${DATASET:-tb2-zai}"
DATASET_DIR="${TB2_ROOT}/${DATASET_NAME}"
AGENT="${AGENT:-terminus-2}"
N_CONCURRENT="${N:-24}"
# 2x ceiling for long-running tasks. Fast tasks exit early anyway.
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2}"
# Tasks we can't run on Daytona Tier 2 (disk cap / blocked egress).
# See tb2/README.md. Override: EXCLUDE="" to include them.
EXCLUDE="${EXCLUDE:-pytorch-model-cli,count-dataset-tokens,mcmc-sampling-stan,rstan-to-pystan}"

# Pick up DAYTONA_API_KEY from the daytona runner if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
    DAYTONA_API_KEY=$(grep -oP 'DAYTONA_API_KEY=\K\S+' "${TB2_ROOT}/run_harbor_daytona.sh" 2>/dev/null || true)
fi
: "${DAYTONA_API_KEY:?Set DAYTONA_API_KEY or put it in tb2/run_harbor_daytona.sh}"
export DAYTONA_API_KEY

# ── Model registry ──────────────────────────────────────────────────────────
# Format: "name|path_or_hf_id|is_hf"
#   name       — short name vLLM exposes and Harbor passes as --model openai/<name>
#   path_or_hf — local dir (NFS) OR HuggingFace repo id when is_hf=true
#   is_hf      — "true" to snapshot_download first, else "false"
MERGED_DIR="${CRANE_MERGED_DIR}"
MODELS=(
    # --- EDIT ME ---
    # Example: locally-served merged models
    # "crane-30b-a015-newgsp|${MERGED_DIR}/crane_30b_a015_newgsp|false"
    #
    # Example: stock HF model — snapshot-downloaded into HF_HOME first
    # "qwen3-30b|Qwen/Qwen3-30B-A3B-Instruct-2507|true"
    "qwen3-30b|Qwen/Qwen3-30B-A3B-Instruct-2507|true"
)

# ── Arg parsing ─────────────────────────────────────────────────────────────
EXTRA_HARBOR_ARGS=()
LIMIT=""
TASK_ID=""
while (( $# > 0 )); do
    case "$1" in
        --limit)  LIMIT="$2"; shift 2 ;;
        --task)   TASK_ID="$2"; shift 2 ;;
        *)        EXTRA_HARBOR_ARGS+=("$1"); shift ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
sep()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; echo; }

cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        log "Cleanup: stopping vLLM (pid $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vllm_ready() { curl -sf "http://localhost:${VLLM_PORT}/v1/models" -o /dev/null 2>/dev/null; }

resolve_hf_path() {
    local hf_id="$1"
    python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${hf_id}', cache_dir='${HF_HOME}',
      allow_patterns=['*.safetensors','*.index.json','config.json','*.json','*.model','*.txt']))
"
}

start_vllm() {
    local model_path="$1" model_name="$2" log_file="$3"
    log "Starting vLLM: $model_name on port $VLLM_PORT..."
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub"
    local tp_size="${VLLM_TP_SIZE:-1}"
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( gpu_count > 1 )); then
        tp_size=$gpu_count
        log "  Auto-detected $gpu_count GPUs, tensor-parallel-size=$tp_size"
    fi
    local max_model_len="${VLLM_MAX_MODEL_LEN:-131072}"
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    vllm serve "$model_path" \
        --served-model-name "$model_name" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --dtype bfloat16 --max-model-len "$max_model_len" \
        --gpu-memory-utilization 0.90 --tensor-parallel-size "$tp_size" \
        --enable-expert-parallel \
        --trust-remote-code --enable-prefix-caching \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --generation-config vllm \
        >> "$RESULTS_DIR/$log_file" 2>&1 &
    VLLM_PID=$!
    log "vLLM started (pid $VLLM_PID), waiting..."
    local WAIT=0 MAX=420 STEP=120
    until vllm_ready; do
        kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM died. See: $RESULTS_DIR/$log_file"
        (( WAIT >= MAX )) && MAX=$(( MAX + STEP )) && log "  Extending wait to ${MAX}s"
        sleep 5; WAIT=$(( WAIT + 5 ))
        log "  Waiting... (${WAIT}s)"
    done
    log "vLLM ready: $model_name"
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        log "Stopping vLLM (pid $VLLM_PID)..."
        pkill -9 -P "$VLLM_PID" 2>/dev/null || true
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    local tries=0
    while (( tries < 30 )); do
        local gpu_max
        gpu_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                  | tr -d ' ' | sort -rn | head -1)
        if [[ -z "$gpu_max" ]] || (( gpu_max <= 100 )); then
            log "GPU memory free (${gpu_max:-n/a} MiB)."
            break
        fi
        log "  GPU still ${gpu_max} MiB used, waiting..."
        sleep 5
        tries=$(( tries + 1 ))
    done
    sync
    sleep 3
}

run_tb2_sweep() {
    local model_name="$1" job_stamp="$2"
    local api_base="http://localhost:${VLLM_PORT}/v1"

    log "Harbor sweep: agent=$AGENT  model=openai/$model_name  dataset=$DATASET_NAME  n=$N_CONCURRENT"

    # Harbor's terminus agent runs LOCALLY (on this HPC) and calls LiteLLM; it
    # reads OPENAI_API_KEY from our env, not from --agent-env (which only
    # propagates into the Daytona sandbox, not into the local agent process).
    export OPENAI_API_BASE="${api_base}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"
    local harbor_args=(
        --path "$DATASET_DIR"
        --env daytona
        --agent "$AGENT"
        --model "openai/$model_name"
        --n-concurrent "$N_CONCURRENT"
        --jobs-dir "$JOBS_DIR"
        --job-name "${job_stamp}"
        --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT}"
        --agent-kwarg "api_base=${api_base}"
    )
    if [[ -n "$TASK_ID" ]]; then
        harbor_args=( --path "${DATASET_DIR}/${TASK_ID}" "${harbor_args[@]:2}" )
    fi
    if [[ -n "$LIMIT" ]]; then
        harbor_args+=( -l "$LIMIT" )
    fi
    if [[ -n "${EXCLUDE}" && -z "$TASK_ID" ]]; then
        IFS=',' read -ra _excl <<< "${EXCLUDE}"
        for t in "${_excl[@]}"; do
            [[ -n "$t" ]] && harbor_args+=( --exclude-task-name "$t" )
        done
    fi

    if harbor run "${harbor_args[@]}" "${EXTRA_HARBOR_ARGS[@]}"; then
        log "  Sweep ok."
    else
        log "  WARNING: harbor exited $?, continuing."
    fi
}

# ── Main ────────────────────────────────────────────────────────────────────

OVERALL_START=$(date +%s)
NUM_MODELS=${#MODELS[@]}
sep "TB2.0 vLLM eval: ${NUM_MODELS} models × ${DATASET_NAME} (harbor port ${VLLM_PORT})"
log "Agent:   $AGENT"
log "Dataset: $DATASET_NAME  ($DATASET_DIR)"
log "Models:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path is_hf <<< "$entry"
    log "  - $name  ($path)  hf=$is_hf"
done
[[ -n "$LIMIT"   ]] && log "Limit:  $LIMIT tasks"
[[ -n "$TASK_ID" ]] && log "Single: $TASK_ID"
log "Harbor extra args: ${EXTRA_HARBOR_ARGS[*]:-<none>}"
echo

declare -a MODEL_NAMES
MODEL_IDX=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path is_hf <<< "$entry"
    MODEL_NAMES+=("$model_name")
    MODEL_IDX=$(( MODEL_IDX + 1 ))
    sep "Model ${MODEL_IDX}/${NUM_MODELS}: ${model_name}"

    if [[ "$is_hf" == "true" ]]; then
        log "Resolving HuggingFace model: $model_path"
        model_path=$(resolve_hf_path "$model_path") || die "snapshot_download failed for $model_path"
        log "Resolved to: $model_path"
    fi
    [[ -d "$model_path" ]] || die "Model dir not found: $model_path"

    vllm_log="vllm-tb2-${model_name}.log"
    start_vllm "$model_path" "$model_name" "$vllm_log"

    job_stamp="tb2-${model_name}-$(date +%Y%m%d_%H%M%S)"
    run_tb2_sweep "$model_name" "$job_stamp"

    stop_vllm
done

# ── Summary ─────────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - OVERALL_START ))
sep "DONE  (total ${ELAPSED}s ≈ $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"

echo
printf "  %-40s  %-12s  %-12s  %-10s\n" "Model" "Pass" "Total" "Accuracy"
printf "  %-40s  %-12s  %-12s  %-10s\n" "$(printf '%*s' 40 '' | tr ' ' -)" "----" "-----" "--------"

for model_name in "${MODEL_NAMES[@]}"; do
    # Find the most recent job dir matching this model
    job_dir=$(ls -td "${JOBS_DIR}"/tb2-"${model_name}"-*/ 2>/dev/null | head -1)
    if [[ -z "$job_dir" || ! -f "$job_dir/result.json" ]]; then
        printf "  %-40s  %-12s  %-12s  %-10s\n" "$model_name" "—" "—" "—"
        continue
    fi
    read pass total acc <<< $(python3 -c "
import json
d = json.load(open('${job_dir}/result.json'))
stats = d.get('stats', {})
evals = stats.get('evals', {})
# Harbor groups evals by '<agent>__<adhoc|dataset>'; grab whichever is present
agg = next(iter(evals.values()), {})
metrics = agg.get('metrics', [{}])
mean = metrics[0].get('mean', 0.0) if metrics else 0.0
n = agg.get('n_trials', stats.get('n_trials', 0))
# pass count from reward_stats if present
rs = agg.get('reward_stats', {}).get('reward', {})
p = len(rs.get('1.0', []))
print(f'{p} {n} {mean:.4f}')
" 2>/dev/null)
    printf "  %-40s  %-12s  %-12s  %-10s\n" "$model_name" "${pass:-?}" "${total:-?}" "${acc:-?}"
done
echo
echo "Job dirs under ${JOBS_DIR}"
