#!/usr/bin/env bash
#SBATCH -A ${SBATCH_ACCOUNT:-default}
#SBATCH -p ${SBATCH_PARTITION:-default}
#SBATCH -q ${SBATCH_QOS:-default}
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=48
#SBATCH -J roo-rain
#SBATCH -o ${CRANE_LOG_DIR}/sbatch_log/%x-%j.out
#SBATCH -e ${CRANE_LOG_DIR}/sbatch_log/%x-%j.err

#
# Roo Test: RAIN-merged checkpoints on Python only.
#
# Models (produced by baseline/rain.py; see baseline/RAIN_README.md):
#   1. baseline-rain-30b      — RAIN-qkvo on Qwen3-30B Base + Instruct + Thinking,
#                                MoE FFN frozen. ~60 GB BF16; TP=4.
#   2. baseline-rain-80b-next — RAIN on Qwen3-Next-80B (proxy base = Thinking,
#                                hybrid arch; only 7 full_attention layers in
#                                the tail-27 receive RAIN updates). ~149 GB BF16; TP=4.
#
# Sampling/serving config:
#   - max_attempts:  11
#   - iterations:    3
#   - context_window:90000
#   - temperature:   0.6
#   - top_p:         0.8
#   - top_k:         20
#   - concurrency:   64
#   - timeout:       300s/attempt, 900s total
#
# Usage:
#   sbatch sh/run_rain.sh                           # full python eval, both models
#   bash   sh/run_rain.sh                           # interactive, no SBATCH
#   bash   sh/run_rain.sh --only 30b                # 30B only
#   bash   sh/run_rain.sh --only 80b                # 80B only
#   bash   sh/run_rain.sh --language go             # different single language
#   bash   sh/run_rain.sh --limit 3                 # sanity-check first 3 ex.

set -euo pipefail

ROOT_DIR="${CRANE_REPO_ROOT}/roo_test"
source "${ROOT_DIR}/sh/env.sh"

EVAL_RUNNER="${ROOT_DIR}/roo-eval.py"
VLLM_PORT=18016
VLLM_PID=""
LANGUAGES=(python)

# ── Model registry ──────────────────────────────────────────────────────────
BASELINE_DIR="${CRANE_REPO_ROOT}/baseline/baseline_model"
# Format: served_name|local_path|tp_size
MODELS=(
    "baseline-rain-30b|${BASELINE_DIR}/rain/unified_model_merge/rain_merged|4"
    "baseline-rain-30b-a030|${BASELINE_DIR}/rain_a030/unified_model_merge/rain_merged_a030|4"
    "baseline-rain-30b-box05|${BASELINE_DIR}/rain_box05/unified_model_merge/rain_merged_box05|4"
    "baseline-rain-30b-pycalib|${BASELINE_DIR}/rain_pycalib/unified_model_merge/rain_merged_pycalib|4"
    "baseline-rain-30b-pycalib-a030|${BASELINE_DIR}/rain_pycalib_a030/unified_model_merge/rain_merged_pycalib_a030|4"
    "baseline-rain-30b-pycalib-qkvof-a030|${BASELINE_DIR}/rain_pycalib_qkvof_a030/unified_model_merge/rain_merged_pycalib_qkvof_a030|4"
    "baseline-rain-30b-agentcalib-a030|${BASELINE_DIR}/rain_agentcalib_a030/unified_model_merge/rain_merged_agentcalib_a030|4"
    "baseline-rain-30b-pycalib-qkvof-planA-a030|${BASELINE_DIR}/rain_pycalib_qkvof_planA_a030/unified_model_merge/rain_merged_pycalib_qkvof_planA_a030|4"
    "baseline-rain-80b-next|${BASELINE_DIR}/rain_qwen3_next_80b/unified_model_merge/rain_merged|4"
    "baseline-rain-80b-next-pycalib-a030|${BASELINE_DIR}/rain_qwen3_next_80b_pycalib_a030/unified_model_merge/rain_merged_pycalib_a030|4"
    "baseline-rain-80b-next-agentcalib-a030|${BASELINE_DIR}/rain_qwen3_next_80b_agentcalib_a030/unified_model_merge/rain_merged_agentcalib_a030|4"
    "baseline-rain-80b-next-pycalib-qkvof-planA-a030|${BASELINE_DIR}/rain_qwen3_next_80b_pycalib_qkvof_planA_a030/unified_model_merge/rain_merged_pycalib_qkvof_planA_a030|4"
)

# ── Parse args ──────────────────────────────────────────────────────────────
EXTRA_ARGS=()
SINGLE_LANG=""
ONLY=""
while (( $# )); do
    case "$1" in
        --language) shift; SINGLE_LANG="$1" ;;
        --only)     shift; ONLY="$1" ;;
        *)          EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

if [[ -n "$SINGLE_LANG" ]]; then
    LANGUAGES=("$SINGLE_LANG")
fi

if [[ -n "$ONLY" ]]; then
    case "$ONLY" in
        30b)                       MODELS=("${MODELS[0]}") ;;
        30b-a030)                  MODELS=("${MODELS[1]}") ;;
        30b-box05)                 MODELS=("${MODELS[2]}") ;;
        30b-pycalib)               MODELS=("${MODELS[3]}") ;;
        30b-pycalib-a030)          MODELS=("${MODELS[4]}") ;;
        30b-pycalib-qkvof-a030)    MODELS=("${MODELS[5]}") ;;
        30b-agentcalib-a030)       MODELS=("${MODELS[6]}") ;;
        30b-planA)                 MODELS=("${MODELS[7]}") ;;
        80b|80b-next)              MODELS=("${MODELS[8]}") ;;
        80b-pycalib-a030)          MODELS=("${MODELS[9]}") ;;
        80b-agentcalib-a030)       MODELS=("${MODELS[10]}") ;;
        80b-planA)                 MODELS=("${MODELS[11]}") ;;
        *)                         echo "ERROR: see header for valid --only values" >&2; exit 2 ;;
    esac
fi

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
    pkill -9 -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -u "$USER" -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

vllm_ready() { curl -sf "http://localhost:${VLLM_PORT}/v1/models" -o /dev/null 2>/dev/null; }

start_vllm() {
    local model_path="$1" served_name="$2" tp_size="$3" log_file="$4"
    log "Starting vLLM: ${served_name} on port ${VLLM_PORT} (TP=${tp_size})..."
    mkdir -p "$ROOT_DIR/eval-results"
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    vllm serve "$model_path" \
        --served-model-name "$served_name" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --dtype bfloat16 --max-model-len 90000 \
        --gpu-memory-utilization 0.90 --tensor-parallel-size "$tp_size" \
        --trust-remote-code --enable-prefix-caching \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --generation-config vllm \
        >> "$ROOT_DIR/eval-results/$log_file" 2>&1 &
    VLLM_PID=$!
    log "vLLM started (pid $VLLM_PID), waiting..."
    local WAIT_SECS=0 MAX_WAIT=600 EXTEND=120
    until vllm_ready; do
        kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM died. Check: $ROOT_DIR/eval-results/$log_file"
        (( WAIT_SECS >= MAX_WAIT )) && MAX_WAIT=$(( MAX_WAIT + EXTEND )) && log "  Extending to ${MAX_WAIT}s"
        sleep 5; WAIT_SECS=$(( WAIT_SECS + 5 ))
        log "  Waiting... (${WAIT_SECS}s)"
    done
    log "vLLM ready: ${served_name}"
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        log "Stopping vLLM (pid $VLLM_PID) and all children..."
        pkill -9 -P "$VLLM_PID" 2>/dev/null || true
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    pkill -9 -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -u "$USER" -f "vllm serve" 2>/dev/null || true
    local tries=0
    while (( tries < 30 )); do
        local gpu_max
        gpu_max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                  | tr -d ' ' | sort -rn | head -1)
        if [[ -z "$gpu_max" ]] || (( gpu_max <= 200 )); then
            log "GPU memory free (${gpu_max:-n/a} MiB)."
            break
        fi
        log "  GPU still has ${gpu_max} MiB used, waiting..."
        sleep 5
        tries=$(( tries + 1 ))
    done
    sync; sleep 3
}

# ── Main ────────────────────────────────────────────────────────────────────
OVERALL_START=$(date +%s)
NUM_MODELS=${#MODELS[@]}
sep "RAIN Roo eval: ${NUM_MODELS} models × ${#LANGUAGES[@]} languages (port ${VLLM_PORT})"
log "Languages: ${LANGUAGES[*]}"
log "Models:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r served path tp <<< "$entry"
    log "  - ${served} (TP=${tp})  ${path}"
done
log "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

# Pre-flight: every model dir must exist before we touch vLLM.
for entry in "${MODELS[@]}"; do
    IFS='|' read -r _ path _ <<< "$entry"
    [[ -d "$path" ]] || die "Model dir not found: $path  (run baseline/rain.py first)"
    [[ -f "$path/config.json" ]] || die "Missing config.json in $path"
done

MODEL_IDX=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r served_name model_path tp_size <<< "$entry"
    MODEL_IDX=$(( MODEL_IDX + 1 ))
    sep "Model ${MODEL_IDX}/${NUM_MODELS}: ${served_name}"

    local_log="vllm-rain-${served_name}.log"
    start_vllm "$model_path" "$served_name" "$tp_size" "$local_log"

    for lang in "${LANGUAGES[@]}"; do
        log "── Eval: ${served_name} / ${lang} ──"

        # Clean up residual processes between languages.
        pkill -9 -u "$USER" -f "pnpm"          2>/dev/null || true
        pkill -9 -u "$USER" -f "gradle"        2>/dev/null || true
        pkill -9 -u "$USER" -f "cargo test"    2>/dev/null || true
        pkill -9 -u "$USER" -f "go test"       2>/dev/null || true
        pkill -9 -u "$USER" -f "node.*roo-cli" 2>/dev/null || true
        rm -rf /tmp/roo-cli-* /tmp/roo-eval-* 2>/dev/null || true
        sync
        log "  Memory cleanup done."

        # max_attempts defaults to 11 inside roo-eval.py.
        tmp_config="/tmp/eval-rain-${served_name}-${lang}.yaml"
        cat > "$tmp_config" <<YAML
provider: openai
model: "${served_name}"
api_key: "EMPTY"
base_url: "http://localhost:${VLLM_PORT}/v1"
languages: [${lang}]
evals_repo: "${ROOT_DIR}/evals"
concurrency: 64
iterations: 3
context_window: 90000
timeout_seconds: 300
total_timeout_seconds: 900
temperature: 0.6
top_p: 0.8
top_k: 20
YAML

        if python3 "$EVAL_RUNNER" --config "$tmp_config" "${EXTRA_ARGS[@]}" 2>&1; then
            log "  ${served_name} / ${lang} complete."
        else
            log "  WARNING: ${served_name} / ${lang} exited with code $?, continuing..."
        fi
        rm -f "$tmp_config"
    done

    stop_vllm
done

# ── Summary ─────────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - OVERALL_START ))
sep "DONE  (total: ${ELAPSED}s ≈ $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"

echo
printf "  %-30s" "Model"
for lang in "${LANGUAGES[@]}"; do printf "  %-12s" "$lang"; done
echo
printf "  %-30s" "------------------------------"
for lang in "${LANGUAGES[@]}"; do printf "  %-12s" "------------"; done
echo

for entry in "${MODELS[@]}"; do
    IFS='|' read -r served_name _path _tp <<< "$entry"
    printf "  %-30s" "$served_name"
    for lang in "${LANGUAGES[@]}"; do
        FOUND=false
        for d in $(ls -td "$ROOT_DIR"/eval-results/2026*/ 2>/dev/null); do
            [[ -f "$d/results.json" ]] || continue
            result=$(python3 -c "
import json, sys
d = json.load(open('${d}results.json'))
if isinstance(d, dict) and d.get('overall',{}).get('model','') == '$served_name':
    tasks = d.get('tasks', [])
    lang_tasks = [t for t in tasks if t.get('language') == '$lang']
    if lang_tasks:
        exs = {}
        for t in lang_tasks:
            exs.setdefault(t['exercise'], []).append(t['passed'])
        p3 = sum(1 for v in exs.values() if any(v))
        print(f'{p3}/{len(exs)}')
        sys.exit(0)
" 2>/dev/null)
            if [[ -n "$result" ]]; then
                printf "  %-12s" "$result"
                FOUND=true
                break
            fi
        done
        $FOUND || printf "  %-12s" "—"
    done
    echo
done
echo
