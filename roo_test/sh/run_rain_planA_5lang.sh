#!/usr/bin/env bash
#
# Roo eval for the two RAIN-Plan-A checkpoints across 4 remaining languages
# (javascript, go, java, rust). Python is run separately by run_rain.sh.
#
# Sampling/serving config: TP=4, max-model-len 90000, concurrency 64,
# iterations 3, temperature 0.6, top_p 0.8, top_k 20, per-attempt timeout
# 300s, total timeout 900s.
#
# Usage:
#   bash sh/run_rain_planA_5lang.sh                  # both models, all 4 langs
#   bash sh/run_rain_planA_5lang.sh --only 30b       # 30B only
#   bash sh/run_rain_planA_5lang.sh --only 80b       # 80B only
#   bash sh/run_rain_planA_5lang.sh --languages "javascript go"
#
set -euo pipefail

ROOT_DIR="${CRANE_REPO_ROOT}/roo_test"
source "${ROOT_DIR}/sh/env.sh"

EVAL_RUNNER="${ROOT_DIR}/roo-eval.py"
VLLM_PORT=18019
VLLM_PID=""
LANGUAGES=(javascript go java rust)

BASELINE_DIR="${CRANE_REPO_ROOT}/baseline/baseline_model"
MODELS=(
    "baseline-rain-30b-pycalib-qkvof-planA-a030|${BASELINE_DIR}/rain_pycalib_qkvof_planA_a030/unified_model_merge/rain_merged_pycalib_qkvof_planA_a030|4"
    "baseline-rain-80b-next-pycalib-qkvof-planA-a030|${BASELINE_DIR}/rain_qwen3_next_80b_pycalib_qkvof_planA_a030/unified_model_merge/rain_merged_pycalib_qkvof_planA_a030|4"
)

ONLY=""
EXTRA_ARGS=()
while (( $# )); do
    case "$1" in
        --only)      shift; ONLY="$1" ;;
        --languages) shift; IFS=' ' read -r -a LANGUAGES <<< "$1" ;;
        *)           EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

case "$ONLY" in
    "")     ;;
    30b)    MODELS=("${MODELS[0]}") ;;
    80b)    MODELS=("${MODELS[1]}") ;;
    *)      echo "ERROR: --only must be 30b or 80b" >&2; exit 2 ;;
esac

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
sep()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; echo; }

cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -9 -u "$USER" -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -u "$USER" -f "vllm serve"       2>/dev/null || true
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
    local WAIT_SECS=0 MAX_WAIT=900 EXTEND=120
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
    pkill -9 -u "$USER" -f "vllm serve"       2>/dev/null || true
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

OVERALL_START=$(date +%s)
NUM_MODELS=${#MODELS[@]}
sep "RAIN Plan A roo eval: ${NUM_MODELS} models × ${#LANGUAGES[@]} languages (port ${VLLM_PORT})"
log "Languages: ${LANGUAGES[*]}"
log "Models:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path tp <<< "$entry"
    log "  - ${name} (TP=${tp})  ${path}"
done
echo

# Pre-flight check
for entry in "${MODELS[@]}"; do
    IFS='|' read -r _ path _ <<< "$entry"
    [[ -d "$path" ]] || die "Model dir not found: $path"
    [[ -f "$path/config.json" ]] || die "Missing config.json in $path"
done

MODEL_IDX=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r served_name model_path tp_size <<< "$entry"
    MODEL_IDX=$(( MODEL_IDX + 1 ))
    sep "Model ${MODEL_IDX}/${NUM_MODELS}: ${served_name}"

    local_log="vllm-rain-planA-${served_name}.log"
    start_vllm "$model_path" "$served_name" "$tp_size" "$local_log"

    for lang in "${LANGUAGES[@]}"; do
        log "── Eval: ${served_name} / ${lang} ──"

        pkill -9 -u "$USER" -f "pnpm"          2>/dev/null || true
        pkill -9 -u "$USER" -f "gradle"        2>/dev/null || true
        pkill -9 -u "$USER" -f "cargo test"    2>/dev/null || true
        pkill -9 -u "$USER" -f "go test"       2>/dev/null || true
        pkill -9 -u "$USER" -f "node.*roo-cli" 2>/dev/null || true
        rm -rf /tmp/roo-cli-* /tmp/roo-eval-* 2>/dev/null || true
        sync
        log "  Memory cleanup done."

        tmp_config="/tmp/eval-rain-planA-${served_name}-${lang}.yaml"
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

ELAPSED=$(( $(date +%s) - OVERALL_START ))
sep "DONE  (total: ${ELAPSED}s ≈ $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"

echo
printf "  %-50s" "Model"
for lang in "${LANGUAGES[@]}"; do printf "  %-12s" "$lang"; done
echo
printf "  %-50s" "$(printf '%.0s-' {1..50})"
for lang in "${LANGUAGES[@]}"; do printf "  %-12s" "------------"; done
echo

for entry in "${MODELS[@]}"; do
    IFS='|' read -r served_name _path _tp <<< "$entry"
    printf "  %-50s" "$served_name"
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
