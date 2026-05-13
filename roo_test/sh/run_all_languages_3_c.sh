#!/usr/bin/env bash
#SBATCH -A ${SBATCH_ACCOUNT:-default}
#SBATCH -p ${SBATCH_PARTITION:-default}
#SBATCH -q ${SBATCH_QOS:-default}
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=48
#SBATCH -J roo-3c
#SBATCH -o ${CRANE_LOG_DIR}/sbatch_log/%x-%j.out
#SBATCH -e ${CRANE_LOG_DIR}/sbatch_log/%x-%j.err

#
# Roo Test: all languages — Part 3c (Qwen3-Next 80B-A3B MoE, CRANE merges)
#
# Models: derived from Qwen3-Next-80B-A3B Instruct × Thinking via
# crane_merge.py (norm-unlock patch + 80B GSP).
#
# Architecture: 80B MoE (3B activated), 512 experts / top-10,
# hybrid Gated DeltaNet + full attention. Requires vLLM >= 0.10.2.
#
# 4× H100 80GB strategy: TP=4 + expert-parallel.
#
# Usage:
#   sbatch sh/run_all_languages_3_c.sh                   # full run
#   bash sh/run_all_languages_3_c.sh --limit 3           # smoke-test
#   bash sh/run_all_languages_3_c.sh --language python   # single language

set -euo pipefail

ROOT_DIR="${CRANE_REPO_ROOT}/roo_test"
source "${ROOT_DIR}/sh/env.sh"

EVAL_RUNNER="${ROOT_DIR}/roo-eval.py"
VLLM_PORT=18015
VLLM_PID=""
LANGUAGES=(python javascript go java rust)
# ── Model registry ──────────────────────────────────────────────────────────
MERGED_DIR="${CRANE_MERGED_DIR}"
MODELS=(
    "crane-next-80b-base|${MERGED_DIR}/crane_next_80b_base|false"
    "crane-next-80b|${MERGED_DIR}/crane_next_80b|false"
)

# ── Parse args ──────────────────────────────────────────────────────────────
EXTRA_ARGS=()
SINGLE_LANG=""
for arg in "$@"; do
    case "$arg" in
        --language) SINGLE_LANG="__next__" ;;
        *)
            if [[ "$SINGLE_LANG" == "__next__" ]]; then
                SINGLE_LANG="$arg"
            else
                EXTRA_ARGS+=("$arg")
            fi
            ;;
    esac
done

if [[ -n "$SINGLE_LANG" && "$SINGLE_LANG" != "__next__" ]]; then
    LANGUAGES=("$SINGLE_LANG")
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
    mkdir -p "$ROOT_DIR/eval-results"
    # Detect available GPUs for tensor-parallel
    local tp_size="${VLLM_TP_SIZE:-1}"
    local gpu_count
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( gpu_count > 1 )); then
        tp_size=$gpu_count
        log "  Auto-detected $gpu_count GPUs, using tensor-parallel-size=$tp_size"
    fi
    local max_model_len="${VLLM_MAX_MODEL_LEN:-90000}"
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
        >> "$ROOT_DIR/eval-results/$log_file" 2>&1 &
    VLLM_PID=$!
    log "vLLM started (pid $VLLM_PID), waiting..."
    local WAIT_SECS=0 MAX_WAIT=300 EXTEND=120
    until vllm_ready; do
        kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM died. Check: $ROOT_DIR/eval-results/$log_file"
        (( WAIT_SECS >= MAX_WAIT )) && MAX_WAIT=$(( MAX_WAIT + EXTEND )) && log "  Extending to ${MAX_WAIT}s"
        sleep 5; WAIT_SECS=$(( WAIT_SECS + 5 ))
        log "  Waiting... (${WAIT_SECS}s)"
    done
    log "vLLM ready: $model_name"
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        log "Stopping vLLM (pid $VLLM_PID) and all children..."
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
        log "  GPU still has ${gpu_max} MiB used, waiting..."
        sleep 5
        tries=$(( tries + 1 ))
    done
    sync
    sleep 3
}

# ── Main ────────────────────────────────────────────────────────────────────

OVERALL_START=$(date +%s)
NUM_MODELS=${#MODELS[@]}
sep "All-Languages Eval Part 3c: ${NUM_MODELS} models × ${#LANGUAGES[@]} languages (port ${VLLM_PORT})"
log "Languages: ${LANGUAGES[*]}"
log "Models:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path _is_hf <<< "$entry"
    log "  - $name ($path)"
done
log "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

MODEL_IDX=0
declare -A MODEL_RESULT_DIRS

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path is_hf <<< "$entry"
    MODEL_IDX=$(( MODEL_IDX + 1 ))
    sep "Model ${MODEL_IDX}/${NUM_MODELS}: ${model_name}"

    if [[ "$is_hf" == "true" ]]; then
        log "Resolving HuggingFace model: $model_path"
        model_path=$(resolve_hf_path "$model_path")
        log "Resolved to: $model_path"
    fi
    [[ -d "$model_path" ]] || die "Model dir not found: $model_path"

    local_log="vllm-allang3c-${model_name}.log"
    start_vllm "$model_path" "$model_name" "$local_log"

    for lang in "${LANGUAGES[@]}"; do
        log "── Eval: ${model_name} / ${lang} ──"

        # Clean up residual processes and page cache between languages
        pkill -9 -f "pnpm" 2>/dev/null || true
        pkill -9 -f "gradle" 2>/dev/null || true
        pkill -9 -f "cargo test" 2>/dev/null || true
        pkill -9 -f "go test" 2>/dev/null || true
        pkill -9 -f "node.*roo-cli" 2>/dev/null || true
        rm -rf /tmp/roo-cli-* /tmp/roo-eval-* 2>/dev/null || true
        sync
        log "  Memory cleanup done."

        # Warm NFS metadata cache for the per-language toolchain to avoid
        # ENOENT under 64-way concurrency.
        case "$lang" in
            go)
                ls ${GOROOT}/bin/ >/dev/null 2>&1 || true
                ls ${GOROOT}/pkg/tool/linux_amd64/ >/dev/null 2>&1 || true
                ${GOROOT}/bin/go version >/dev/null 2>&1 || true
                ;;
            rust)
                ls ${CARGO_HOME}/bin/ >/dev/null 2>&1 || true
                ls ${RUSTUP_HOME}/toolchains/ >/dev/null 2>&1 || true
                ${CARGO_HOME}/bin/cargo --version >/dev/null 2>&1 || true
                ;;
            javascript)
                ls ${NODE_HOME}/bin/ >/dev/null 2>&1 || true
                ${NODE_HOME}/bin/node --version >/dev/null 2>&1 || true
                ;;
            java)
                ls /usr/lib/jvm/java-11-openjdk-11.0.25.0.9-2.el9.x86_64/bin/ >/dev/null 2>&1 || true
                ;;
            python)
                ls ${CRANE_REPO_ROOT}/.venv/bin/ >/dev/null 2>&1 || true
                ;;
        esac
        log "  NFS toolchain prewarm (${lang}) done."

        tmp_config="/tmp/eval-allang3c-${model_name}-${lang}.yaml"
        cat > "$tmp_config" <<YAML
provider: openai
model: "${model_name}"
api_key: "EMPTY"
base_url: "http://localhost:${VLLM_PORT}/v1"
languages: [${lang}]
evals_repo: "${CRANE_REPO_ROOT}/roo_test/evals"
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
            log "  ${model_name} / ${lang} complete."
        else
            log "  WARNING: ${model_name} / ${lang} exited with code $?, continuing..."
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
    IFS='|' read -r model_name _path _hf <<< "$entry"
    printf "  %-30s" "$model_name"
    for lang in "${LANGUAGES[@]}"; do
        FOUND=false
        for d in $(ls -td "$ROOT_DIR"/eval-results/2026*/ 2>/dev/null); do
            [ -f "$d/results.json" ] || continue
            result=$(python3 -c "
import json, sys
d = json.load(open('${d}results.json'))
if isinstance(d, dict) and d.get('overall',{}).get('model','') == '$model_name':
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
