#!/usr/bin/env bash
#SBATCH -A ${SBATCH_ACCOUNT:-default}
#SBATCH -p ${SBATCH_PARTITION:-gpu}
#SBATCH -q ${SBATCH_QOS:-default}
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH -J swe-lite-80b
#SBATCH -o ${CRANE_LOG_DIR}/sbatch_log/%x-%j.out
#SBATCH -e ${CRANE_LOG_DIR}/sbatch_log/%x-%j.err
#
# End-to-end SWE-bench Verified eval driver for 80B-class models.
# Parallel sibling of run_swebench_30b.sh — same agent/harness flow, independent
# MODELS registry. Kept as two files instead of one with a flag because the
# 30B and 80B batches run on different days / nodes and we want each script
# to be self-contained and diff-able.
#
# Usage:
#   sbatch  run_swebench_80b.sh                             # full run on all 80b models
#   bash    run_swebench_80b.sh --model qwen3-next-80b-instruct
#   bash    run_swebench_80b.sh --model qwen3-next-80b-instruct|Qwen/Qwen3-Next-80B-A3B-Instruct|true
#   bash    run_swebench_80b.sh --subset verified --split test
#   bash    run_swebench_80b.sh --limit 1                   # smoke
#   bash    run_swebench_80b.sh --skip-agent --skip-eval    # just boot vllm for debugging
#
# Add new 80b models by editing the MODELS array below.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

VLLM_PORT="${VLLM_PORT:-18020}"
VLLM_PID=""
RESULTS_DIR="${ROOT_DIR}/eval_results"
VLLM_LOG_DIR="${RESULTS_DIR}/vllm"
SBATCH_LOG_DIR="${RESULTS_DIR}/sbatch_log"
mkdir -p "$VLLM_LOG_DIR" "$SBATCH_LOG_DIR"

# ── Model registry (80B-class) ──────────────────────────────────────────────
# Format: "shortname|hf_id_or_local_dir|is_hf(true|false)"
MERGED_DIR="${CRANE_MERGED_DIR}"
BASELINE_DIR="${CRANE_REPO_ROOT}/baseline/baseline_model/qwen3_next_80b"
MODELS=(
    "qwen3-next-80b-instruct|Qwen/Qwen3-Next-80B-A3B-Instruct|true"
    "qwen3-next-80b-thinking|Qwen/Qwen3-Next-80B-A3B-Thinking|true"
    "crane-next-80b|${MERGED_DIR}/crane_next_80b|false"
    "qwen3-next-80b-ta|${BASELINE_DIR}/task_arithmetic|false"
    "qwen3-next-80b-ties|${BASELINE_DIR}/ties|false"
    "qwen3-next-80b-slerp|${BASELINE_DIR}/slerp|false"
    "qwen3-next-80b-aim-ta|${BASELINE_DIR}/aim_ta|false"
    "qwen3-next-80b-aim-ties|${BASELINE_DIR}/aim_ties|false"
    "qwen3-next-80b-lewis|${BASELINE_DIR}/lewis|false"
)

# ── Default eval knobs ──────────────────────────────────────────────────────
SUBSET="verified"
# SUBSET="lite"
SPLIT="test"
INSTANCE=""
LIMIT=""
WORKERS="${AGENT_WORKERS:-24}"
HARNESS_WORKERS="${HARNESS_WORKERS:-24}"
AGENT="${AGENT:-openhands}"   # mini | swe-agent | openhands
MINI_CFG="${AGENT_VENV}/lib/python3.11/site-packages/minisweagent/config/benchmarks/swebench.yaml"
SWE_AGENT_CFG="${ROOT_DIR}/vendors/SWE-agent/config/benchmarks/250526_anthropic_filemap_simple_review_sbl.yaml"
# 80B-class default iteration cap. Bumped from the 30b script because larger
# models sometimes need more iters to converge on a valid patch.
OPENHANDS_MAX_ITER="${OPENHANDS_MAX_ITER:-100}"
MODEL_FILTER=""
SKIP_AGENT=false
SKIP_EVAL=false
KEEP_VLLM=false

# ── Parse args ──────────────────────────────────────────────────────────────
while (($#)); do
    case "$1" in
        --subset) SUBSET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --harness-workers) HARNESS_WORKERS="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --agent-config) MINI_CFG="$2"; SWE_AGENT_CFG="$2"; shift 2 ;;
        --model) MODEL_FILTER="$2"; shift 2 ;;
        --skip-agent) SKIP_AGENT=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --keep-vllm) KEEP_VLLM=true; shift ;;
        --port) VLLM_PORT="$2"; shift 2 ;;
        -h|--help) sed -n '16,29p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }
sep()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; echo; }

vllm_ready() { curl -sf "http://localhost:${VLLM_PORT}/v1/models" -o /dev/null 2>/dev/null; }

resolve_hf_path() {
    local hf_id="$1"
    "${VLLM_VENV}/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${hf_id}', cache_dir='${HF_HUB_CACHE%/hub}',
      allow_patterns=['*.safetensors','*.index.json','config*.json','*.json','*.model','tokenizer*','*.txt']))
"
}

vllm_serves_model() {
    local want="$1"
    curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
        | grep -q "\"$want\"" 2>/dev/null
}

start_vllm() {
    local model_path="$1" served_name="$2" log_file="$3"
    if vllm_serves_model "$served_name"; then
        log "vLLM already serving ${served_name} on port ${VLLM_PORT}; reusing."
        VLLM_PID=""
        return 0
    fi
    source "${VLLM_VENV}/bin/activate"
    local tp_size="${VLLM_TP_SIZE:-1}"
    local gpu_count; gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
    if (( gpu_count > 1 )); then
        tp_size=$gpu_count
        log "  Auto-detected $gpu_count GPUs, tensor-parallel-size=$tp_size"
    fi
    local max_model_len="${VLLM_MAX_MODEL_LEN:-131072}"
    local gpu_util="${VLLM_GPU_UTIL:-0.90}"
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
    log "Launching vLLM: $served_name @ port $VLLM_PORT (tp=$tp_size, len=$max_model_len)"
    vllm serve "$model_path" \
        --served-model-name "$served_name" \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --dtype bfloat16 --max-model-len "$max_model_len" \
        --gpu-memory-utilization "$gpu_util" \
        --tensor-parallel-size "$tp_size" \
        --trust-remote-code \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --generation-config vllm \
        >> "$log_file" 2>&1 &
    VLLM_PID=$!
    log "vLLM pid=$VLLM_PID, waiting for /v1/models..."
    local waited=0 budget=300
    until vllm_ready; do
        kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM died early. See $log_file"
        (( waited >= budget )) && budget=$(( budget + 120 )) && log "  extending to ${budget}s"
        sleep 5; waited=$(( waited + 5 ))
        (( waited % 30 == 0 )) && log "  waiting... (${waited}s)"
    done
    log "vLLM ready: $served_name"
    deactivate 2>/dev/null || true
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]]; then
        log "Stopping vLLM (pid $VLLM_PID) and children..."
        pkill -9 -P "$VLLM_PID" 2>/dev/null || true
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    local tries=0
    while (( tries < 20 )); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
               | tr -d ' ' | sort -rn | head -1)
        if [[ -z "$used" ]] || (( used <= 200 )); then
            log "GPU free (${used:-n/a} MiB)."; break
        fi
        log "  GPU still ${used} MiB, waiting..."
        sleep 5; tries=$(( tries + 1 ))
    done
}

run_mini_agent() {
    local model_name="$1" out_dir="$2"
    source "${AGENT_VENV}/bin/activate"
    export MSWEA_DOCKER_EXECUTABLE=podman
    export MSWEA_COST_TRACKING=ignore_errors
    export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENAI_BASE_URL="$OPENAI_API_BASE"
    export OPENAI_API_KEY=dummy
    declare -a agent_args=(
        --subset "$SUBSET" --split "$SPLIT"
        -c "$MINI_CFG"
        -c "model.model_class=litellm"
        -c "model.model_kwargs.temperature=0.7"
        -c "model.model_kwargs.top_p=0.8"
        -c "model.model_kwargs.extra_body.top_k=20"
        -m "openai/${model_name}"
        --workers "$WORKERS"
        -o "$out_dir"
    )
    [[ -n "$INSTANCE" ]] && agent_args+=(--filter "^${INSTANCE}\$")
    [[ -n "$LIMIT" ]] && agent_args+=(--slice "0:${LIMIT}")
    mini-extra swebench "${agent_args[@]}" 2>&1 | tee -a "${out_dir}/run.log"
    deactivate 2>/dev/null || true
}

warm_bases_for_run() {
    log "Warming base images via ghcr mirror for ${SUBSET}/${SPLIT}..."
    local iids
    iids=$("${AGENT_VENV}/bin/python" - <<PY
from datasets import load_dataset
M = {'lite':'princeton-nlp/SWE-Bench_Lite','verified':'princeton-nlp/SWE-Bench_Verified','full':'princeton-nlp/SWE-Bench','multimodal':'princeton-nlp/SWE-Bench_Multimodal'}
ds = load_dataset(M.get('${SUBSET}','${SUBSET}'), split='${SPLIT}')
instance = '${INSTANCE}'
limit = '${LIMIT}'
ids = [r['instance_id'] for r in ds]
if instance:
    ids = [i for i in ids if i == instance]
elif limit:
    ids = ids[:int(limit)]
print('\n'.join(ids))
PY
    )
    [[ -z "$iids" ]] && { log "  (no instances match — skipping warm)"; return 0; }
    local n; n=$(echo "$iids" | wc -l)
    log "  $n instance(s); parallel=4"
    echo "$iids" | pull_bases_via_ghcr --parallel 4 | tail -20 >&2
    log "  warm done."
}

ensure_swerex_builder_ready() {
    if ! podman image exists localhost/swerex-builder:latest 2>/dev/null; then
        log "Prewarming swerex builder image (one-off, ~3-5 min on a free node)..."
        bash "${SCRIPT_DIR}/warm_builder.sh"
    else
        log "swerex builder image cached — skip prewarm."
    fi
    bash "${SCRIPT_DIR}/use_prebuilt_builder.sh" >/dev/null 2>&1 || true
}

run_openhands_agent() {
    local model_name="$1" out_dir="$2"
    bash "${SCRIPT_DIR}/cleanup_podman.sh" >/dev/null 2>&1 || true
    [[ -d "$OPENHANDS_VENV" ]] || die "OpenHands venv missing: $OPENHANDS_VENV (run: uv venv --python=3.12 $OPENHANDS_VENV && uv pip install openhands-sdk openhands-tools datasets pandas)"
    export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENAI_API_KEY=dummy
    declare -a oh_args=(
        --subset "$SUBSET" --split "$SPLIT"
        --model "$model_name"
        --api-base "$OPENAI_API_BASE"
        --api-key "$OPENAI_API_KEY"
        --max-input-tokens "${VLLM_MAX_MODEL_LEN:-131072}"
        --max-iter "$OPENHANDS_MAX_ITER"
        --workers "$WORKERS"
        --output-dir "$out_dir"
    )
    [[ -n "$INSTANCE" ]] && oh_args+=(--instance "$INSTANCE")
    [[ -n "$LIMIT" ]] && oh_args+=(--limit "$LIMIT")
    "${OPENHANDS_VENV}/bin/python" "${SCRIPT_DIR}/openhands_swebench.py" "${oh_args[@]}" \
        2>&1 | tee -a "${out_dir}/run.log"
    bash "${SCRIPT_DIR}/cleanup_podman.sh" >/dev/null 2>&1 || true
}

run_swe_agent() {
    local model_name="$1" out_dir="$2"
    ensure_swerex_builder_ready
    bash "${SCRIPT_DIR}/cleanup_podman.sh" >/dev/null 2>&1 || true
    source "${AGENT_VENV}/bin/activate"
    export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENAI_BASE_URL="$OPENAI_API_BASE"
    export OPENAI_API_KEY=dummy
    declare -a sw_args=(
        --instances.type swe_bench
        --instances.subset "$SUBSET"
        --instances.split "$SPLIT"
        --instances.evaluate=false
        --config "$SWE_AGENT_CFG"
        --agent.type default
        --agent.model.name "openai/${model_name}"
        --agent.model.api_base "$OPENAI_API_BASE"
        --agent.model.api_key dummy
        --agent.model.per_instance_cost_limit 0
        --agent.model.total_cost_limit 0
        --agent.model.max_input_tokens "${VLLM_MAX_MODEL_LEN:-131072}"
        --agent.model.temperature 0.7
        --agent.model.top_p 0.8
        --instances.deployment.type docker
        --instances.deployment.container_runtime podman
        --output_dir "$out_dir"
        --num_workers "$WORKERS"
        --progress_bar False
    )
    [[ -n "$INSTANCE" ]] && sw_args+=(--instances.filter "^${INSTANCE}\$")
    [[ -n "$LIMIT" ]] && sw_args+=(--instances.slice ":${LIMIT}")
    sweagent run-batch "${sw_args[@]}" 2>&1 | tee -a "${out_dir}/run.log"
    deactivate 2>/dev/null || true
    bash "${SCRIPT_DIR}/cleanup_podman.sh" >/dev/null 2>&1 || true
}

start_podman_service() {
    pkill -f "podman system service" 2>/dev/null || true
    sleep 1
    podman system service --time=0 "unix://${PODMAN_SOCK}" \
        > "${RESULTS_DIR}/podman-svc.log" 2>&1 &
    PODMAN_SVC_PID=$!
    sleep 3
    [[ -S "$PODMAN_SOCK" ]] || die "Podman socket not ready: $PODMAN_SOCK"
    log "podman service pid=$PODMAN_SVC_PID, sock=$PODMAN_SOCK"
}

stop_podman_service() {
    [[ -n "${PODMAN_SVC_PID:-}" ]] && kill "$PODMAN_SVC_PID" 2>/dev/null || true
    PODMAN_SVC_PID=""
}

cleanup() {
    if ! $KEEP_VLLM; then
        stop_vllm || true
    fi
    stop_podman_service || true
    [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
    bash "${SCRIPT_DIR}/cleanup_podman.sh" --all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

PODMAN_SOCK="/tmp/${USER}-podman/podman.sock"

# Per-invocation timestamp suffix on run_tag / out_dir so a re-run never
# stomps the previous run's preds.json / traj files / harness reports.
# Override with `RUN_TS=<value>` to merge a resumed run into one dir.
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

# ── Select models ───────────────────────────────────────────────────────────
# --model accepts either a shortname (looked up in MODELS above) or a full
# `shortname|path|is_hf` tuple.
if [[ -n "$MODEL_FILTER" ]]; then
    if [[ "$MODEL_FILTER" == *"|"* ]]; then
        MODELS=("$MODEL_FILTER")
    else
        FILTERED=()
        for entry in "${MODELS[@]}"; do
            IFS='|' read -r name _p _h <<< "$entry"
            [[ "$name" == "$MODEL_FILTER" ]] && FILTERED+=("$entry")
        done
        (( ${#FILTERED[@]} > 0 )) || die "Model '$MODEL_FILTER' not in registry"
        MODELS=("${FILTERED[@]}")
    fi
fi

# ── Go ──────────────────────────────────────────────────────────────────────
OVERALL_START=$(date +%s)
sep "SWE-bench ${SUBSET}/${SPLIT}  |  ${#MODELS[@]} model(s)  |  port ${VLLM_PORT}  |  80B class"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path _ <<< "$entry"
    log "  • $name ($path)"
done

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path is_hf <<< "$entry"
    sep "Model: ${model_name}"

    if [[ "$is_hf" == "true" ]]; then
        log "Resolving HF snapshot: $model_path"
        model_path=$(resolve_hf_path "$model_path")
        log "Snapshot: $model_path"
    fi
    [[ -d "$model_path" ]] || die "Model dir not found: $model_path"

    vllm_log="${VLLM_LOG_DIR}/${model_name}.log"
    start_vllm "$model_path" "$model_name" "$vllm_log"

    run_tag="${model_name}_${SUBSET}_${SPLIT}_${RUN_TS}"
    out_dir="${RESULTS_DIR}/${run_tag}"
    preds_file="${out_dir}/preds.json"
    mkdir -p "$out_dir"

    if ! $SKIP_AGENT; then
        warm_bases_for_run
        bash "${SCRIPT_DIR}/watchdog_stuck_containers.sh" \
            > "${out_dir}/watchdog.log" 2>&1 &
        WATCHDOG_PID=$!
        log "Container watchdog pid=$WATCHDOG_PID (timeout=${CONTAINER_TIMEOUT_MIN:-30}m)"
        agent_start=$(date +%s)
        case "$AGENT" in
            mini)
                log "── [mini] ${model_name} → $out_dir ──"
                run_mini_agent "$model_name" "$out_dir"
                ;;
            swe-agent|sweagent)
                log "── [swe-agent] ${model_name} → $out_dir ──"
                run_swe_agent "$model_name" "$out_dir"
                ;;
            openhands|oh)
                log "── [openhands] ${model_name} → $out_dir ──"
                run_openhands_agent "$model_name" "$out_dir"
                ;;
            *) die "Unknown --agent '$AGENT' (expected: mini | swe-agent | openhands)" ;;
        esac
        echo $(( $(date +%s) - agent_start )) > "${out_dir}/.agent_wall_seconds"
        kill "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
    else
        log "Skipping agent run (--skip-agent)"
    fi

    if ! $KEEP_VLLM; then
        stop_vllm
    else
        log "Keeping vLLM alive (--keep-vllm)."
    fi

    if ! $SKIP_EVAL; then
        [[ -f "$preds_file" ]] || die "Predictions missing: $preds_file"
        n_preds=$(python3 -c "import json;print(len(json.load(open('$preds_file'))))")
        log "── swebench.harness ${model_name}: ${n_preds} preds ──"

        source "${AGENT_VENV}/bin/activate"
        mkdir -p "$(dirname "$PODMAN_SOCK")"
        start_podman_service
        export DOCKER_HOST="unix://${PODMAN_SOCK}"

        run_id="$run_tag"
        declare -a harness_args=(
            --dataset_name "SWE-bench/SWE-bench_$(python3 -c "print({'lite':'Lite','verified':'Verified','full':'Full'}.get('$SUBSET','$SUBSET'.capitalize()))")"
            --predictions_path "$preds_file"
            --run_id "$run_id"
            --max_workers "$HARNESS_WORKERS"
            --cache_level instance
        )
        [[ -n "$INSTANCE" ]] && harness_args+=(--instance_ids "$INSTANCE")

        pushd "$out_dir" >/dev/null
        python -m swebench.harness.run_evaluation "${harness_args[@]}" \
            2>&1 | tee harness.log
        popd >/dev/null

        stop_podman_service
        deactivate 2>/dev/null || true
    else
        log "Skipping harness eval (--skip-eval)"
    fi

    # Per-model results.{md,json} (pass rate + token usage + vLLM cache stats).
    log "Aggregating results for ${model_name}..."
    "${AGENT_VENV}/bin/python" "${SCRIPT_DIR}/aggregate_model_results.py" \
        "$out_dir" --model "$model_name" --vllm-port "$VLLM_PORT" \
        2>&1 | tee -a "${out_dir}/run.log" || true

    # Between-models cleanup (matches 30b script): drop any lingering oh-*
    # / sweb.eval.* / build processes so the next model gets a clean graph.
    log "Inter-model cleanup..."
    bash "${SCRIPT_DIR}/cleanup_podman.sh" --all >/dev/null 2>&1 || true
done

ELAPSED=$(( $(date +%s) - OVERALL_START ))
sep "DONE (${ELAPSED}s ≈ $(( ELAPSED / 60 ))m)"

# ── Summary table ──────────────────────────────────────────────────────────
printf "\n  %-32s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s  %s\n" \
    "Model" "Agent" "Resolved" "Total" "Tokens-In" "Tokens-Out" "Cost(USD)" "Wall(s)"
printf "  %s\n" "$(printf '%.0s-' {1..140})"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name _ _ <<< "$entry"
    run_tag="${model_name}_${SUBSET}_${SPLIT}_${RUN_TS}"
    out_dir="${RESULTS_DIR}/${run_tag}"
    report=$(ls "${out_dir}/"*".${run_tag}.json" 2>/dev/null | head -1)
    wall=$(cat "${out_dir}/.agent_wall_seconds" 2>/dev/null || echo "—")
    read -r resolved total toks_in toks_out cost < <(AGENT="$AGENT" OUT_DIR="$out_dir" REPORT="$report" python3 - <<'PY'
import json, os, glob
out_dir = os.environ["OUT_DIR"]
report  = os.environ["REPORT"]
agent   = os.environ["AGENT"]

resolved, total = "—", "—"
if report and os.path.exists(report):
    d = json.load(open(report))
    resolved = d.get("resolved_instances", 0)
    total    = d.get("total_instances", 0)

toks_in = toks_out = 0; cost = 0.0; seen = False
if agent == "mini":
    for fp in glob.glob(f"{out_dir}/*.traj.json"):
        t = json.load(open(fp))
        ms = t.get("info", {}).get("model_stats", {}) or {}
        cost += ms.get("instance_cost", 0.0) or 0.0
        for m in t.get("messages", []):
            u = m.get("usage") or {}
            toks_in  += u.get("prompt_tokens", 0)     or 0
            toks_out += u.get("completion_tokens", 0) or 0
        seen = True
elif agent in ("swe-agent", "sweagent"):
    for fp in glob.glob(f"{out_dir}/**/*.traj", recursive=True):
        try:
            t = json.load(open(fp))
        except Exception:
            continue
        info = t.get("info", {})
        mstats = info.get("model_stats", {}) or {}
        cost += mstats.get("instance_cost", 0.0) or 0.0
        toks_in  += mstats.get("tokens_sent", 0)     or 0
        toks_out += mstats.get("tokens_received", 0) or 0
        seen = True

def fmt_n(n):
    return f"{n}" if not seen else f"{n}"
print(resolved, total, toks_in, toks_out, f"{cost:.4f}")
PY
    )
    printf "  %-32s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s  %s\n" \
        "$model_name" "$AGENT" "$resolved" "$total" "$toks_in" "$toks_out" "$cost" "$wall"
done
echo
