#!/usr/bin/env bash
# Calibration-set robustness sweep for 30B Taylor S_reason.
#
# Runs `crane_s_reason_auto.py` over a queue of public calibration sets,
# parallel-pairing across cuda:0 and cuda:1. Each job is single-GPU; 30B fits
# in one H100 (~60GB bf16 + grads).
#
# Usage:
#   bash run_robustness_30b.sh [SET ...]
#
# With no arguments, runs the full default queue (4 sources × 3 sizes
# + 5 mix-seed + 5 bootstrap = 22 jobs). Otherwise runs only the
# explicit calibration_set names provided.
set -u

FINAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${CRANE_LOG_DIR}/robustness_30b"
STATS_DIR="${CRANE_DATA_DIR}/robustness"
CLAIM_DIR="$STATS_DIR/.claims"
mkdir -p "$LOG_DIR" "$STATS_DIR" "$CLAIM_DIR"

# GPU_LIST defaults to "0 1" but can be overridden via env, e.g.
# `GPU_LIST="2 3" bash run_robustness_30b.sh` to run a second dispatcher on
# GPUs 2/3 against the same queue. Multiple dispatchers can be safely run in
# parallel: each job is claimed via an atomic `mkdir` under $CLAIM_DIR.
GPU_LIST_ENV="${GPU_LIST:-0 1}"
DISPATCHER_TAG="${DISPATCHER_TAG:-d$$}"

source ${CRANE_REPO_ROOT}/roo_test/sh/env.sh

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/dispatcher.log"; }

DEFAULT_SETS=(
    # 5 mix-seeds (already-built, code-domain reasoning)
    public_mix_seed0 public_mix_seed1 public_mix_seed2 public_mix_seed3 public_mix_seed4
    # 4 sources × 3 sizes (cross-domain robustness)
    public_gsm8k_seed42_n36 public_gsm8k_seed42_n64 public_gsm8k_seed42_n100
    public_math_seed42_n36 public_math_seed42_n64 public_math_seed42_n100
    public_bbh_seed42_n36 public_bbh_seed42_n64 public_bbh_seed42_n100
    public_openorca_seed42_n36 public_openorca_seed42_n64 public_openorca_seed42_n100
    # 5 D_R/D_T bootstrap (within-set noise)
    public_bootstrap_seed0 public_bootstrap_seed1
    public_bootstrap_seed2 public_bootstrap_seed3
    public_bootstrap_seed4
)

if (( $# > 0 )); then
    SETS=("$@")
else
    SETS=("${DEFAULT_SETS[@]}")
fi

run_one() {
    local cal_set="$1"
    local gpu="$2"
    local out="$STATS_DIR/phase2_stats_30b_${cal_set}.json"
    local log="$LOG_DIR/${cal_set}.log"
    local claim="$CLAIM_DIR/${cal_set}.lock"

    if [[ -f "$out" ]]; then
        log "[skip ${DISPATCHER_TAG}] $cal_set — output already exists at $out"
        return 0
    fi
    # Atomic claim: only one dispatcher across the cluster will succeed.
    if ! mkdir "$claim" 2>/dev/null; then
        log "[skip ${DISPATCHER_TAG}] $cal_set — already claimed by another dispatcher"
        return 0
    fi
    echo "$DISPATCHER_TAG cuda:$gpu pid=$$" > "$claim/owner"
    log "[start ${DISPATCHER_TAG} cuda:$gpu] $cal_set → $out"
    CUDA_VISIBLE_DEVICES="$gpu" python "$FINAL_DIR/crane_s_reason_auto.py" \
        --model-preset qwen3-30b \
        --output "$out" \
        --device cuda:0 \
        --calibration-set "$cal_set" \
        --max-new-r 4096 --max-new-t 2048 \
        --decode-batch 4 \
        > "$log" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        log "[done  ${DISPATCHER_TAG} cuda:$gpu] $cal_set (rc=0)"
    else
        log "[FAIL  ${DISPATCHER_TAG} cuda:$gpu] $cal_set (rc=$rc) — see $log"
        # Release the claim on failure so another dispatcher can retry.
        rm -rf "$claim"
    fi
    return $rc
}

# ── Pair-parallel scheduler ────────────────────────────────────────────────
# One slot per GPU listed in GPU_LIST_ENV. Track PIDs in associative arrays.
declare -A SLOT_PID    # gpu -> pid
declare -A SLOT_SET    # gpu -> calibration_set
read -r -a GPU_LIST <<< "$GPU_LIST_ENV"

dispatch_next() {
    local gpu="$1"
    if [[ -z "${SLOT_PID[$gpu]:-}" ]]; then
        if (( ${#QUEUE[@]} > 0 )); then
            local nxt="${QUEUE[0]}"
            QUEUE=("${QUEUE[@]:1}")
            run_one "$nxt" "$gpu" &
            SLOT_PID[$gpu]=$!
            SLOT_SET[$gpu]="$nxt"
        fi
    fi
}

QUEUE=("${SETS[@]}")
log "=== Robustness sweep starting: ${#QUEUE[@]} jobs across ${#GPU_LIST[@]} GPUs ==="
log "queue: ${QUEUE[*]}"

for gpu in "${GPU_LIST[@]}"; do dispatch_next "$gpu"; done

while true; do
    any_running=0
    for gpu in "${GPU_LIST[@]}"; do
        pid="${SLOT_PID[$gpu]:-}"
        if [[ -n "$pid" ]]; then
            if kill -0 "$pid" 2>/dev/null; then
                any_running=1
            else
                wait "$pid"
                rc=$?
                cal="${SLOT_SET[$gpu]}"
                log "[reaped cuda:$gpu] $cal rc=$rc"
                unset SLOT_PID[$gpu]
                unset SLOT_SET[$gpu]
                dispatch_next "$gpu"
                if [[ -n "${SLOT_PID[$gpu]:-}" ]]; then
                    any_running=1
                fi
            fi
        fi
    done
    if (( any_running == 0 )); then
        break
    fi
    sleep 30
done

log "=== Robustness sweep complete ==="
ls -lh "$STATS_DIR"/phase2_stats_30b_*.json 2>/dev/null | tee -a "$LOG_DIR/dispatcher.log"
