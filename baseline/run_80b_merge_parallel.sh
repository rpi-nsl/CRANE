#!/usr/bin/env bash
# Run the 6 baseline merges for Qwen3-Next-80B-A3B on a single 4× H100 node.
#
# Strategy — 4 phases:
#   Phase 1 (parallel, 3 GPUs):  TA, TIES, SLERP  (streaming tensor merges)
#   Phase 2 (serial, 4 GPUs ea): AIM importance (instruct), Lewis importance
#                                (instruct), Lewis importance (thinking)
#   Phase 3 (parallel, 3 GPUs):  AIM-TA, AIM-TIES, Lewis merge
#   (Output: 6 merged models under baseline_model/qwen3_next_80b/)
#
# Total est: 4–7 h merge + 12–16 h eval (eval is handled by sh/run_all_languages_3_b.sh).
#
# Usage:
#   bash run_80b_merge_parallel.sh               # full run
#   bash run_80b_merge_parallel.sh --phase 1     # only phase 1 (TA/TIES/SLERP)
#   bash run_80b_merge_parallel.sh --phase 2     # only importance extraction
#   bash run_80b_merge_parallel.sh --phase 3     # only AIM-TA / AIM-TIES / Lewis

set -euo pipefail

# env.sh resets SCRIPT_DIR to its own location, so we save our real dir first.
MERGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MM_ROOT="$(cd "$MERGE_DIR/.." && pwd)"
source "$MM_ROOT/roo_test/sh/env.sh"
SCRIPT_DIR="$MERGE_DIR"
ROOT_DIR="$MM_ROOT"

ALPHA=0.15
PRESET="qwen3-next-80b"
OUT_DIR="$SCRIPT_DIR/baseline_model/qwen3_next_80b"
IMP_DIR="$OUT_DIR/importance"
mkdir -p "$OUT_DIR" "$IMP_DIR"

# ── Parse args ──────────────────────────────────────────────────────────────
PHASES="1 2 3"
for (( i=1; i <= $#; i++ )); do
    case "${!i}" in
        --phase) i=$((i+1)); PHASES="${!i}" ;;
        *) echo "Unknown arg: ${!i}" >&2; exit 2 ;;
    esac
done
echo "Phases to run: $PHASES"

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
sep()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; echo; }
overall_start=$(date +%s)

wait_pid() {
    local pid=$1 name=$2
    if wait "$pid"; then
        log "  $name (pid $pid) done."
        return 0
    else
        local rc=$?
        log "  $name (pid $pid) FAILED (exit $rc)."
        return $rc
    fi
}

FAIL=0

# ── Phase 1: TA / TIES / SLERP (parallel, 3 GPUs) ───────────────────────────
if [[ " $PHASES " == *" 1 "* ]]; then
    sep "Phase 1: TA / TIES / SLERP (parallel on GPUs 0/1/2)"
    phase_start=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/task_arithmetic.py" \
        --preset "$PRESET" --alpha "$ALPHA" --device cuda:0 \
        > "$OUT_DIR/task_arithmetic.log" 2>&1 &
    PID_TA=$!
    log "GPU 0: Task Arithmetic  (pid $PID_TA)  log=$OUT_DIR/task_arithmetic.log"

    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/ties.py" \
        --preset "$PRESET" --alpha "$ALPHA" --device cuda:0 \
        > "$OUT_DIR/ties.log" 2>&1 &
    PID_TIES=$!
    log "GPU 1: TIES            (pid $PID_TIES)  log=$OUT_DIR/ties.log"

    CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/slerp.py" \
        --preset "$PRESET" --t "$ALPHA" --device cuda:0 \
        > "$OUT_DIR/slerp.log" 2>&1 &
    PID_SLERP=$!
    log "GPU 2: SLERP           (pid $PID_SLERP)  log=$OUT_DIR/slerp.log"

    wait_pid $PID_TA    "TaskArithmetic" || FAIL=$((FAIL+1))
    wait_pid $PID_TIES  "TIES"           || FAIL=$((FAIL+1))
    wait_pid $PID_SLERP "SLERP"          || FAIL=$((FAIL+1))

    log "Phase 1 complete in $(( $(date +%s) - phase_start ))s. Cumulative failures: $FAIL"
fi

# ── Phase 2: importance extraction (serial, 4 GPUs each) ────────────────────
# Each job loads the full 80B model with device_map="auto" (~40 GB/GPU) and
# does forward passes over the calibration set.  Running multiple in parallel
# would OOM — so we serialize.
if [[ " $PHASES " == *" 2 "* ]]; then
    sep "Phase 2: importance extraction (serial, 4 GPUs each)"
    phase_start=$(date +%s)
    unset CUDA_VISIBLE_DEVICES

    log "AIM importance (instruct, 4 GPUs)..."
    python3 "$SCRIPT_DIR/aim_importance.py" \
        --preset "$PRESET" \
        --output "$IMP_DIR/aim_instruct.pt" \
        > "$OUT_DIR/aim_importance.log" 2>&1
    if [[ $? -eq 0 ]]; then
        log "  AIM importance done."
    else
        log "  AIM importance FAILED."; FAIL=$((FAIL+1))
    fi

    log "Lewis importance (instruct, 4 GPUs)..."
    python3 "$SCRIPT_DIR/lewis_importance.py" \
        --preset "${PRESET}-instruct" \
        --output "$IMP_DIR/lewis_instruct.pt" \
        > "$OUT_DIR/lewis_importance_instruct.log" 2>&1
    if [[ $? -eq 0 ]]; then
        log "  Lewis importance (instruct) done."
    else
        log "  Lewis importance (instruct) FAILED."; FAIL=$((FAIL+1))
    fi

    log "Lewis importance (thinking, 4 GPUs)..."
    python3 "$SCRIPT_DIR/lewis_importance.py" \
        --preset "${PRESET}-thinking" \
        --output "$IMP_DIR/lewis_thinking.pt" \
        > "$OUT_DIR/lewis_importance_thinking.log" 2>&1
    if [[ $? -eq 0 ]]; then
        log "  Lewis importance (thinking) done."
    else
        log "  Lewis importance (thinking) FAILED."; FAIL=$((FAIL+1))
    fi

    log "Phase 2 complete in $(( $(date +%s) - phase_start ))s. Cumulative failures: $FAIL"
fi

# ── Phase 3: AIM-TA / AIM-TIES / Lewis merge (parallel, 3 GPUs) ─────────────
if [[ " $PHASES " == *" 3 "* ]]; then
    sep "Phase 3: AIM-TA / AIM-TIES / Lewis merge (parallel on GPUs 0/1/2)"
    phase_start=$(date +%s)

    # AIM-TA: TA with --aim-importance, output to aim_ta/
    if [[ -f "$IMP_DIR/aim_instruct.pt" ]]; then
        CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/task_arithmetic.py" \
            --preset "$PRESET" --alpha "$ALPHA" --device cuda:0 \
            --aim-importance "$IMP_DIR/aim_instruct.pt" \
            --output-dir "$OUT_DIR/aim_ta" \
            > "$OUT_DIR/aim_ta.log" 2>&1 &
        PID_AIM_TA=$!
        log "GPU 0: AIM-TA    (pid $PID_AIM_TA)  log=$OUT_DIR/aim_ta.log"
    else
        log "SKIP AIM-TA: $IMP_DIR/aim_instruct.pt missing"; FAIL=$((FAIL+1))
        PID_AIM_TA=""
    fi

    # AIM-TIES: TIES with --aim-importance, output to aim_ties/
    if [[ -f "$IMP_DIR/aim_instruct.pt" ]]; then
        CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/ties.py" \
            --preset "$PRESET" --alpha "$ALPHA" --device cuda:0 \
            --aim-importance "$IMP_DIR/aim_instruct.pt" \
            --output-dir "$OUT_DIR/aim_ties" \
            > "$OUT_DIR/aim_ties.log" 2>&1 &
        PID_AIM_TIES=$!
        log "GPU 1: AIM-TIES  (pid $PID_AIM_TIES)  log=$OUT_DIR/aim_ties.log"
    else
        log "SKIP AIM-TIES: $IMP_DIR/aim_instruct.pt missing"; FAIL=$((FAIL+1))
        PID_AIM_TIES=""
    fi

    # Lewis merge
    if [[ -f "$IMP_DIR/lewis_instruct.pt" && -f "$IMP_DIR/lewis_thinking.pt" ]]; then
        CUDA_VISIBLE_DEVICES=2 python3 "$SCRIPT_DIR/lewis.py" \
            --preset "$PRESET" --alpha "$ALPHA" --device cuda:0 \
            --importance-instruct "$IMP_DIR/lewis_instruct.pt" \
            --importance-thinking "$IMP_DIR/lewis_thinking.pt" \
            > "$OUT_DIR/lewis.log" 2>&1 &
        PID_LEWIS=$!
        log "GPU 2: Lewis     (pid $PID_LEWIS)  log=$OUT_DIR/lewis.log"
    else
        log "SKIP Lewis merge: importance files missing"; FAIL=$((FAIL+1))
        PID_LEWIS=""
    fi

    [[ -n "$PID_AIM_TA" ]]   && { wait_pid $PID_AIM_TA   "AIM-TA"   || FAIL=$((FAIL+1)); }
    [[ -n "$PID_AIM_TIES" ]] && { wait_pid $PID_AIM_TIES "AIM-TIES" || FAIL=$((FAIL+1)); }
    [[ -n "$PID_LEWIS" ]]    && { wait_pid $PID_LEWIS    "Lewis"    || FAIL=$((FAIL+1)); }

    log "Phase 3 complete in $(( $(date +%s) - phase_start ))s. Cumulative failures: $FAIL"
fi

sep "All done. Total: $(( $(date +%s) - overall_start ))s. Failures: $FAIL"
log "Outputs:"
ls -la "$OUT_DIR"/*/merge_config.json 2>/dev/null || true
exit $FAIL
