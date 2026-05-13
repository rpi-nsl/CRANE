#!/bin/bash
# Two-phase prebuild for SWE-Bench Verified:
#   phase 1: pull base images from Epoch AI's ghcr mirror (no Docker Hub rate
#            limit) and retag to the canonical docker.io/swebench name that
#            prebuild_wrappers.sh + swebench.harness expect.
#   phase 2: build wrappers at parallel=10 (CPU-bound, local build, no pulls).
#
# Epoch AI's mirror: ghcr.io/epoch-research/swe-bench.eval.x86_64.<iid>:latest
# (raw instance_id; no `__` → `_1776_` mangling — that's Docker Hub-only).
# Reference: https://epoch.ai/blog/swebench-docker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
export PODMAN_IGNORE_CGROUPSV1_WARNING=1

LOG_DIR="${SCRIPT_DIR}/../eval_results/prebuild_log"
TSV="${LOG_DIR}/verified_images.tsv"
PULL_LOG="${LOG_DIR}/verified_pull.log"
BUILD_LOG="${LOG_DIR}/verified_build.log"
PULL_PARALLEL=${PULL_PARALLEL:-4}
BUILD_PARALLEL=${BUILD_PARALLEL:-10}

mkdir -p "${LOG_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -s "${TSV}" ]]; then
    log "Generating verified image list..."
    "${AGENT_VENV}/bin/python" - > "${TSV}" <<'PY'
from datasets import load_dataset
ds = load_dataset('princeton-nlp/SWE-Bench_Verified', split='test')
for r in ds:
    iid = r['instance_id']
    img = f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest".lower()
    print(f"{iid}\t{img}")
PY
fi

TOTAL=$(wc -l < "${TSV}")
log "=== Phase 1: parallel=${PULL_PARALLEL} pull from ghcr mirror for ${TOTAL} images ==="

pull_one() {
    local iid="$1" canon="$2"
    local ghcr="ghcr.io/epoch-research/swe-bench.eval.x86_64.${iid}:latest"
    if podman image exists "$canon" 2>/dev/null; then
        echo "[skip] $iid"
        return 0
    fi
    local start=$(date +%s)
    if podman pull -q "$ghcr" >/dev/null 2>&1; then
        podman tag "$ghcr" "$canon" 2>/dev/null
        local dur=$(( $(date +%s) - start ))
        echo "[ok]   $iid  (${dur}s)"
    else
        echo "[FAIL] $iid  ghcr=$ghcr"
    fi
}
export -f pull_one

: > "${PULL_LOG}"
awk -v OFS=' ' '{print $1, $2}' "${TSV}" \
    | xargs -n 2 -P "${PULL_PARALLEL}" bash -c 'pull_one "$0" "$1"' \
    | tee -a "${PULL_LOG}"

OK=$(grep -c '^\[ok\]'   "${PULL_LOG}" || true)
SKIP=$(grep -c '^\[skip\]' "${PULL_LOG}" || true)
FAIL=$(grep -c '^\[FAIL\]' "${PULL_LOG}" || true)
log "Phase 1 done: ok=${OK} skip=${SKIP} fail=${FAIL}"

if (( FAIL > 0 )); then
    log "WARNING: ${FAIL} pulls failed — inspect ${PULL_LOG} before running phase 2"
    exit 1
fi

log "=== Phase 2: parallel=${BUILD_PARALLEL} wrapper build ==="
bash "${SCRIPT_DIR}/prebuild_wrappers.sh" --subset verified --parallel "${BUILD_PARALLEL}" 2>&1 | tee "${BUILD_LOG}"

log "=== All phases complete ==="
