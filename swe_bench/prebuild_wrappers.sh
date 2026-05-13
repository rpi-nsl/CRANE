#!/bin/bash
# Pre-build the SWE-ReX wrapper image for every SWE-bench instance in a subset,
# so that `sweagent run-batch` can skip the podman build step and go straight
# to `podman run`. Safe to re-run: already-built wrappers are detected by a
# sentinel tag and skipped.
#
# Usage:
#   bash sh/prebuild_wrappers.sh                       # lite/test, 4-way parallel
#   bash sh/prebuild_wrappers.sh --subset verified
#   bash sh/prebuild_wrappers.sh --slice :20           # first 20 only
#   bash sh/prebuild_wrappers.sh --parallel 8
#   bash sh/prebuild_wrappers.sh --rebuild             # force rebuild
#
# Prerequisites:
#   bash sh/warm_builder.sh          # builds localhost/swerex-builder:latest
#   bash sh/use_prebuilt_builder.sh  # patches swerex to FROM localhost/swerex-builder
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
export PODMAN_IGNORE_CGROUPSV1_WARNING=1

SUBSET="lite"
SPLIT="test"
SLICE=":"
PULL_PARALLEL=2     # podman pulls fight for bandwidth; keep low
BUILD_PARALLEL=8    # builds are CPU/IO once base is local; keep high
REBUILD=false
SKIP_PULL=false
SKIP_BUILD=false

while (($#)); do
    case "$1" in
        --subset) SUBSET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --slice) SLICE="$2"; shift 2 ;;
        --pull-parallel) PULL_PARALLEL="$2"; shift 2 ;;
        --build-parallel) BUILD_PARALLEL="$2"; shift 2 ;;
        --parallel) PULL_PARALLEL="$2"; BUILD_PARALLEL="$2"; shift 2 ;;
        --rebuild) REBUILD=true; shift ;;
        --skip-pull) SKIP_PULL=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        -h|--help) sed -n '4,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

log()  { echo "[$(date +%H:%M:%S)] $*"; }

# Make sure the builder layer exists (one-off Python compile).
if ! podman image exists localhost/swerex-builder:latest 2>/dev/null; then
    log "Prewarming localhost/swerex-builder:latest..."
    bash "${SCRIPT_DIR}/warm_builder.sh"
fi
bash "${SCRIPT_DIR}/use_prebuilt_builder.sh" >/dev/null 2>&1 || true

# Collect (instance_id, base_image) pairs via the agent venv (has `datasets`).
MAP=$(
    "${AGENT_VENV}/bin/python" - <<PY
from datasets import load_dataset
MAP = {'lite':'princeton-nlp/SWE-Bench_Lite','verified':'princeton-nlp/SWE-Bench_Verified','full':'princeton-nlp/SWE-Bench','multimodal':'princeton-nlp/SWE-Bench_Multimodal'}
ds = load_dataset(MAP.get('${SUBSET}', '${SUBSET}'), split='${SPLIT}')
sl = '${SLICE}'
start, stop = 0, len(ds)
if ':' in sl:
    a, b = sl.split(':', 1)
    if a: start = int(a)
    if b: stop  = int(b)
for i in range(start, min(stop, len(ds))):
    r = ds[i]
    iid = r['instance_id']
    img = f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest".lower()
    print(f"{iid}\t{img}")
PY
)

TOTAL=$(echo "$MAP" | wc -l)
log "Targeting ${TOTAL} instances (subset=${SUBSET})."

# ── Phase 1: pull all base images (network-bound, low parallel) ─────────────
# Uses the ghcr mirror via env.sh (see `pull_base_via_ghcr_one`) to avoid
# Docker Hub's 100/6h anon rate limit. Pulls from ghcr and retags to `$base`
# (canonical docker.io name that the harness and sweagent look up).
pull_one() {
    local iid="$1" base="$2"
    if podman image exists "$base" 2>/dev/null; then
        echo "[pull-skip] $iid"
        return 0
    fi
    local ghcr; ghcr="$(sweb_ghcr_tag "$iid")"
    if podman pull -q "$ghcr" >/dev/null 2>&1 && podman tag "$ghcr" "$base" 2>/dev/null; then
        echo "[pull-ok]   $iid"
    else
        echo "[pull-FAIL] $iid  (ghcr=$ghcr)"
    fi
}
export -f pull_one

if ! $SKIP_PULL; then
    log "Phase 1: pulling base images, parallel=${PULL_PARALLEL}..."
    echo "$MAP" | awk -v OFS=' ' '{print $1, $2}' \
        | xargs -n 2 -P "$PULL_PARALLEL" bash -c 'pull_one "$0" "$1"'
    log "Phase 1 done. Disk: $(du -sh /tmp/${USER}-podman 2>/dev/null | cut -f1)"
fi

# ── Phase 2: build wrappers in parallel (CPU/IO-bound, high parallel) ──────
# Produce the swerex glibc dockerfile by asking the patched swerex directly.
DOCKERFILE=$(
    "${AGENT_VENV}/bin/python" - <<'PY'
from swerex.deployment.docker import DockerDeployment
from swerex.deployment.config import DockerDeploymentConfig
cfg = DockerDeploymentConfig(image='BASE_IMAGE_PLACEHOLDER', container_runtime='podman', python_standalone_dir='/root')
print(DockerDeployment.from_config(cfg).glibc_dockerfile)
PY
)

build_one() {
    local iid="$1" base="$2"
    local tag="localhost/sweb-wrapper-${iid}:latest"
    if ! $REBUILD && podman image exists "$tag" 2>/dev/null; then
        echo "[skip] $iid"
        return 0
    fi
    if ! podman image exists "$base" 2>/dev/null; then
        echo "[FAIL] $iid (base $base not local; rerun with pull phase)"
        return 0
    fi
    echo "$DOCKERFILE" | podman build -q --build-arg "BASE_IMAGE=$base" -t "$tag" - \
        > "/tmp/prebuild-${iid}.log" 2>&1 && echo "[ok]   $iid" || echo "[FAIL] $iid (see /tmp/prebuild-${iid}.log)"
}

export -f build_one
export REBUILD DOCKERFILE

if ! $SKIP_BUILD; then
    log "Phase 2: building wrappers, parallel=${BUILD_PARALLEL}..."
    echo "$MAP" | awk -v OFS=' ' '{print $1, $2}' \
        | xargs -n 2 -P "$BUILD_PARALLEL" bash -c 'build_one "$0" "$1"'
    log "Phase 2 done. Disk: $(du -sh /tmp/${USER}-podman 2>/dev/null | cut -f1)"
fi

log "Prebuild complete."
