#!/usr/bin/env bash
# SWE-bench-Verified environment. Sources the top-level env.sh and adds the
# podman/docker-host wiring + ghcr->dockerhub retag helpers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
source "$(cd "$ROOT_DIR/.." && pwd)/env.sh"

export PODMAN_IGNORE_CGROUPSV1_WARNING=1

# Per-process file descriptor limit. Default 4096 is too low for swebench
# harness with --max_workers 24: each worker holds open per-instance log
# FileHandlers + docker-py HTTP pool to podman socket, and the totals
# accumulate across the run. Manifests as `OSError: [Errno 23] Too many
# open files in system` mid-run. Hard limit on most hosts is 65535+.
ulimit -n 65535 2>/dev/null || true

# Two venv layout: vLLM (py3.11 + vllm 0.19.0) lives at $CRANE_REPO_ROOT/.venv
# (shared with Roo-Eval); the OpenHands SDK + swebench harness need py3.12 and
# different litellm/pydantic pins, so they live at $CRANE_REPO_ROOT/.venv-openhands.
VLLM_VENV="${VLLM_VENV:-$CRANE_REPO_ROOT/.venv}"
AGENT_VENV="${AGENT_VENV:-$CRANE_REPO_ROOT/.venv-swebench}"
OPENHANDS_VENV="${OPENHANDS_VENV:-$CRANE_REPO_ROOT/.venv-openhands}"

export ROOT_DIR VLLM_VENV AGENT_VENV OPENHANDS_VENV

# ── SWE-bench base-image registry ──────────────────────────────────────────
# Docker Hub rate-limits anonymous pulls (100 per 6h per IP). Epoch AI publishes
# a smaller mirror of all SWE-bench eval images on ghcr.io with no anon
# rate limit. We pull from ghcr and retag to the canonical docker.io/swebench
# name that swebench.harness + sweagent look up.
#   ghcr:  ghcr.io/epoch-research/swe-bench.eval.x86_64.<iid>:latest
#   canon: docker.io/swebench/sweb.eval.x86_64.<mangled>:latest  (iid with __→_1776_)
# Reference: https://epoch.ai/blog/swebench-docker

sweb_canon_tag() {
    local iid="$1"
    echo "docker.io/swebench/sweb.eval.x86_64.${iid//__/_1776_}:latest" \
        | tr '[:upper:]' '[:lower:]'
}

sweb_ghcr_tag() {
    echo "ghcr.io/epoch-research/swe-bench.eval.x86_64.$1:latest"
}

pull_base_via_ghcr_one() {
    local iid="$1"
    local canon ghcr
    canon="$(sweb_canon_tag "$iid")"
    ghcr="$(sweb_ghcr_tag "$iid")"
    if podman image exists "$canon" 2>/dev/null; then
        echo "[skip] $iid"
        return 0
    fi
    local start=$(date +%s)
    if podman pull -q "$ghcr" >/dev/null 2>&1; then
        podman tag "$ghcr" "$canon" 2>/dev/null
        echo "[ok]   $iid  ($(( $(date +%s) - start ))s)"
    else
        echo "[FAIL] $iid  ghcr=$ghcr"
        return 1
    fi
}
export -f sweb_canon_tag sweb_ghcr_tag pull_base_via_ghcr_one

pull_bases_via_ghcr() {
    local parallel=4
    if [[ "${1:-}" == "--parallel" ]]; then parallel="$2"; shift 2; fi
    local src
    if (( $# > 0 )); then
        src=$(printf '%s\n' "$@")
    else
        src=$(cat)
    fi
    echo "$src" | xargs -n 1 -P "$parallel" bash -c 'pull_base_via_ghcr_one "$0"'
}
