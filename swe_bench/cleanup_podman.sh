#!/bin/bash
# Kill any lingering swebench / swe-rex containers and runaway podman build
# processes. Idempotent; safe to run before or after an agent batch.
#
# Usage: bash sh/cleanup_podman.sh [--all]
#   no flag  — only kill swe-bench eval & swe-rex wrapper containers
#   --all    — additionally kill stray 'podman build' and buildah-oci-runtime
set -euo pipefail
MODE="${1:-}"

kill_matching() {
    local pattern="$1"
    local n=0
    while read -r name; do
        [[ -z "$name" ]] && continue
        podman kill "$name" >/dev/null 2>&1 || true
        podman rm -f "$name" >/dev/null 2>&1 || true
        n=$(( n + 1 ))
    done < <(podman ps -a --format '{{.Names}}' 2>/dev/null | grep -E "$pattern" || true)
    echo "  killed $n containers matching $pattern"
}

echo "[cleanup] swebench / swe-rex / openhands containers..."
# `^oh-` covers OpenHands per-instance containers (openhands_swebench.py uses
# `oh-<iid>-<rand>` naming). The other patterns: `sweb.eval.*` is the harness
# per-instance container, `minisweagent-` is mini-swe-agent's, and the typo'd
# `docker.ioswebench` catches a legacy form.
kill_matching "^oh-|sweb\.eval\.|minisweagent-|docker\.ioswebench"

if [[ "$MODE" == "--all" ]]; then
    echo "[cleanup] podman build / buildah processes..."
    for pid in $(pgrep -f "podman build\|buildah-oci-runtime\|slirp4netns" 2>/dev/null | grep -vE "pgrep|node-export|dcgm" || true); do
        kill -9 "$pid" 2>/dev/null || true
    done
fi

echo "[cleanup] done."
