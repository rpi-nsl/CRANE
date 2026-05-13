#!/usr/bin/env bash
# Terminal-Bench v2 environment. Sources the top-level env.sh and configures
# the Daytona / GHCR registry layout for the openhands reference agent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$SCRIPT_DIR/.." && pwd)/env.sh"

# --- EDIT THESE BEFORE FIRST RUN ---
# GHCR_USER / GHCR_NAMESPACE / IMAGE_TAG control where your prebuilt task images
# live. The build step pushes ghcr.io/$GHCR_USER/$GHCR_NAMESPACE-<task>:$IMAGE_TAG.
# TASKS_SUBSET is space-separated to limit which of the 89 (default) tasks run.
export GHCR_USER="${GHCR_USER:?Set GHCR_USER to your github user/org (lowercased)}"
export GHCR_NAMESPACE="${GHCR_NAMESPACE:-tb}"
export IMAGE_TAG="${IMAGE_TAG:-v1}"
export TASKS_SUBSET="${TASKS_SUBSET:-}"
# ------------------------------------

export REGISTRY="ghcr.io/${GHCR_USER}"

# UPSTREAM_DIR points at a clone of laude-institute/terminal-bench; see
# UPSTREAM.md in this directory for the pinned commit.
export TBENCH_ROOT="$SCRIPT_DIR"
export UPSTREAM_DIR="${TBENCH_UPSTREAM:-$TBENCH_ROOT/upstream}"
export TASKS_DIR="${UPSTREAM_DIR}/original-tasks"

# Daytona credentials (cloud sandboxes, only required at run time).
# Generate at https://app.daytona.io and export DAYTONA_API_KEY in your shell.
[ -n "${DAYTONA_API_KEY:-}" ] || echo "WARNING: DAYTONA_API_KEY not set." >&2

image_for_task() {
    echo "${REGISTRY}/${GHCR_NAMESPACE}-$1:${IMAGE_TAG}"
}
