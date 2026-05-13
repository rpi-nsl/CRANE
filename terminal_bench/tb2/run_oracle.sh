#!/usr/bin/env bash
# Run terminal-bench oracle agent on a v0.2.x task using rootless podman
# as the docker backend. No Mac-side build/push needed: task images build
# from the public GHCR base images (no rate limits).
#
# Usage:
#   ./tb2/run_oracle.sh                        # default: fix-permissions
#   ./tb2/run_oracle.sh chess-best-move        # any v0.2.x task id
set -euo pipefail

TB2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="${TB2_ROOT}/tasks-v0.2/tasks"
TASK_ID="${1:-fix-permissions}"

[[ -d "${TASKS_DIR}/${TASK_ID}" ]] || {
  echo "ERROR: task '${TASK_ID}' not in ${TASKS_DIR}" >&2
  exit 1
}

export PODMAN_IGNORE_CGROUPSV1_WARNING=1
# BuildKit needs cgroups v2; this cluster is cgroups v1 + rootless podman.
# Force classic builder for docker compose.
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
SOCK="/tmp/${USER}-podman.sock"

# Start rootless podman docker-compat socket if not running
if ! pgrep -u "${USER}" -f "podman system service" >/dev/null; then
  echo "Starting podman system service on ${SOCK}"
  nohup podman system service --time=0 "unix://${SOCK}" \
    > "/tmp/${USER}-podman.log" 2>&1 &
  disown
  for _ in 1 2 3 4 5; do sleep 1; [[ -S "${SOCK}" ]] && break; done
fi
export DOCKER_HOST="unix://${SOCK}"

# Load cluster docker CLI (v26 w/ compose v2) into PATH
source /etc/profile.d/modules.sh 2>/dev/null || true
module load docker/26.1.5 2>/dev/null || true

command -v docker >/dev/null || { echo "ERROR: docker CLI not in PATH" >&2; exit 1; }
docker version --format '{{.Server.Version}}' >/dev/null 2>&1 || {
  echo "ERROR: docker CLI can't reach podman at ${DOCKER_HOST}" >&2
  exit 1
}

# Run tbench with oracle agent on the picked task
cd "${TB2_ROOT}"
exec uv tool run --python 3.12 terminal-bench run \
  --dataset-path "${TASKS_DIR}" \
  --agent oracle \
  --task-id "${TASK_ID}" \
  --n-concurrent 1 \
  --global-test-timeout-sec 300 \
  "${@:2}"
