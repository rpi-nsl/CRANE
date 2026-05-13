#!/usr/bin/env bash
# Run oracle agent against ALL v0.2.x tasks (94), rootless podman backend.
# Assumes patch_compose.py has already been applied.
#
# Usage:
#   ./tb2/run_oracle_all.sh                 # default 4 concurrent
#   N=6 ./tb2/run_oracle_all.sh             # override concurrency
set -euo pipefail

TB2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="${TB2_ROOT}/tasks-v0.2/tasks"
N="${N:-4}"

export PODMAN_IGNORE_CGROUPSV1_WARNING=1
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
SOCK="/tmp/${USER}-podman.sock"

if ! pgrep -u "${USER}" -f "podman system service" >/dev/null; then
  nohup podman system service --time=0 "unix://${SOCK}" \
    > "/tmp/${USER}-podman.log" 2>&1 &
  disown
  for _ in 1 2 3 4 5; do sleep 1; [[ -S "${SOCK}" ]] && break; done
fi
export DOCKER_HOST="unix://${SOCK}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load docker/26.1.5 2>/dev/null || true

command -v docker >/dev/null || { echo "ERROR: docker CLI not in PATH" >&2; exit 1; }
docker version --format '{{.Server.Version}}' >/dev/null 2>&1 || {
  echo "ERROR: docker CLI can't reach podman at ${DOCKER_HOST}" >&2
  exit 1
}

cd "${TB2_ROOT}"
exec uv tool run --python 3.12 terminal-bench run \
  --dataset-path "${TASKS_DIR}" \
  --agent oracle \
  --n-concurrent "${N}" \
  --global-test-timeout-sec 300 \
  "${@}"
