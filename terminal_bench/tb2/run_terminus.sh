#!/usr/bin/env bash
# Smoke-test a v0.2.x task with the terminus-2 agent, driving a local
# OpenAI-compatible endpoint (e.g. vLLM-served Qwen).
#
# Prereq:
#   - ./tb2/patch_compose.py <task_id>   (already applied for fix-permissions)
#   - vLLM (or any OpenAI-compat server) listening on OPENAI_API_BASE
#
# Usage:
#   OPENAI_API_BASE=http://localhost:8000/v1 \
#   OPENAI_API_KEY=placeholder \
#   MODEL=openai/qwen3-30b \
#   ./tb2/run_terminus.sh fix-permissions
set -euo pipefail

TB2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="${TB2_ROOT}/tasks-v0.2/tasks"
TASK_ID="${1:-fix-permissions}"
MODEL="${MODEL:-openai/qwen3-30b}"
MAX_EPISODES="${MAX_EPISODES:-20}"

: "${OPENAI_API_BASE:?set OPENAI_API_BASE to your vLLM /v1 URL (e.g. http://localhost:8000/v1)}"
: "${OPENAI_API_KEY:=placeholder}"     # vLLM doesn't check it, but LiteLLM insists on one
export OPENAI_API_BASE OPENAI_API_KEY

[[ -d "${TASKS_DIR}/${TASK_ID}" ]] || {
  echo "ERROR: task '${TASK_ID}' not in ${TASKS_DIR}" >&2
  exit 1
}

# Sanity-check the endpoint BEFORE spinning up podman
echo "Pinging ${OPENAI_API_BASE}/models ..."
if ! curl -sf -m 5 "${OPENAI_API_BASE}/models" >/dev/null; then
  echo "ERROR: ${OPENAI_API_BASE}/models not reachable" >&2
  exit 1
fi
echo "OK."

export PODMAN_IGNORE_CGROUPSV1_WARNING=1
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
SOCK="/tmp/${USER}-podman.sock"

if ! pgrep -u "${USER}" -f "podman system service" >/dev/null; then
  echo "Starting podman system service on ${SOCK}"
  nohup podman system service --time=0 "unix://${SOCK}" \
    > "/tmp/${USER}-podman.log" 2>&1 &
  disown
  for _ in 1 2 3 4 5; do sleep 1; [[ -S "${SOCK}" ]] && break; done
fi
export DOCKER_HOST="unix://${SOCK}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load docker/26.1.5 2>/dev/null || true

# Pre-warm podman /version so docker SDK's 60s timeout inside tb doesn't fire
docker version --format '{{.Server.Version}}' >/dev/null 2>&1

cd "${TB2_ROOT}"
exec uv tool run --python 3.12 terminal-bench run \
  --dataset-path "${TASKS_DIR}" \
  --agent terminus-2 \
  --model "${MODEL}" \
  --agent-kwarg "api_base=${OPENAI_API_BASE}" \
  --agent-kwarg "temperature=0.6" \
  --agent-kwarg 'llm_call_kwargs={"top_p":0.95,"extra_body":{"top_k":20}}' \
  --agent-kwarg "max_episodes=${MAX_EPISODES}" \
  --task-id "${TASK_ID}" \
  --n-concurrent 1 \
  --global-test-timeout-sec 300 \
  "${@:2}"
