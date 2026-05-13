#!/usr/bin/env bash
# Smoke-test one TB 2.0 task via Harbor on Daytona cloud sandboxes.
#
# Prereq:
#   
#
# Usage:
#   ./tb2/run_harbor_daytona.sh chess-best-move                # oracle agent
#   AGENT=terminus-2 MODEL=openai/qwen3-30b \
#     OPENAI_API_BASE=http://<host>:8000/v1 OPENAI_API_KEY=sk-x \
#     ./tb2/run_harbor_daytona.sh chess-best-move
set -euo pipefail
TB2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to zai-verified (env + instruction fixes, same 89 tasks, xiangyangli images).
# Override with DATASET=tb2-official to use the laude-institute baseline.
DATASET_NAME="${DATASET:-tb2-zai}"
OFFICIAL="${TB2_ROOT}/${DATASET_NAME}"
TASK_ID="${1:-chess-best-move}"
AGENT="${AGENT:-oracle}"
MODEL="${MODEL:-}"
JOBS_DIR="${JOBS_DIR:-/tmp/${USER}-harbor-jobs}"
# Default x2 covers long tasks like count-dataset-tokens (agent.timeout 900 → 1800).
# Tasks that finish early aren't penalized — they exit fast regardless of ceiling.
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-2}"
# Reap any Daytona sandbox older than this (minutes) on exit (normal, Ctrl+C, error).
CLEANUP_AGE_MIN="${CLEANUP_AGE_MIN:-45}"

: "${DAYTONA_API_KEY:?Export DAYTONA_API_KEY first}"

[[ -d "${OFFICIAL}/${TASK_ID}" ]] || {
  echo "ERROR: task '${TASK_ID}' not in ${OFFICIAL}" >&2
  echo "Available (${DATASET_NAME}):"
  ls "${OFFICIAL}" | head
  exit 1
}

mkdir -p "${JOBS_DIR}"

EXTRA=()
if [[ -n "${MODEL}" ]]; then
  EXTRA+=(--model "${MODEL}")
fi
# If the agent uses LiteLLM (terminus-*), pass api_base + ensure API key is in
# LOCAL env (terminus runs on HPC, not in the sandbox, so --agent-env is not
# enough — LiteLLM reads from the parent process's env).
if [[ "${AGENT}" == terminus* && -n "${OPENAI_API_BASE:-}" ]]; then
  EXTRA+=(--agent-kwarg "api_base=${OPENAI_API_BASE}")
  EXTRA+=(--agent-kwarg "temperature=0.6")
  EXTRA+=(--agent-kwarg 'llm_call_kwargs={"top_p":0.95,"extra_body":{"top_k":20}}')
  export OPENAI_API_BASE
  export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"
fi

cd "${TB2_ROOT}"

HARBOR_PY="${MZ_CACHE}/uv-tools/harbor/bin/python3"
cleanup_on_exit() {
  [[ -x "$HARBOR_PY" ]] || return 0
  "$HARBOR_PY" "${TB2_ROOT}/cleanup_daytona.py" --age "${CLEANUP_AGE_MIN}" --delete \
      >> "${JOBS_DIR}/cleanup-daytona.log" 2>&1 || true
}
trap cleanup_on_exit EXIT INT TERM

harbor run \
  --path "${OFFICIAL}/${TASK_ID}" \
  --env daytona \
  --agent "${AGENT}" \
  --n-concurrent 1 \
  --jobs-dir "${JOBS_DIR}" \
  --agent-timeout-multiplier "${AGENT_TIMEOUT_MULT}" \
  "${EXTRA[@]}"
