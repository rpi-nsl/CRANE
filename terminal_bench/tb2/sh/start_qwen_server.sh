#!/usr/bin/env bash
# Start vLLM (Qwen3-30B-A3B-Instruct-2507) behind a Cloudflare named tunnel.
# Portable: the only state that differs per machine is the tunnel credentials
# JSON; everything else is computed at runtime.
#
# Required on the target machine:
#   - 4× GPU (or adjust VLLM_TP_SIZE)
#   - vllm venv (default ${CRANE_REPO_ROOT}/.venv, override VENV)
#   - cloudflared binary at ~/bin/cloudflared (or override CLOUDFLARED)
#   - tunnel credentials JSON at ~/.cloudflared/qwen-tb2.json
#     (override with CF_TUNNEL_CREDS=/path/to/qwen-tb2.json)
#
# When the creds file is missing but CF_API_TOKEN + CF_ACCOUNT_ID + TUNNEL_ID +
# TUNNEL_SECRET are set in env, it will rebuild the creds JSON automatically —
# useful if you don't want to ship a secret file around (pass it via env instead).
#
# Usage:
#   ./tb2/sh/start_qwen_server.sh                      # foreground, Ctrl+C to stop
#   bash ./tb2/sh/start_qwen_server.sh &               # background
#   MODEL_HF_ID=Qwen/Qwen3-30B-A3B-Thinking-2507 ./tb2/sh/start_qwen_server.sh
#
# Env you'll commonly tweak:
#   SERVED_NAME    — name vllm exposes (default qwen3-30b)
#   PUBLIC_URL     — the tunnel's public hostname (default your-vllm.example.com)
#   VLLM_PORT      — localhost port vllm binds (default 18016)
#   VLLM_TP_SIZE   — tensor-parallel size (default #GPUs)
#   VLLM_MAX_MODEL_LEN — default 131072
#
# Output: prints `PUBLIC_API_BASE=https://<host>/v1` once ready, then blocks.
set -euo pipefail

# ── Paths / settings ────────────────────────────────────────────────────────
VENV="${VENV:-${CRANE_REPO_ROOT}/.venv}"
CLOUDFLARED="${CLOUDFLARED:-$HOME/bin/cloudflared}"
CF_TUNNEL_CREDS="${CF_TUNNEL_CREDS:-$HOME/.cloudflared/qwen-tb2.json}"
PUBLIC_URL="${PUBLIC_URL:-your-vllm.example.com}"

SERVED_NAME="${SERVED_NAME:-qwen3-30b}"
VLLM_PORT="${VLLM_PORT:-18016}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
MODEL_HF_ID="${MODEL_HF_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

LOG_DIR="${LOG_DIR:-/tmp/tb2-server}"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
CF_LOG="$LOG_DIR/cloudflared.log"

# ── Helpers ─────────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*"; exit 1; }

VLLM_PID=""
CF_PID=""
cleanup() {
  log "cleanup: stopping vLLM + cloudflared"
  [[ -n "$CF_PID"   ]] && kill -TERM "$CF_PID"   2>/dev/null || true
  [[ -n "$VLLM_PID" ]] && kill -TERM "$VLLM_PID" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "vllm serve"       2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Resolve tunnel credentials (file OR env) ────────────────────────────────
if [[ ! -f "$CF_TUNNEL_CREDS" ]]; then
  log "no creds file at $CF_TUNNEL_CREDS; trying env (CF_ACCOUNT_ID / TUNNEL_ID / TUNNEL_SECRET)..."
  : "${CF_ACCOUNT_ID:?CF_ACCOUNT_ID required when creds file is missing}"
  : "${TUNNEL_ID:?TUNNEL_ID required when creds file is missing}"
  : "${TUNNEL_SECRET:?TUNNEL_SECRET required when creds file is missing}"
  mkdir -p "$(dirname "$CF_TUNNEL_CREDS")"
  cat > "$CF_TUNNEL_CREDS" <<EOF
{"AccountTag":"$CF_ACCOUNT_ID","TunnelID":"$TUNNEL_ID","TunnelName":"$(basename "$CF_TUNNEL_CREDS" .json)","TunnelSecret":"$TUNNEL_SECRET"}
EOF
  chmod 600 "$CF_TUNNEL_CREDS"
  log "wrote creds to $CF_TUNNEL_CREDS from env"
fi
TUNNEL_NAME=$(python3 -c "import json,sys;print(json.load(open('$CF_TUNNEL_CREDS'))['TunnelName'])")
log "tunnel: $TUNNEL_NAME → $PUBLIC_URL  (creds: $CF_TUNNEL_CREDS)"

# ── Activate vLLM venv + resolve model path ─────────────────────────────────
[[ -f "$VENV/bin/activate" ]] || die "venv not found at $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
command -v vllm >/dev/null || die "vllm not on PATH after activating $VENV"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  # Look for cached snapshot of $MODEL_HF_ID
  snapshot_dir="$HF_HOME/hub/models--${MODEL_HF_ID//\//--}/snapshots"
  if [[ -d "$snapshot_dir" ]]; then
    MODEL_PATH=$(ls -d "$snapshot_dir"/*/ 2>/dev/null | head -1)
  fi
  if [[ -z "$MODEL_PATH" ]]; then
    log "snapshot-downloading $MODEL_HF_ID to $HF_HOME (once) ..."
    MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_HF_ID', cache_dir='$HF_HOME',
  allow_patterns=['*.safetensors','*.index.json','*.json','*.model','*.txt','tokenizer*']))
")
  fi
fi
[[ -d "$MODEL_PATH" ]] || die "model dir missing: $MODEL_PATH"
log "model: $MODEL_PATH"

# ── TP auto-detect ──────────────────────────────────────────────────────────
TP="${VLLM_TP_SIZE:-}"
if [[ -z "$TP" ]]; then
  TP=$(nvidia-smi -L 2>/dev/null | wc -l)
  (( TP < 1 )) && TP=1
fi
log "tensor-parallel: $TP"

# ── Start vLLM ──────────────────────────────────────────────────────────────
log "starting vLLM on :$VLLM_PORT (log: $VLLM_LOG)"
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$VLLM_PORT" \
  --dtype bfloat16 --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 --tensor-parallel-size "$TP" \
  --enable-expert-parallel \
  --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --enable-prompt-tokens-details \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
log "vllm pid=$VLLM_PID"

# ── Start tunnel (can run in parallel with vllm boot) ───────────────────────
log "starting cloudflared tunnel (log: $CF_LOG)"
[[ -x "$CLOUDFLARED" ]] || die "cloudflared binary missing at $CLOUDFLARED"
"$CLOUDFLARED" tunnel --credentials-file "$CF_TUNNEL_CREDS" \
  --url "http://localhost:$VLLM_PORT" run "$TUNNEL_NAME" \
  > "$CF_LOG" 2>&1 &
CF_PID=$!
log "cloudflared pid=$CF_PID"

# ── Wait for vllm ready ─────────────────────────────────────────────────────
log "waiting for vllm /v1/models ..."
t=0
until curl -sf -m 3 "http://localhost:$VLLM_PORT/v1/models" -o /dev/null 2>/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || die "vllm died; see $VLLM_LOG"
  sleep 5; t=$((t+5))
  (( t % 30 == 0 )) && log "  vllm still loading ... (${t}s)"
done
log "vllm ready"

# ── Wait for tunnel (public URL returns 200) ────────────────────────────────
log "waiting for https://$PUBLIC_URL/v1/models ..."
t=0
until curl -sf -m 5 "https://$PUBLIC_URL/v1/models" -o /dev/null 2>/dev/null; do
  kill -0 "$CF_PID" 2>/dev/null || die "cloudflared died; see $CF_LOG"
  sleep 5; t=$((t+5))
  if (( t >= 180 )); then
    log "  tunnel still unreachable after 3 min — first run on a new hostname can take up to 15 min for Universal SSL; will keep trying"
    t=0
  fi
done
log "tunnel ready"

cat <<READY

====================================================================
  vLLM + Cloudflare tunnel READY
  PUBLIC_API_BASE=https://$PUBLIC_URL/v1
  served model:   $SERVED_NAME
  local port:     http://localhost:$VLLM_PORT
  logs:           $LOG_DIR

  Point harbor at it:
    OPENAI_API_BASE="https://$PUBLIC_URL/v1" \\
    OPENAI_API_KEY=placeholder \\
    AGENT=terminus-2 MODEL=openai/$SERVED_NAME \\
      ./tb2/sh/run_tb2_vllm_models.sh
====================================================================

READY

# ── Block on both children ──────────────────────────────────────────────────
wait
