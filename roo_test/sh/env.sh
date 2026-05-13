#!/usr/bin/env bash
# Roo-Eval environment. Sources the top-level env.sh and adds Roo-specific paths.
# Source this from any roo_test eval script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$(cd "$ROOT_DIR/.." && pwd)/env.sh"

# Activate eval venv if present (vLLM 0.19.0 + flashinfer + transformers).
# By convention put the eval venv at $CRANE_REPO_ROOT/.venv (see top-level
# README.md installation step).
VENV_DIR="${CRANE_VENV:-$CRANE_REPO_ROOT/.venv}"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Verify node is on PATH (Roo-Code CLI requires Node 24+ and pnpm).
if ! command -v node &>/dev/null; then
    echo "WARNING: node not found on PATH (Roo-Eval needs Node 24+)." >&2
fi

# ── Per-language workspaces ─────────────────────────────────────────────────
# Some Roo-Eval languages need extra toolchains (Go, Rust, Java).
# Top-level env.sh already adds them to PATH if NODE_HOME/CARGO_HOME/GOROOT
# are set; nothing to do here unless your install layout differs.

export ROOT_DIR
