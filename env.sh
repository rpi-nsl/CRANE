#!/usr/bin/env bash
# Top-level environment for the CRANE reproduction repo.
# Source from any benchmark script: source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
#
# All other env.sh files in the repo (roo_test/sh/env.sh, swe_bench/env.sh,
# terminal_bench/env.sh) source this one and then add benchmark-specific paths.
#
# Override anything from your shell:
#   export CRANE_REPO_ROOT=/path/to/code_repo
#   export CRANE_DATA_DIR=/path/to/cached/calib_targets_and_phase2_stats
#   export CRANE_CHECKPOINT_DIR=/path/to/downloaded/checkpoints
#   export HF_HOME=/path/to/huggingface_cache
#   export MZ_CACHE=/path/to/scratch_cache    # for pip/uv/triton/torch caches

# Repo root (this file's directory) ───────────────────────────────────────────
export CRANE_REPO_ROOT="${CRANE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Data, checkpoints, logs ─────────────────────────────────────────────────────
# These hold artifacts that are too large to ship in the repo:
#   - CRANE_DATA_DIR: pre-computed phase2_stats_*.json, format_projectors_*.pt,
#     calib_targets/, robustness/. Provided as Google-Drive download
#     (see checkpoints/README.md).
#   - CRANE_CHECKPOINT_DIR: paired Qwen3 Instruct/Thinking checkpoints + the
#     merged outputs you produce.
#   - CRANE_LOG_DIR: per-run logs (eval-results, sbatch_log, etc.)
export CRANE_DATA_DIR="${CRANE_DATA_DIR:-$CRANE_REPO_ROOT/data}"
export CRANE_CHECKPOINT_DIR="${CRANE_CHECKPOINT_DIR:-$CRANE_REPO_ROOT/checkpoints}"
export CRANE_LOG_DIR="${CRANE_LOG_DIR:-$CRANE_REPO_ROOT/logs}"
export CRANE_MERGED_DIR="${CRANE_MERGED_DIR:-$CRANE_REPO_ROOT/merged_model}"

# Heavy caches (pip/uv/triton/torch_extensions/HuggingFace) ──────────────────
# Default to ~/.cache/crane unless the user points us at a faster scratch.
export MZ_CACHE="${MZ_CACHE:-$HOME/.cache/crane}"
export HF_HOME="${HF_HOME:-$MZ_CACHE/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MZ_CACHE/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MZ_CACHE/uv}"
export TORCH_HOME="${TORCH_HOME:-$MZ_CACHE/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$MZ_CACHE/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MZ_CACHE/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$MZ_CACHE/nv}"

# Toolchains used by Roo-Eval ─────────────────────────────────────────────────
# If you have system installs of node/go/rust, leave these unset and the env
# scripts will fall back to whatever is on $PATH. Otherwise point each
# variable at a portable install (e.g. nvm-managed node, rustup home, etc.).
# CUDA toolkit (used by flashinfer JIT against vLLM):
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
# Node.js (only needed for Roo-Eval):
[ -n "${NODE_HOME:-}" ] && [ -d "$NODE_HOME/bin" ] && export PATH="$NODE_HOME/bin:$PATH"
# Rust (only needed for Rust language Roo-Eval rows):
if [ -n "${CARGO_HOME:-}" ]; then
    export PATH="$CARGO_HOME/bin:$PATH"
fi
# Go (only needed for Go language Roo-Eval rows):
if [ -n "${GOROOT:-}" ]; then
    export PATH="$GOROOT/bin:$PATH"
fi

# Per-Slurm-job scratch on node-local fs (clusters vary; safe to skip) ───────
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${TMPDIR:-}" ]; then
    candidate="/tmp/$SLURM_JOB_ID"
    mkdir -p "$candidate" 2>/dev/null && export TMPDIR="$candidate"
fi
