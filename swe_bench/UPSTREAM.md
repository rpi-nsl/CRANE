# SWE-bench-Verified upstream dependencies

Four upstream agent frameworks are vendored into
`swe_bench/vendors/<name>/` at the commits below, with build artifacts,
dependency caches, and large evaluation-result blobs stripped (`.git/`,
`node_modules/`, `__pycache__/`, `dist/`, `build/`, `*.egg-info/`,
`.venv*/`, `*.vsix`, `*.tar.gz`, `.env*`).

The OpenHands SDK adapter in this directory (`openhands_swebench.py`)
delegates to the [OpenHands](https://github.com/All-Hands-AI/OpenHands)
agent loop. The harness wrapper (`harness_fast.py`) wraps the
[swebench](https://github.com/princeton-nlp/SWE-bench) harness with the
podman-friendly patches documented in Appendix §A.1.2.

For ablation comparisons that use the SWE-agent or moatless scaffolds,
the original sweeps also vendored those upstreams; pinned commits are
listed for traceability but only OpenHands is **required** to reproduce
the headline rows of Table 2.

## Pinned commits

| Repo | URL | Commit | Required for headline? |
|------|-----|--------|:---:|
| OpenHands | https://github.com/All-Hands-AI/OpenHands.git | `fa0da8f3bd68c98f8948d8f1bab383fd0149dc6e` | Yes |
| SWE-agent | https://github.com/SWE-agent/SWE-agent.git | `0f4f3bba990e01ca8460b9963abdcd89e38042f2` | No |
| moatless-tools | https://github.com/aorwall/moatless-tools.git | `011ead57a5c81664e9c45e07e1f50b17e695cc63` | No |
| openhands-benchmarks | https://github.com/OpenHands/benchmarks.git | `5a1a44ee07fe49305a516772a61a1f2c2b54d585` | No |

## Stripped vendored assets

A few oversized cache/result blobs were removed from the vendored
copies. They are not needed to reproduce the paper, but if you want to
inspect moatless's own internal evaluation history, re-fetch them from
upstream at the pinned commit:

| Vendor | Removed |
|--------|---------|
| moatless | `moatless/evaluation/datasets/dataset_index.json` (~13 MB), `moatless/evaluation/swebench_verified_all_evaluations.json` (~9 MB) |
| SWE-agent | `assets/warning.png` (~6 MB) |

## Install

```bash
# Required: OpenHands SDK + swebench harness in a dedicated py3.12 venv
# (separate from the vLLM venv to avoid litellm/pydantic pin conflicts).
python3.12 -m venv "$CRANE_REPO_ROOT/.venv-openhands"
source "$CRANE_REPO_ROOT/.venv-openhands/bin/activate"
pip install -e "$CRANE_REPO_ROOT/swe_bench/vendors/OpenHands/openhands-sdk"
pip install -e "$CRANE_REPO_ROOT/swe_bench/vendors/OpenHands/openhands-tools"
pip install swebench==4.1.0 datasets litellm pyyaml
python -m swe_bench.swebench_lchown_patch  # apply the rootless-podman fix
```

For ablations that need the optional scaffolds:

```bash
pip install -e "$CRANE_REPO_ROOT/swe_bench/vendors/SWE-agent"
pip install -e "$CRANE_REPO_ROOT/swe_bench/vendors/moatless"
pip install -e "$CRANE_REPO_ROOT/swe_bench/vendors/openhands-benchmarks"
```
