# Terminal-Bench v2 upstream dependency

The official [laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench)
repository is vendored into `terminal_bench/upstream/` at the commit
listed below, with build artifacts, dependency caches, and a few
oversized task-data assets stripped (`.git/`, `node_modules/`,
`__pycache__/`, `dist/`, `build/`, `runs/`, `*.vsix`, `*.tar.gz`,
`.env*`).

## Pinned commit

| Repo | URL | Commit |
|------|-----|--------|
| terminal-bench | https://github.com/laude-institute/terminal-bench.git | `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b` |

## Stripped task assets (re-fetch for these tasks)

The following per-task data files were too large to ship in this repo
and have been removed from the vendored copy. Re-fetch them from the
upstream commit if you intend to run the affected tasks:

| Task | Removed file |
|------|--------------|
| `fmri-encoding-r` | `original-tasks/fmri-encoding-r/fMRIdata.RData` (~81 MB) |
| `train-fasttext` | `original-tasks/train-fasttext/tests/private_test.txt` (~29 MB) |
| `weighted-max-sat-solver` | `original-tasks/weighted-max-sat-solver/test_instance.wcnf` (~7.5 MB) |

```bash
# Re-fetch only the assets you need (sparse checkout):
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/laude-institute/terminal-bench.git /tmp/tb-upstream
git -C /tmp/tb-upstream checkout 1a6ffa9674b571da0ed040c470cb40c4d85f9b9b -- \
    original-tasks/fmri-encoding-r/fMRIdata.RData \
    original-tasks/train-fasttext/tests/private_test.txt \
    original-tasks/weighted-max-sat-solver/test_instance.wcnf
cp -r /tmp/tb-upstream/original-tasks/fmri-encoding-r/fMRIdata.RData \
      "$CRANE_REPO_ROOT/terminal_bench/upstream/original-tasks/fmri-encoding-r/"
# (and similarly for the other two)
```

## Reporting subset

Per Appendix §A.1.3 of the paper, the public reporting denominator is
**89 tasks** out of the 94 in the `tb2-zai` dataset. Five tasks are
excluded because they fail to launch reliably under the configured
Daytona sandbox budget (each excluded task counts as failed for every
model, matching the public Terminal-Bench leaderboard convention):

- `pytorch-model-cli`
- `count-dataset-tokens`
- `mcmc-sampling-stan`
- `rstan-to-pystan`
- `reshard-c4-data`

The patch scripts in `tb2/patch_*.py` rewrite the upstream tasks'
Dockerfiles, compose.yml, and test.sh shims to fit the GHCR mirror
layout and the Daytona sandbox specs documented in the paper.
