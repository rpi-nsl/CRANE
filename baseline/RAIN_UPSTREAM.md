# RAIN-Merging upstream dependency

The RAIN baseline (`rain.py`) wraps the original RAIN-Merging
implementation. The upstream is vendored into `baseline/rain_upstream/`
at the commit below, with `.git/`, `__pycache__/`, and `.venv*/`
stripped. `data/` and `data_python/` (the calibration JSON/JSONL files
RAIN needs at run time) are kept.

## Pinned commit

| Repo | URL | Commit |
|------|-----|--------|
| RAIN-Merging | https://github.com/K1nght/RAIN-Merging | `05003cc1afd594d4a00026222ce79fbe1c08c5f1` |

## Install

```bash
source env.sh   # activates $CRANE_REPO_ROOT/.venv
pip install -r "$CRANE_REPO_ROOT/baseline/rain_upstream/requirements.txt"
pip install -e "$CRANE_REPO_ROOT/baseline/rain_upstream"
```

`rain.py` expects the upstream at `$CRANE_REPO_ROOT/baseline/rain_upstream/`
by default; override via the `RAIN_UPSTREAM` env var.

See `RAIN_README.md` in this directory for the calibration-set
construction step that the upstream's `data/` provides.
