# Checkpoints and pre-computed artifacts

Checkpoints is available on Google Drive:
https://drive.google.com/drive/folders/1JfD02zes5hCW6zFGhcO900yP-HbHaVk7?usp=sharing




## Expected layout

After downloading and unpacking, this `checkpoints/` directory should
contain:

```
checkpoints/
├── README.md                           ← this file
├── qwen3-30b-instruct/                 ← HF snapshot (Qwen/Qwen3-30B-A3B-Instruct-2507)
├── qwen3-30b-thinking/                 ← HF snapshot (Qwen/Qwen3-30B-A3B-Thinking-2507)
├── qwen3-next-80b-instruct/            ← HF snapshot (Qwen/Qwen3-Next-80B-A3B-Instruct)
├── qwen3-next-80b-thinking/            ← HF snapshot (Qwen/Qwen3-Next-80B-A3B-Thinking)
└── precomputed/                        ← outputs of the long Taylor / GSP runs
    ├── format_projectors_30b.pt        ← Π_τ projector for 30B (≈3 min on 4×H100 to recompute)
    ├── format_projectors_80b_next.pt   ← Π_τ projector for 80B-Next (≈18 min)
    ├── gsp_80b_next_linattn_inner/
    │   └── format_projectors_linattn_inner.pt
    ├── phase2_stats_30b.json           ← S_reason coefficients for 30B (≈6 min)
    ├── phase2_stats_80b_next.json      ← S_reason coefficients for 80B-Next (≈32 min)
    ├── phase2_stats_30b.json
    ├── phase2_stats_80b.json
    └── calib_targets/                  ← cached Thinking-side decode targets (D_R)
```

You can also rebuild any `precomputed/` artifact from scratch using the
scripts under `src/` (see top-level `README.md`, "Reproduce CRANE"
section). The Google Drive copy exists only to skip the
≈40 minutes-per-model setup for reviewers who want a quick sanity check.

## Pointing the code at this directory

By default the code looks for these files under
`${CRANE_DATA_DIR}` (set in `env.sh`, defaults to
`${CRANE_REPO_ROOT}/data`). Either:

- symlink: `ln -s "$(pwd)/checkpoints/precomputed" "$CRANE_DATA_DIR"`
- or pass an explicit path on the CLI: each `crane_*.py` script accepts
  `--targets-cache`, `--stats`, `--gsp`, etc.

For base/merged checkpoints, override `CRANE_CHECKPOINT_DIR`:

```bash
export CRANE_CHECKPOINT_DIR=/path/to/checkpoints
```

`src/_common.py` reads the model paths from `PRESETS` (declared at the
top of that file); update those four entries if you want to load the HF
snapshots from a non-default location.
