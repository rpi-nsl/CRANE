# `src/` — Architecture-Adaptive CRANE Components

This directory re-implements the CRANE merge-analysis pipeline in a way that
works across three Qwen3 model families (dense / MoE / Qwen3-Next hybrid)
**without modifying `crane/`**. The three components — Taylor
S_reason, GSP format projectors, and Φ(δ) denoise — are fully independent
and can be run in any subset.

The merge formula targeted by this pipeline is:

```
θ_M = θ_I + Π_τ( α_c · Φ(δ) )
```

| Term     | Source component                                | Artifact                               |
| -------- | ----------------------------------------------- | -------------------------------------- |
| α_c      | Taylor S_reason (`crane_s_reason_auto.py`)      | `phase2_stats_*.json`                  |
| Φ(δ)     | Denoise (`crane_denoise.py`)                    | —  (pure inline function, no artifact) |
| Π_τ      | GSP (`crane_gsp.py`)                            | `format_projectors.pt`                 |

Fisher contrast `(F_R − F_T)/(F_R + F_T + ε)` and decision-head `D(l, h)`
are **intentionally not produced**; the current method does not use them.

---

## File layout

```
src/
├── __init__.py                 # marker, empty
├── _common.py                  # shared: PRESETS, family detection, classifiers,
│                               #         ThinkingWeightLoader (arch-adaptive)
├── crane_s_reason_auto.py      # Component 1 — Taylor S_reason
├── crane_gsp.py                # Component 2 — Format nullspace projectors
├── crane_denoise.py            # Component 3 — Φ(δ) median-threshold
└── README.md                   # (this file)
```

External dependencies imported from `../crane/` via `sys.path` (so no code
duplication, and any upstream bugfix is picked up automatically):

- `crane/model_arch.py::load_arch` — reads `config.json`, returns a frozen
  `ModelArch(num_layers, num_experts, model_type, …)` dataclass. This file
  was extended (backward-compatibly) to add a `model_type` field.
- `crane/crane_calibration.py`, `crane_calibration_clean.py`,
  `crane_calibration.py` — tokenizer-driven text generators. Shared
  chat template between Qwen3 4B / 30B / Next, so these are arch-agnostic.
- `crane/crane_nullspace_format5_5.py` — the GSP building blocks
  (`_sample_hermes`, `_sample_ultrachat`, `build_long_context_calibration`,
  `build_completion_calibration`, `FormatFeatureCollector`,
  `compute_projector_svd_with_sigma`, etc.). `crane_gsp.py` imports these
  directly so numerics are bit-for-bit identical under default args.

---

## Supported model families

| Preset key          | model_type   | num_experts | Attention layout                   | Expert tensors (instruct)                               | bf16 single-GPU fit |
| ------------------- | ------------ | ----------- | ---------------------------------- | ------------------------------------------------------- | ------------------- |
| `qwen3-4b`          | `qwen3`      | 0           | full `self_attn.{q,k,v,o}_proj`    | —                                                       | ✓ (4B × 2B ≈ 8 GB)  |
| `qwen3-30b`         | `qwen3_moe`  | 128         | full self-attn                     | **fused** `mlp.experts.{gate_up_proj, down_proj}`       | ✓ (30B × 2B ≈ 60 GB)|
| `qwen3-next-80b`    | `qwen3_next` | 512         | 3:1 `linear_attn` / `self_attn`    | **unfused** `mlp.experts.{i}.{gate,up,down}_proj`       | ✗ (~160 GB, needs multi-GPU or `device_map="auto"`) |

Presets are defined in [`_common.py`](_common.py) under `PRESETS`. Each
entry sets the HuggingFace model IDs for both variants plus the default
`layer_chunk` (memory-budget knob for S_reason) and the default
`baseline_component` (`ffn` for dense, `expert` for MoE / Next — see
§Baseline below).

### Adding a new preset

Edit `_common.py::PRESETS` and add an entry; every component inherits the
new preset automatically. Override `instruct_id` / `thinking_id` via CLI
for one-off experiments without editing code.

---

## Shared utilities — `_common.py`

Exports (see [`_common.py`](_common.py) for signatures):

| Symbol                      | Purpose |
| --------------------------- | ------- |
| `PRESETS`                   | Model IDs + per-family defaults (see above) |
| `CACHE_DIR`                 | Resolved `HF_HOME` or fallback path |
| `detect_family(arch)`       | Returns `"dense" \| "moe" \| "next"` from `model_type + num_experts` |
| `baseline_component(family)`| Returns `"ffn"` for dense, `"expert"` otherwise |
| `classify_component(name)`  | Per-tensor group label; see table below |
| `classify_subcomponent(name)`| Finer-grained group label |
| `SUBCOMPS_BY_GROUP`         | Mapping used by `backfill_phase2_stats` |
| `layer_index(name)`         | Extracts layer index from `"model.layers.N.…"` or `None` |
| `ThinkingWeightLoader(model_dir, num_experts)` | Lazy safetensors reader with optional fused-expert reconstruction |
| `load_arch_and_family(id)`  | Convenience wrapper: `(arch, family)` |

### Component classification

`classify_component(name)` routes every parameter tensor into one of nine
groups. Only the groups that actually appear in a given model are
populated; unused groups are naturally empty.

```
linear_attn       linear_attn.*                          (Qwen3-Next only)
attention         self_attn.*
shared_expert     mlp.shared_expert.*                    (Qwen3-Next only)
router            mlp.gate.weight  |  shared_expert_gate (MoE + Next)
expert            mlp.experts.*                          (MoE + Next)
ffn               mlp.{gate,up,down}_proj                (dense only)
embedding         embed_tokens
lm_head           lm_head
norm              *norm*
other             anything else
```

`classify_subcomponent(name)` splits those further, e.g. `attention` →
`{attn_q, attn_k, attn_v, attn_o, attn_q_norm, attn_k_norm}`, `expert` →
`{expert_down, expert_up}` (where `expert_up` deliberately covers both the
unfused `gate_proj`/`up_proj` and the 30B-instruct-fused `gate_up_proj`),
`linear_attn` → `{linattn_qkvz, linattn_ba, linattn_conv, linattn_out,
linattn_norm, linattn_A_log, linattn_dt_bias}`, etc.

### `ThinkingWeightLoader` — adaptive fused-expert reconstruction

Critical detail that makes the loader work across all three families:

1. Always tries `_raw(key)` first — direct lookup in the thinking-side
   safetensors index.
2. If the direct lookup misses **and** `num_experts > 0` **and** the key
   matches the Qwen3-30B-instruct fused pattern
   `r".*\.mlp\.experts\.(down_proj|gate_up_proj)$"`, it reconstructs the
   fused tensor by stacking `num_experts` unfused per-expert tensors from
   the thinking side (`mlp.experts.{i}.{gate_proj,up_proj,down_proj}`).
   This is the *only* place where the old `_NUM_EXPERTS=128` hardcode
   lives, and it's now parameterized on `num_experts` from `ModelArch`.
3. Otherwise returns `None`.

Result:

- **Dense (4B)**: direct lookup always hits (thinking and instruct are
  structurally identical); fused branch is never reached.
- **MoE (30B)**: thinking is unfused, instruct is fused → direct lookup
  misses for fused keys → fused reconstruction path triggers with
  `num_experts=128`.
- **Next (80B)**: thinking and instruct are both unfused → direct lookup
  always hits.

---

## Component 1 — `crane_s_reason_auto.py` (Taylor S_reason)

Computes per-component and per-layer Pareto-gated Taylor improvements of
the instruct→thinking delta `δ_j = θ_think,j − θ_inst,j` on two calibration
sets. For every scalar parameter `j`:

```
improvement_R(j) = − grad_R,j · δ_j          grad_R: reasoning loss gradient
improvement_T(j) = − grad_T,j · δ_j          grad_T: tool-use loss gradient
g_j              = max(0, min(improvement_R(j), improvement_T(j)))
S_raw(c)         = Σ_{j∈c} g_j / sqrt(Σ_{j∈c} θ_j²)
S_reason(c)      = S_raw(c) / S_raw(BASELINE)
```

`BASELINE` defaults to `"ffn"` for dense and `"expert"` for MoE/Next —
overridable at runtime through `baseline_component(family)` in
`_common.py`.

Both `grad_R` and `grad_T` are measured against behavior targets: the
thinking model's greedy decode of D_R prompts (reasoning) and the instruct
model's greedy decode of D_T prompts (tool-use). Loss is evaluated only on
assistant-span tokens (`labels=-100` over prompt + pad). Calibration texts
come from `crane_calibration*.py` — dataset is selected via
`--calibration-set {default, public_*}`.

Memory strategy (reused from upstream): layers are unfrozen in chunks of
`layer_chunk` (preset default), two backward passes per chunk (D_R then
D_T), gradient checkpointing enabled, grad_R clones stashed on CPU in bf16
to free GPU memory between the two passes, and thinking-side weights
pulled lazily per-tensor from safetensors.

Decode target caches are written under `--targets-cache` (default
`${CRANE_DATA_DIR%/*}/data/calib_targets`) and keyed by the resolved
model IDs, tokenizer ID, calibration set, prompt digest, `max_seq_len`,
and `max_new_tokens`; this avoids silently reusing stale targets when
doing one-off `--instruct-id` / `--thinking-id` / truncation overrides.

### CLI

```
python crane_s_reason_auto.py \
    --model-preset {qwen3-4b | qwen3-30b | qwen3-next-80b} \
    [--instruct-id <hf-id>]                       # override preset
    [--thinking-id <hf-id>]                       # override preset
    --output <path/to/phase2_stats.json> \
    [--device cuda:0]
    [--calibration-set {default, public_*}]            # default: old
    [--max-seq-len 2048]
    [--max-new-tokens 8192]                       # global cap
    [--max-new-r <n>] [--max-new-t <n>]           # per-set overrides
    [--decode-batch 20]
    [--layer-chunk <n>]                           # default from preset
    [--targets-cache <dir>]
    [--force-decode]                              # bypass cached targets
```

### Output JSON schema

```jsonc
{
  "per_component": {                              // P/I/N/S buckets per subcomponent
    "attn_q":   {"P": 0, "I": 247783478, "N": 129703882, "S": 0, "total": 377487360},
    "ffn_gate": {"P": 0, "I": 896532480, "N": 0,         "S": 0, "total": 896532480},
    …
  },
  "per_layer_S_reason": {                         // {layer_idx: {component: float}}
    "0":  {"attention": 0.88, "ffn": 1.0, "norm": 0.07},
    …
  },
  "per_component_S_reason": {                     // aggregated across layers
    "attention": 0.656, "ffn": 1.0, "norm": 0.001
  },
  "metadata": {
    "model_preset": "qwen3-4b",
    "instruct_id":  "Qwen/Qwen3-4B-Instruct-2507",
    "thinking_id":  "Qwen/Qwen3-4B-Thinking-2507",
    "family":       "dense",
    "baseline_component": "ffn",
    "calibration_set": "old",
    "num_d_r": 20, "num_d_t": 10,
    "layer_chunk":   8,
    "num_layers":    36,
    "num_experts":   0,
    "model_type":    "qwen3",
    "main_device":   "cuda:0",
    "d_r_decode_device": "cuda:0"
  }
}
```

`metadata.baseline_component` is load-bearing: downstream code must read it
to know which component was used as the denominator. Never hard-code
`"expert"`.

### Reference runtimes (single H100 80GB)

| Preset      | max_new_r / max_new_t | Decode | 1 chunk backward | Total     |
| ----------- | --------------------- | ------ | ---------------- | --------- |
| qwen3-4b    | 128 / 128             | ~14 s  | ~13 s (3 chunks, 12 layers each) | ~65 s   |
| qwen3-30b   | 8192 / 2048           | ~5 min | ~10 s (24 chunks, 2 layers each) | ~8 min  |
| qwen3-next-80b | (requires multi-GPU) | —     | —                | —         |

---

## Component 2 — `crane_gsp.py` (GSP format projectors)

Builds per-layer per-component V_r + σ singular vectors that span the
instruct model's hidden-state subspace on format-token neighborhoods
(`<tool_call>`, `<|im_start|>`, etc.). The merge script then projects
δ-updates away from this subspace so format tokens survive merging.

### Regression invariant (critical)

`${CRANE_DATA_DIR%/*}/data/format_projectors.pt` (192 MB, produced
2026-04-13) is the projector file used by the current merge model. It was
produced by `crane/crane_nullspace_format5_5.py` under its default
arguments. `crane_gsp.py` wraps the exact same building blocks and
deliberately keeps identical defaults:

| Parameter         | Default | CLI flag             |
| ----------------- | ------- | -------------------- |
| `max_seq_len`     | 4096    | `--max-seq-len`      |
| `svd_rank`        | 256     | `--svd-rank`         |
| `threshold`       | 1e-5    | `--threshold`        |
| `radius`          | 3       | `--radius`           |
| `n_tool`          | 300     | `--n-tool`           |
| `n_chat`          | 70      | `--n-chat`           |
| `n_long_context`  | 30      | `--n-long-context`   |
| `n_completion`    | 30      | `--n-completion`     |
| long-context range | 4000 – 16000 tokens | `--long-context-min/--max` |
| completion range   | 2000 – 12000 tokens | `--completion-min/--max`   |
| `seed`            | 42      | `--seed`             |

Calibration sources (both fixed):

1. `NousResearch/hermes-function-calling-v1` (5 subsets: `func_calling_singleturn`, `func_calling`, `glaive_func_calling`, `json_mode_agentic`, `json_mode_singleturn`)
2. `HuggingFaceH4/ultrachat_200k` split `train_sft`
3. Synthetic long-context multi-turn tool-use conversations
4. Synthetic `attempt_completion` conversations

Six format patterns, fixed order: `<tool_call>`, `</tool_call>`,
`<tool_response>`, `</tool_response>`, `<|im_start|>`, `<|im_end|>`.

**All of the above are CLI-overridable, but the defaults must not drift.**
Running with `--model-preset qwen3-30b` and no other flags is required to
bit-for-bit reproduce the existing `.pt`:

```bash
python src/crane_gsp.py --model-preset qwen3-30b --out /tmp/gsp_regression

python -c "
import torch
a = torch.load('${CRANE_DATA_DIR%/*}/data/format_projectors.pt',
               map_location='cpu', weights_only=True)
b = torch.load('/tmp/gsp_regression/format_projectors.pt',
               map_location='cpu', weights_only=True)
assert set(a) == set(b), f'key diff: {set(a) ^ set(b)}'
for k in a:
    # V_r and sigma are both torch tensors
    for field in ('V_r', 'sigma'):
        assert torch.allclose(a[k][field], b[k][field], atol=1e-5, rtol=1e-4), k
print('GSP regression OK')
"
```

Short-context degradation (e.g. `--max-seq-len 512`) is the most
frequently-recurring bug in this area: format token features at token
position 15k drift from those at position 500, so projectors built without
long-context calibration fail to protect long tool-use conversations.
Never move `max_seq_len` below 4096 on the acceptance path.

### CLI

```
python crane_gsp.py \
    --model-preset {qwen3-4b | qwen3-30b | qwen3-next-80b} \
    [--instruct-id <hf-id>]
    --out <output-dir> \
    [--device cuda:0]
    [--device-map auto]                           # for 80B multi-GPU sharding
    [--svd-rank 256 --threshold 1e-5 --radius 3 --max-seq-len 4096]
    [--n-tool 300 --n-chat 70 --n-long-context 30 --n-completion 30]
    [--long-context-min 4000 --long-context-max 16000]
    [--completion-min  2000 --completion-max  12000]
    [--seed 42]
```

### Outputs

```
<out>/
├── format_projectors.pt     # dict {"{component}_layer_{i}": {"V_r": Tensor, "sigma": Tensor}}
└── format_stats.json        # hyperparameters + per-component rank/sigma stats
```

---

## Component 3 — `crane_denoise.py` (Φ(δ) median threshold)

Element-wise denoise used in the merge formula. For a delta tensor δ:

```
Φ(δ)_j = δ_j · 2     if |δ_j| > median(|δ|)
         0           otherwise
```

Parameter-free (the median is the unique threshold that requires no
selection), architecture-agnostic (operates on individual tensors), no
precomputed artifact.

### Library use

```python
from src.crane_denoise import median_threshold
delta_clean = median_threshold(theta_think - theta_inst)
```

### CLI stat mode

Reports `||δ||_F`, `||Φ(δ)||_F`, non-zero fraction, and max|δ| per tensor
— useful for sanity-checking a new model preset before running a full
merge.

```
python crane_denoise.py \
    --model-preset {qwen3-4b | qwen3-30b | qwen3-next-80b} \
    [--instruct-id <hf-id>] [--thinking-id <hf-id>]
    --stat-only
    [--tensor-filter "<fnmatch glob>"]            # e.g. "model.layers.0.*"
```

Without `--stat-only` the script runs a self-check on a random tensor and
exits.

### Expected statistics

For weight matrices (FFN proj, attention proj), non-zero fraction hovers
at ~0.5 by construction. For small parameter tensors (layer norms with
O(10³) elements), kthvalue ties make the non-zero fraction deviate from
0.5 — this is expected and not a bug.

---

## Usage patterns

### Minimum pipeline for a new model family (e.g. Qwen3-4B)

```bash
cd ${CRANE_REPO_ROOT}
source .venv/bin/activate

# 1) Taylor S_reason (α_c)
python src/crane_s_reason_auto.py --model-preset qwen3-4b \
    --output ${CRANE_DATA_DIR%/*}/data/phase2_stats_auto_4b.json \
    --device cuda:0 --calibration-set old

# 2) GSP projectors (Π_τ)
python src/crane_gsp.py --model-preset qwen3-4b \
    --out ${CRANE_DATA_DIR%/*}/data/gsp_4b \
    --device cuda:0

# 3) Denoise is pure inline — no artifact to pre-compute.
#    It will be imported from crane_denoise at merge time.
```

The three commands are fully independent; run any subset in any order.

### 30B regression path

```bash
python src/crane_s_reason_auto.py --model-preset qwen3-30b \
    --output /tmp/s_reason_30b_new.json --calibration-set old

python src/crane_gsp.py --model-preset qwen3-30b --out /tmp/gsp_regression
```

The first should match the existing
`${CRANE_DATA_DIR%/*}/data/phase2_stats_auto.json` to within 1e-4 per
component. The second must `allclose(atol=1e-5, rtol=1e-4)` the existing
`${CRANE_DATA_DIR%/*}/data/format_projectors.pt`.

### 80B caveat

`qwen3-next-80b` requires multi-GPU (`device_map="auto"`). The current
S_reason script sets `device_map={"": device}` (single GPU) and will OOM
on a single 80GB card — support for multi-GPU S_reason is deliberately
out of scope for this iteration. GSP already uses `device_map="auto"` by
default and should work. Denoise operates on single tensors at CPU and is
size-independent.

---

## Design notes

### Why three separate scripts, not a single pipeline?

Each component has very different runtime / memory profiles:

- **S_reason** needs backward passes through the instruct model with
  gradient checkpointing and a two-pass D_R/D_T schedule per layer chunk.
- **GSP** needs `device_map="auto"` forward-only hooks over 430+ long
  calibration texts. No gradients.
- **Denoise** is O(numel) per tensor at CPU.

Coupling them would force the heaviest component's constraints onto the
lighter ones. Independent scripts also let each be re-run in isolation
when its inputs or defaults change.

### Why keep `expert`/`router`/`linear_attn` groups in `SUBCOMPS_BY_GROUP` for dense models?

`backfill_phase2_stats` iterates `SUBCOMPS_BY_GROUP.items()` and skips any
subcomponent whose `sub_numel` is 0 (via `if n == 0: continue`). Dense
models simply populate zero entries for MoE-only groups; no special-casing
needed.

### Why `model_type` on `ModelArch`?

To distinguish Qwen3-Next (`model_type="qwen3_next"`, `num_experts=512`,
hybrid attention) from standard Qwen3-MoE (`model_type="qwen3_moe"`,
`num_experts=128`, full attention only). Both have `is_moe == True`, but
the hybrid family needs different component handling. The field was added
to `crane/model_arch.py` with a default empty string so existing callers
in `crane/` continue to work unchanged.

### What about `tie_word_embeddings`?

Qwen3-4B has `tie_word_embeddings=true` — the safetensors index typically
stores only `embed_tokens.weight`, not a separate `lm_head.weight`. S_reason
only iterates over `model.model.layers[i]` (never the embedding or
lm_head), so tied weights are silently unaffected. If a future component
ever does touch embeddings, `ThinkingWeightLoader.get("lm_head.weight",
...)` will correctly return `None` under tying.

### Relationship to `crane/`

`src/` does **not** replace `crane/`. `crane_nullspace_format5_5.py` and
the calibration loaders in `crane/` are imported by `src/` for the
pieces that are shared between architectures.
