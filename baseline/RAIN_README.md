# RAIN-Merging for Qwen3-30B-A3B-2507 and Qwen3-Next-80B-A3B

Adapter + patches that turn the upstream
[K1nght/RAIN-Merging](https://github.com/K1nght/RAIN-Merging) (ICLR 2026)
implementation into **two adapted baselines** that merge the
**Instruct ↔ Thinking** pair of two Qwen3 model families on this 4×H100
cluster. **Neither preset is a faithful paper reproduction**; both
deviate from the published recipe in specific ways called out below.

| Family | Label | Base used | ITM (instruct) | Target (gets modified) | Result size |
|---|---|---|---|---|---|
| `qwen3-30b` | RAIN-qkvo (MoE FFN frozen) | `Qwen/Qwen3-30B-A3B-Base` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | ~60 GB BF16 |
| `qwen3-next-80b` | RAIN-adapted (proxy base, hybrid arch, 7/48 layers) | `Qwen/Qwen3-Next-80B-A3B-Thinking` (proxy — no Base on HF) | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `Qwen/Qwen3-Next-80B-A3B-Thinking` | ~149 GB BF16 |

For each preset:
```
τ           = ITM − BASE                 (computed in Stage 1)
τ_proj      = nullspace_project(τ)       (preserves reasoning forward features
                                          on a calibration set of LRM prompts)
α*          = QP-optimize α              (Stage 2, two-pass forward on
                                          instruction calibration prompts;
                                          truncated at QP_MAX_SEQ_LEN — see below)
merged      = target + α* · τ_proj       (Stage 3, written back as HF safetensors)
```

## Known deviations from paper RAIN

These are real differences vs. the published recipe — read before treating
either checkpoint as a paper baseline:

1. **30B: FFN is *not* merged.**
   The paper's `run_stage1.sh` defaults `MERGE_TYPES=qkvof` (Q, K, V, O,
   *and* FFN); we default to `qkvo`. Reason: the upstream FFN handler
   assumes a dense `mlp.{gate,up,down}_proj` layout and breaks on the
   128-expert MoE FFN of Qwen3-30B-A3B. The MoE FFN of the target model
   is therefore copied verbatim. Pass `--merge-types qkvof` only after
   patching the FFN handler for `mlp.experts[i].{gate,up,down}_proj`.

2. **30B + 80B: instruction-calibration samples are truncated to 1536
   tokens.** Upstream's `qp_true_forward_fast.py` hard-truncates each
   instruction-calibration sample to `QP_MAX_SEQ_LEN=1536` to avoid 21 GB
   single-allocation OOMs in the two-pass forward. With the bundled
   instruction set and the Thinking tokenizers, **19/365 (≈ 5.2%)
   samples exceed this limit**. Spans are extracted *before* truncation,
   so any token-index ≥ 1536 is silently dropped, biasing α toward
   shorter samples. We surface this as `--qp-max-seq-len` (recorded in
   `merge_config.json`); set it to 4096+ if you have GPU headroom.

3. **80B: there is no published Base.** We reuse Thinking as both the
   proxy Base and the target, so τ = Instruct − Thinking and the merged
   model is `Thinking + α* · proj(Instruct − Thinking)`. This is *the
   only setup with the two published checkpoints that yields a non-zero
   task vector* — naively setting `base = instruct = Instruct` (which
   the user originally asked for) gives τ ≡ 0 and a no-op merge.

4. **80B: only 7/48 layers receive RAIN updates.** Qwen3-Next is a
   hybrid model — 12 of 48 layers are `full_attention`, the other 36 are
   `linear_attention` (Gated DeltaNet) with no `q/k/v/o_proj`. Patch B
   below filters `selected_layers` against `config.layer_types`; with
   the default `--layers-tail 27` we pick up the full-attention layers
   in the tail, which is `[23, 27, 31, 35, 39, 43, 47]` (7 layers).
   Linear-attention layers, q-gate parameters, and the 512-expert MoE
   FFN are all copied verbatim from the target.

The 80B preset additionally required six small patches to make the
upstream code run at all on Qwen3-Next (gated-Q layout, partial-RoPE,
fp32-leak in accelerate dispatch, etc.) — see "What we changed vs.
upstream" below.

## Quick start

```bash
cd ${CRANE_REPO_ROOT}/baseline
source ../roo_test/sh/env.sh                # uv venv, HF_HOME, CUDA 12.9

# RAIN-qkvo on 30B (≈ 45–60 min)
python rain.py --preset qwen3-30b

# RAIN-adapted on 80B (≈ 30 min after the parallel-load+GPU patch — was 4+ hr before)
python rain.py --preset qwen3-next-80b \
    --max-samples 256 --max-seq-len 4096 \
    --qk-device cuda:0 --vo-device cuda:1 --ffn-device cuda:2

# Resume just Stage 2+3 from a saved Stage 1 pickle
python rain.py --preset qwen3-next-80b --stages 2,3 …

# Stage 3 alone — works after the alpha-discovery fix; previously fell
# back to scaling_factor=1.0 because upstream's two_pass writes
# `alpha_true_forward_align_leak.{pt,json}`, not `…_two_pass.*`.
python rain.py --preset qwen3-next-80b --stages 3

# Reduce QP truncation bias (raise from default 1536 → 4096; ≈ 5.2 % of
# instruction-calib samples no longer get cut). Stored in merge_config.json.
python rain.py --preset qwen3-30b --qp-max-seq-len 4096
```

Output for each run:

```
baseline_model/<output_dir>/
├── projected_task_vectors.pkl         # Stage 1 (~470 MB on 80B)
├── projected_task_vectors_config.json
├── qp_optimization/
│   ├── alpha_true_forward_align_leak.{pt,json}      # Stage 2 α*
│   └── true_forward_qp_result_align_leak.pt         # QP solver state
├── unified_model_merge/
│   └── rain_merged/                   # Stage 3 — drop-in HF checkpoint
│       ├── model-0000{1..N}-of-N.safetensors
│       ├── model.safetensors.index.json
│       ├── config.json, tokenizer_config.json, …
│       └── MERGE_NOTES.md             # auto-generated caveats
├── merge_config.json                  # full provenance (commits, args)
└── unified_merge_stats.json           # per-(layer,head) Δ statistics
```

`rain_merged/` loads cleanly with `transformers.AutoModelForCausalLM` and
serves under `vllm serve` with no further conversion.

## Files in this directory

```
rain.py                                 — adapter / orchestrator (this script)
RAIN_README.md                          — this file
rain_upstream/                          — vendored K1nght/RAIN-Merging
├── nullspace_projection_compute.py     — Stage 1 (CG nullspace projection)
├── nullspace_merge_qkvo_ffn.py         — per-head Q/K/V/O constraint math
├── qp_true_forward_fast.py             — Stage 2 (α QP, two-pass forward)
├── unified_model_merge.py              — Stage 3 (apply α·τ_proj, save)
└── data/{reasoning,instruction}_calibration_set.{json,jsonl}
```

`_merge_io.py` (one level up) supplies `resolve_model_snapshot` so the same
HF caching/revision-pinning logic the other baselines use applies here too.

## What we changed vs. upstream

Six surgical patches; everything else is upstream-unchanged. All six are
**no-ops for 30B** (which is dense + standard GQA + full RoPE on every
layer) so the 30B run produces the same projected pickle and α as
running upstream directly **with `MERGE_TYPES=qkvo`** — but that's still
not the paper recipe (which is `qkvof`), and Stage 2 still applies the
shared QP_MAX_SEQ_LEN truncation. See "Known deviations" above.

### Patch A — `qwen3-next-80b` preset + `MERGE_NOTES.md` writer
*[rain.py](rain.py)*

Adds a preset entry pointing at Thinking as both the BASE proxy *and* the
target (so τ = Instruct − Thinking is non-zero), and emits a
`MERGE_NOTES.md` inside `rain_merged/` listing the three caveats every
downstream consumer needs to know:

1. **Proxy base**: `Qwen3-Next-80B-A3B-Base` is not published; we reuse
   Thinking as both BASE and target. The merged checkpoint is therefore
   `Thinking + α* · proj(Instruct − Thinking)` — Thinking nudged toward
   Instruct in null-space-allowed (= reasoning-preserving) directions.
2. **Hybrid architecture**: only the 12 `full_attention` layers receive
   RAIN's per-head Q/K/V/O updates. The 36 `linear_attention` (Gated
   DeltaNet) layers expose no `q_proj`/`k_proj`/`v_proj`/`o_proj` and are
   copied verbatim from the target.
3. **MoE FFN**: `--merge-types qkvo` (the default) skips the FFN entirely.
   Merging the 512-expert MoE FFN would require extending the upstream
   FFN handler for `mlp.experts[i].{gate,up,down}_proj`.

### Patch B — filter `selected_layers` by `config.layer_types`
*[rain_upstream/nullspace_projection_compute.py:751-770](rain_upstream/nullspace_projection_compute.py#L751-L770)*

Upstream picks the last `--layers-tail` (default 27) layers. On Qwen3-Next
the corresponding range is `[21..47]`, but most of those are
`linear_attention` layers without the q/k/v/o projections RAIN's math
requires. We post-filter `selected_layers` against `config.layer_types`
to keep only `full_attention` layers — for 80B/layers_tail=27 that yields
`[23, 27, 31, 35, 39, 43, 47]`.

### Patch C — defensive `layer_types` guard in Stage 3
*[rain_upstream/unified_model_merge.py:495-523](rain_upstream/unified_model_merge.py#L495-L523)*

Mirror of Patch B inside `apply_weighted_merge_to_model`. Patch B already
guarantees the projected pickle never references a non-full-attention
layer, but this guard is a safety net for someone splicing in a stale
pickle (`--projected-file`) on a hybrid model.

### Patch D — gated-Q slice in attention-weight computation
*[rain_upstream/nullspace_merge_qkvo_ffn.py:355-378](rain_upstream/nullspace_merge_qkvo_ffn.py#L355-L378)*
*[rain_upstream/nullspace_projection_compute.py:211-232](rain_upstream/nullspace_projection_compute.py#L211-L232)*

Qwen3-Next full_attention layers use **gated query attention**:

```
q_proj.weight       shape = [2 · num_heads · head_dim,  hidden_size]
                          = [8192, 2048]            # not [4096, 2048]
q_full              = q_proj(x)                     # [B, T, 8192]
q_content, q_gate   = q_full.split(num_heads · head_dim, dim=-1)
attn_out            = sigmoid(q_gate) * SDPA(q_content, k, v)
```

So `q_proj`'s output dim is `2 × (16 × 256) = 8192`, not `4096`. The
upstream code does `q_out.view(T, num_heads, head_dim)` which crashes
with `shape '[T, 16, 256]' is invalid for input of size T*8192`. The
patch detects this (`q_out.shape[-1] == 2 · num_heads · head_dim`) and
slices the q-content half. Both copies of this function (the inline one
in `compute_attention_weights_from_qkv` and the cached-recompute version
inside `nullspace_projection_compute.py`) are patched.

The per-head Q application slicing in `unified_model_merge.py:537-542`
naturally lands in the q-content rows (`h * head_dim : (h+1) * head_dim`
for `h ∈ [0, 15]` covers indices `[0, 4096)` — the q-content half) and
*never* touches the q-gate rows `[4096, 8192)`, so it didn't need its
own patch.

### Patch E — bf16 recast for fp32-leaked tensors
*[rain_upstream/nullspace_projection_compute.py:752-771](rain_upstream/nullspace_projection_compute.py#L752-L771)*

With `device_map="auto"` and `torch_dtype=torch.bfloat16`, accelerate's
hook dispatch on Qwen3-Next leaves a non-deterministic subset of
parameters in fp32 (we observed 2 of them on one run, 111 on another).
Forward then fails with `"expected mat1 and mat2 to have the same dtype,
BFloat16 != float"` inside `Qwen3NextLinearAttention.in_proj_qkvz`. We
walk `model.parameters()` + `.buffers()` after load and cast everything
non-integral to bf16, with a count printed (e.g. `recast 111 target
tensors to bf16 (transformers/accelerate left them in fp32 unexpectedly)`).

This is a workaround, not a fix — feel free to remove once
transformers/accelerate stop leaking fp32 on Qwen3-Next loads.

### Patch F — partial-RoPE-aware `apply_rotary_pos_emb`
*[rain_upstream/qp_true_forward_fast.py:30-46](rain_upstream/qp_true_forward_fast.py#L30-L46)*

Qwen3-Next has `partial_rotary_factor = 0.25`, so `head_dim = 256` but
the rotary embedding only acts on the first `int(256 · 0.25) = 64` dims
of each head. Upstream imports `apply_rotary_pos_emb` from `qwen3_moe`
which does full-dim rotation (`q * cos`) and crashes with `"size of
tensor a (256) must match the size of tensor b (64) at non-singleton
dimension 3"`. We prefer `qwen3_next.modeling_qwen3_next.apply_rotary_pos_emb`
which slices `q[..., :rotary_dim]`, applies RoPE, then concatenates the
unrotated remainder back. It's a strict generalization of the qwen3_moe
version (when `rotary_dim == head_dim` the two are byte-identical), so
it's a safe drop-in for 30B too.

## Performance / utilization patches (not algorithm changes)

These three changes are pure performance: same math, much less wall-clock.

### Parallel 3-model load
*[rain_upstream/nullspace_projection_compute.py:734-760](rain_upstream/nullspace_projection_compute.py#L734-L760)*

Upstream loads base/instruct/target sequentially, all on CPU. On 80B
that's ≈ 3 × 30 min of NFS reads = 1.5 hr just for loading. We use a
`ThreadPoolExecutor(max_workers=3)` to overlap them; in practice the
slowest run dominates (~30 min from cold cache, ~3 min when files are
in OS page cache).

### `target` on GPU via `device_map="auto"`
*same edit*

The bottleneck operation is the calibration forward through the *target*
LRM. Upstream runs it CPU-only on the originally-loaded copy, which is
unusable for 80B (it has to copy back to GPU per-tensor). We load
`model_target` directly with `device_map="auto"` so accelerate shards it
across the 4 H100s (≈ 40 GB / GPU for 80B BF16) and the forward runs at
GPU speed.

### Dedupe `model_R_shared`
*[rain_upstream/nullspace_projection_compute.py:168-176](rain_upstream/nullspace_projection_compute.py#L168-L176)*

Upstream creates a *second* GPU copy of the target inside
`compute_nullspace_projections` (loading another ~80 GB of weights from
disk) just so it can keep the first copy on CPU. Now that we already
have target on GPU, the second copy is pure waste — we alias
`model_R_shared = model_target`.

Net effect on 80B Stage 1: 4+ hr → ≈ 40 min (cold) / ≈ 12 min (warm).

## Patches D, E, F never fire on 30B

D is gated on `q_proj` output width = 2 × expected (only true for
Qwen3-Next), E walks parameters and finds none in fp32 on Qwen3-30B-A3B,
F prefers qwen3_next's rotary helper which reduces to qwen3_moe's
behavior when `rotary_dim == head_dim`. So if you run **upstream**
(K1nght/RAIN-Merging) directly with `MERGE_TYPES=qkvo` on the same
30B checkpoints, you'll get the same projected pickle and α our 30B
baseline produces. The deviations from *paper-faithful* RAIN
(`qkvof` + un-truncated calibration) apply equally to upstream and to
us — see "Known deviations" at the top.

## Calibration data

Both stages use the exact calibration sets shipped in
`rain_upstream/data/`:

- **Stage 1**: `reasoning_calibration_set.json` — **150 reasoning
  prompts** (NOT 1000 as the upstream README implies; the bundled file
  has 150). Sources:
  - `open-r1/OpenR1-Math-220k` (50): math word problems / equations / geometry.
  - `open-r1/codeforces-cots` (50): competitive programming with chain-of-thought solutions.
  - `nvidia/Llama-Nemotron-Post-Training-Dataset` (50): mixed reasoning.
  Used to define the nullspace constraint — projected τ should not
  perturb the LRM's forward features at thinking-special-tokens on
  these prompts.
- **Stage 2**: `instruction_calibration_set.jsonl` — **365 IFEval
  prompts** (Instruction Following Eval). Categories: detectable_format
  (100), keywords (97), length_constraints (96), change_case (55),
  startend (51), combination (50), punctuation (43), detectable_content
  (40). Examples: *"Write in Shakespearean style. No commas."*,
  *"Include at least 12 placeholders [name], [address]."*, *"Wrap in
  double angular brackets `<<title>>`."* Only **30 / 365** mention code
  keywords. Used in the two-pass forward (anchor: target as-is, post:
  target with α_prior · τ_proj applied) to estimate per-(layer, head)
  attention deltas Δa, Δu and solve a box-constrained QP for α*.

**Domain caveat (important when reading the eval results)**: the
instruction set is *format/style compliance*, not multi-turn coding. The
α coefficients are optimized to make the merged model better at IFEval-
style instructions (write in X style, include exactly N items, wrap
output in Y) — they are *not* optimized to make it better at Roo Code
agent loops, file editing, debugging, or Python-specific coding. We
inherit upstream's calibration choices unchanged so our results are
comparable to the paper's; if your downstream task is coding agents,
swapping in code-task calibration prompts is the most-leveraged knob to
turn (more leverage than the box-tightening or FFN-merge fixes in § 8
of [RAIN_RESULTS.md](RAIN_RESULTS.md)).

## Memory budget on this cluster

| Phase | CPU RAM | GPU memory (per H100) | GPU total |
|---|---|---|---|
| 80B Stage 1 model load | 320 GB (base+instruct on CPU) | ≈ 40 GB (target sharded ×4) | 160 GB |
| 80B Stage 1 forward + per-head CG | 320 GB | ≈ 45 GB | 180 GB |
| 80B Stage 2 alpha QP | 0 GB (release CPU copies) | ≈ 45 GB | 180 GB |
| 80B Stage 3 apply | 160 GB (target reload to apply δ in place) | ≈ 0 GB (CPU-only apply) | 0 GB |
| 80B vLLM serve, TP=4 | — | ≈ 70 GB | 280 GB |

Cluster has 2 TiB RAM and 4 × H100-80GB = 320 GB total VRAM, so the
budget fits with comfortable headroom on every stage.

## Verifying the merged checkpoint

```bash
# Loads cleanly?
python -c "from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(
    './baseline_model/rain_qwen3_next_80b/unified_model_merge/rain_merged',
    torch_dtype='bfloat16', trust_remote_code=True)
print(m.config.architectures, m.config.layer_types[:8])"

# vLLM smoke (TP=4 needed for 80B; TP=2 enough for 30B)
vllm serve ./baseline_model/<dir>/unified_model_merge/rain_merged \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --max-model-len 8192 --trust-remote-code --port 8012 \
    --served-model-name rain-80b --gpu-memory-utilization 0.85

curl -sS http://localhost:8012/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"rain-80b","messages":[{"role":"user","content":"…"}],"max_tokens":256}'
```

End-to-end Roo Code eval: see [../roo_test/sh/run_rain_smoke.sh](../roo_test/sh/run_rain_smoke.sh).
