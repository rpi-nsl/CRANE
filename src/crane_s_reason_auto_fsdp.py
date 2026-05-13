#!/usr/bin/env python3
"""FSDP-parallel Pareto-Taylor S_reason for very large models (80B).

This is a **separate script** that does NOT modify `crane_s_reason_auto.py`.
It imports helpers from the original and replaces only two things:

1.  **Model loading** is done via `torch.distributed` + FSDP `FULL_SHARD`,
    so forward/backward parallelize across all ranks (as opposed to the
    `device_map="auto"` pipeline-parallel mode in the original, where only
    one GPU is active at a time).
2.  **Pareto-Taylor accumulation** uses `FSDP.summon_full_params(
    offload_to_cpu=True, rank0_only=True, with_grads=True)` to bring the
    per-chunk parameters and gradients to rank-0 CPU, where the per-element
    Pareto math runs sequentially (it's cheap compared to forward/backward).

Decoded calibration targets MUST be pre-cached. Use the single-process
`crane_s_reason_auto.py --multi-gpu` to warm caches first; this script
fails fast if `.pt` cache files are missing, rather than attempting
FSDP-mode generation (tricky + slow).

Launch::

    torchrun --nproc_per_node=4 \\
      ${CRANE_REPO_ROOT}/src/crane_s_reason_auto_fsdp.py \\
      --model-preset qwen3-next-80b \\
      --calibration-set default \\
      --layer-chunk 4 \\
      --output ${CRANE_DATA_DIR}/phase2_stats_auto_80b.json
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Reuse heavy lifting from the single-process script (no edits there).
from crane_s_reason_auto import (  # noqa: E402
    MAX_SEQ_LEN,
    PUBLIC_CALIBRATION_SETS,
    _load_targets,
    backfill_phase2_stats,
    build_target_cache_meta,
    build_target_cache_path,
    freeze_all,
    get_calibration_texts,
    resolve_snapshot,
    run_backward,
    to_device,
    unfreeze_layer_range,
)
from _common import (  # noqa: E402
    CACHE_DIR,
    PRESETS,
    ThinkingWeightLoader,
    baseline_component,
    classify_component,
    classify_subcomponent,
    detect_family,
    layer_index,
    load_arch,
)


# ── Distributed init ───────────────────────────────────────────────────────

def init_dist() -> Tuple[int, int, int]:
    """Init torch.distributed via torchrun env vars; pin current CUDA device."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _log(rank: int, *msg):
    if rank == 0:
        print(*msg, flush=True)


# ── FSDP wrap helpers ──────────────────────────────────────────────────────

def _decoder_layer_classes(model) -> set:
    """Return the set of per-layer transformer block classes to pass to the
    FSDP auto-wrap policy. Qwen3-Next may mix linear-attn and full-attn
    layer classes — grab whichever classes appear directly under
    ``model.layers.{i}``.
    """
    classes = set()
    for name, mod in model.named_modules():
        # matches exactly `model.layers.<int>` (3 dots)
        if name.startswith("model.layers.") and name.count(".") == 2:
            try:
                int(name.rsplit(".", 1)[-1])
            except ValueError:
                continue
            classes.add(type(mod))
    return classes


def wrap_with_fsdp(model, local_rank: int):
    rank = dist.get_rank()
    layer_classes = _decoder_layer_classes(model)
    _log(rank, f"  FSDP wrap classes: {sorted(c.__name__ for c in layer_classes)}")

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,   # model already identical across ranks
    )

    # Use HF gradient_checkpointing (FSDP's apply_activation_checkpointing
    # drops grads for non-expert params in Qwen3-Next MoE).
    fsdp_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    fsdp_model.config = model.config
    fsdp_model.config.use_cache = False
    return fsdp_model


# ── Taylor accumulation (FSDP version) ─────────────────────────────────────

def _remap_comp(comp: str, mode: str) -> str:
    """Optionally merge component buckets before aggregation.

    - "none"  (default): return comp as-is
    - "mode1": shared_expert → expert  (linear_attn / attention stay split)
    - "mode2": shared_expert → expert AND linear_attn → attention
    """
    if mode == "mode1":
        if comp == "shared_expert":
            return "expert"
        return comp
    if mode == "mode2":
        if comp == "shared_expert":
            return "expert"
        if comp == "linear_attn":
            return "attention"
        return comp
    return comp


def accumulate_pareto_fsdp(
    fsdp_model,
    toks_R, toks_T,
    think_loader: ThinkingWeightLoader,
    layer_chunk: int,
    comp_unify_mode: str = "none",
) -> Tuple[dict, dict]:
    """FSDP-parallel forward/backward + rank-0 Pareto math with
    summon_full_params(offload_to_cpu=True, rank0_only=True)."""

    rank = dist.get_rank()
    n_R = len(toks_R)
    n_T = len(toks_T)

    comp_agg = defaultdict(lambda: {
        "pareto_sum": 0.0,
        "cost_sum":   0.0,   # Σ (imp_R² + imp_T²) — for auto_alpha
        "delta_sq":   0.0,   # Σ δ² — for diagnostics
        "numel": 0,
        "impR_sum": 0.0,
        "impT_sum": 0.0,
        "theta_sq": 0.0,
    })
    sub_numel = defaultdict(int)

    num_layers = fsdp_model.config.num_hidden_layers
    num_chunks = math.ceil(num_layers / layer_chunk)

    freeze_all(fsdp_model)
    fsdp_model.config.use_cache = False

    for ci in range(num_chunks):
        lo = ci * layer_chunk
        hi = min(lo + layer_chunk, num_layers)
        t0 = time.time()
        unfreeze_layer_range(fsdp_model, lo, hi)

        # ── D_R backward (all ranks cooperatively via FSDP) ────────────────
        run_backward(fsdp_model, toks_R)

        # Per-layer summon (one layer at a time, bounded memory). FSDP doesn't
        # support with_grads + use_orig_params + offload_to_cpu together, so
        # keep results on GPU and stash gR to CPU manually.
        grad_R_cpu: Dict[str, torch.Tensor] = {}
        for i in range(lo, hi):
            layer_module = fsdp_model.model.layers[i]
            with FSDP.summon_full_params(
                layer_module, with_grads=True, writeback=False,
            ):
                prefix = f"model.layers.{i}."
                for name, p in layer_module.named_parameters():
                    if p.grad is None:
                        continue
                    grad_R_cpu[prefix + name] = (
                        (p.grad.detach().float() / n_R)
                        .to(dtype=torch.bfloat16, device="cpu")
                    )

        fsdp_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # ── D_T backward ───────────────────────────────────────────────────
        run_backward(fsdp_model, toks_T)

        missing = 0
        for i in range(lo, hi):
            layer_module = fsdp_model.model.layers[i]
            with FSDP.summon_full_params(
                layer_module, with_grads=True, writeback=False,
            ):
                prefix = f"model.layers.{i}."
                # All ranks see the same summoned data; only rank 0 records.
                if rank != 0:
                    continue
                p_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                for name, p in layer_module.named_parameters():
                    full = prefix + name
                    gR_cpu = grad_R_cpu.get(full)
                    if gR_cpu is None or p.grad is None:
                        missing += 1
                        continue
                    t_think = think_loader.get(
                        full, device=p_device, dtype=torch.bfloat16,
                    )
                    if t_think is None or t_think.shape != p.shape:
                        missing += 1
                        continue
                    delta = t_think - p.data.detach().to(torch.bfloat16)
                    del t_think
                    delta_sq = float((delta.float() ** 2).sum().item())
                    gR = gR_cpu.to(device=p_device, non_blocking=True)
                    gT = (p.grad.detach() / n_T).to(torch.bfloat16)

                    imp_R = -(gR * delta)
                    imp_T = -(gT * delta)
                    del delta, gR, gT
                    pareto = torch.clamp(torch.minimum(imp_R, imp_T), min=0.0)

                    # cost_sum contribution for auto_alpha: Σ (imp_R² + imp_T²)
                    cost_contrib = float(
                        (imp_R.float() ** 2).sum().item()
                        + (imp_T.float() ** 2).sum().item()
                    )

                    comp = _remap_comp(classify_component(full), comp_unify_mode)
                    sub = classify_subcomponent(full)
                    layer = layer_index(full)
                    key = (comp, layer)
                    comp_agg[key]["pareto_sum"] += float(pareto.float().sum().item())
                    comp_agg[key]["cost_sum"]   += cost_contrib
                    comp_agg[key]["delta_sq"]   += delta_sq
                    comp_agg[key]["impR_sum"]  += float(imp_R.float().sum().item())
                    comp_agg[key]["impT_sum"]  += float(imp_T.float().sum().item())
                    comp_agg[key]["numel"]     += p.numel()
                    comp_agg[key]["theta_sq"]  += float(
                        (p.data.detach().float() ** 2).sum().item()
                    )
                    sub_numel[sub]             += p.numel()

                    del imp_R, imp_T, pareto

        grad_R_cpu.clear()
        fsdp_model.zero_grad(set_to_none=True)
        for i in range(lo, hi):
            for p in fsdp_model.model.layers[i].parameters():
                p.requires_grad_(False)
                p.grad = None
        torch.cuda.empty_cache()
        gc.collect()

        if rank == 0:
            gpu0 = torch.cuda.memory_allocated(torch.cuda.current_device()) / 1024**3
            print(
                f"    chunk {ci+1}/{num_chunks} (layers {lo}-{hi-1}): "
                f"{time.time()-t0:.1f}s, missing={missing}, rank0_GPU={gpu0:.1f} GB",
                flush=True,
            )
        dist.barrier()

    return dict(comp_agg), dict(sub_numel)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--instruct-id", default=None)
    ap.add_argument("--thinking-id", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--calibration-set",
        default="default",
        help="default = canonical D_R/D_T. Override with public_<name> for any "
             "${CRANE_DATA_DIR}/calibration_public/<name>.jsonl (§A.5 robustness sweep).",
    )
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-new-r", type=int, default=None)
    ap.add_argument("--max-new-t", type=int, default=None)
    ap.add_argument("--layer-chunk", type=int, default=None)
    ap.add_argument(
        "--targets-cache",
        default="${CRANE_DATA_DIR}/calib_targets",
    )
    ap.add_argument(
        "--comp-unify-mode",
        choices=["none", "mode1", "mode2"],
        default="none",
        help=("Pre-aggregation component merging. "
              "none = no merging (6 comps: attention, linear_attn, expert, shared_expert, router, norm). "
              "mode1 = shared_expert merged into expert (5 comps, linear_attn/attention still split). "
              "mode2 = additionally linear_attn merged into attention (4 comps)."),
    )
    args = ap.parse_args()

    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")

    preset = PRESETS[args.model_preset]
    instruct_id = args.instruct_id or preset["instruct_id"]
    thinking_id = args.thinking_id or preset["thinking_id"]

    # Snapshots (all ranks — safe, hub cache is shared)
    inst_dir = resolve_snapshot(instruct_id)
    think_dir = resolve_snapshot(thinking_id)
    arch = load_arch(instruct_id, cache_dir=CACHE_DIR)
    family = detect_family(arch)
    BASELINE = baseline_component(family)
    layer_chunk = (
        args.layer_chunk if args.layer_chunk is not None
        else int(preset["default_layer_chunk"])
    )
    max_new_r = args.max_new_r if args.max_new_r is not None else args.max_new_tokens
    max_new_t = args.max_new_t if args.max_new_t is not None else 2048

    _log(rank, "=" * 64)
    _log(rank, f"  CRANE FSDP S_reason — world_size={world_size}")
    _log(rank, f"  preset: {args.model_preset}   family: {family}   "
                f"baseline: {BASELINE}")
    _log(rank, f"  arch: {arch.num_layers}L, hidden={arch.hidden_dim}, "
                f"experts={arch.num_experts}, model_type={arch.model_type}")
    _log(rank, f"  layer_chunk: {layer_chunk}   calibration: {args.calibration_set}")
    _log(rank, f"  output: {args.output}")
    _log(rank, "=" * 64)

    # ── Require cached targets ───────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(instruct_id, cache_dir=CACHE_DIR)
    d_r_texts, d_t_texts, d_t_label = get_calibration_texts(tok, args.calibration_set)

    dr_meta = build_target_cache_meta(
        role="d_r_thinking", model_preset=args.model_preset,
        calibration_set=args.calibration_set, model_id=thinking_id,
        tokenizer_id=instruct_id, prompts=d_r_texts,
        max_new_tokens=max_new_r, max_seq_len=args.max_seq_len,
    )
    dt_meta = build_target_cache_meta(
        role="d_t_instruct", model_preset=args.model_preset,
        calibration_set=args.calibration_set, model_id=instruct_id,
        tokenizer_id=instruct_id, prompts=d_t_texts,
        max_new_tokens=max_new_t, max_seq_len=args.max_seq_len,
    )
    dr_path = build_target_cache_path(args.targets_cache, dr_meta)
    dt_path = build_target_cache_path(args.targets_cache, dt_meta)

    if not (os.path.exists(dr_path) and os.path.exists(dt_path)):
        if rank == 0:
            print(
                f"\n  ERROR: required target caches missing. Warm them first via:\n"
                f"    python crane_s_reason_auto.py --model-preset {args.model_preset} "
                f"--multi-gpu --calibration-set {args.calibration_set} "
                f"--output /tmp/warm_decode_only.json\n"
                f"  (s_reason will error out after decode; that's fine — caches "
                f"will exist.)\n"
                f"  Missing: DR={not os.path.exists(dr_path)} DT={not os.path.exists(dt_path)}",
                flush=True,
            )
        dist.destroy_process_group()
        sys.exit(1)

    _log(rank, f"  D_R cache: {dr_path}")
    _log(rank, f"  D_T cache: {dt_path}")

    triples_R = _load_targets(dr_path)
    triples_T = _load_targets(dt_path)
    _log(rank, f"  |D_R| = {len(triples_R)} batches, |D_T| = {len(triples_T)} batches")

    # ── Load model on CPU, then FSDP-shard to GPUs ───────────────────────
    _log(rank, "\n  Loading instruct model (CPU, low_cpu_mem_usage=True) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        instruct_id, cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _log(rank, f"  loaded in {time.time()-t0:.1f}s")

    _log(rank, "\n  Wrapping with FSDP FULL_SHARD ...")
    t0 = time.time()
    fsdp_model = wrap_with_fsdp(model, local_rank)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    _log(rank, f"  FSDP wrap in {time.time()-t0:.1f}s "
                f"(rank0 GPU {torch.cuda.memory_allocated(device)/1024**3:.1f} GB)")

    # Move triples to current device
    toks_R = to_device(triples_R, device)
    toks_T = to_device(triples_T, device)

    think_loader = ThinkingWeightLoader(think_dir, num_experts=arch.num_experts)
    _log(rank, f"\n  thinking weight_map: {len(think_loader.weight_map)} keys "
                f"(num_experts={arch.num_experts})")

    _log(rank, f"\n  Accumulating per-element Pareto-Taylor (FSDP parallel; "
                f"comp_unify_mode={args.comp_unify_mode}) ...")
    comp_agg, sub_numel = accumulate_pareto_fsdp(
        fsdp_model, toks_R, toks_T, think_loader, layer_chunk=layer_chunk,
        comp_unify_mode=args.comp_unify_mode,
    )

    # ── Rank 0 post-process + save JSON ──────────────────────────────────
    if rank == 0:
        # Per-layer S_raw and per-layer S_reason (normalized to BASELINE).
        s_raw_pl: Dict[Tuple[str, int], float] = {}
        for (comp, layer), v in comp_agg.items():
            if layer is None:
                continue
            tn = math.sqrt(v.get("theta_sq", 0.0))
            s_raw_pl[(comp, layer)] = (v["pareto_sum"] / tn) if tn > 0 else 0.0

        layers = sorted({l for (_, l) in s_raw_pl})
        components_seen = sorted({c for (c, _) in s_raw_pl})

        s_reason_pl: Dict[int, Dict[str, float]] = {}
        for l in layers:
            baseline_raw_l = s_raw_pl.get((BASELINE, l), 0.0) or 1e-12
            layer_dict = {c: s_raw_pl.get((c, l), 0.0) / baseline_raw_l
                          for c in components_seen}
            layer_dict[BASELINE] = 1.0
            s_reason_pl[l] = layer_dict

        # Aggregated per-component (across layers)
        agg_global = defaultdict(lambda: {
            "pareto_sum": 0.0, "cost_sum": 0.0, "delta_sq": 0.0, "theta_sq": 0.0
        })
        for (comp, layer), v in comp_agg.items():
            agg_global[comp]["pareto_sum"] += v["pareto_sum"]
            agg_global[comp]["cost_sum"]   += v["cost_sum"]
            agg_global[comp]["delta_sq"]   += v["delta_sq"]
            agg_global[comp]["theta_sq"]   += v["theta_sq"]
        s_raw = {
            c: (v["pareto_sum"] / math.sqrt(v["theta_sq"])) if v["theta_sq"] > 0 else 0.0
            for c, v in agg_global.items()
        }
        baseline_raw = s_raw.get(BASELINE, 0.0) or 1e-12
        s_reason = {c: v / baseline_raw for c, v in s_raw.items()}
        s_reason[BASELINE] = 1.0

        print("\n  Per-component (aggregated across layers)")
        for c in sorted(s_reason):
            print(f"    {c:<14s} {s_reason[c]:>8.4f}")

        # ── auto_alpha (data-driven Pareto-Taylor α) ─────────────────────
        # α_auto = L_total / Q_total with S(c) = S_reason(c), unit S_arch.
        #   L_per(c) = S(c) · pareto_sum(c)
        #   Q_per(c) = S(c)² · cost_sum(c)
        eps = 1e-12
        L_per, Q_per, pareto_per, cost_per = {}, {}, {}, {}
        for c, v in agg_global.items():
            S = float(s_reason.get(c, 0.0))
            pareto_per[c] = float(v["pareto_sum"])
            cost_per[c]   = float(v["cost_sum"])
            L_per[c] = S * pareto_per[c]
            Q_per[c] = (S * S) * cost_per[c]
        L_total = sum(L_per.values())
        Q_total = sum(Q_per.values())
        if Q_total < eps:
            alpha_auto = None
            fallback_reason = f"Q_total={Q_total:.3e} < {eps:.3e}"
        else:
            alpha_auto = L_total / Q_total
            fallback_reason = None
        print(f"\n  auto_alpha = {alpha_auto!r}  (L={L_total:.3e}, Q={Q_total:.3e})")

        stats = backfill_phase2_stats(s_reason, dict(sub_numel))
        stats["per_layer_S_reason"] = {
            str(l): {c: float(v) for c, v in d.items()}
            for l, d in s_reason_pl.items()
        }
        stats["per_component_S_reason"] = {c: float(v) for c, v in s_reason.items()}
        stats["auto_alpha"] = {
            "alpha_auto":                 alpha_auto,
            "L_total":                    float(L_total),
            "Q_total":                    float(Q_total),
            "L_per_component":            {c: float(v) for c, v in L_per.items()},
            "Q_per_component":            {c: float(v) for c, v in Q_per.items()},
            "pareto_sum_per_component":   pareto_per,
            "cost_sum_per_component":     cost_per,
            "fallback_reason":            fallback_reason,
            "note": "S_arch=1.0 for all components (80B-Next); "
                    "L = Σ S_reason·pareto_sum, Q = Σ S_reason²·cost_sum",
        }
        # Also expose at top level for merge-script convenience
        if alpha_auto is not None:
            stats["recommended_alpha"] = float(alpha_auto)
        stats["metadata"] = {
            "source": "crane_s_reason_auto_fsdp.py",
            "model_preset": args.model_preset,
            "instruct_id": instruct_id,
            "thinking_id": thinking_id,
            "family": family,
            "baseline_component": BASELINE,
            "calibration_set": args.calibration_set,
            "comp_unify_mode": args.comp_unify_mode,
            "num_d_r": len(d_r_texts),
            "num_d_t": len(d_t_texts),
            "world_size": world_size,
            "layer_chunk": layer_chunk,
            "num_layers": arch.num_layers,
            "num_experts": arch.num_experts,
            "model_type": arch.model_type,
        }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  wrote {args.output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
