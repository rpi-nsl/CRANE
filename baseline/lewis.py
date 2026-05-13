#!/usr/bin/env python3
"""LEWIS-Merging baseline (arXiv:2503.03874).

Builds a per-(layer, projection) TIES density schedule from |Δ| of two
LEWIS importance files (base + fine-tuned), then runs a TIES merge with
that schedule instead of a single global density. Defaults: γ=0.3, ε=0.8,
α=0.3 (paper CLI example).
"""

import argparse
import gc
import json
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRANE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "crane")
sys.path.insert(0, CRANE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _merge_io import (
    copy_non_weight_files,
    float_in_range,
    iter_merge_by_instruct_shards,
    pick_device,
    prepare_output_dir,
    resolve_model_snapshot,
    validate_importance_sidecar,
    write_index_json,
)
from _lewis import (
    build_density_schedule,
    load_lewis_importance,
    make_density_lookup,
)
from ties import ties_disjoint_merge, ties_elect_sign, ties_trim

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")
BASELINE_MODEL_DIR = os.path.join(SCRIPT_DIR, "baseline_model")

MODEL_PRESETS = {
    "qwen3-30b": {
        "model_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "lewis"),
    },
    "qwen3-4b": {
        "model_instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_4b", "lewis"),
    },
    "qwen3-next-80b": {
        "model_instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "model_thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_next_80b", "lewis"),
    },
}


def merge_lewis(
    model_instruct: str,
    model_thinking: str,
    importance_instruct: str,
    importance_thinking: str,
    output_dir: str,
    alpha: float = 0.3,
    gamma: float = 0.3,
    epsilon: float = 0.8,
    default_density: float = 0.5,
    dry_run: bool = False,
    revision_instruct: str | None = None,
    revision_thinking: str | None = None,
    strict_provenance: bool = True,
    device: str | None = None,
):
    dev = pick_device(device)
    print(f"\n{'='*60}")
    print(f"  LEWIS-Merging  (TIES with per-layer density schedule)")
    print(f"  α = {alpha},  γ = {gamma},  ε = {epsilon},  default density = {default_density}")
    print(f"  device = {dev}")
    print(f"  Instruct: {model_instruct}" + (f" @ {revision_instruct}" if revision_instruct else ""))
    print(f"  Thinking: {model_thinking}" + (f" @ {revision_thinking}" if revision_thinking else ""))
    print(f"  Importance (base):    {importance_instruct}")
    print(f"  Importance (target):  {importance_thinking}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Build the density schedule before loading weights so a broken schedule
    # (no overlap, all-zero deltas) fails fast.
    imp_base = load_lewis_importance(importance_instruct)
    imp_target = load_lewis_importance(importance_thinking)
    schedule, dbg = build_density_schedule(imp_base, imp_target, gamma=gamma, epsilon=epsilon)
    density_lookup = make_density_lookup(schedule, default_density=default_density)

    print(f"[LEWIS] base modules: {len(imp_base)},  target modules: {len(imp_target)}")
    print(f"[LEWIS] schedule buckets: {len(schedule)}  "
          f"(skipped {dbg['skipped_target_keys_not_in_base']} target keys missing from base)")
    if schedule:
        ds = list(schedule.values())
        print(f"[LEWIS] density range: [{min(ds):.3f}, {max(ds):.3f}], "
              f"mean={sum(ds)/len(ds):.3f}")

    dir_i, commit_i = resolve_model_snapshot(model_instruct, CACHE_DIR, revision_instruct)
    dir_t, commit_t = resolve_model_snapshot(model_thinking, CACHE_DIR, revision_thinking)
    print(f"Instruct dir: {dir_i}  (commit: {commit_i or 'N/A'})")
    print(f"Thinking dir: {dir_t}  (commit: {commit_t or 'N/A'})")

    # Verify each sidecar against its corresponding model snapshot.
    validate_importance_sidecar(
        importance_instruct, commit_i, dir_i, role="LEWIS base",
        strict=strict_provenance,
    )
    validate_importance_sidecar(
        importance_thinking, commit_t, dir_t, role="LEWIS target",
        strict=strict_provenance,
    )

    prepare_output_dir(output_dir, dir_i)

    # Stat buckets — also track the density actually applied per param.
    total_params = 0
    eligible_params = 0
    nonzero_tv_params = 0
    injected_params = 0
    density_used_sum = 0.0
    density_used_n = 0
    scheduled_keys = 0
    default_keys = 0

    def merge_fn(key, p_i, p_t):
        nonlocal total_params, eligible_params, nonzero_tv_params, injected_params
        nonlocal density_used_sum, density_used_n, scheduled_keys, default_keys
        total_params += p_i.numel()

        if p_t is None:
            return None
        if p_i.shape != p_t.shape:
            print(f"  SKIP {key}: shape mismatch {p_i.shape} vs {p_t.shape}")
            return None

        # Per-key density; `hit` distinguishes scheduled vs default fallback.
        d, hit = density_lookup(key)
        if hit:
            scheduled_keys += 1
        else:
            default_keys += 1
        density_used_sum += d
        density_used_n += 1

        p_i_d = p_i.to(dev, non_blocking=True)
        p_t_d = p_t.to(dev, non_blocking=True)
        task_vector = p_t_d.float() - p_i_d.float()
        eligible_params += task_vector.numel()
        nonzero_tv_params += (task_vector != 0).sum().item()

        trimmed_tv = ties_trim(task_vector, d)
        sign = ties_elect_sign(trimmed_tv)
        merged_tv = ties_disjoint_merge([trimmed_tv], sign)

        if alpha != 0.0:
            injected_params += (merged_tv != 0).sum().item()

        return (p_i_d.float() + alpha * merged_tv).to(p_i.dtype).cpu()

    stats = iter_merge_by_instruct_shards(
        dir_i=dir_i,
        dir_t=dir_t,
        merge_fn=merge_fn,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not dry_run:
        copy_non_weight_files(dir_i, output_dir)
        write_index_json(output_dir, stats["weight_map"])

    elapsed = time.time() - t0

    denom = max(1, nonzero_tv_params)
    density_actual = injected_params / denom
    avg_density_used = density_used_sum / max(1, density_used_n)

    print(f"\n{'='*60}")
    print(f"  LEWIS merge complete")
    print(f"  Total params:         {total_params:,}")
    print(f"  Eligible (same-shape): {eligible_params:,} ({100*eligible_params/max(1,total_params):.1f}% of total)")
    print(f"  Non-zero task vector: {nonzero_tv_params:,} ({100*nonzero_tv_params/max(1,eligible_params):.1f}% of eligible)")
    print(f"  Injected delta:       {injected_params:,} ({100*injected_params/max(1,eligible_params):.1f}% of eligible)")
    print(f"  Effective density:    {density_actual:.3f}  (avg scheduled density: {avg_density_used:.3f})")
    print(f"  Schedule coverage:    {scheduled_keys}/{scheduled_keys + default_keys} weight keys "
          f"({100*scheduled_keys/max(1,scheduled_keys+default_keys):.1f}%)")
    print(f"  Tensors: {stats['merged_tensors']}/{stats['total_tensors']} merged, "
          f"{len(stats['thinking_only_tensors'])} thinking-only skipped")
    print(f"  Time: {elapsed:.1f}s")
    if not dry_run:
        print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    if not dry_run:
        cfg = {
            "method": "lewis",
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "default_density": default_density,
            "effective_density": round(density_actual, 4),
            "avg_scheduled_density": round(avg_density_used, 4),
            "scheduled_weight_keys": scheduled_keys,
            "default_weight_keys": default_keys,
            "model_instruct": model_instruct,
            "model_thinking": model_thinking,
            "instruct_revision": revision_instruct,
            "thinking_revision": revision_thinking,
            "instruct_resolved_commit": commit_i,
            "thinking_resolved_commit": commit_t,
            "instruct_resolved_path": dir_i,
            "thinking_resolved_path": dir_t,
            "importance_instruct": importance_instruct,
            "importance_thinking": importance_thinking,
            "lewis_debug": dbg,
            "total_params": total_params,
            "eligible_params": eligible_params,
            "nonzero_task_vector_params": nonzero_tv_params,
            "injected_params": injected_params,
            "total_tensors": stats["total_tensors"],
            "merged_tensors": stats["merged_tensors"],
            "thinking_only_tensors": stats["thinking_only_tensors"],
            "elapsed_s": round(elapsed, 1),
        }
        with open(os.path.join(output_dir, "merge_config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="LEWIS-Merging baseline")
    parser.add_argument("--preset", default="qwen3-30b", choices=MODEL_PRESETS.keys())
    parser.add_argument("--instruct-model", "--model-instruct", dest="model_instruct",
                        type=str, default=None,
                        help="Override the preset instruct model id / local path")
    parser.add_argument("--thinking-model", "--model-thinking", dest="model_thinking",
                        type=str, default=None,
                        help="Override the preset thinking model id / local path")
    parser.add_argument(
        "--alpha",
        type=float_in_range(-10.0, 10.0, "--alpha"),
        default=0.3,
    )
    parser.add_argument(
        "--gamma",
        type=float_in_range(0.0, 1.0, "--gamma"),
        default=0.3,
        help="LEWIS density floor γ ∈ [0, 1] (default: 0.3 — paper)",
    )
    parser.add_argument(
        "--epsilon",
        type=float_in_range(0.0, 1.0, "--epsilon"),
        default=0.8,
        help="LEWIS density ceiling ε ∈ [0, 1] (default: 0.8 — paper)",
    )
    parser.add_argument(
        "--default-density",
        type=float_in_range(0.0, 1.0, "--default-density"),
        default=0.5,
        help="Density used for parameters not covered by the LEWIS schedule "
             "(embeddings, lm_head, biases, layernorms, MoE experts that "
             "never fired in either calibration). Default: 0.5",
    )
    parser.add_argument("--importance-instruct", type=str, required=True,
                        help="Path to LEWIS importance .pt for the BASE (instruct) model")
    parser.add_argument("--importance-thinking", type=str, required=True,
                        help="Path to LEWIS importance .pt for the FINE-TUNED (thinking) model")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--revision-instruct", type=str, default=None)
    parser.add_argument("--revision-thinking", type=str, default=None)
    parser.add_argument("--strict-provenance", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Fail (not just warn) if either --importance-* file was "
                             "computed against a different model snapshot. Default: "
                             "enabled. Use --no-strict-provenance for ablations.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device for the merge math: auto/cpu/cuda/cuda:N")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.gamma <= args.epsilon:
        parser.error(f"--gamma ({args.gamma}) must be <= --epsilon ({args.epsilon})")

    preset = MODEL_PRESETS[args.preset]
    model_instruct = args.model_instruct or preset["model_instruct"]
    model_thinking = args.model_thinking or preset["model_thinking"]
    output_dir = args.output_dir or preset["output_dir"]

    merge_lewis(
        model_instruct=model_instruct,
        model_thinking=model_thinking,
        importance_instruct=args.importance_instruct,
        importance_thinking=args.importance_thinking,
        output_dir=output_dir,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        default_density=args.default_density,
        dry_run=args.dry_run,
        revision_instruct=args.revision_instruct,
        revision_thinking=args.revision_thinking,
        strict_provenance=args.strict_provenance,
        device=args.device,
    )


if __name__ == "__main__":
    main()
