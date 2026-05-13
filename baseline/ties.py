#!/usr/bin/env python3
"""
Baseline: TIES-Merging (Trim, Elect Sign, Disjoint Merge).

    1. Trim: zero out low-magnitude task vector elements (keep top-k%)
    2. Elect Sign: resolve sign conflicts by majority vote
    3. Disjoint Merge: average only the elements that agree on sign

    θ_M = θ_I + α · ties_merge(θ_T - θ_I)

Reference: Yadav et al., "Resolving Interference When Merging Models", NeurIPS 2023.

Single task vector variant (instruct → thinking), which simplifies to:
trim + scale, since there's no sign conflict with only one task vector.
For completeness, the full TIES algorithm is implemented to support
future multi-model merging.
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
from model_arch import load_arch
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
from _aim import load_aim_importance, make_aim_scaler

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")
BASELINE_MODEL_DIR = os.path.join(SCRIPT_DIR, "baseline_model")

MODEL_PRESETS = {
    "qwen3-30b": {
        "model_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "ties"),
    },
    "qwen3-4b": {
        "model_instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_4b", "ties"),
    },
    "qwen3-next-80b": {
        "model_instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "model_thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_next_80b", "ties"),
    },
}


def ties_trim(task_vector: torch.Tensor, density: float) -> torch.Tensor:
    """
    Step 1 - Trim: keep only the top `density` fraction of elements by magnitude.
    The rest are zeroed out.

    density=1.0 is a no-op (keep all); density=0.0 zeroes the task vector
    entirely. Values outside [0, 1] raise ValueError rather than being
    silently clamped — ablations rely on these boundary semantics being exact.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    if density >= 1.0:
        return task_vector
    if density <= 0.0:
        return torch.zeros_like(task_vector)

    flat = task_vector.abs().flatten()
    k = int(flat.numel() * density)
    if k <= 0:
        # Fewer than one element would be kept after rounding down → zero out.
        return torch.zeros_like(task_vector)
    threshold = flat.kthvalue(flat.numel() - k + 1).values.item()

    mask = task_vector.abs() >= threshold
    return task_vector * mask


def ties_elect_sign(*task_vectors: torch.Tensor) -> torch.Tensor:
    """
    Step 2 - Elect Sign: for each element, determine the dominant sign
    across task vectors by majority vote (weighted by magnitude).

    Returns a sign tensor (+1 or -1).
    """
    if len(task_vectors) == 1:
        return task_vectors[0].sign()

    # Magnitude-weighted sign vote
    vote = torch.zeros_like(task_vectors[0])
    for tv in task_vectors:
        vote += tv.sign() * tv.abs()

    return vote.sign()


def ties_disjoint_merge(task_vectors: list[torch.Tensor], elected_sign: torch.Tensor) -> torch.Tensor:
    """
    Step 3 - Disjoint Merge: for each element, average only the task vectors
    that agree with the elected sign. Elements that disagree are excluded.
    """
    result = torch.zeros_like(task_vectors[0])
    count = torch.zeros_like(task_vectors[0])

    for tv in task_vectors:
        agree = (tv.sign() == elected_sign) & (tv != 0)
        result += tv * agree
        count += agree.float()

    # Average where count > 0
    count = count.clamp(min=1)
    return result / count


def merge_ties(
    model_instruct: str,
    model_thinking: str,
    output_dir: str,
    alpha: float = 0.3,
    density: float = 0.5,
    dry_run: bool = False,
    revision_instruct: str | None = None,
    revision_thinking: str | None = None,
    aim_importance_path: str | None = None,
    aim_omega: float = 0.4,
    strict_provenance: bool = True,
    device: str | None = None,
):
    """
    TIES merge: trim task vector, elect sign, disjoint merge, then scale by α.

    With --aim-importance, the final α-scaled merged delta is additionally
    rescaled by AIM's activation-informed relaxation factor before being
    added to the base weights. AIM is a strict post-processing step — TIES
    still trims + sign-elects + averages as usual; AIM only shrinks the
    per-input-channel magnitude of the result on channels that are
    important to the base model.
    """
    aim_scale = None
    if aim_importance_path is not None:
        importance = load_aim_importance(aim_importance_path)
        aim_scale = make_aim_scaler(importance, aim_omega)
        print(f"[AIM] loaded importance for {len(importance)} modules, ω={aim_omega}")

    dev = pick_device(device)

    print(f"\n{'='*60}")
    print(f"  TIES-Merging" + (" + AIM" if aim_scale else ""))
    print(f"  α = {alpha}, density = {density}"
          + (f",  ω = {aim_omega}" if aim_scale else ""))
    print(f"  device = {dev}")
    print(f"  Instruct: {model_instruct}" + (f" @ {revision_instruct}" if revision_instruct else ""))
    print(f"  Thinking: {model_thinking}" + (f" @ {revision_thinking}" if revision_thinking else ""))
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    dir_i, commit_i = resolve_model_snapshot(model_instruct, CACHE_DIR, revision_instruct)
    dir_t, commit_t = resolve_model_snapshot(model_thinking, CACHE_DIR, revision_thinking)
    print(f"Instruct dir: {dir_i}  (commit: {commit_i or 'N/A'})")
    print(f"Thinking dir: {dir_t}  (commit: {commit_t or 'N/A'})")

    if aim_importance_path is not None:
        validate_importance_sidecar(
            aim_importance_path, commit_i, dir_i, role="AIM",
            strict=strict_provenance,
        )

    prepare_output_dir(output_dir, dir_i)

    # Per-element stat buckets (effective-density denominator + numerators).
    total_params = 0
    eligible_params = 0
    nonzero_tv_params = 0
    injected_params = 0

    def merge_fn(key, p_i, p_t):
        nonlocal total_params, eligible_params, nonzero_tv_params, injected_params
        total_params += p_i.numel()

        if p_t is None:
            return None
        if p_i.shape != p_t.shape:
            print(f"  SKIP {key}: shape mismatch {p_i.shape} vs {p_t.shape}")
            return None

        # ties_trim's kthvalue is much faster on GPU than CPU for big shards.
        p_i_d = p_i.to(dev, non_blocking=True)
        p_t_d = p_t.to(dev, non_blocking=True)
        task_vector = p_t_d.float() - p_i_d.float()
        eligible_params += task_vector.numel()
        nonzero_tv_params += (task_vector != 0).sum().item()

        # Step 1: Trim
        trimmed_tv = ties_trim(task_vector, density)

        # Step 2: Elect sign (trivial with single task vector)
        sign = ties_elect_sign(trimmed_tv)

        # Step 3: Disjoint merge (trivial with single task vector)
        merged_tv = ties_disjoint_merge([trimmed_tv], sign)

        # θ_M = θ_I + α * merged_tv
        if alpha != 0.0:
            injected_params += (merged_tv != 0).sum().item()

        delta = alpha * merged_tv
        if aim_scale is not None:
            delta = aim_scale(key, delta)
        return (p_i_d.float() + delta).to(p_i.dtype).cpu()

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

    # Copy non-weight files and emit a fresh shard index.
    if not dry_run:
        copy_non_weight_files(dir_i, output_dir)
        write_index_json(output_dir, stats["weight_map"])

    elapsed = time.time() - t0

    # Effective density: of the params that COULD be injected (eligible AND
    # had a non-zero task vector to begin with), how many actually were.
    # Reporting against `nonzero_tv_params` is the honest denominator — if the
    # task vector was already zero, trimming never "dropped" anything.
    denom = max(1, nonzero_tv_params)
    density_actual = injected_params / denom

    print(f"\n{'='*60}")
    print(f"  TIES merge complete")
    print(f"  Total params:         {total_params:,}")
    print(f"  Eligible (same-shape): {eligible_params:,} ({100*eligible_params/max(1,total_params):.1f}% of total)")
    print(f"  Non-zero task vector: {nonzero_tv_params:,} ({100*nonzero_tv_params/max(1,eligible_params):.1f}% of eligible)")
    print(f"  Injected delta:       {injected_params:,} ({100*injected_params/max(1,eligible_params):.1f}% of eligible)")
    print(f"  Effective density:    {density_actual:.3f}  (target: {density})")
    print(f"  Tensors: {stats['merged_tensors']}/{stats['total_tensors']} merged, "
          f"{len(stats['thinking_only_tensors'])} thinking-only skipped")
    print(f"  Time: {elapsed:.1f}s")
    if not dry_run:
        print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    if not dry_run:
        cfg = {
            "method": "ties",
            "alpha": alpha,
            "density": density,
            "effective_density": round(density_actual, 4),
            "model_instruct": model_instruct,
            "model_thinking": model_thinking,
            "instruct_revision": revision_instruct,
            "thinking_revision": revision_thinking,
            "instruct_resolved_commit": commit_i,
            "thinking_resolved_commit": commit_t,
            "instruct_resolved_path": dir_i,
            "thinking_resolved_path": dir_t,
            "total_params": total_params,
            "eligible_params": eligible_params,
            "nonzero_task_vector_params": nonzero_tv_params,
            "injected_params": injected_params,
            "total_tensors": stats["total_tensors"],
            "merged_tensors": stats["merged_tensors"],
            "thinking_only_tensors": stats["thinking_only_tensors"],
            "aim_enabled": aim_scale is not None,
            "aim_importance_path": aim_importance_path,
            "aim_omega": aim_omega if aim_scale is not None else None,
            "elapsed_s": round(elapsed, 1),
        }
        with open(os.path.join(output_dir, "merge_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TIES-Merging baseline")
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
        help="Merge intensity in [-10, 10] (default: 0.3)",
    )
    parser.add_argument(
        "--density",
        type=float_in_range(0.0, 1.0, "--density"),
        default=0.5,
        help="Trim density in [0, 1] — fraction of task vector to keep (default: 0.5)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--revision-instruct", type=str, default=None,
                        help="Pin the instruct model to this HF revision (commit / tag / branch)")
    parser.add_argument("--revision-thinking", type=str, default=None,
                        help="Pin the thinking model to this HF revision (commit / tag / branch)")
    parser.add_argument("--aim-importance", type=str, default=None,
                        help="Path to AIM importance .pt (from aim_importance.py). "
                             "Enables activation-informed delta rescaling.")
    parser.add_argument("--aim-omega",
                        type=float_in_range(0.0, 1.0, "--aim-omega"),
                        default=0.4,
                        help="AIM relaxation floor ω ∈ [0, 1] (default: 0.4)")
    parser.add_argument("--strict-provenance", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Fail (not just warn) if --aim-importance was computed "
                             "against a different model snapshot. Default: enabled. "
                             "Use --no-strict-provenance to downgrade to warnings.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device for the merge math: auto/cpu/cuda/cuda:N")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    model_instruct = args.model_instruct or preset["model_instruct"]
    model_thinking = args.model_thinking or preset["model_thinking"]
    output_dir = args.output_dir or preset["output_dir"]

    merge_ties(
        model_instruct=model_instruct,
        model_thinking=model_thinking,
        output_dir=output_dir,
        alpha=args.alpha,
        density=args.density,
        dry_run=args.dry_run,
        revision_instruct=args.revision_instruct,
        revision_thinking=args.revision_thinking,
        aim_importance_path=args.aim_importance,
        aim_omega=args.aim_omega,
        strict_provenance=args.strict_provenance,
        device=args.device,
    )


if __name__ == "__main__":
    main()
