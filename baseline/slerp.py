#!/usr/bin/env python3
"""
Baseline: SLERP (Spherical Linear Interpolation) merge.

    θ_M = slerp(θ_I, θ_T, t)
         = sin((1-t)·Ω)/sin(Ω) · θ_I + sin(t·Ω)/sin(Ω) · θ_T

    where Ω = arccos(cos_sim(θ_I, θ_T))

Reference: Goddard et al., "Arcee's MergeKit", 2024.
           White, "Sampling Generative Networks", 2016.

SLERP interpolates along the geodesic on the hypersphere, preserving
the norm of weight tensors better than linear interpolation.
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
    write_index_json,
)

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")
BASELINE_MODEL_DIR = os.path.join(SCRIPT_DIR, "baseline_model")

MODEL_PRESETS = {
    "qwen3-30b": {
        "model_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "slerp"),
    },
    "qwen3-4b": {
        "model_instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_4b", "slerp"),
    },
    "qwen3-next-80b": {
        "model_instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "model_thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_next_80b", "slerp"),
    },
}


def slerp_tensor(t1: torch.Tensor, t2: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    For 1D tensors, SLERP is applied directly.
    For higher-dim tensors, SLERP is applied per-row (flattening all but the first dim).
    Falls back to linear interpolation when vectors are nearly parallel.
    """
    orig_shape = t1.shape
    orig_dtype = t1.dtype

    # Flatten to 2D: (num_rows, dim)
    if t1.dim() <= 1:
        v1 = t1.float().unsqueeze(0)
        v2 = t2.float().unsqueeze(0)
    else:
        v1 = t1.float().reshape(t1.shape[0], -1)
        v2 = t2.float().reshape(t2.shape[0], -1)

    # Norms (not clamped — needed to detect near-zero rows)
    n1 = v1.norm(dim=-1, keepdim=True)
    n2 = v2.norm(dim=-1, keepdim=True)

    # Rows with near-zero norm can't be normalized safely → lerp fallback
    zero_norm_mask = (n1 < eps) | (n2 < eps)

    # Safe normalization for non-zero rows
    v1_unit = v1 / n1.clamp(min=eps)
    v2_unit = v2 / n2.clamp(min=eps)

    # Cosine similarity per row
    cos_omega = (v1_unit * v2_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    # SLERP is unstable when:
    #   - cos ≈  1 (near-parallel): sin(omega) ≈ 0, coefficients 0/0
    #   - cos ≈ -1 (near-antiparallel): sin(omega) ≈ 0, coefficients blow up,
    #     and the interpolation path is ill-defined (any great circle works)
    # Fall back to plain lerp in both cases.
    # Use a generous threshold (1e-3) so we also avoid the high-coefficient regime.
    unstable_mask = (sin_omega.abs() < 1e-3) | zero_norm_mask
    safe_sin = sin_omega.clamp(min=eps)

    # SLERP coefficients (may be garbage on unstable rows — masked out below)
    c1 = torch.sin((1.0 - t) * omega) / safe_sin
    c2 = torch.sin(t * omega) / safe_sin

    # Interpolated direction (on the unit sphere), then rescale by lerp of norms
    direction = c1 * v1_unit + c2 * v2_unit
    n_interp = (1.0 - t) * n1 + t * n2
    result = direction * n_interp

    # Fallback: plain linear interpolation for unstable rows
    if unstable_mask.any():
        lerp_result = (1.0 - t) * v1 + t * v2
        result = torch.where(unstable_mask, lerp_result, result)

    return result.reshape(orig_shape).to(orig_dtype)


def merge_slerp(
    model_instruct: str,
    model_thinking: str,
    output_dir: str,
    t: float = 0.3,
    dry_run: bool = False,
    revision_instruct: str | None = None,
    revision_thinking: str | None = None,
    device: str | None = None,
):
    """
    SLERP merge: θ_M = slerp(θ_I, θ_T, t)
    """
    dev = pick_device(device)

    print(f"\n{'='*60}")
    print(f"  SLERP Merge")
    print(f"  t = {t}")
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

    prepare_output_dir(output_dir, dir_i)

    total_params = 0
    merged_params = 0

    def merge_fn(key, p_i, p_t):
        nonlocal total_params, merged_params
        total_params += p_i.numel()

        if p_t is None:
            return None
        if p_i.shape != p_t.shape:
            print(f"  SKIP {key}: shape mismatch {p_i.shape} vs {p_t.shape}")
            return None

        merged_params += p_i.numel()
        # SLERP needs per-row norms and acos — both meaningfully faster
        # on a GPU once tensors are large. Move once, compute, return CPU.
        p_i_d = p_i.to(dev, non_blocking=True)
        p_t_d = p_t.to(dev, non_blocking=True)
        return slerp_tensor(p_i_d, p_t_d, t).cpu()

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

    # Copy non-safetensors files (config, tokenizer, etc.), always overwriting.
    # Then write a fresh index json matching the shards we wrote.
    if not dry_run:
        copy_non_weight_files(dir_i, output_dir)
        write_index_json(output_dir, stats["weight_map"])

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SLERP merge complete")
    print(f"  Merged {merged_params:,} / {total_params:,} params ({100*merged_params/max(1,total_params):.1f}%)")
    print(f"  Tensors: {stats['merged_tensors']}/{stats['total_tensors']} merged, "
          f"{len(stats['thinking_only_tensors'])} thinking-only skipped")
    print(f"  Time: {elapsed:.1f}s")
    if not dry_run:
        print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    if not dry_run:
        cfg = {
            "method": "slerp",
            "t": t,
            "model_instruct": model_instruct,
            "model_thinking": model_thinking,
            "instruct_revision": revision_instruct,
            "thinking_revision": revision_thinking,
            "instruct_resolved_commit": commit_i,
            "thinking_resolved_commit": commit_t,
            "instruct_resolved_path": dir_i,
            "thinking_resolved_path": dir_t,
            "total_params": total_params,
            "merged_params": merged_params,
            "total_tensors": stats["total_tensors"],
            "merged_tensors": stats["merged_tensors"],
            "thinking_only_tensors": stats["thinking_only_tensors"],
            "elapsed_s": round(elapsed, 1),
        }
        with open(os.path.join(output_dir, "merge_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SLERP merge baseline")
    parser.add_argument("--preset", default="qwen3-30b", choices=MODEL_PRESETS.keys())
    parser.add_argument("--instruct-model", "--model-instruct", dest="model_instruct",
                        type=str, default=None,
                        help="Override the preset instruct model id / local path")
    parser.add_argument("--thinking-model", "--model-thinking", dest="model_thinking",
                        type=str, default=None,
                        help="Override the preset thinking model id / local path")
    parser.add_argument(
        "--t",
        type=float_in_range(0.0, 1.0, "--t"),
        default=0.3,
        help="Interpolation factor in [0, 1] (0=instruct, 1=thinking, default: 0.3)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--revision-instruct", type=str, default=None,
                        help="Pin the instruct model to this HF revision (commit / tag / branch)")
    parser.add_argument("--revision-thinking", type=str, default=None,
                        help="Pin the thinking model to this HF revision (commit / tag / branch)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device for the merge math: auto/cpu/cuda/cuda:N")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    model_instruct = args.model_instruct or preset["model_instruct"]
    model_thinking = args.model_thinking or preset["model_thinking"]
    output_dir = args.output_dir or preset["output_dir"]

    merge_slerp(
        model_instruct=model_instruct,
        model_thinking=model_thinking,
        output_dir=output_dir,
        t=args.t,
        dry_run=args.dry_run,
        revision_instruct=args.revision_instruct,
        revision_thinking=args.revision_thinking,
        device=args.device,
    )


if __name__ == "__main__":
    main()
