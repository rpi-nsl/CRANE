#!/usr/bin/env python3
"""
Baseline: Task Arithmetic (TA) merge.

    θ_M = θ_I + α · (θ_T - θ_I)

Reference: Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023.

Merges a thinking model into an instruct model by adding a scaled task vector.
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
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "task_arithmetic"),
    },
    "qwen3-4b": {
        "model_instruct": "Qwen/Qwen3-4B-Instruct-2507",
        "model_thinking": "Qwen/Qwen3-4B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_4b", "task_arithmetic"),
    },
    "qwen3-next-80b": {
        "model_instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "model_thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "qwen3_next_80b", "task_arithmetic"),
    },
}


def merge_task_arithmetic(
    model_instruct: str,
    model_thinking: str,
    output_dir: str,
    alpha: float = 0.3,
    dry_run: bool = False,
    revision_instruct: str | None = None,
    revision_thinking: str | None = None,
    aim_importance_path: str | None = None,
    aim_omega: float = 0.4,
    strict_provenance: bool = True,
    device: str | None = None,
):
    """
    Task Arithmetic merge: θ_M = θ_I + α · (θ_T - θ_I)

    With --aim-importance, the per-parameter delta is additionally rescaled
    by AIM's input-channel relaxation factor before being applied:
        Δ_final = α · (θ_T − θ_I) ⊙ (1 − s · (1 − ω))
    where s ∈ [0,1] is the normalized input-activation magnitude of the
    base (instruct) model on a calibration set.
    """
    aim_scale = None
    if aim_importance_path is not None:
        importance = load_aim_importance(aim_importance_path)
        aim_scale = make_aim_scaler(importance, aim_omega)
        print(f"[AIM] loaded importance for {len(importance)} modules, ω={aim_omega}")

    dev = pick_device(device)

    print(f"\n{'='*60}")
    print(f"  Task Arithmetic Merge" + (" + AIM" if aim_scale else ""))
    print(f"  α = {alpha}" + (f",  ω = {aim_omega}" if aim_scale else ""))
    print(f"  device = {dev}")
    print(f"  Instruct: {model_instruct}" + (f" @ {revision_instruct}" if revision_instruct else ""))
    print(f"  Thinking: {model_thinking}" + (f" @ {revision_thinking}" if revision_thinking else ""))
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Resolve models — also capture the resolved snapshot commit so the
    # exact upstream revision can be recorded in merge_config.json.
    dir_i, commit_i = resolve_model_snapshot(model_instruct, CACHE_DIR, revision_instruct)
    dir_t, commit_t = resolve_model_snapshot(model_thinking, CACHE_DIR, revision_thinking)
    print(f"Instruct dir: {dir_i}  (commit: {commit_i or 'N/A'})")
    print(f"Thinking dir: {dir_t}  (commit: {commit_t or 'N/A'})")

    # AIM importance is captured on the BASE (instruct) model, so verify
    # the sidecar references that snapshot. Mismatched provenance here
    # produces a well-formed but subtly-wrong merge; fail loudly.
    if aim_importance_path is not None:
        validate_importance_sidecar(
            aim_importance_path, commit_i, dir_i, role="AIM",
            strict=strict_provenance,
        )

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

        # Task Arithmetic: θ_M = θ_I + α * (θ_T - θ_I)
        # Move both shards onto the compute device once and do all the
        # math there. The result is sent back to CPU for safetensors save.
        p_i_d = p_i.to(dev, non_blocking=True)
        p_t_d = p_t.to(dev, non_blocking=True)
        task_vector = p_t_d.float() - p_i_d.float()
        delta = alpha * task_vector
        if aim_scale is not None:
            delta = aim_scale(key, delta)
        merged_params += p_i.numel()
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

    # Copy non-safetensors files (config, tokenizer, etc.), always overwriting.
    # Then write a fresh index json that matches the shard layout we produced.
    if not dry_run:
        copy_non_weight_files(dir_i, output_dir)
        write_index_json(output_dir, stats["weight_map"])

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Task Arithmetic merge complete")
    print(f"  Merged {merged_params:,} / {total_params:,} params ({100*merged_params/max(1,total_params):.1f}%)")
    print(f"  Tensors: {stats['merged_tensors']}/{stats['total_tensors']} merged, "
          f"{len(stats['thinking_only_tensors'])} thinking-only skipped")
    print(f"  Time: {elapsed:.1f}s")
    if not dry_run:
        print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Save merge config
    if not dry_run:
        cfg = {
            "method": "task_arithmetic",
            "alpha": alpha,
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
            "aim_enabled": aim_scale is not None,
            "aim_importance_path": aim_importance_path,
            "aim_omega": aim_omega if aim_scale is not None else None,
            "elapsed_s": round(elapsed, 1),
        }
        with open(os.path.join(output_dir, "merge_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Task Arithmetic merge baseline")
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
                             "against a different model snapshot than the one being "
                             "merged. Default: enabled. Disable for cross-revision "
                             "ablations: --no-strict-provenance.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device for the merge math: auto/cpu/cuda/cuda:N "
                             "(default auto = cuda:0 if available)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    model_instruct = args.model_instruct or preset["model_instruct"]
    model_thinking = args.model_thinking or preset["model_thinking"]
    output_dir = args.output_dir or preset["output_dir"]

    merge_task_arithmetic(
        model_instruct=model_instruct,
        model_thinking=model_thinking,
        output_dir=output_dir,
        alpha=args.alpha,
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
