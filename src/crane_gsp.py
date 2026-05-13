#!/usr/bin/env python3
"""Architecture-adaptive GSP (format nullspace) projector builder.

Wraps `crane/crane_nullspace_format5_5.py` with a `--model-preset` flag.
Defaults match the upstream constants byte-for-byte (max_seq_len=4096,
svd_rank=256, threshold=1e-5, radius=3, n_tool=300, n_chat=70,
n_long_context=30, n_completion=30, seed=42).
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _common import CACHE_DIR, PRESETS  # noqa: E402

# Reuse the building blocks unchanged so default-run output is bit-identical.
from crane_nullspace_format5_5 import (  # noqa: E402
    DEFAULT_N_CHAT,
    DEFAULT_N_COMPLETION,
    DEFAULT_N_LONG_CONTEXT,
    DEFAULT_N_TOOL,
    DEFAULT_RADIUS,
    DEFAULT_SVD_RANK,
    DEFAULT_THRESHOLD,
    FORMAT_PATTERNS,
    FormatFeatureCollector,
    MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN,
    _sample_hermes,
    _sample_ultrachat,
    build_completion_calibration,
    build_format_neighborhoods,
    build_long_context_calibration,
    compute_projector_svd_with_sigma,
    locate_format_tokens,
)


def resolve_model_ids(args):
    if args.model_preset not in PRESETS:
        raise SystemExit(f"Unknown --model-preset {args.model_preset!r}; "
                         f"known: {list(PRESETS)}")
    p = PRESETS[args.model_preset]
    instruct_id = args.instruct_id or p["instruct_id"]
    return instruct_id


def main():
    ap = argparse.ArgumentParser(
        description="CRANE GSP (format nullspace) — arch-adaptive; defaults "
                    "reproduce the existing format_projectors.pt on qwen3-30b."
    )
    ap.add_argument("--model-preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--instruct-id", default=None,
                    help="Override the preset's instruct model id")
    ap.add_argument("--out", required=True, help="Output directory "
                    "(format_projectors.pt + format_stats.json written here)")
    ap.add_argument("--device", default="cuda:0")

    # GSP hyper-parameters — defaults must equal the crane script's defaults
    ap.add_argument("--svd-rank",        type=int,   default=DEFAULT_SVD_RANK)
    ap.add_argument("--threshold",       type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--radius",          type=int,   default=DEFAULT_RADIUS)
    ap.add_argument("--max-seq-len",     type=int,   default=DEFAULT_MAX_SEQ_LEN)
    ap.add_argument("--n-tool",          type=int,   default=DEFAULT_N_TOOL)
    ap.add_argument("--n-chat",          type=int,   default=DEFAULT_N_CHAT)
    ap.add_argument("--n-long-context",  type=int,   default=DEFAULT_N_LONG_CONTEXT)
    ap.add_argument("--n-completion",    type=int,   default=DEFAULT_N_COMPLETION)
    ap.add_argument("--long-context-min", type=int,  default=4000)
    ap.add_argument("--long-context-max", type=int,  default=16000)
    ap.add_argument("--completion-min",  type=int,   default=2000)
    ap.add_argument("--completion-max",  type=int,   default=12000)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--device-map", default="auto",
                    help="HF device_map for model loading (default 'auto' to "
                         "shard large models across GPUs)")
    args = ap.parse_args()

    instruct_id = resolve_model_ids(args)

    print("=" * 70)
    print("  CRANE GSP (format nullspace projectors) — arch-adaptive")
    print("=" * 70)
    print(f"  preset:           {args.model_preset}")
    print(f"  instruct:         {instruct_id}")
    print(f"  out_dir:          {args.out}")
    print(f"  svd_rank:         {args.svd_rank}")
    print(f"  threshold:        {args.threshold}")
    print(f"  radius:           {args.radius}")
    print(f"  max_seq_len:      {args.max_seq_len}")
    print(f"  n_tool / n_chat / n_long / n_completion = "
          f"{args.n_tool} / {args.n_chat} / {args.n_long_context} / {args.n_completion}")
    print(f"  seed:             {args.seed}")
    print(f"  format_patterns:  {FORMAT_PATTERNS}")

    device = torch.device(args.device)
    print(f"\n  GPU: {torch.cuda.get_device_name(device)}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device.index or 0)
        print(f"  Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    print(f"\n  Loading tokenizer: {instruct_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        instruct_id, cache_dir=CACHE_DIR, trust_remote_code=True,
    )

    # Calibration texts ─ identical order / sources to crane_nullspace_format5_5::main
    rng = random.Random(args.seed)

    print(f"\n  === Building calibration data ===")
    print(f"  [1/4] Standard tool-calling ({args.n_tool}) ...")
    texts = _sample_hermes(tokenizer, args.n_tool, rng)

    print(f"  [2/4] Multi-turn chat ({args.n_chat}) ...")
    texts += _sample_ultrachat(tokenizer, args.n_chat, rng)

    print(f"  [3/4] Synthetic long-context ({args.n_long_context}) ...")
    texts += build_long_context_calibration(
        tokenizer, args.n_long_context, rng,
        target_tokens_range=(args.long_context_min, args.long_context_max),
    )

    print(f"  [4/4] Completion-context ({args.n_completion}) ...")
    texts += build_completion_calibration(
        tokenizer, args.n_completion, rng,
        target_tokens_range=(args.completion_min, args.completion_max),
    )

    rng.shuffle(texts)
    print(f"\n  Total calibration: {len(texts)} texts")
    lengths = [len(tokenizer.encode(t)) for t in texts]
    print(f"  Token lengths: min={min(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")

    print(f"\n  Loading model: {instruct_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        instruct_id, torch_dtype=torch.bfloat16,
        device_map=args.device_map, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")

    num_layers = model.config.num_hidden_layers
    os.makedirs(args.out, exist_ok=True)

    # Step 1: locate format-token positions
    print(f"\n  === Step 1: Locating format tokens ===")
    all_positions = {}
    total_format = 0
    total_neighborhood = 0
    for idx, text in enumerate(texts):
        format_indices = locate_format_tokens(text, tokenizer)
        if not format_indices:
            continue
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_len = enc["input_ids"].shape[1]
        neighborhood = build_format_neighborhoods(
            format_indices, seq_len, args.radius
        )
        all_positions[idx] = {
            "text": text,
            "input_ids": enc["input_ids"],
            "format_indices": format_indices,
            "neighborhood": neighborhood,
        }
        total_format += len(format_indices)
        total_neighborhood += len(neighborhood)

    print(f"  Texts with format tokens: {len(all_positions)}/{len(texts)}")
    print(f"  Total format token positions: {total_format}")
    print(f"  Total neighborhood positions: {total_neighborhood}")

    # Step 2: feature collection via forward hooks
    print(f"\n  === Step 2: Collecting features ===")
    collector = FormatFeatureCollector(model, num_layers)
    model.eval()
    with torch.no_grad():
        for idx, pos_data in all_positions.items():
            input_ids = pos_data["input_ids"]
            if input_ids.shape[1] > args.max_seq_len:
                input_ids = input_ids[:, :args.max_seq_len]
                positions = [p for p in pos_data["neighborhood"] if p < args.max_seq_len]
            else:
                positions = pos_data["neighborhood"]

            collector.set_positions(positions)
            input_ids = input_ids.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=input_ids)

            del input_ids
            if (idx + 1) % 50 == 0:
                print(f"    [{idx+1}/{len(all_positions)}] texts processed")

    print(f"  All {len(all_positions)} texts processed")
    features = collector.get_features()
    collector.remove_hooks()
    print(f"  Collected features for {len(features)} components")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: per-component SVD projectors
    print(f"\n  === Step 3: Computing projectors (V_r + sigma) ===")
    projectors = {}
    stats = {"total_components": 0, "per_component": {}}

    for key in sorted(features.keys()):
        feat = features[key]
        t0 = time.time()
        V_r, sigma, r = compute_projector_svd_with_sigma(
            feat, args.svd_rank, args.threshold, device
        )
        elapsed = time.time() - t0

        if r > 0:
            projectors[key] = {"V_r": V_r, "sigma": sigma}
            stats["total_components"] += 1
            stats["per_component"][key] = {
                "n_tokens": feat.shape[0],
                "d_in": feat.shape[1],
                "effective_rank": r,
                "sigma_1": sigma[0].item(),
                "sigma_r": sigma[-1].item(),
                "ratio": (sigma[-1] / sigma[0]).item(),
                "svd_time": elapsed,
            }
            m = re.search(r"layer_(\d+)", key)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx % 12 == 0 or layer_idx == num_layers - 1:
                    print(f"    {key}: features ({feat.shape[0]}, {feat.shape[1]}) → "
                          f"rank {r}, σ1={sigma[0].item():.4f}, σr={sigma[-1].item():.6f}, "
                          f"ratio={sigma[-1].item()/sigma[0].item():.6f} [{elapsed:.1f}s]")

    proj_path = os.path.join(args.out, "format_projectors.pt")
    torch.save(projectors, proj_path)
    fsize = os.path.getsize(proj_path) / 1024**2
    print(f"\n  Saved projectors → {proj_path} ({fsize:.1f} MB)")
    print(f"  Total components: {stats['total_components']}")

    stats_path = os.path.join(args.out, "format_stats.json")
    stats_json = {
        "model_preset": args.model_preset,
        "instruct_id": instruct_id,
        "svd_rank": args.svd_rank,
        "threshold": args.threshold,
        "radius": args.radius,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "n_texts": len(texts),
        "n_tool": args.n_tool,
        "n_chat": args.n_chat,
        "n_long_context": args.n_long_context,
        "n_completion": args.n_completion,
        "long_context_range": [args.long_context_min, args.long_context_max],
        "completion_range": [args.completion_min, args.completion_max],
        "format_patterns": FORMAT_PATTERNS,
        "total_format_tokens": total_format,
        "total_neighborhood_tokens": total_neighborhood,
        "total_components": stats["total_components"],
        "per_component": stats["per_component"],
    }
    with open(stats_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print(f"\n  === GSP Projector Generation Complete ===")
    print(f"  Projectors: {proj_path}")
    print(f"  Stats:      {stats_path}")


if __name__ == "__main__":
    main()
