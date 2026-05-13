#!/usr/bin/env python3
"""
Build GSP format-nullspace projectors for Qwen3-Next linear_attn inner state.

Companion to crane_gsp.py.  The existing GSP file only captures hidden-state
(d_model) subspace at each layer's input_layernorm / post_attention_layernorm
outputs.  For Qwen3-Next hybrid MoE, that covers:
  - self_attn.{q,k,v}_proj   (input side == d_model) ✓
  - linear_attn.in_proj_*    (input side == d_model) ✓
  - mlp/expert gate/up       (input side == d_model) ✓
but leaves linear_attn.out_proj (input side == value_dim ≠ d_model) and
self_attn.o_proj (input side == num_q_heads*head_dim ≠ d_model) out of
Π_τ's reach.

This script adds a second set of projectors keyed `layer_{L}_linattn_inner`
by pre-hooking each Qwen3NextGatedDeltaNet.out_proj module.  The resulting
file is additive — load it alongside format_projectors.pt and route
linear_attn.out_proj to the new key in crane_merge.py.

Hybrid MoE only: full-attn layers have no `.linear_attn` attribute, so they
are silently skipped.  Expected key count on Qwen3-Next-80B = 36.

Usage:
    python crane_gsp_linattn_inner.py \
        --model-preset qwen3-next-80b \
        --out ${CRANE_DATA_DIR}/gsp_80b_next_linattn_inner
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _common import CACHE_DIR, PRESETS  # noqa: E402

from crane_nullspace_format5_5 import (  # noqa: E402
    DEFAULT_N_CHAT,
    DEFAULT_N_COMPLETION,
    DEFAULT_N_LONG_CONTEXT,
    DEFAULT_N_TOOL,
    DEFAULT_RADIUS,
    DEFAULT_SVD_RANK,
    DEFAULT_THRESHOLD,
    FORMAT_PATTERNS,
    MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN,
    _sample_hermes,
    _sample_ultrachat,
    build_completion_calibration,
    build_format_neighborhoods,
    build_long_context_calibration,
    compute_projector_svd_with_sigma,
    locate_format_tokens,
)


class LinAttnInnerCollector:
    """Pre-forward hook on every Qwen3NextGatedDeltaNet.out_proj."""

    def __init__(self, model):
        self._hooks = []
        self._features = defaultdict(list)
        self._current_positions = None

        n_registered = 0
        for l, layer in enumerate(model.model.layers):
            if not hasattr(layer, "linear_attn"):
                continue
            linear_attn = layer.linear_attn
            if not hasattr(linear_attn, "out_proj"):
                continue
            key = f"layer_{l}_linattn_inner"
            hook = linear_attn.out_proj.register_forward_pre_hook(
                self._make_pre_hook(key)
            )
            self._hooks.append(hook)
            n_registered += 1
        print(f"  Registered {n_registered} linattn_inner hooks "
              f"(linear_attn layers only)")

    def _make_pre_hook(self, key):
        def hook_fn(module, inputs):
            if self._current_positions is None or len(self._current_positions) == 0:
                return
            x = inputs[0]
            if x.dim() == 3:
                x = x[0]
            positions = [p for p in self._current_positions if p < x.shape[0]]
            if positions:
                pos_tensor = torch.tensor(positions, device=x.device)
                feat = x[pos_tensor].detach().float().cpu()
                self._features[key].append(feat)
        return hook_fn

    def set_positions(self, positions):
        self._current_positions = positions

    def get_features(self):
        return {k: torch.cat(ts, dim=0) for k, ts in self._features.items() if ts}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def resolve_instruct(args):
    if args.model_preset not in PRESETS:
        raise SystemExit(f"Unknown --model-preset {args.model_preset!r}; "
                         f"known: {list(PRESETS)}")
    p = PRESETS[args.model_preset]
    return args.instruct_id or p["instruct_id"]


def main():
    ap = argparse.ArgumentParser(
        description="CRANE GSP (linear_attn inner subspace) — Qwen3-Next only"
    )
    ap.add_argument("--model-preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--instruct-id", default=None)
    ap.add_argument("--out", required=True,
                    help="Output dir (format_projectors_linattn_inner.pt + stats.json)")
    ap.add_argument("--device", default="cuda:0",
                    help="Primary device (feature collection target)")

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
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    instruct_id = resolve_instruct(args)

    print("=" * 70)
    print("  CRANE GSP linattn_inner — Qwen3-Next Gated DeltaNet out_proj")
    print("=" * 70)
    print(f"  preset:           {args.model_preset}")
    print(f"  instruct:         {instruct_id}")
    print(f"  out_dir:          {args.out}")
    print(f"  svd_rank:         {args.svd_rank}")
    print(f"  threshold:        {args.threshold}")
    print(f"  radius:           {args.radius}")
    print(f"  max_seq_len:      {args.max_seq_len}")
    print(f"  n_tool/chat/long/completion = "
          f"{args.n_tool}/{args.n_chat}/{args.n_long_context}/{args.n_completion}")
    print(f"  seed:             {args.seed}")
    print(f"  device_map:       {args.device_map}")
    print(f"  format_patterns:  {FORMAT_PATTERNS}")

    device = torch.device(args.device)
    if device.type == "cuda":
        n = torch.cuda.device_count()
        print(f"\n  Visible GPUs: {n}")
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            print(f"    cuda:{i}  {torch.cuda.get_device_name(i)}  "
                  f"{free/1024**3:.1f}G free / {total/1024**3:.1f}G total")

    print(f"\n  Loading tokenizer: {instruct_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        instruct_id, cache_dir=CACHE_DIR, trust_remote_code=True,
    )

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

    print(f"\n  === Step 2: Collecting linattn_inner features ===")
    collector = LinAttnInnerCollector(model)
    with torch.no_grad():
        for idx, pos_data in all_positions.items():
            input_ids = pos_data["input_ids"]
            if input_ids.shape[1] > args.max_seq_len:
                input_ids = input_ids[:, :args.max_seq_len]
                positions = [p for p in pos_data["neighborhood"]
                             if p < args.max_seq_len]
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
                if layer_idx % 8 == 3 or layer_idx == num_layers - 1:
                    print(f"    {key}: features ({feat.shape[0]}, {feat.shape[1]}) → "
                          f"rank {r}, σ1={sigma[0].item():.4f}, σr={sigma[-1].item():.6f}, "
                          f"ratio={sigma[-1].item()/sigma[0].item():.6f} "
                          f"[{elapsed:.1f}s]")

    proj_path = os.path.join(args.out, "format_projectors_linattn_inner.pt")
    torch.save(projectors, proj_path)
    fsize = os.path.getsize(proj_path) / 1024**2
    print(f"\n  Saved projectors → {proj_path} ({fsize:.1f} MB)")
    print(f"  Total components: {stats['total_components']}")

    stats_path = os.path.join(args.out, "format_stats_linattn_inner.json")
    stats_json = {
        "model_preset": args.model_preset,
        "instruct_id": instruct_id,
        "hook_target": "linear_attn.out_proj (pre-forward, input[0])",
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

    print(f"\n  === GSP linattn_inner Generation Complete ===")
    print(f"  Projectors: {proj_path}")
    print(f"  Stats:      {stats_path}")


if __name__ == "__main__":
    main()
