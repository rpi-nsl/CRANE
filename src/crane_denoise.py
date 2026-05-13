#!/usr/bin/env python3
"""T(δ) median-magnitude thresholding (library + CLI stat).

Zero entries with |δ| below the median, rescale survivors by 2× so
||T(δ)||₁ ≈ ||δ||₁. Architecture-agnostic — operates per tensor.
"""

import argparse
import fnmatch
import os
import sys
from typing import Iterable, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _common import CACHE_DIR, PRESETS  # noqa: E402


# ── Core function ──────────────────────────────────────────────────────────

def median_threshold(delta: torch.Tensor) -> torch.Tensor:
    """Median-magnitude threshold: zero |δ| below median, rescale survivors by 2×."""
    flat = delta.flatten()
    k = max(1, flat.numel() // 2)
    threshold = torch.kthvalue(flat.abs(), k).values
    mask = (flat.abs() > threshold).to(flat.dtype)
    return (flat * mask * 2.0).reshape(delta.shape)


# ── CLI stat helpers ───────────────────────────────────────────────────────

def _resolve_dir(model_id: str) -> str:
    return snapshot_download(
        repo_id=model_id, cache_dir=CACHE_DIR,
        allow_patterns=["*.safetensors", "*.index.json", "config.json",
                        "*.json", "*.model", "*.txt"],
    )


def _open_shards(model_dir: str):
    """Return (weight_map, file_cache_dict). weight_map: key -> shard filename."""
    import json
    index = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index) as f:
        wmap = json.load(f)["weight_map"]
    return wmap, {}


def _get_tensor(model_dir: str, wmap, cache, key: str) -> Optional[torch.Tensor]:
    fname = wmap.get(key)
    if fname is None:
        return None
    f = cache.get(fname)
    if f is None:
        f = safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu")
        cache[fname] = f
    try:
        return f.get_tensor(key)
    except Exception:
        return None


def _filter_keys(keys: Iterable[str], glob: Optional[str]):
    if not glob:
        return list(keys)
    return [k for k in keys if fnmatch.fnmatch(k, glob)]


def stat_mode(instruct_id: str, thinking_id: str, glob: Optional[str]):
    inst_dir = _resolve_dir(instruct_id)
    think_dir = _resolve_dir(thinking_id)
    print(f"  instruct dir: {inst_dir}")
    print(f"  thinking dir: {think_dir}")

    i_wmap, i_cache = _open_shards(inst_dir)
    t_wmap, t_cache = _open_shards(think_dir)

    keys = sorted(set(i_wmap) & set(t_wmap))
    keys = _filter_keys(keys, glob)
    print(f"  matched {len(keys)} shared keys (filter={glob!r})")
    print()
    hdr = f"  {'key':<60s}  {'||δ||_F':>10s}  {'||T(δ)||_F':>10s}  {'nnz':>6s}  {'max|δ|':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for k in keys:
        t_i = _get_tensor(inst_dir, i_wmap, i_cache, k)
        t_t = _get_tensor(think_dir, t_wmap, t_cache, k)
        if t_i is None or t_t is None or t_i.shape != t_t.shape:
            continue
        delta = (t_t.float() - t_i.float())
        td = median_threshold(delta)
        fro_before = float(delta.norm().item())
        fro_after = float(td.norm().item())
        nnz = float((td != 0).float().mean().item())
        mx = float(delta.abs().max().item())
        print(f"  {k:<60s}  {fro_before:>10.4f}  {fro_after:>10.4f}  "
              f"{nnz:>6.2f}  {mx:>8.4f}")


def main():
    ap = argparse.ArgumentParser(
        description="T(δ) median-threshold denoise — library + CLI stat."
    )
    ap.add_argument("--model-preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--instruct-id", default=None)
    ap.add_argument("--thinking-id", default=None)
    ap.add_argument("--stat-only", action="store_true",
                    help="Print |δ|_F / nnz stats before and after T(δ) per tensor")
    ap.add_argument("--tensor-filter", default=None,
                    help="fnmatch glob to restrict which keys are scanned "
                         '(e.g. "model.layers.0.*")')
    args = ap.parse_args()

    p = PRESETS[args.model_preset]
    inst = args.instruct_id or p["instruct_id"]
    think = args.thinking_id or p["thinking_id"]

    print("=" * 64)
    print(f"  preset: {args.model_preset}")
    print(f"  instruct: {inst}")
    print(f"  thinking: {think}")
    print("=" * 64)

    if args.stat_only:
        stat_mode(inst, think, args.tensor_filter)
        return

    # Default action: self-check on a random tensor
    print("  (No --stat-only flag given; running self-check on a random tensor.)")
    torch.manual_seed(0)
    x = torch.randn(4096)
    y = median_threshold(x)
    nnz = (y != 0).float().mean().item()
    print(f"  self-check: input ||x||_2={x.norm():.3f}, T(x) ||y||_2={y.norm():.3f}, "
          f"nnz={nnz:.3f} (expected ~0.5)")


if __name__ == "__main__":
    main()
