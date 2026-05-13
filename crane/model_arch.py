#!/usr/bin/env python3
"""
Load architecture constants from model config.json instead of hardcoding.

Usage:
    from model_arch import load_arch
    arch = load_arch("Qwen/Qwen3-30B-A3B-Instruct-2507", cache_dir="${HF_HOME}")
    print(arch.num_layers, arch.num_q_heads, arch.head_dim)
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelArch:
    """Architecture constants derived from config.json."""
    num_layers: int           # num_hidden_layers
    hidden_dim: int           # hidden_size
    head_dim: int             # head_dim (or hidden_size // num_attention_heads)
    num_q_heads: int          # num_attention_heads
    num_kv_heads: int         # num_key_value_heads
    num_experts: int           # num_experts (0 for dense models)
    num_active_experts: int    # num_experts_per_tok
    moe_intermediate_size: int # moe_intermediate_size (expert inner dim)
    intermediate_size: int     # intermediate_size (shared expert / dense FFN)
    vocab_size: int
    model_type: str = ""      # model_type from config.json ("qwen3", "qwen3_moe", "qwen3_next", ...)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0


def _find_config_json(model_name: str, cache_dir: str) -> str:
    """Locate config.json in HuggingFace cache structure."""
    # Try direct path first (already downloaded)
    slug = model_name.replace("/", "--")
    models_dir = os.path.join(cache_dir, f"models--{slug}")
    if os.path.isdir(models_dir):
        snapshots = os.path.join(models_dir, "snapshots")
        if os.path.isdir(snapshots):
            for snap in sorted(os.listdir(snapshots)):
                cfg = os.path.join(snapshots, snap, "config.json")
                if os.path.isfile(cfg):
                    return cfg
    # Try hub/ subdirectory
    hub_dir = os.path.join(cache_dir, "hub", f"models--{slug}")
    if os.path.isdir(hub_dir):
        snapshots = os.path.join(hub_dir, "snapshots")
        if os.path.isdir(snapshots):
            for snap in sorted(os.listdir(snapshots)):
                cfg = os.path.join(snapshots, snap, "config.json")
                if os.path.isfile(cfg):
                    return cfg
    raise FileNotFoundError(
        f"Cannot find config.json for {model_name} in {cache_dir}")


def load_arch(model_name: str, cache_dir: str = "${HF_HOME}") -> ModelArch:
    """Load architecture from model config.json."""
    cfg_path = _find_config_json(model_name, cache_dir)
    with open(cfg_path) as f:
        c = json.load(f)

    num_q = c["num_attention_heads"]
    hidden = c["hidden_size"]
    head_dim = c.get("head_dim", hidden // num_q)

    return ModelArch(
        num_layers=c["num_hidden_layers"],
        hidden_dim=hidden,
        head_dim=head_dim,
        num_q_heads=num_q,
        num_kv_heads=c.get("num_key_value_heads", num_q),
        num_experts=c.get("num_experts", 0),
        num_active_experts=c.get("num_experts_per_tok", 0),
        moe_intermediate_size=c.get("moe_intermediate_size", 0),
        intermediate_size=c.get("intermediate_size", 0),
        vocab_size=c.get("vocab_size", 0),
        model_type=c.get("model_type", ""),
    )
