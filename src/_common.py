"""
Shared utilities for src/ — architecture-adaptive CRANE components.

Supports three Qwen3 families:
  - dense  (Qwen3-4B-2507, Qwen3ForCausalLM, model_type="qwen3")
  - moe    (Qwen3-30B-A3B-2507, Qwen3MoeForCausalLM, model_type="qwen3_moe")
  - next   (Qwen3-Next-80B-A3B, Qwen3NextForCausalLM, model_type="qwen3_next")

Provides:
  - PRESETS                 : model-id + defaults per preset
  - detect_family(arch)     : "dense" | "moe" | "next"
  - classify_component/subcomponent(name) : one regex set covering all three
  - SUBCOMPS_BY_GROUP       : mapping used by backfill_phase2_stats
  - ThinkingWeightLoader    : lazy safetensors reader with optional fused-expert
                              reconstruction (for 30B instruct-side keys only)
  - crane helpers           : re-exported from the crane/ directory via sys.path
"""

import json
import os
import re
import sys
from typing import Dict, Literal, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

# ── Locate sibling crane/ directory and import its helpers ──────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CRANE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "crane"))
if _CRANE_DIR not in sys.path:
    sys.path.insert(0, _CRANE_DIR)

# NOTE: we import from crane/ rather than copying to avoid silent drift.
from model_arch import ModelArch, load_arch  # noqa: E402


# ── Presets ─────────────────────────────────────────────────────────────────

Family = Literal["dense", "moe", "next"]

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")

PRESETS: Dict[str, Dict[str, object]] = {
    "qwen3-4b": {
        "instruct_id": "Qwen/Qwen3-4B-Instruct-2507",
        "thinking_id": "Qwen/Qwen3-4B-Thinking-2507",
        "family_hint": "dense",
        "default_layer_chunk": 8,
        "baseline_component": "ffn",
    },
    "qwen3-30b": {
        "instruct_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "thinking_id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "family_hint": "moe",
        "default_layer_chunk": 2,
        "baseline_component": "expert",
    },
    "qwen3-next-80b": {
        "instruct_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "thinking_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "family_hint": "next",
        "default_layer_chunk": 1,
        "baseline_component": "expert",
    },
}


def detect_family(arch: ModelArch) -> Family:
    """Choose dense / moe / next from ModelArch."""
    mt = (arch.model_type or "").lower()
    if mt == "qwen3_next":
        return "next"
    if arch.is_moe:
        return "moe"
    return "dense"


def baseline_component(family: Family) -> str:
    """Which component's S_raw is the normalization denominator."""
    return "ffn" if family == "dense" else "expert"


# ── Component / subcomponent classification ─────────────────────────────────

def classify_component(name: str) -> str:
    if "linear_attn" in name:
        return "linear_attn"
    if "self_attn" in name:
        return "attention"
    if "mlp.shared_expert_gate" in name:
        return "router"
    if "mlp.shared_expert" in name:
        return "shared_expert"
    if "mlp.gate.weight" in name and "experts" not in name:
        return "router"
    if "mlp.experts" in name:
        return "expert"
    if "mlp." in name:
        return "ffn"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "norm"
    return "other"


def classify_subcomponent(name: str) -> str:
    # Linear attention (Qwen3-Next Gated DeltaNet)
    if "linear_attn.in_proj_qkvz" in name: return "linattn_qkvz"
    if "linear_attn.in_proj_ba"  in name: return "linattn_ba"
    if "linear_attn.conv1d"      in name: return "linattn_conv"
    if "linear_attn.out_proj"    in name: return "linattn_out"
    if "linear_attn.norm"        in name: return "linattn_norm"
    if "linear_attn.A_log"       in name: return "linattn_A_log"
    if "linear_attn.dt_bias"     in name: return "linattn_dt_bias"

    # Standard self-attention
    if "self_attn.q_norm" in name: return "attn_q_norm"
    if "self_attn.k_norm" in name: return "attn_k_norm"
    if "self_attn.q_proj" in name: return "attn_q"
    if "self_attn.k_proj" in name: return "attn_k"
    if "self_attn.v_proj" in name: return "attn_v"
    if "self_attn.o_proj" in name: return "attn_o"

    # MoE routed experts (fused 30B instruct: gate_up_proj / down_proj ;
    # unfused 30B thinking and Next: {gate,up,down}_proj per expert)
    if "mlp.experts" in name:
        if "down_proj" in name: return "expert_down"
        return "expert_up"  # gate_proj, up_proj, or fused gate_up_proj

    # Qwen3-Next shared expert
    if "mlp.shared_expert_gate" in name: return "shared_gate"
    if "mlp.shared_expert.gate_proj" in name: return "shared_ffn_gate"
    if "mlp.shared_expert.up_proj"   in name: return "shared_ffn_up"
    if "mlp.shared_expert.down_proj" in name: return "shared_ffn_down"

    # Dense FFN
    if "mlp.gate_proj" in name: return "ffn_gate"
    if "mlp.up_proj"   in name: return "ffn_up"
    if "mlp.down_proj" in name: return "ffn_down"

    # Router
    if "mlp.gate.weight" in name: return "router_gate"

    if "norm" in name:
        return "norm"
    return "other"


SUBCOMPS_BY_GROUP: Dict[str, list] = {
    "attention":     ["attn_q", "attn_k", "attn_v", "attn_o", "attn_q_norm", "attn_k_norm"],
    "linear_attn":   ["linattn_qkvz", "linattn_ba", "linattn_conv", "linattn_out",
                      "linattn_norm", "linattn_A_log", "linattn_dt_bias"],
    "ffn":           ["ffn_gate", "ffn_up", "ffn_down"],
    "expert":        ["expert_down", "expert_up"],
    "shared_expert": ["shared_ffn_gate", "shared_ffn_up", "shared_ffn_down"],
    "router":        ["router_gate", "shared_gate"],
    "norm":          ["norm"],
}


def layer_index(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ── Thinking-side weight lazy loader ────────────────────────────────────────

class ThinkingWeightLoader:
    """
    Lazy reader for the thinking-side safetensors shards. Exposes .get(name)
    that returns the tensor corresponding to the instruct-model parameter
    `name` (or None if unavailable).

    Fused-expert reconstruction path is only used when `num_experts > 0` AND
    the requested key matches `mlp.experts.(gate_up_proj|down_proj)` — this
    is the Qwen3-30B instruct-side naming. For dense (4B) and Qwen3-Next
    (two-sided unfused) models, _raw direct lookup always hits.
    """

    _FUSED_KEY = re.compile(r"(.*\.mlp\.experts)\.(down_proj|gate_up_proj)$")

    def __init__(self, model_dir: str, num_experts: int):
        self.model_dir = model_dir
        self.num_experts = num_experts
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._file_cache: Dict[str, object] = {}

    def _file(self, fname: str):
        f = self._file_cache.get(fname)
        if f is None:
            f = safe_open(
                os.path.join(self.model_dir, fname),
                framework="pt", device="cpu",
            )
            self._file_cache[fname] = f
        return f

    def _raw(self, key: str, device, dtype):
        fname = self.weight_map.get(key)
        if fname is None:
            return None
        try:
            return self._file(fname).get_tensor(key).to(device=device, dtype=dtype)
        except Exception:
            return None

    def get(self, name: str, device: torch.device, dtype=torch.float32):
        direct = self._raw(name, device, dtype)
        if direct is not None:
            return direct

        # Only attempt fused reconstruction when the model has MoE experts
        # AND the key matches the 30B instruct-side fused naming.
        if self.num_experts <= 0:
            return None
        m = self._FUSED_KEY.match(name)
        if m is None:
            return None
        prefix, kind = m.group(1), m.group(2)

        if kind == "down_proj":
            parts = []
            for e in range(self.num_experts):
                t = self._raw(f"{prefix}.{e}.down_proj.weight", device, dtype)
                if t is None:
                    return None
                parts.append(t)
            return torch.stack(parts, dim=0)

        # gate_up_proj: concat [gate, up] along dim 0 then stack per expert
        parts = []
        for e in range(self.num_experts):
            gate = self._raw(f"{prefix}.{e}.gate_proj.weight", device, dtype)
            up = self._raw(f"{prefix}.{e}.up_proj.weight", device, dtype)
            if gate is None or up is None:
                return None
            parts.append(torch.cat([gate, up], dim=0))
        return torch.stack(parts, dim=0)


# ── Convenience: arch + family bundle ───────────────────────────────────────

def load_arch_and_family(instruct_id: str, cache_dir: str = CACHE_DIR):
    snapshot_download(
        repo_id=instruct_id,
        cache_dir=cache_dir,
        allow_patterns=["config.json"],
    )
    arch = load_arch(instruct_id, cache_dir=cache_dir)
    family = detect_family(arch)
    return arch, family


__all__ = [
    "PRESETS",
    "CACHE_DIR",
    "Family",
    "detect_family",
    "baseline_component",
    "classify_component",
    "classify_subcomponent",
    "SUBCOMPS_BY_GROUP",
    "layer_index",
    "ThinkingWeightLoader",
    "load_arch_and_family",
    "ModelArch",
    "load_arch",
]
