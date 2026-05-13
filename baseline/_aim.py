"""Shared helpers for Activation-Informed Merging (AIM, ahnobari/AIM).

Per-parameter soft relaxation: Δ_final = Δ ⊙ (1 − s · (1 − ω)) broadcast
across the output dim, where s ∈ [0,1]^in is the input-channel activation
importance of the BASE (instruct) model. ω=0.4 by default.
"""

from __future__ import annotations

import os
from typing import Callable

import torch


# Skip non-Linear params (no meaningful input-channel importance).
_AIM_SKIP_SUBSTR = ("embed_tokens", "norm", "rotary", "bias")


def _param_key_to_module_name(param_key: str) -> str | None:
    """Strip `.weight` and skip non-Linear params; returns None to fall through."""
    if not param_key.endswith(".weight"):
        return None
    for skip in _AIM_SKIP_SUBSTR:
        if skip in param_key:
            return None
    return param_key[: -len(".weight")]


def load_aim_importance(path: str) -> dict[str, torch.Tensor]:
    """Load `{module_name: (in_features,) tensor}` produced by `aim_importance.py`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AIM importance file not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a dict of importances")
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 1:
            continue
        out[k] = v.float()
    if not out:
        raise ValueError(f"{path} contained no 1-D importance tensors")
    return out


def make_aim_scaler(
    importance: dict[str, torch.Tensor],
    omega: float,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """Return `scale(key, delta) -> delta_scaled` applying Δ * (1 − s·(1−ω))
    where s = importance / importance.max(). Returns delta unchanged if the
    key has no importance (AIM is opt-in per param). ω must be in [0, 1].
    """
    if not 0.0 <= omega <= 1.0:
        raise ValueError(f"aim omega must be in [0, 1], got {omega}")

    def scale(key: str, delta: torch.Tensor) -> torch.Tensor:
        module_name = _param_key_to_module_name(key)
        if module_name is None:
            return delta
        imp = importance.get(module_name)
        if imp is None:
            return delta
        # Linear weight is (out, in); importance is (in,).
        if delta.ndim != 2 or delta.shape[-1] != imp.numel():
            return delta
        imp_max = imp.max()
        if not torch.isfinite(imp_max) or imp_max.item() <= 0.0:
            return delta
        s = (imp / imp_max).to(delta.device, delta.dtype)
        factor = 1.0 - s * (1.0 - omega)   # shape (in,), broadcasts over dim 0
        return delta * factor

    return scale


def aim_coverage_report(
    importance: dict[str, torch.Tensor],
    all_weight_keys: list[str],
) -> dict:
    """Summarize AIM coverage for diagnostic logging."""
    covered = 0
    matchable = 0
    missing: list[str] = []
    for key in all_weight_keys:
        mod = _param_key_to_module_name(key)
        if mod is None:
            continue
        matchable += 1
        if mod in importance:
            covered += 1
        else:
            missing.append(mod)
    return {
        "matchable_linear_weights": matchable,
        "covered_by_importance": covered,
        "coverage_fraction": covered / max(1, matchable),
        "example_missing": missing[:10],
    }
