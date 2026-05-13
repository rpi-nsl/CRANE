"""LEWIS density-schedule helpers (arXiv:2503.03874).

Build a per-(layer, projection) density schedule in [γ, ε] from two
LEWIS importance dicts (base / fine-tuned). MoE per-expert Linears are
mean-collapsed by projection before normalization to match the paper's
per-projection assumption.
"""

from __future__ import annotations

import os
import re
from typing import Callable

import torch


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"experts\.\d+\.")


def _layer_idx(module_name: str) -> int | None:
    m = _LAYER_RE.search(module_name)
    return int(m.group(1)) if m else None


def _projection_type(module_name: str) -> str | None:
    """Strip the `layers.N.` prefix and collapse `experts.K.` for MoE buckets."""
    m = _LAYER_RE.search(module_name)
    if not m:
        return None
    rest = module_name[m.end():]
    return _EXPERT_RE.sub("experts.", rest)


def _module_name_from_param_key(param_key: str) -> str | None:
    """`...q_proj.weight` -> `...q_proj` ; non-`.weight` keys -> None."""
    if not param_key.endswith(".weight"):
        return None
    return param_key[: -len(".weight")]


def load_lewis_importance(path: str) -> dict[str, float]:
    """Load `{module_name: scalar importance}` from a `lewis_importance.py` run."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LEWIS importance file not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a dict")
    return {str(k): float(v) for k, v in raw.items()}


def build_density_schedule(
    importance_base: dict[str, float],
    importance_target: dict[str, float],
    gamma: float,
    epsilon: float,
) -> tuple[dict[str, float], dict]:
    """Compute the `(layer_idx, proj_type) -> density` table in [γ, ε]."""
    if not 0.0 <= gamma <= epsilon <= 1.0:
        raise ValueError(f"need 0 ≤ gamma ≤ epsilon ≤ 1, got γ={gamma}, ε={epsilon}")

    # Step 1: per (layer, proj_type) accumulate |Δ|, averaged across experts.
    deltas: dict[tuple[int, str], list[float]] = {}
    skipped_no_match = 0
    for name, imp_t in importance_target.items():
        imp_b = importance_base.get(name)
        if imp_b is None:
            skipped_no_match += 1
            continue
        li = _layer_idx(name)
        pt = _projection_type(name)
        if li is None or pt is None:
            continue
        deltas.setdefault((li, pt), []).append(abs(imp_t - imp_b))

    if not deltas:
        raise ValueError("no overlapping (layer, projection) entries between the two importance files")

    # Mean across experts (or 1-element list for non-MoE).
    delta_lc: dict[tuple[int, str], float] = {
        k: (sum(v) / len(v)) for k, v in deltas.items()
    }

    # Step 2: per projection type, normalize across layers (sum_l = 1)
    # then clip to [0, 1] (parity with the paper).
    by_proj: dict[str, list[tuple[int, float]]] = {}
    for (li, pt), d in delta_lc.items():
        by_proj.setdefault(pt, []).append((li, d))

    normalized: dict[tuple[int, str], float] = {}
    proj_stats: dict[str, dict] = {}
    for pt, items in by_proj.items():
        total = sum(d for _, d in items)
        if total <= 0.0:
            # All deltas zero for this projection — assign the [γ, ε] midpoint.
            for li, _ in items:
                normalized[(li, pt)] = 0.0
            proj_stats[pt] = {"sum": 0.0, "n_layers": len(items), "constant": True}
            continue
        for li, d in items:
            normalized[(li, pt)] = max(0.0, min(1.0, d / total))
        proj_stats[pt] = {"sum": total, "n_layers": len(items), "constant": False}

    # Step 3: global min-max scale into [γ, ε].
    values = list(normalized.values())
    vmin = min(values)
    vmax = max(values)
    schedule: dict[tuple[int, str], float] = {}
    if vmax - vmin <= 1e-12:
        # Degenerate: every (l, c) has the same normalized delta — use the
        # [γ, ε] midpoint so it behaves as constant-density TIES.
        mid = 0.5 * (gamma + epsilon)
        for k in normalized:
            schedule[k] = mid
    else:
        scale = (epsilon - gamma) / (vmax - vmin)
        for k, v in normalized.items():
            schedule[k] = gamma + (v - vmin) * scale

    debug = {
        "gamma": gamma,
        "epsilon": epsilon,
        "min_normalized_delta": vmin,
        "max_normalized_delta": vmax,
        "n_buckets": len(schedule),
        "skipped_target_keys_not_in_base": skipped_no_match,
        "projection_stats": proj_stats,
    }
    return {f"{li}|{pt}": d for (li, pt), d in schedule.items()}, debug


def make_density_lookup(
    schedule: dict[str, float],
    default_density: float,
) -> Callable[[str], tuple[float, bool]]:
    """Return `density(param_key) -> (density, hit_schedule)`.

    Keys not covered by the schedule fall through to `default_density`
    (so LEWIS degrades to TIES on uncovered params). hit_schedule is True
    only when the schedule actually had an entry for this key.
    """
    if not 0.0 <= default_density <= 1.0:
        raise ValueError(f"default_density must be in [0, 1], got {default_density}")

    def density(param_key: str) -> tuple[float, bool]:
        mod = _module_name_from_param_key(param_key)
        if mod is None:
            return default_density, False
        li = _layer_idx(mod)
        pt = _projection_type(mod)
        if li is None or pt is None:
            return default_density, False
        bucket = f"{li}|{pt}"
        if bucket in schedule:
            return schedule[bucket], True
        return default_density, False

    return density


def schedule_coverage(
    schedule_keys: set[str],
    weight_keys: list[str],
) -> dict:
    """How many transformer-block Linears does the schedule actually cover?"""
    matchable = 0
    covered = 0
    for k in weight_keys:
        mod = _module_name_from_param_key(k)
        if mod is None:
            continue
        li = _layer_idx(mod)
        pt = _projection_type(mod)
        if li is None or pt is None:
            continue
        matchable += 1
        if f"{li}|{pt}" in schedule_keys:
            covered += 1
    return {
        "matchable_block_linear_weights": matchable,
        "covered_by_schedule": covered,
        "coverage_fraction": covered / max(1, matchable),
    }
