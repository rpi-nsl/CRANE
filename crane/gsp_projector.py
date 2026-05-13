#!/usr/bin/env python3
"""Graduated Sigmoidal Projection (GSP) for format-token protection.

    w_i = sigmoid(k · (σ_i / σ_1 − τ))

Strong directions (σ_i ≫ τ σ_1) get w_i ≈ 1 (near-exact protection); weak
directions decay smoothly with no jump discontinuity. 48-layer accumulated
residual is < 0.001%.
"""

import math
import re
from typing import Dict, Optional, Tuple

import torch


# ── GSP Core ────────────────────────────────────────────────────────────────

def compute_gsp_weights(
    sigma: torch.Tensor,
    tau: float = 0.03,
    k: float = 200.0,
) -> torch.Tensor:
    """
    Compute per-direction protection weights using graduated sigmoid.

    Args:
        sigma: (r,) singular values, sorted descending
        tau:   sigmoid center point (relative to sigma_1)
        k:     sigmoid steepness (higher = sharper transition)

    Returns:
        weights: (r,) protection weights in [0, 1]
    """
    if len(sigma) == 0:
        return torch.zeros(0)

    sigma_1 = sigma[0]
    if sigma_1 == 0:
        return torch.zeros_like(sigma)

    # Normalized singular values
    s_normalized = sigma / sigma_1  # (r,) in [0, 1]

    # Sigmoid: w_i = 1 / (1 + exp(-k * (s_i/s_1 - tau)))
    weights = torch.sigmoid(k * (s_normalized - tau))

    return weights


def compute_adaptive_tau(
    sigma: torch.Tensor,
    tau_base: float = 0.03,
    alpha_entropy: float = 0.5,
    _entropy_median_cache: dict = {},
) -> float:
    """
    Compute per-layer adaptive tau based on energy spectrum entropy.

    Layers with concentrated energy (few dominant directions) get lower tau
    (tighter protection). Layers with diffuse energy get higher tau.

    Args:
        sigma: (r,) singular values
        tau_base: base threshold
        alpha_entropy: entropy adaptation strength
        _entropy_median_cache: mutable cache for median computation

    Returns:
        tau_l: adapted threshold for this layer
    """
    if len(sigma) < 2:
        return tau_base

    # Energy distribution
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    if total == 0:
        return tau_base

    p = sigma_sq / total  # (r,)
    # Avoid log(0)
    p_safe = p.clamp(min=1e-12)
    entropy = -(p_safe * p_safe.log()).sum().item()

    # Use cached median or default
    entropy_median = _entropy_median_cache.get("median", entropy)

    tau_l = tau_base * (1.0 + alpha_entropy * (entropy - entropy_median))
    # Clamp to reasonable range
    tau_l = max(0.005, min(0.20, tau_l))

    return tau_l


def precompute_entropy_median(
    projectors: Dict[str, dict],
    tau_base: float = 0.03,
    alpha_entropy: float = 0.5,
) -> float:
    """
    Compute median entropy across all projectors for adaptive tau calibration.

    Args:
        projectors: dict of {key: {"V_r": Tensor, "sigma": Tensor}}
        tau_base: base threshold
        alpha_entropy: adaptation strength

    Returns:
        entropy_median: median entropy value
    """
    entropies = []
    for key, proj_data in projectors.items():
        if not isinstance(proj_data, dict) or "sigma" not in proj_data:
            continue
        sigma = proj_data["sigma"]
        if len(sigma) < 2:
            continue
        sigma_sq = sigma ** 2
        total = sigma_sq.sum()
        if total == 0:
            continue
        p = sigma_sq / total
        p_safe = p.clamp(min=1e-12)
        entropy = -(p_safe * p_safe.log()).sum().item()
        entropies.append(entropy)

    if not entropies:
        return 0.0

    entropies.sort()
    median = entropies[len(entropies) // 2]
    return median


def apply_gsp_projection(
    delta: torch.Tensor,
    V_r: torch.Tensor,
    sigma: torch.Tensor,
    tau: float = 0.03,
    k: float = 200.0,
    device: torch.device = torch.device("cuda:0"),
) -> torch.Tensor:
    """
    Apply Graduated Sigmoidal Projection to a weight delta.

    Removes components of the delta that lie in format-token-important
    directions, with smooth sigmoid weighting.

    Formula:
        Δ_projected = Δ - Δ @ V_r @ diag(w) @ V_r^T

    Args:
        delta:  (d_out, d_in) weight delta
        V_r:    (d_in, r) right singular vectors from format token SVD
        sigma:  (r,) singular values, sorted descending
        tau:    sigmoid center point
        k:      sigmoid steepness
        device: compute device

    Returns:
        delta_projected: (d_out, d_in)
    """
    V_r = V_r.to(device=device, dtype=torch.float32)
    sigma = sigma.to(device=device, dtype=torch.float32)

    # Compute GSP weights
    weights = compute_gsp_weights(sigma, tau=tau, k=k)  # (r,)

    # Project: Δ' = Δ - Δ @ V_r @ diag(w) @ V_r^T
    delta_f = delta.float().to(device)
    DV = delta_f @ V_r                       # (d_out, r)
    DVw = DV * weights.unsqueeze(0)          # (d_out, r)
    projection = DVw @ V_r.T                 # (d_out, d_in)

    result = (delta_f - projection).to(delta.dtype)

    del V_r, sigma, delta_f, DV, DVw, projection, weights
    return result


def apply_hard_nullspace(
    delta: torch.Tensor,
    V_r: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Hard null-space projection (delta − V_r V_rᵀ delta)."""
    V_r = V_r.to(device=device, dtype=torch.float32)
    delta_f = delta.float().to(device)
    projection = delta_f @ V_r @ V_r.T
    result = (delta_f - projection).to(delta.dtype)
    del V_r, delta_f, projection
    return result


# ── Diagnostics ─────────────────────────────────────────────────────────────

def log_gsp_stats(
    key: str,
    sigma: torch.Tensor,
    tau: float,
    k: float,
) -> str:
    """Generate diagnostic string for a projector's GSP configuration."""
    weights = compute_gsp_weights(sigma, tau=tau, k=k)

    r = len(sigma)
    s1 = sigma[0].item()
    sr = sigma[-1].item()

    # Count effectively protected directions (w > 0.99)
    n_hard = (weights > 0.99).sum().item()
    # Count transition band (0.01 < w < 0.99)
    n_transition = ((weights > 0.01) & (weights <= 0.99)).sum().item()
    # Count effectively open (w < 0.01)
    n_open = (weights <= 0.01).sum().item()

    # Energy coverage of protected directions
    sigma_sq = sigma ** 2
    total_energy = sigma_sq.sum().item()
    protected_energy = (sigma_sq * (weights > 0.99).float()).sum().item()
    energy_frac = protected_energy / total_energy if total_energy > 0 else 0

    return (f"{key}: r={r}, σ1={s1:.4f}, σr={sr:.6f}, "
            f"protected={n_hard} transition={n_transition} open={n_open}, "
            f"protected_energy={energy_frac:.4f}")


def verify_48_layer_residual(
    projectors: Dict[str, dict],
    tau: float = 0.03,
    k: float = 200.0,
) -> dict:
    """
    Verify that 48-layer accumulated residual is within tolerance.

    For each projector, compute max residual weight for the strongest direction.
    The accumulated residual across 48 layers (×2 hooks) = 96 projectors is:
        total_residual ≈ 1 - (1 - max_residual)^96

    Returns:
        dict with per-layer and total residual stats
    """
    max_residuals = []
    for key, proj_data in sorted(projectors.items()):
        if not isinstance(proj_data, dict) or "sigma" not in proj_data:
            continue
        sigma = proj_data["sigma"]
        weights = compute_gsp_weights(sigma, tau=tau, k=k)
        if len(weights) > 0:
            # Residual for strongest direction = 1 - w_1
            max_residual = (1.0 - weights[0]).item()
            max_residuals.append((key, max_residual))

    if not max_residuals:
        return {"status": "no projectors", "total_residual": 0}

    worst_key, worst_residual = max(max_residuals, key=lambda x: x[1])
    avg_residual = sum(r for _, r in max_residuals) / len(max_residuals)

    # Accumulated: 1 - (1-r)^n ≈ n*r for small r
    n = len(max_residuals)
    accumulated = 1.0 - (1.0 - avg_residual) ** n

    return {
        "n_projectors": n,
        "worst_residual": worst_residual,
        "worst_key": worst_key,
        "avg_residual": avg_residual,
        "accumulated_residual": accumulated,
        "status": "OK" if accumulated < 0.01 else "WARNING",
    }
