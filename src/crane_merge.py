#!/usr/bin/env python3
"""Architecture-adaptive merge: θ_M = θ_I + Π_τ( α_c · T(δ) )

Building blocks (each toggleable via --no-taylor / --no-denoise / --no-gsp):
  - α_c    : per-component (per-layer) scaling from Taylor S_reason JSON
  - T(δ)   : median-magnitude thresholding (crane_denoise.median_threshold)
  - Π_τ    : GSP format-nullspace projection (gsp_projector.apply_gsp_projection)

Boundary tensors (embedding / lm_head / router) get explicit special-token
protection to keep the merged checkpoint stable at sequence boundaries.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _common import (  # noqa: E402
    CACHE_DIR,
    PRESETS,
    classify_component,
    layer_index,
    load_arch_and_family,
)
from crane_denoise import median_threshold  # noqa: E402

# From crane/
from gsp_projector import apply_gsp_projection  # noqa: E402


# ── Structural constants ──────────────────────────────────────────────────

# Qwen3 tokenizer special-token IDs; protected during embedding / lm_head merge.
SPECIAL_TOKEN_IDS = [t for t in range(151643, 151669) if t not in (151667, 151668)]
THINK_TOKEN_IDS = [151667, 151668]   # <think> open / close
THINK_TOKEN_ALPHA = 0.80
LM_HEAD_ALPHA = 0.03
SLERP_DOT_THRESHOLD = 0.9995


# ── α_c resolution from phase2_stats.json ──────────────────────────────────

class AlphaSource:
    """Resolve α_c from per_layer_S_reason (preferred) or per_component fallback."""

    def __init__(
        self,
        stats_path: Optional[str],
        alpha_base: float,
        use_taylor: bool,
    ):
        self.alpha_base = alpha_base
        self.use_taylor = use_taylor
        self.per_layer: Dict[int, Dict[str, float]] = {}
        self.per_component: Dict[str, float] = {}
        self.baseline_component: Optional[str] = None
        self.family: Optional[str] = None

        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            self.per_layer = {
                int(k): v for k, v in stats.get("per_layer_S_reason", {}).items()
            }
            self.per_component = stats.get("per_component_S_reason", {})
            meta = stats.get("metadata", {})
            self.baseline_component = meta.get("baseline_component")
            self.family = meta.get("family")
            print(f"  α source: {stats_path}")
            print(f"    family={self.family}  baseline={self.baseline_component}  "
                  f"per_layer_keys={len(self.per_layer)}")
        else:
            if use_taylor:
                raise SystemExit(
                    f"--taylor requires a valid --stats JSON (got: {stats_path!r})"
                )
            print("  α source: flat (Taylor disabled)")

    def get(self, comp: str, layer: Optional[int]) -> float:
        if not self.use_taylor:
            return self.alpha_base
        s: Optional[float] = None
        if layer is not None and layer in self.per_layer:
            s = self.per_layer[layer].get(comp)
        if s is None:
            s = self.per_component.get(comp)
        if s is None:
            return self.alpha_base
        return float(s) * self.alpha_base


# ── GSP projection helper ──────────────────────────────────────────────────

def gsp_project(
    delta: torch.Tensor,
    proj_data: Dict,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    V_r = proj_data["V_r"]
    sigma = proj_data.get("sigma")
    if delta.shape[-1] != V_r.shape[0] or sigma is None:
        return delta
    K = 4.6 / tau if tau > 0 else 200.0
    return apply_gsp_projection(delta, V_r, sigma, tau=tau, k=K, device=device)


# ── SLERP for router ───────────────────────────────────────────────────────

def slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    if a.dim() == 1:
        return torch.lerp(a, b, t)
    orig = a.shape
    af, bf = a.flatten().float(), b.flatten().float()
    na, nb = af.norm(), bf.norm()
    if na == 0 or nb == 0:
        return b if na == 0 else a
    cos = (torch.dot(af, bf) / (na * nb)).clamp(-1, 1)
    if cos.item() > SLERP_DOT_THRESHOLD:
        return torch.lerp(a, b, t).reshape(orig)
    th = torch.acos(cos)
    sn = torch.sin(th)
    return ((torch.sin((1-t)*th)/sn)*af + (torch.sin(t*th)/sn)*bf).to(a.dtype).reshape(orig)


# ── Merge one tensor ───────────────────────────────────────────────────────

def merge_tensor(
    name: str,
    t_inst: torch.Tensor,
    t_think: torch.Tensor,
    device: torch.device,
    alpha_src: AlphaSource,
    projectors: Optional[Dict],
    tau: float,
    use_denoise: bool,
    use_gsp: bool,
    slerp_router: bool = True,
    gsp_router: bool = False,
    _logged: Optional[set] = None,
) -> torch.Tensor:
    comp = classify_component(name)
    layer = layer_index(name)
    t_i = t_inst.to(device=device, dtype=torch.bfloat16)
    t_t = t_think.to(device=device, dtype=torch.bfloat16)

    if comp == "embedding":
        result = t_i.clone()
        # Special tokens (non-<think>): keep instruct
        for tid in SPECIAL_TOKEN_IDS:
            if tid < result.shape[0]:
                result[tid] = t_i[tid]
        for tid in THINK_TOKEN_IDS:
            if tid < result.shape[0]:
                result[tid] = t_i[tid] + (t_t[tid] - t_i[tid]) * THINK_TOKEN_ALPHA
        return result.cpu()

    if comp == "lm_head":
        delta = t_t - t_i
        result = t_i + delta * LM_HEAD_ALPHA
        for tid in SPECIAL_TOKEN_IDS:
            if tid < result.shape[0]:
                result[tid] = t_i[tid]
        for tid in THINK_TOKEN_IDS:
            if tid < result.shape[0]:
                result[tid] = t_i[tid] + (t_t[tid] - t_i[tid]) * THINK_TOKEN_ALPHA
        return result.cpu()

    if comp == "router" and slerp_router:
        # Router is sensitive; SLERP with heavy damping. With gsp_router on,
        # the SLERP target is the Π_τ-projected thinking weights (format-safe).
        t_target = t_t
        if gsp_router and use_gsp and projectors and layer is not None:
            proj_key = f"layer_{layer}_ffn_input"
            if proj_key in projectors:
                d = t_t - t_i
                if use_denoise:
                    d = median_threshold(d)
                d = gsp_project(d, projectors[proj_key], tau, device)
                t_target = t_i + d
                if _logged is not None:
                    _logged.add(proj_key + "[router]")
        return slerp_tensor(t_i, t_target, alpha_src.alpha_base / 3.0).cpu()

    # ── Main formula θ_M = θ_I + Π_τ( α_c · T(δ) ) ──
    delta = t_t - t_i
    if use_denoise:
        delta = median_threshold(delta)

    w = alpha_src.get(comp, layer)
    scaled = delta * w

    # GSP projection if compatible key exists
    if use_gsp and projectors and layer is not None:
        proj_key = None
        if "linear_attn.out_proj" in name:
            proj_key = f"layer_{layer}_linattn_inner"
        elif comp in ("attention", "linear_attn"):
            proj_key = f"layer_{layer}_attn_input"
        elif comp in ("ffn", "expert", "shared_expert"):
            proj_key = f"layer_{layer}_ffn_input"
        elif comp == "router" and gsp_router:
            # Router input is post_attention_layernorm(h), same as ffn_input.
            proj_key = f"layer_{layer}_ffn_input"
        if proj_key and proj_key in projectors:
            scaled = gsp_project(scaled, projectors[proj_key], tau, device)
            if _logged is not None and proj_key not in _logged:
                _logged.add(proj_key)

    result = t_i + scaled
    del t_i, t_t, delta, scaled
    return result.cpu()


# ── Model I/O ──────────────────────────────────────────────────────────────

def resolve_dir(model_id: str) -> str:
    return snapshot_download(
        repo_id=model_id, cache_dir=CACHE_DIR,
        allow_patterns=["*.safetensors", "*.index.json", "config.json",
                        "*.json", "*.model", "*.txt"],
    )


def get_weight_map(model_dir: str) -> Dict[str, str]:
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["weight_map"]
    # single-shard fallback
    files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if len(files) == 1:
        from safetensors import safe_open
        keys = {}
        with safe_open(os.path.join(model_dir, files[0]),
                       framework="pt", device="cpu") as sf:
            for k in sf.keys():
                keys[k] = files[0]
        return keys
    raise FileNotFoundError(f"No safetensors index in {model_dir}")


def merge_all(
    dir_inst: str,
    dir_think: str,
    out_dir: str,
    device: torch.device,
    alpha_src: AlphaSource,
    projectors: Optional[Dict],
    tau: float,
    use_denoise: bool,
    use_gsp: bool,
    slerp_router: bool = True,
    gsp_router: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    wmap = get_weight_map(dir_inst)
    logged = set()

    fk = defaultdict(list)
    for k in sorted(wmap):
        fk[wmap[k]].append(k)

    total = 0
    t0 = time.time()
    files = sorted(fk)
    for idx, fname in enumerate(files):
        keys = fk[fname]
        ts = time.time()
        si = load_file(os.path.join(dir_inst, fname))
        st_path = os.path.join(dir_think, fname)
        st = load_file(st_path) if os.path.exists(st_path) else {}

        merged = {}
        for k in keys:
            if k not in si:
                merged[k] = st.get(k)
                continue
            if k not in st:
                merged[k] = si[k]
                continue
            if si[k].shape != st[k].shape:
                # Shape mismatch (e.g. MoE fused vs unfused) → keep instruct
                merged[k] = si[k]
                continue
            merged[k] = merge_tensor(
                k, si[k], st[k], device,
                alpha_src=alpha_src,
                projectors=projectors, tau=tau,
                use_denoise=use_denoise, use_gsp=use_gsp,
                slerp_router=slerp_router,
                gsp_router=gsp_router,
                _logged=logged,
            )
        total += len(merged)
        save_file(merged, os.path.join(out_dir, fname))
        print(f"  [{idx+1}/{len(files)}] {fname}: {len(keys)} tensors "
              f"in {time.time()-ts:.1f}s")
        del si, st, merged
        gc.collect()

    print(f"\n  Total: {total} tensors in {time.time()-t0:.1f}s")
    if use_gsp:
        print(f"  GSP applied to {len(logged)} layer/component slots")

    # Copy tokenizer / config files from instruct (prefer instruct's chat template)
    for fn in os.listdir(dir_inst):
        if fn.endswith((".json", ".txt", ".model")):
            dst = os.path.join(out_dir, fn)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(dir_inst, fn), dst)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Architecture-adaptive merge: θ_M = θ_I + Π_τ(α_c · T(δ))"
    )
    ap.add_argument("--model-preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--instruct-id", default=None)
    ap.add_argument("--thinking-id", default=None)
    ap.add_argument("--stats", default=None,
                    help="phase2_stats JSON produced by crane_s_reason_auto.py "
                         "(required unless --no-taylor)")
    ap.add_argument("--gsp", default=None,
                    help="format_projectors.pt produced by crane_gsp.py "
                         "(required unless --no-gsp)")
    ap.add_argument("--gsp-linattn", default=None,
                    help="Optional additional format_projectors_linattn_inner.pt "
                         "(produced by crane_gsp_linattn_inner.py); its keys are "
                         "merged into the main --gsp dict and cover "
                         "linear_attn.out_proj's value_dim subspace.")
    ap.add_argument("--out", required=True, help="Output merged-model dir")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="α_base scalar (multiplies per-component S_reason)")
    ap.add_argument("--tau", type=float, default=0.03,
                    help="GSP sigmoid center")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-taylor", action="store_true",
                    help="Disable Taylor α_c (use flat --alpha everywhere)")
    ap.add_argument("--no-denoise", action="store_true",
                    help="Disable T(δ) median thresholding")
    ap.add_argument("--no-gsp", action="store_true",
                    help="Disable Π_τ format nullspace projection")
    ap.add_argument("--no-slerp-router", action="store_true",
                    help="Merge router via the standard θ_I + Π_τ(α_c · T(δ)) "
                         "path instead of SLERP(t=α_base/3)")
    ap.add_argument("--gsp-router", action="store_true",
                    help="Apply Π_τ (ffn_input projector) to router δ. "
                         "With SLERP (default): SLERP target becomes θ_I + Π_τ(T(δ)), "
                         "so format-token routing stays close to instruct. "
                         "With --no-slerp-router: router joins the standard GSP path. "
                         "Shape-compatible because router input == post_attn_layernorm(h), "
                         "same as ffn_input (hidden_dim=2048).")
    args = ap.parse_args()

    # Resolve model IDs
    p = PRESETS[args.model_preset]
    inst_id = args.instruct_id or p["instruct_id"]
    think_id = args.thinking_id or p["thinking_id"]

    use_taylor = not args.no_taylor
    use_denoise = not args.no_denoise
    use_gsp = not args.no_gsp
    slerp_router = not args.no_slerp_router
    gsp_router = args.gsp_router

    device = torch.device(args.device)

    alpha_src = AlphaSource(args.stats, args.alpha, use_taylor=use_taylor)

    projectors = None
    if use_gsp:
        if args.gsp is None or not os.path.exists(args.gsp):
            raise SystemExit(f"--gsp is required unless --no-gsp "
                             f"(got: {args.gsp!r})")
        projectors = torch.load(args.gsp, map_location="cpu", weights_only=True)
        print(f"  loaded {len(projectors)} GSP components from {args.gsp}")
        if args.gsp_linattn:
            if not os.path.exists(args.gsp_linattn):
                raise SystemExit(f"--gsp-linattn file not found: {args.gsp_linattn!r}")
            extra = torch.load(args.gsp_linattn, map_location="cpu", weights_only=True)
            projectors.update(extra)
            print(f"  merged {len(extra)} linattn_inner components from {args.gsp_linattn}")

    print("\n  Resolving snapshots ...")
    dir_inst = resolve_dir(inst_id)
    dir_think = resolve_dir(think_id)
    arch, family = load_arch_and_family(inst_id)

    print("=" * 70)
    print("  CRANE final merge — θ_M = θ_I + Π_τ(α_c · T(δ))")
    print(f"  preset:   {args.model_preset}  family={family}")
    print(f"  instruct: {inst_id}")
    print(f"  thinking: {think_id}")
    print(f"  arch: {arch.num_layers}L, hidden={arch.hidden_dim}, "
          f"experts={arch.num_experts}")
    print(f"  components: taylor={use_taylor}  denoise={use_denoise}  gsp={use_gsp}  slerp_router={slerp_router}  gsp_router={gsp_router}")
    print(f"  alpha={args.alpha}  tau={args.tau}")
    print(f"  stats: {args.stats}")
    print(f"  gsp:   {args.gsp}")
    print(f"  out:   {args.out}")
    print("=" * 70)
    print(f"    instruct dir: {dir_inst}")
    print(f"    thinking dir: {dir_think}")

    print("\n  Merging ...")
    merge_all(
        dir_inst=dir_inst,
        dir_think=dir_think,
        out_dir=args.out,
        device=device,
        alpha_src=alpha_src,
        projectors=projectors,
        tau=args.tau,
        use_denoise=use_denoise,
        use_gsp=use_gsp,
        slerp_router=slerp_router,
        gsp_router=gsp_router,
    )
    print(f"\n  Wrote merged model → {args.out}")


if __name__ == "__main__":
    main()
