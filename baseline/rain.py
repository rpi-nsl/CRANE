#!/usr/bin/env python3
"""RAIN-Merging adapter — see RAIN_README.md for full algorithm and caveats.

Drives the vendored upstream (https://github.com/K1nght/RAIN-Merging) Stage
1/2/3 with the right CLI flags for Qwen3-30B-A3B and Qwen3-Next-80B-A3B.
Requires cvxpy and osqp (Stage 2 QP).
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(SCRIPT_DIR, "rain_upstream")
sys.path.insert(0, SCRIPT_DIR)
from _merge_io import resolve_model_snapshot  # noqa: E402

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")
BASELINE_MODEL_DIR = os.path.join(SCRIPT_DIR, "baseline_model")

# RAIN's three-model setup. The "target" is what gets modified — for our
# instruct→thinking merge direction this is the thinking model. The base
# is the un-instructed pretrained checkpoint, which RAIN uses only to
# form τ = ITM − BASE. If Qwen3-30B-A3B-Base is unavailable on your hub,
# point --base-model at a local snapshot.
MODEL_PRESETS = {
    "qwen3-30b": {
        "base":     "Qwen/Qwen3-30B-A3B-Base",
        "instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "target":   "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "rain"),
    },
    # Proxy-base ablation: BASE replaced with Thinking-2507 to mirror the
    # `qwen3-next-80b` proxy-base setup on a model where a real Base exists.
    # τ = Instruct − Thinking; merged = Thinking + α·proj(τ).
    "qwen3-30b-proxy": {
        "base":     "Qwen/Qwen3-30B-A3B-Thinking-2507",   # PROXY base (= target)
        "instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "target":   "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "output_dir": os.path.join(
            BASELINE_MODEL_DIR, "rain_30b_proxy_pycalib_qkvof_planA_a030"
        ),
    },
    # Qwen3-Next-80B-A3B: no Base published, so use Thinking as the BASE proxy
    # (= target). τ = Instruct − Thinking; merged = Thinking + α·proj(τ). The
    # `selected_layers` filter restricts updates to the 12 full_attention
    # layers; linear_attention layers and the MoE FFN are inherited from
    # Thinking. See RAIN_README.md.
    "qwen3-next-80b": {
        "base":     "Qwen/Qwen3-Next-80B-A3B-Thinking",   # PROXY base (same as target)
        "instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "target":   "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "output_dir": os.path.join(BASELINE_MODEL_DIR, "rain_qwen3_next_80b"),
    },
}

REASONING_CALIB = os.path.join(UPSTREAM_DIR, "data", "reasoning_calibration_set.json")
INSTRUCTION_CALIB = os.path.join(UPSTREAM_DIR, "data", "instruction_calibration_set.jsonl")


def _check_upstream():
    """Hard-fail early if the vendored repo or its calibration data is missing."""
    needed = [
        os.path.join(UPSTREAM_DIR, "nullspace_projection_compute.py"),
        os.path.join(UPSTREAM_DIR, "qp_true_forward_fast.py"),
        os.path.join(UPSTREAM_DIR, "unified_model_merge.py"),
        REASONING_CALIB,
        INSTRUCTION_CALIB,
    ]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        sys.stderr.write(
            "[rain] missing required upstream files:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n  re-clone with: git clone --depth 1 "
              "https://github.com/K1nght/RAIN-Merging "
            + UPSTREAM_DIR + "\n"
        )
        sys.exit(2)


def _check_deps():
    """Hard-fail early if cvxpy/osqp aren't installed."""
    missing = []
    for mod in ("cvxpy", "osqp"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.stderr.write(
            f"[rain] missing Python dependencies for Stage 2: {', '.join(missing)}\n"
            f"  install with (into the active uv venv only):\n"
            f"      uv pip install {' '.join(missing)}\n"
        )
        sys.exit(2)


def _write_merge_notes(merged_dir: str, *, preset: str | None,
                       base_model: str, instruct_model: str, target_model: str,
                       merge_types: str, layers_tail: int,
                       layer_types: list[str] | None,
                       used_full_attn_layers: list[int] | None) -> None:
    """Drop a MERGE_NOTES.md inside the merged HF dir documenting limitations."""
    if not os.path.isdir(merged_dir):
        return
    is_hybrid = layer_types is not None and "linear_attention" in layer_types
    lines: list[str] = []
    lines.append(f"# RAIN-Merging notes for `{os.path.basename(merged_dir)}`\n")
    lines.append(f"- Method: RAIN-Merging (K1nght/RAIN-Merging, ICLR 2026)")
    lines.append(f"- Preset: `{preset or 'custom'}`")
    lines.append(f"- Base model:     `{base_model}`")
    lines.append(f"- Instruct model: `{instruct_model}`")
    lines.append(f"- Target model:   `{target_model}` (the merged weights are this model with RAIN deltas applied)")
    lines.append(f"- merge_types:    `{merge_types}`  (q/k/v/o flags; 'f' = FFN, omitted for MoE by default)")
    lines.append(f"- layers_tail:    `{layers_tail}` (RAIN considers the last N transformer layers)")
    if is_hybrid:
        n_full = sum(1 for t in (layer_types or []) if t == "full_attention")
        n_lin = sum(1 for t in (layer_types or []) if t == "linear_attention")
        lines.append("")
        lines.append("## Hybrid architecture caveat")
        lines.append("")
        lines.append(f"This model has **{n_full} `full_attention` layers** and **{n_lin} `linear_attention` "
                     "layers** (Gated DeltaNet). RAIN's per-head Q/K/V/O QP only applies to the full-attention "
                     "layers — linear-attention layers expose no `q_proj`/`k_proj`/`v_proj`/`o_proj`, so they "
                     "are copied **verbatim** from the target.")
        if used_full_attn_layers is not None:
            lines.append("")
            lines.append(f"Layers that actually received RAIN updates: `{used_full_attn_layers}`")
    if base_model == target_model and base_model != instruct_model:
        lines.append("")
        lines.append("## Proxy-base caveat")
        lines.append("")
        lines.append(f"No pretrained Base checkpoint is published for this family, so the target "
                     f"model (`{target_model}`) **is also used as the proxy Base** in RAIN's "
                     f"τ = ITM − BASE formulation. The resulting task vector is therefore "
                     f"τ = Instruct − Thinking, and the merged model is "
                     f"`Thinking + α · proj(Instruct − Thinking)` — i.e. the Thinking checkpoint "
                     f"nudged in reasoning-preserving directions toward Instruct. This is a "
                     f"deviation from the paper (which assumes a pretrained, un-instructed base) "
                     f"but is the only setup with the two published checkpoints that yields a "
                     f"non-degenerate task vector.")
    lines.append("")
    lines.append("## MoE FFN")
    lines.append("")
    lines.append("Default `merge_types='qkvo'` skips the FFN. The MoE FFN of the target model is "
                 "therefore copied unchanged. The paper's default is `qkvof` (FFN merged); we "
                 "deviate because the upstream FFN handler assumes dense gate/up/down and breaks "
                 "on MoE. To merge MoE FFN you would need to extend "
                 "`nullspace_merge_qkvo_ffn.py` for `mlp.experts[i].{gate,up,down}_proj`. "
                 "→ this checkpoint is **RAIN-qkvo (MoE FFN frozen)**, not a paper-faithful merge.")
    lines.append("")
    lines.append("## Stage 2 calibration truncation")
    lines.append("")
    lines.append("Upstream `qp_true_forward_fast.py` truncates each instruction-calibration sample "
                 "to `QP_MAX_SEQ_LEN=1536` tokens (override via env var) to avoid 21 GB single-"
                 "allocation OOMs during the two-pass forward. With our calibration set "
                 "(`rain_upstream/data/instruction_calibration_set.jsonl`, 365 samples) and "
                 "the target tokenizer, **7/365 samples (≈ 1.9 %)** exceed 1536 tokens. Their "
                 "spans are extracted on the un-truncated text but token indices past 1536 are "
                 "silently dropped — α is therefore biased toward shorter samples. Set "
                 "`--qp-max-seq-len 4096` (or higher) to remove the bias if you have GPU memory "
                 "to spare; the value used for this checkpoint is recorded in `merge_config.json`.")
    lines.append("")
    notes_path = os.path.join(merged_dir, "MERGE_NOTES.md")
    try:
        with open(notes_path, "w") as f:
            f.write("\n".join(lines))
        print(f"[rain] merge notes written to {notes_path}")
    except OSError as e:
        # Non-fatal: the merged weights are already written.
        sys.stderr.write(f"[rain] WARN: could not write {notes_path}: {e}\n")


def _run(cmd: list[str], stage: str, dry_run: bool,
         extra_env: dict | None = None) -> None:
    """Run a stage subprocess in the upstream working directory."""
    print(f"\n[rain] === {stage} ===")
    if extra_env:
        env_str = " ".join(f"{k}={v}" for k, v in extra_env.items())
        print(f"[rain] env: {env_str}")
    print("[rain] " + " ".join(cmd))
    if dry_run:
        print("[rain] (dry-run, not executing)")
        return
    t0 = time.time()
    env = dict(os.environ)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    result = subprocess.run(cmd, cwd=UPSTREAM_DIR, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        sys.stderr.write(f"[rain] {stage} failed (exit {result.returncode}, {elapsed:.0f}s)\n")
        sys.exit(result.returncode)
    print(f"[rain] {stage} ok ({elapsed:.0f}s)")


def run_rain(
    base_model: str,
    instruct_model: str,
    target_model: str,
    output_dir: str,
    stages: list[int],
    *,
    preset: str | None,
    max_samples: int,
    reasoning_calib: str,
    layers_tail: int,
    merge_types: str,
    max_seq_len: int,
    lambda_ridge: float,
    cg_maxit: int,
    cg_tol: float,
    compute_precision: str,
    qk_device: str,
    vo_device: str,
    ffn_device: str,
    qp_device: str,
    qp_variant: str,
    qp_max_seq_len: int,
    prior_scalar: float,
    l2_prior: float,
    l1_reg: float,
    box_lo: float,
    box_hi: float,
    h_lambda: float,
    h_mu: float,
    rho_du: float,
    kappa_a: float,
    kappa_u: float,
    decouple_qk: bool,
    scaling_factor: float | None,
    model_name: str,
    revision_base: str | None,
    revision_instruct: str | None,
    revision_target: str | None,
    explicit_projected_file: str | None,
    explicit_alpha_file: str | None,
    dry_run: bool,
):
    _check_upstream()
    if 2 in stages:
        _check_deps()

    print(f"\n{'='*60}")
    print(f"  RAIN-Merging  (vendored upstream, stages={stages})")
    print(f"  base:     {base_model}")
    print(f"  instruct: {instruct_model}")
    print(f"  target:   {target_model}")
    print(f"  output:   {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Resolve all three model snapshots; pass absolute paths to upstream
    # so the cwd-change to rain_upstream/ doesn't confuse HF id resolution.
    dir_b, commit_b = resolve_model_snapshot(base_model, CACHE_DIR, revision_base)
    dir_i, commit_i = resolve_model_snapshot(instruct_model, CACHE_DIR, revision_instruct)
    dir_t, commit_t = resolve_model_snapshot(target_model, CACHE_DIR, revision_target)
    print(f"  resolved base:     {dir_b}  (commit: {commit_b or 'N/A'})")
    print(f"  resolved instruct: {dir_i}  (commit: {commit_i or 'N/A'})")
    print(f"  resolved target:   {dir_t}  (commit: {commit_t or 'N/A'})")

    # Default artifact paths sit under output_dir; --projected-file /
    # --alpha-file allow splicing in artifacts from a different run.
    if explicit_projected_file:
        projected_file = os.path.abspath(explicit_projected_file)
    else:
        projected_file = os.path.abspath(os.path.join(output_dir, "projected_task_vectors.pkl"))
    qp_dir = os.path.abspath(os.path.join(output_dir, "qp_optimization"))
    merge_dir = os.path.abspath(os.path.join(output_dir, "unified_model_merge"))
    os.makedirs(qp_dir, exist_ok=True)
    os.makedirs(merge_dir, exist_ok=True)

    # ── Stage 1 ────────────────────────────────────────────────────────
    if 1 in stages:
        cmd1 = [
            sys.executable, "nullspace_projection_compute.py",
            "--base",           dir_b,
            "--instruct",       dir_i,
            "--target",         dir_t,
            "--texts_r",        reasoning_calib,   # reasoning calib only here
            "--output_file",    projected_file,
            "--max_samples_r",  str(max_samples),
            "--layers_tail",    str(layers_tail),
            "--heads",          "all",
            "--merge_types",    merge_types,
            "--lambda_ridge",   str(lambda_ridge),
            "--cg_maxit",       str(cg_maxit),
            "--cg_tol",         str(cg_tol),
            "--compute_precision", compute_precision,
            "--qk_device",      qk_device,
            "--vo_device",      vo_device,
            "--ffn_device",     ffn_device,
            "--max_seq_len",    str(max_seq_len),
            "--use_hooks",
        ]
        _run(cmd1, "Stage 1: null-space projection", dry_run)

    # Stages 2 and 3 require projected_file to exist (built by Stage 1).
    if (2 in stages or 3 in stages) and not dry_run:
        if not os.path.exists(projected_file):
            sys.stderr.write(
                f"[rain] required projected file does not exist: {projected_file}\n"
                f"  run --stages 1 first, or pass --projected-file <path>\n"
            )
            sys.exit(2)

    # ── Stage 2 ────────────────────────────────────────────────────────
    # alpha_file resolution priority: explicit --alpha-file → file produced
    # by Stage 2 this run → auto-discovery from qp_dir → scaling_factor.
    alpha_file: str | None = None
    if explicit_alpha_file:
        alpha_file = os.path.abspath(explicit_alpha_file)
        if not os.path.exists(alpha_file) and not dry_run:
            sys.stderr.write(f"[rain] --alpha-file does not exist: {alpha_file}\n")
            sys.exit(2)
        print(f"[rain] using explicit alpha file: {alpha_file}")

    if 2 in stages:
        cmd2 = [
            sys.executable, "qp_true_forward_fast.py",
            "--projected_file", projected_file,
            "--base_model",     dir_t,             # NB: "base" here = LRM, not pretrained
            "--json_data",      INSTRUCTION_CALIB, # instruction calib only here
            "--layers",         "all",
            "--heads",          "all",
            "--prior_scalar",   str(prior_scalar),
            "--l2_prior",       str(l2_prior),
            "--l1",             str(l1_reg),
            "--box_lo",         str(box_lo),
            "--box_hi",         str(box_hi),
            "--device",         qp_device,
            "--out",            qp_dir,
            "--qp_variant",     qp_variant,
            "--H_lambda",       str(h_lambda),
            "--H_mu",           str(h_mu),
            "--rho_du",         str(rho_du),
            "--verbose",
        ]
        # κ_a / κ_u only apply to single-state variants (two_pass derives
        # them from Δ-statistics). Mirrors upstream run_stage2.sh.
        if qp_variant in ("anchor_only", "post_only"):
            cmd2 += ["--kappa_a", str(kappa_a), "--kappa_u", str(kappa_u)]
        if decouple_qk:
            cmd2.append("--decouple_qk")
        _run(cmd2, "Stage 2: QP α-optimization", dry_run,
             extra_env={"QP_MAX_SEQ_LEN": qp_max_seq_len})

        # Upstream's two_pass variant writes `alpha_true_forward_align_leak.*`,
        # NOT `alpha_true_forward_two_pass.*` — check both spellings.
        if alpha_file is None:
            alt_name = "align_leak" if qp_variant == "two_pass" else qp_variant
            for cand in (
                f"alpha_true_forward_{qp_variant}.pt",
                f"alpha_true_forward_{qp_variant}.json",
                f"alpha_true_forward_{alt_name}.pt",
                f"alpha_true_forward_{alt_name}.json",
            ):
                p = os.path.join(qp_dir, cand)
                if os.path.exists(p):
                    alpha_file = p
                    break
            if alpha_file is None and not dry_run:
                sys.stderr.write(f"[rain] Stage 2 did not produce an alpha file in {qp_dir}\n")
                sys.exit(2)
    elif 3 in stages and alpha_file is None:
        # Auto-discover an alpha file from a previous --stages 2 run.
        # two_pass variant writes `alpha_true_forward_align_leak.*` upstream.
        alt_name = "align_leak" if qp_variant == "two_pass" else qp_variant
        for cand in (
            f"alpha_true_forward_{qp_variant}.pt",
            f"alpha_true_forward_{qp_variant}.json",
            f"alpha_true_forward_{alt_name}.pt",
            f"alpha_true_forward_{alt_name}.json",
            # Last-resort fallback: accept any variant.
            "alpha_true_forward_align_leak.pt",
            "alpha_true_forward_two_pass.pt",
            "alpha_true_forward_anchor_only.pt",
            "alpha_true_forward_post_only.pt",
        ):
            p = os.path.join(qp_dir, cand)
            if os.path.exists(p):
                alpha_file = p
                print(f"[rain] auto-discovered alpha file from prior Stage 2: {alpha_file}")
                break

    # ── Stage 3 ────────────────────────────────────────────────────────
    if 3 in stages and alpha_file is None and scaling_factor is None:
        # No QP α and no explicit scaling: upstream falls back to
        # scaling_factor=1.0 which is NOT the paper merge. Warn loudly.
        print(
            "[rain] WARN: Stage 3 has neither an alpha file nor an "
            "explicit --scaling-factor; falling back to scaling_factor=1.0. "
            "This is NOT the QP-coefficient RAIN merge — to reproduce the "
            "paper, run Stage 2 first or pass --alpha-file <path>."
        )

    if 3 in stages:
        cmd3 = [
            sys.executable, "unified_model_merge.py",
            "--projected_file", projected_file,
            "--base_model",     dir_t,             # NB: again, target LRM
            "--output_dir",     merge_dir,
            "--model_name",     model_name,
            "--verbose",
        ]
        if alpha_file is not None:
            cmd3 += ["--alpha_file", alpha_file]
        if scaling_factor is not None:
            cmd3 += ["--scaling_factor", str(scaling_factor)]
        if alpha_file is None and scaling_factor is None:
            # Default to 1.0 like upstream's stage3.sh does
            cmd3 += ["--scaling_factor", "1.0"]
        _run(cmd3, "Stage 3: unified merge", dry_run)

    # Write merge_config.json provenance manifest (matches the other baselines).
    if not dry_run:
        manifest = {
            "method": "rain",
            "upstream": "https://github.com/K1nght/RAIN-Merging",
            "stages_run": stages,
            "base_model": base_model,
            "instruct_model": instruct_model,
            "target_model": target_model,
            "base_resolved_commit": commit_b,
            "instruct_resolved_commit": commit_i,
            "target_resolved_commit": commit_t,
            "base_resolved_path": dir_b,
            "instruct_resolved_path": dir_i,
            "target_resolved_path": dir_t,
            "projected_file": projected_file,
            "alpha_file": alpha_file,
            "merged_model_dir": os.path.join(merge_dir, model_name),
            "stage1": {
                "reasoning_calib": reasoning_calib,
                "max_samples": max_samples,
                "layers_tail": layers_tail,
                "merge_types": merge_types,
                "max_seq_len": max_seq_len,
                "lambda_ridge": lambda_ridge,
                "cg_maxit": cg_maxit,
                "cg_tol": cg_tol,
                "compute_precision": compute_precision,
            },
            "stage2": {
                "qp_variant": qp_variant,
                "qp_max_seq_len": qp_max_seq_len,
                "prior_scalar": prior_scalar,
                "l2_prior": l2_prior,
                "l1": l1_reg,
                "box": [box_lo, box_hi],
                "H_lambda": h_lambda,
                "H_mu": h_mu,
                "rho_du": rho_du,
                "kappa_a": kappa_a,
                "kappa_u": kappa_u,
                "decouple_qk": decouple_qk,
                "explicit_alpha_file": explicit_alpha_file,
                "explicit_projected_file": explicit_projected_file,
            },
            "stage3": {
                "scaling_factor": scaling_factor,
                "model_name": model_name,
            },
        }
        with open(os.path.join(output_dir, "merge_config.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[rain] manifest written to {os.path.join(output_dir, 'merge_config.json')}")

        # Drop MERGE_NOTES.md in the merged HF dir (only if Stage 3 ran).
        merged_dir = os.path.join(merge_dir, model_name)
        if 3 in stages and os.path.isdir(merged_dir):
            layer_types = None
            used_full_attn_layers: list[int] | None = None
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(dir_t, trust_remote_code=True)
                layer_types = list(getattr(cfg, "layer_types", []) or []) or None
                if layer_types is not None:
                    n_layers = len(layer_types)
                    tail = list(range(max(0, n_layers - layers_tail), n_layers))
                    used_full_attn_layers = [li for li in tail
                                              if layer_types[li] == "full_attention"]
            except Exception as e:  # noqa: BLE001
                print(f"[rain] WARN: could not introspect target config for notes: {e}")
            _write_merge_notes(
                merged_dir,
                preset=preset,
                base_model=base_model,
                instruct_model=instruct_model,
                target_model=target_model,
                merge_types=merge_types,
                layers_tail=layers_tail,
                layer_types=layer_types,
                used_full_attn_layers=used_full_attn_layers,
            )


def main():
    parser = argparse.ArgumentParser(
        description="RAIN-Merging baseline (vendored from K1nght/RAIN-Merging)",
    )
    parser.add_argument("--preset", default="qwen3-30b", choices=MODEL_PRESETS.keys())
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override the preset base (pretrained) model id / path")
    parser.add_argument("--instruct-model", type=str, default=None,
                        help="Override the preset ITM (instruction) model id / path")
    parser.add_argument("--target-model", type=str, default=None,
                        help="Override the preset LRM (target / thinking) model id / path")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--stages", type=str, default="1,2,3",
                        help="Comma-separated subset of {1,2,3} to run (default: all)")

    # Stage 1 — null-space projection
    parser.add_argument("--reasoning-calib", type=str, default=REASONING_CALIB,
                        help="Stage 1 reasoning calibration JSON file. Default is "
                             "the upstream-bundled set (50 OpenR1-Math + 50 codeforces-cots "
                             "+ 50 Llama-Nemotron-mixed = 150 samples). For coding-task "
                             "calibration, point this at e.g. "
                             "rain_upstream/data_python/reasoning_calibration_set_python.json "
                             "(150 Python samples from nvidia/OpenCodeReasoning).")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Reasoning calibration samples (upstream default 1000; "
                             "lower for memory)")
    parser.add_argument("--layers-tail", type=int, default=27,
                        help="Process the last N transformer layers (default 27)")
    parser.add_argument("--merge-types", type=str, default="qkvo",
                        help="Subset of q/k/v/o/f. Default 'qkvo' (omits FFN) because "
                             "the upstream FFN path was not validated on MoE.")
    parser.add_argument("--max-seq-len", type=int, default=7168)
    parser.add_argument("--lambda-ridge", type=float, default=1e-4)
    parser.add_argument("--cg-maxit", type=int, default=100)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--compute-precision", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--qk-device", default="auto")
    parser.add_argument("--vo-device", default="auto")
    parser.add_argument("--ffn-device", default="auto")

    # Stage 2 — QP
    parser.add_argument("--qp-device", default="cuda:0")
    parser.add_argument("--qp-variant",
                        choices=["two_pass", "anchor_only", "post_only"],
                        default="two_pass")
    parser.add_argument("--prior-scalar", type=float, default=1.0)
    parser.add_argument("--l2-prior", type=float, default=0.1)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--box-lo", type=float, default=0.0)
    parser.add_argument("--box-hi", type=float, default=1.5)
    parser.add_argument("--h-lambda", type=float, default=1.0)
    parser.add_argument("--h-mu", type=float, default=1.0)
    parser.add_argument("--rho-du", type=float, default=0.5)
    parser.add_argument("--kappa-a", type=float, default=1.0,
                        help="Alignment-score scale for anchor_only / post_only "
                             "QP variants (ignored by two_pass). Default 1.0.")
    parser.add_argument("--kappa-u", type=float, default=1.0,
                        help="Leakage-score scale for anchor_only / post_only "
                             "QP variants (ignored by two_pass). Default 1.0.")
    parser.add_argument("--decouple-qk", action="store_true")
    parser.add_argument("--qp-max-seq-len", type=int, default=1536,
                        help="Stage-2 hard truncation length (env QP_MAX_SEQ_LEN). "
                             "Upstream's two-pass forward OOMs on long calibration "
                             "samples without this. The default 1536 truncates "
                             "≈ 7/365 (1.9%%) of the bundled instruction "
                             "calibration set, biasing α toward shorter samples. "
                             "Raise to 4096+ if you have GPU memory.")

    # Stage 3 — apply
    parser.add_argument("--scaling-factor", type=float, default=None,
                        help="Optional global α multiplier; combined with the QP α "
                             "if Stage 2 was run, otherwise replaces it")
    parser.add_argument("--projected-file", type=str, default=None,
                        help="Resume from an existing Stage 1 output instead of the "
                             "default <output_dir>/projected_task_vectors.pkl")
    parser.add_argument("--alpha-file", type=str, default=None,
                        help="Resume from an existing Stage 2 alpha file. Required "
                             "when running --stages 3 alone if your alpha file lives "
                             "outside <output_dir>/qp_optimization/")
    parser.add_argument("--model-name", type=str, default="rain_merged")

    parser.add_argument("--revision-base", type=str, default=None)
    parser.add_argument("--revision-instruct", type=str, default=None)
    parser.add_argument("--revision-target", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the stage commands without executing them")

    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    base_model     = args.base_model     or preset["base"]
    instruct_model = args.instruct_model or preset["instruct"]
    target_model   = args.target_model   or preset["target"]
    output_dir     = args.output_dir     or preset["output_dir"]

    try:
        stages = sorted({int(s.strip()) for s in args.stages.split(",") if s.strip()})
    except ValueError:
        parser.error(f"--stages must be comma-separated integers, got {args.stages!r}")
    if not stages or any(s not in (1, 2, 3) for s in stages):
        parser.error(f"--stages must be a subset of {{1,2,3}}, got {stages}")

    run_rain(
        base_model=base_model,
        instruct_model=instruct_model,
        target_model=target_model,
        output_dir=output_dir,
        stages=stages,
        preset=args.preset,
        max_samples=args.max_samples,
        reasoning_calib=os.path.abspath(args.reasoning_calib),
        layers_tail=args.layers_tail,
        merge_types=args.merge_types,
        max_seq_len=args.max_seq_len,
        lambda_ridge=args.lambda_ridge,
        cg_maxit=args.cg_maxit,
        cg_tol=args.cg_tol,
        compute_precision=args.compute_precision,
        qk_device=args.qk_device,
        vo_device=args.vo_device,
        ffn_device=args.ffn_device,
        qp_device=args.qp_device,
        qp_variant=args.qp_variant,
        qp_max_seq_len=args.qp_max_seq_len,
        prior_scalar=args.prior_scalar,
        l2_prior=args.l2_prior,
        l1_reg=args.l1,
        box_lo=args.box_lo,
        box_hi=args.box_hi,
        h_lambda=args.h_lambda,
        h_mu=args.h_mu,
        rho_du=args.rho_du,
        kappa_a=args.kappa_a,
        kappa_u=args.kappa_u,
        decouple_qk=args.decouple_qk,
        scaling_factor=args.scaling_factor,
        model_name=args.model_name,
        revision_base=args.revision_base,
        revision_instruct=args.revision_instruct,
        revision_target=args.revision_target,
        explicit_projected_file=args.projected_file,
        explicit_alpha_file=args.alpha_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
