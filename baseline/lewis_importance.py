#!/usr/bin/env python3
"""LEWIS Wanda-style per-Linear importance for one model (arXiv:2503.03874).

Computes Wanda's `|| W * sqrt(scaler_row) ||_F` Frobenius scalar per
nn.Linear inside a transformer block; the merge step (lewis.py) consumes
two of these (base + fine-tuned) to derive a TIES per-layer density
schedule. We hook all transformer-block Linears (works for dense and MoE)
and skip the per-layer pruning of the reference Wanda pipeline since LEWIS
only needs the unpruned scalar.
"""

import argparse
import json
import os
import re
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRANE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "crane")
sys.path.insert(0, CRANE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _merge_io import resolve_model_snapshot  # noqa: E402
from aim_importance import _load_calibration_texts  # noqa: E402

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")

MODEL_PRESETS = {
    "qwen3-30b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-30b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-next-80b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "qwen3-next-80b-thinking": "Qwen/Qwen3-Next-80B-A3B-Thinking",
}


# Only transformer-block Linears (skip embed/lm_head/layernorms/rotary).
_LAYER_RE = re.compile(r"model\.layers\.\d+\.")
_SKIP_SUBSTR = ("embed_tokens", "lm_head", "rotary", "norm")


def _should_hook(module_name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    if not _LAYER_RE.search(module_name):
        return False
    for skip in _SKIP_SUBSTR:
        if skip in module_name:
            return False
    return True


def compute_lewis_importance(
    model_id: str,
    output_path: str,
    calib_source: str = "crane_drdt",
    calib_file: str | None = None,
    num_samples: int = 0,
    max_length: int = 512,
    revision: str | None = None,
    d_t_count: int = 10,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  LEWIS importance extraction (Wanda || W * sqrt(E[x^2]) ||_F)")
    print(f"  Model:  {model_id}" + (f" @ {revision}" if revision else ""))
    print(f"  Calib:  {calib_source}" + (f" ({calib_file})" if calib_file else ""))
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    t0 = time.time()

    dir_base, commit = resolve_model_snapshot(model_id, CACHE_DIR, revision)
    print(f"Resolved: {dir_base}  (commit: {commit or 'N/A'})")

    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(dir_base, trust_remote_code=True)

    print("Loading model (bfloat16, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        dir_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # scaler_row[name]: running sum of x_t^2 per input channel; counts[name]
    # is per-module since MoE routing makes per-expert token counts differ.
    scaler_row: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hook_handles = []

    def make_hook(name: str):
        def _hook(_mod, inputs, _output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if x is None:
                return
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] == 0:
                return
            # Keep on the layer's device (per-Linear vector is a few kB).
            sq_sum = (flat.float() ** 2).sum(dim=0).detach()
            prev = scaler_row.get(name)
            if prev is None:
                scaler_row[name] = sq_sum
            else:
                scaler_row[name] = prev + sq_sum
            counts[name] = counts.get(name, 0) + flat.shape[0]
        return _hook

    hooked_modules: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if _should_hook(name, module):
            hook_handles.append(module.register_forward_hook(make_hook(name)))
            hooked_modules.append((name, module))
    print(f"Registered {len(hooked_modules)} forward hooks on transformer-block Linears")

    texts = _load_calibration_texts(
        source=calib_source,
        calib_file=calib_file,
        num_samples=num_samples,
        tokenizer=tok,
        d_t_count=d_t_count,
    )
    print(f"Calibration: {len(texts)} prompts")

    device = next(model.parameters()).device
    with torch.inference_mode():
        for i, text in enumerate(texts):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
            ids = ids.to(device)
            model(ids)
            if (i + 1) % 5 == 0 or (i + 1) == len(texts):
                print(f"  [{i+1}/{len(texts)}] modules with signal: {len(scaler_row)}")

    for h in hook_handles:
        h.remove()

    # Wanda Frobenius scalar per hooked Linear:
    #   imp = sqrt( sum_{i,j} W[i,j]^2 * E[x[j]^2] )
    # Modules that never received tokens (unrouted MoE experts) are omitted
    # from the output dict; build_density_schedule treats missing keys as
    # fall-through. Saving them as 0.0 would corrupt the |Δ| normalization.
    importance: dict[str, float] = {}
    skipped_no_signal = 0
    for name, module in hooked_modules:
        s = scaler_row.get(name)
        if s is None or counts.get(name, 0) == 0:
            skipped_no_signal += 1
            continue
        mean_sq = (s / counts[name]).to(module.weight.device)
        with torch.no_grad():
            w_sq = module.weight.float() ** 2          # (out, in)
            val = (w_sq * mean_sq.unsqueeze(0)).sum().sqrt().item()
        importance[name] = float(val)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(importance, output_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Saved LEWIS importance for {len(importance)} / {len(hooked_modules)} modules")
    print(f"  Skipped {skipped_no_signal} modules with no calibration signal "
          f"(e.g. unrouted MoE experts)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    meta = {
        "model": model_id,
        "resolved_commit": commit,
        "resolved_path": dir_base,
        "calib_source": calib_source,
        "calib_file": calib_file,
        "num_samples": len(texts),
        "max_length": max_length,
        "hooked_modules": len(hooked_modules),
        "captured_modules": len(importance),
        "skipped_no_signal": skipped_no_signal,
        "elapsed_s": round(elapsed, 1),
    }
    with open(output_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute LEWIS Wanda importance on one model")
    parser.add_argument("--preset", default=None, choices=MODEL_PRESETS.keys(),
                        help="Convenience preset for the qwen3-30b / qwen3-4b instruct/thinking models")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the preset model id / local path")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to write the importance .pt file")
    parser.add_argument("--calib-source", default="crane_drdt",
                        choices=["crane_dr", "crane_drdt", "file"])
    parser.add_argument("--calib-file", type=str, default=None,
                        help="JSON list of strings (used when --calib-source=file)")
    parser.add_argument("--d-t-count", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    if args.model is None and args.preset is None:
        parser.error("must specify --model or --preset")
    model_id = args.model or MODEL_PRESETS[args.preset]

    compute_lewis_importance(
        model_id=model_id,
        output_path=args.output,
        calib_source=args.calib_source,
        calib_file=args.calib_file,
        num_samples=args.num_samples,
        max_length=args.max_length,
        revision=args.revision,
        d_t_count=args.d_t_count,
    )


if __name__ == "__main__":
    main()
