#!/usr/bin/env python3
"""Compute AIM input-channel importance on the BASE (instruct) model.

Reference: Nobari et al., "Activation-Informed Merging of Large Language Models"
(https://github.com/ahnobari/ActivationInformedMerging).

Forward pre-hook captures `x.abs().mean(dim=0)` over each Linear's input
across a calibration set; output is `{module_name: (in_features,) tensor}`
consumed by `_aim.py::load_aim_importance` at merge time.
"""

import argparse
import json
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRANE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "crane")
sys.path.insert(0, CRANE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from _merge_io import resolve_model_snapshot  # noqa: E402

CACHE_DIR = os.environ.get("HF_HOME", "${HF_HOME}")

MODEL_PRESETS = {
    "qwen3-30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-next-80b": "Qwen/Qwen3-Next-80B-A3B-Instruct",
}


# Skip non-Linear weights and rotary buffers (mirrored in _aim.py).
_SKIP_SUBSTR = ("embed_tokens", "lm_head", "rotary")


def _should_hook(module_name: str, module: torch.nn.Module) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    for skip in _SKIP_SUBSTR:
        if skip in module_name:
            return False
    return True


def _load_calibration_texts(
    source: str,
    calib_file: str | None,
    num_samples: int,
    tokenizer,
    d_t_count: int = 10,
    seed: int = 42,
) -> list[str]:
    """Return a list of calibration prompts, already chat-templated.

    - source=crane_dr:     D_R only (20 hand-written reasoning prompts).
    - source=crane_drdt:   D_R + D_T, matching the Fisher calibration set
                           used by crane. D_T samples `d_t_count` SWE-bench
                           issues wrapped in the Roo-Code system prompt
                           (requires tokenizer chat template + the
                           SWE-bench_lite dataset in HF cache).
    - source=file:         JSON list of raw strings at `calib_file`.
                           Strings are used as-is (no chat template applied),
                           so the caller is responsible for formatting.

    D_R and D_T are both passed through the model's chat template so the
    recorded activations reflect actual inference-time inputs (system tag,
    turn markers, generation prompt) — not bare user text. This matches how
    crane's Fisher pipeline uses these sets.

    If `num_samples > 0` we truncate the final list to that length. We do
    NOT cycle: capping at `num_samples < len(pool)` is how the caller asks
    for a cheaper run, and silently cycling would misrepresent the result.
    """
    if source == "file":
        if not calib_file or not os.path.isfile(calib_file):
            raise FileNotFoundError(f"--calib-file not found: {calib_file}")
        with open(calib_file) as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError(f"{calib_file} must contain a JSON list of strings")
        texts = list(data)
    elif source in ("crane_dr", "crane_drdt"):
        from crane_calibration import get_d_r_texts  # type: ignore
        texts = list(get_d_r_texts(tokenizer))
        if source == "crane_drdt":
            # D_T needs the SWE-bench_lite dataset available via HF cache.
            # If it isn't (offline / missing cache), fall back to D_R-only
            # rather than aborting — the user gets a clear warning and the
            # final merge is still sound, just with a narrower calibration.
            try:
                from crane_calibration import get_d_t_texts  # type: ignore
                d_t = list(get_d_t_texts(tokenizer, n=d_t_count, seed=seed))
                texts.extend(d_t)
                print(f"[calib] D_R={len(texts) - len(d_t)}  D_T={len(d_t)}")
            except Exception as e:
                print(f"[calib] WARNING: D_T load failed ({e}); using D_R only")
    else:
        raise ValueError(f"unknown calib source: {source}")

    if not texts:
        raise ValueError("calibration text list is empty")
    if num_samples > 0 and num_samples < len(texts):
        return texts[:num_samples]
    return texts


def compute_importance(
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
    print(f"  AIM importance extraction")
    print(f"  Base model: {model_id}" + (f" @ {revision}" if revision else ""))
    print(f"  Calib:      {calib_source}" + (f" ({calib_file})" if calib_file else ""))
    print(f"  Samples:    {num_samples or 'all'},  max_length={max_length}")
    print(f"  Output:     {output_path}")
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

    # Per-Linear running sums (1-D, in_features), accumulated across samples.
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hook_handles = []

    def make_hook(name: str):
        def _hook(_mod, inputs, _output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if x is None:
                return
            # Flatten (..., in_features) -> (-1, in_features).
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] == 0:
                return
            # Keep the accumulator on the layer's own device under
            # device_map="auto" so we don't sync per step.
            contrib = flat.abs().mean(dim=0).detach().to(torch.float32)
            prev = sums.get(name)
            if prev is None:
                sums[name] = contrib
            else:
                sums[name] = prev + contrib
            counts[name] = counts.get(name, 0) + 1
        return _hook

    hooked = 0
    for name, module in model.named_modules():
        if _should_hook(name, module):
            hook_handles.append(module.register_forward_hook(make_hook(name)))
            hooked += 1
    print(f"Registered {hooked} forward hooks on Linear modules")

    texts = _load_calibration_texts(
        source=calib_source,
        calib_file=calib_file,
        num_samples=num_samples,
        tokenizer=tok,
        d_t_count=d_t_count,
    )
    print(f"Calibration: {len(texts)} prompts")

    # Single-sequence passes (batch=1) — matches the reference impl.
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i, text in enumerate(texts):
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
            ids = ids.to(device)
            model(ids)
            if (i + 1) % 5 == 0 or (i + 1) == len(texts):
                print(f"  [{i+1}/{len(texts)}] hooked modules so far: {len(sums)}")

    for h in hook_handles:
        h.remove()

    # Average per-module: MoE experts divide by their routed-token count
    # (so unused experts get correspondingly low importance, as AIM intends).
    importance: dict[str, torch.Tensor] = {}
    for name, total in sums.items():
        n = max(1, counts.get(name, 1))
        importance[name] = (total / n).contiguous().cpu()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(importance, output_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Saved importance for {len(importance)} modules")
    print(f"  Coverage: {len(importance)}/{hooked} hooked modules produced signal")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Sidecar JSON with provenance so a merged model can be audited without
    # having to reload the .pt file.
    meta = {
        "model": model_id,
        "resolved_commit": commit,
        "resolved_path": dir_base,
        "calib_source": calib_source,
        "calib_file": calib_file,
        "num_samples": len(texts),
        "max_length": max_length,
        "hooked_modules": hooked,
        "captured_modules": len(importance),
        "elapsed_s": round(elapsed, 1),
    }
    with open(output_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute AIM importance on a base model")
    parser.add_argument("--preset", default="qwen3-30b", choices=MODEL_PRESETS.keys())
    parser.add_argument("--model", type=str, default=None,
                        help="Override the preset base model id / local path")
    parser.add_argument("--output", type=str, required=True,
                        help="Where to write the importance .pt file")
    parser.add_argument("--calib-source", default="crane_drdt",
                        choices=["crane_dr", "crane_drdt", "file"],
                        help="crane_dr = D_R only (20 reasoning prompts); "
                             "crane_drdt = D_R + D_T (D_T samples SWE-bench "
                             "wrapped in Roo-Code system prompt — needs the "
                             "SWE-bench_lite dataset in HF cache); "
                             "file = raw JSON list via --calib-file")
    parser.add_argument("--calib-file", type=str, default=None,
                        help="JSON list of strings (used when --calib-source=file)")
    parser.add_argument("--d-t-count", type=int, default=10,
                        help="Number of SWE-bench prompts to sample for D_T "
                             "(only used when --calib-source=crane_drdt)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Cap on calibration prompts (0 = all available)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    model_id = args.model or MODEL_PRESETS[args.preset]
    compute_importance(
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
