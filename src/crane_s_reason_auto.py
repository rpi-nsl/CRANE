#!/usr/bin/env python3
"""Architecture-adaptive Pareto-Taylor S_reason for Qwen3 dense / MoE / Next.

Per parameter j with instruct→thinking delta δ_j:
    improvement_R(j) = − grad_R,j · δ_j           (D_R: reasoning)
    improvement_T(j) = − grad_T,j · δ_j           (D_T: clean tool-use)
    g_j              = max(0, min(improvement_R(j), improvement_T(j)))

Component aggregation:
    S_raw(c)    = Σ_{j ∈ c} g_j / sqrt(Σ_{j ∈ c} θ_j^2)
    S_reason(c) = S_raw(c) / S_raw(BASELINE)

BASELINE is "ffn" for dense, "expert" for MoE / Next.
"""

import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _common import (  # noqa: E402
    CACHE_DIR,
    PRESETS,
    SUBCOMPS_BY_GROUP,
    ThinkingWeightLoader,
    baseline_component,
    classify_component,
    classify_subcomponent,
    detect_family,
    layer_index,
    load_arch,
)

# crane/ calibration texts are pure-Python / tokenizer-driven, arch-agnostic
from crane_calibration import (  # noqa: E402
    get_d_r_texts,
    get_d_t_texts,
)
from crane_calibration_public import (  # noqa: E402
    PUBLIC_CALIBRATION_SETS,
    discover_public_calibration_sets,
    get_public_calibration_texts,
    is_public_calibration_set,
)


MAX_SEQ_LEN = 2048


# ── Per-element Pareto-Taylor accumulation ─────────────────────────────────

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_layer_range(model, lo: int, hi: int):
    for i in range(lo, hi):
        for p in model.model.layers[i].parameters():
            p.requires_grad_(True)


def decode_targets(
    model, tokenizer, prompts: List[str], device: torch.device,
    max_new_tokens: int = 512, max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = 8,
):
    """Batched greedy-decode; returns (input_ids, attention_mask, labels)
    triples where labels has -100 over prompt and post-EOS padding."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id or pad_id

    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    model.eval()

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attn_in = enc["attention_mask"].to(device)
        max_prompt_len = input_ids.shape[1]
        t0 = time.time()
        print(f"    decode batch {bi+1}/{n_batches}: "
              f"{len(batch)} prompts, prompt_len={max_prompt_len}, "
              f"max_new={max_new_tokens} ...", flush=True)

        with torch.inference_mode():
            full = model.generate(
                input_ids=input_ids,
                attention_mask=attn_in,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        gen_lens = []
        for r in range(full.shape[0]):
            gen_part = full[r, max_prompt_len:]
            n_real = int((gen_part != pad_id).sum().item())
            gen_lens.append(n_real)
        print(f"    decode batch {bi+1}/{n_batches} done in "
              f"{time.time()-t0:.1f}s, gen_len min/mean/max = "
              f"{min(gen_lens)}/{sum(gen_lens)//len(gen_lens)}/{max(gen_lens)}",
              flush=True)

        for r in range(full.shape[0]):
            ids = full[r].cpu()
            in_attn = attn_in[r].cpu()
            attn_row = torch.ones_like(ids)
            attn_row[:max_prompt_len] = in_attn
            gen_pad_mask = (ids[max_prompt_len:] == pad_id)
            attn_row[max_prompt_len:][gen_pad_mask] = 0

            labels = ids.clone()
            labels[:max_prompt_len] = -100
            labels[max_prompt_len:][gen_pad_mask] = -100

            out.append((
                ids.unsqueeze(0),
                attn_row.unsqueeze(0),
                labels.unsqueeze(0),
            ))

        del full, input_ids, attn_in
        torch.cuda.empty_cache()

    return out


def to_device(triples, device: torch.device):
    return [
        (a.to(device), b.to(device), c.to(device))
        for a, b, c in triples
    ]


def run_backward(model, toks):
    """Accumulate gradient across calibration triples into param.grad.
    model.train() is required for HF gradient checkpointing to kick in."""
    model.train()
    for input_ids, attention_mask, labels in toks:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
        loss.backward()
        del out, loss


def accumulate_pareto(
    model,
    toks_R, toks_T,
    device: torch.device,
    think_loader: ThinkingWeightLoader,
    layer_chunk: int,
):
    """Per-chunk, per-element Pareto-Taylor accumulation (see module docstring)."""
    n_R = len(toks_R)
    n_T = len(toks_T)

    comp_agg = defaultdict(lambda: {
        "pareto_sum": 0.0,
        "numel": 0,
        "impR_sum": 0.0,
        "impT_sum": 0.0,
        "theta_sq": 0.0,
    })
    sub_numel = defaultdict(int)

    num_layers = model.config.num_hidden_layers
    num_chunks = math.ceil(num_layers / layer_chunk)
    freeze_all(model)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("    grad_checkpointing: ON", flush=True)

    for ci in range(num_chunks):
        lo = ci * layer_chunk
        hi = min(lo + layer_chunk, num_layers)
        t0 = time.time()
        unfreeze_layer_range(model, lo, hi)

        run_backward(model, toks_R)

        grad_R_cpu: Dict[str, torch.Tensor] = {}
        for i in range(lo, hi):
            prefix = f"model.layers.{i}."
            for name, p in model.model.layers[i].named_parameters():
                if p.grad is None:
                    continue
                grad_R_cpu[prefix + name] = (
                    (p.grad.detach().float() / n_R)
                    .to(dtype=torch.bfloat16, device="cpu", non_blocking=False)
                )

        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        run_backward(model, toks_T)

        missing = 0
        for i in range(lo, hi):
            prefix = f"model.layers.{i}."
            for name, p in model.model.layers[i].named_parameters():
                full = prefix + name
                gR_cpu = grad_R_cpu.get(full)
                if gR_cpu is None or p.grad is None:
                    missing += 1
                    continue

                # device_map="auto" places shards across GPUs — route think
                # tensor and saved grad to wherever this param actually lives.
                p_device = p.data.device
                t_think = think_loader.get(
                    full, device=p_device, dtype=torch.bfloat16,
                )
                if t_think is None or t_think.shape != p.shape:
                    missing += 1
                    continue

                delta = (t_think - p.data.detach().to(torch.bfloat16))
                del t_think

                gR = gR_cpu.to(device=p_device, non_blocking=True)
                gT = (p.grad.detach() / n_T).to(torch.bfloat16)

                imp_R = -(gR * delta)
                imp_T = -(gT * delta)
                del delta, gR, gT

                pareto = torch.clamp(torch.minimum(imp_R, imp_T), min=0.0)

                comp = classify_component(full)
                sub = classify_subcomponent(full)
                layer = layer_index(full)
                key = (comp, layer)
                comp_agg[key]["pareto_sum"] += float(pareto.float().sum().item())
                comp_agg[key]["impR_sum"]  += float(imp_R.float().sum().item())
                comp_agg[key]["impT_sum"]  += float(imp_T.float().sum().item())
                comp_agg[key]["numel"]     += p.numel()
                comp_agg[key]["theta_sq"]  += float(
                    (p.data.detach().float() ** 2).sum().item()
                )
                sub_numel[sub]             += p.numel()

                del imp_R, imp_T, pareto
                grad_R_cpu[full] = None
                torch.cuda.empty_cache()

        grad_R_cpu.clear()
        model.zero_grad(set_to_none=True)
        for i in range(lo, hi):
            for p in model.model.layers[i].parameters():
                p.requires_grad_(False)
                p.grad = None
        torch.cuda.empty_cache()
        gc.collect()

        print(
            f"    chunk {ci+1}/{num_chunks} (layers {lo}-{hi-1}): "
            f"{time.time()-t0:.1f}s, missing={missing}, "
            f"GPU {torch.cuda.memory_allocated(device)/1024**3:.1f} GB"
        )

    return dict(comp_agg), dict(sub_numel)


# ── Main ───────────────────────────────────────────────────────────────────

def resolve_snapshot(model_id: str) -> str:
    return snapshot_download(
        repo_id=model_id, cache_dir=CACHE_DIR,
        allow_patterns=["*.safetensors", "*.index.json", "config.json",
                        "*.json", "*.model", "*.txt"],
    )


def backfill_phase2_stats(
    s_reason: Dict[str, float],
    sub_numel: Dict[str, int],
) -> Dict:
    positive = [v for v in s_reason.values() if v > 0]
    max_sr = max(positive) if positive else 1.0
    per_component = {}
    for group, subs in SUBCOMPS_BY_GROUP.items():
        rho = (max(0.0, s_reason.get(group, 0.0) / max_sr)
               if max_sr > 0 else 0.0)
        for sub in subs:
            n = sub_numel.get(sub, 0)
            if n == 0:
                continue
            i_bucket = max(0, min(n, int(round(rho * n))))
            per_component[sub] = {
                "P": 0,
                "I": i_bucket,
                "N": n - i_bucket,
                "S": 0,
                "total": n,
            }
    return {"per_component": per_component}


def _save_targets(path: str, triples, meta: Dict[str, object] | None = None):
    payload = [(a.cpu(), b.cpu(), c.cpu()) for a, b, c in triples]
    if meta is not None:
        payload = {"meta": meta, "triples": payload}
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_targets(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "triples" in payload:
        return payload["triples"]
    return payload


def _prompt_digest(prompts: List[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def build_target_cache_meta(
    *,
    role: str,
    model_preset: str,
    calibration_set: str,
    model_id: str,
    tokenizer_id: str,
    prompts: List[str],
    max_new_tokens: int,
    max_seq_len: int,
) -> Dict[str, object]:
    return {
        "cache_version": 2,
        "role": role,
        "model_preset": model_preset,
        "calibration_set": calibration_set,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "prompt_count": len(prompts),
        "prompt_digest": _prompt_digest(prompts),
        "max_new_tokens": max_new_tokens,
        "max_seq_len": max_seq_len,
    }


def build_target_cache_path(cache_dir: str, meta: Dict[str, object]) -> str:
    digest = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()[:16]
    return os.path.join(
        cache_dir,
        f"{meta['model_preset']}_{meta['calibration_set']}_{meta['role']}"
        f"_seq{meta['max_seq_len']}_new{meta['max_new_tokens']}_{digest}.pt",
    )


def pick_reasoning_device(main_device: torch.device,
                          multi_gpu: bool = False) -> torch.device:
    if multi_gpu:
        return main_device
    if main_device.type != "cuda":
        return main_device
    n_cuda = torch.cuda.device_count()
    if n_cuda < 2:
        return main_device
    main_idx = 0 if main_device.index is None else main_device.index
    for idx in range(n_cuda):
        if idx != main_idx:
            return torch.device(f"cuda:{idx}")
    return main_device


def _device_map_for(device: torch.device, multi_gpu: bool):
    """from_pretrained device_map: sharded vs single-GPU."""
    if multi_gpu:
        return "auto"
    return {"": device}


def decode_targets_to_cache(
    model_id: str,
    tokenizer_id: str,
    prompts: List[str],
    device_str: str,
    cache_path: str,
    cache_meta: Dict[str, object],
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
):
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    triples = decode_targets(
        model,
        tokenizer,
        prompts,
        device,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
    _save_targets(cache_path, triples, meta=cache_meta)


def decode_targets_worker(
    model_id: str,
    tokenizer_id: str,
    prompts: List[str],
    device_str: str,
    cache_path: str,
    cache_meta: Dict[str, object],
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
):
    try:
        print(
            f"  [worker] decoding {len(prompts)} prompts for {model_id} "
            f"on {device_str} → {cache_path}",
            flush=True,
        )
        decode_targets_to_cache(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            prompts=prompts,
            device_str=device_str,
            cache_path=cache_path,
            cache_meta=cache_meta,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
        )
        print(f"  [worker] cached → {cache_path}", flush=True)
    except Exception:
        print("  [worker] D_R decode failed:", flush=True)
        traceback.print_exc()
        raise


def launch_decode_worker(
    model_id: str,
    tokenizer_id: str,
    prompts: List[str],
    device: torch.device,
    cache_path: str,
    cache_meta: Dict[str, object],
    max_new_tokens: int,
    max_seq_len: int,
    batch_size: int,
):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=decode_targets_worker,
        kwargs={
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "prompts": prompts,
            "device_str": str(device),
            "cache_path": cache_path,
            "cache_meta": cache_meta,
            "max_new_tokens": max_new_tokens,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
        },
    )
    proc.start()
    return proc


def get_calibration_texts(tokenizer, calibration_set: str):
    if calibration_set in (None, "", "default"):
        return (
            get_d_r_texts(tokenizer),
            get_d_t_texts(tokenizer),
            "D_T",
        )
    if is_public_calibration_set(calibration_set):
        return get_public_calibration_texts(tokenizer, calibration_set)
    raise SystemExit(
        f"Unknown --calibration-set {calibration_set!r}. Known: "
        f"{['default', *discover_public_calibration_sets()]}"
    )


def resolve_model_ids(args):
    """Return (instruct_id, thinking_id) resolved from preset + CLI override."""
    if args.model_preset not in PRESETS:
        raise SystemExit(f"Unknown --model-preset {args.model_preset!r}; "
                         f"known: {list(PRESETS)}")
    p = PRESETS[args.model_preset]
    inst = args.instruct_id or p["instruct_id"]
    think = args.thinking_id or p["thinking_id"]
    return inst, think, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-preset", required=True,
                   choices=list(PRESETS.keys()),
                   help="Model family preset (qwen3-4b / qwen3-30b / qwen3-next-80b)")
    p.add_argument("--instruct-id", default=None,
                   help="Override the preset's instruct model id")
    p.add_argument("--thinking-id", default=None,
                   help="Override the preset's thinking model id")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--multi-gpu", action="store_true",
                   help="Shard model across all visible GPUs via "
                        "device_map='auto' (required for Qwen3-Next-80B on "
                        "4×80GB; forces sequential D_R/D_T decode).")
    p.add_argument(
        "--calibration-set",
        default="default",
        help="default = canonical D_R/D_T from crane_calibration.py (headline "
             "recipe). Override with public_<name> for any "
             "${CRANE_DATA_DIR}/calibration_public/<name>.jsonl "
             "(e.g. public_mix_seed0, public_gsm8k_seed42_n36, "
             "public_bootstrap_seed0; used for the §A.5 robustness sweep).",
    )
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-new-r", type=int, default=None)
    p.add_argument("--max-new-t", type=int, default=None)
    p.add_argument("--decode-batch", type=int, default=20)
    p.add_argument("--layer-chunk", type=int, default=None,
                   help="Override the preset's default layer chunk size")
    p.add_argument("--targets-cache", default="${CRANE_DATA_DIR}/calib_targets")
    p.add_argument("--force-decode", action="store_true")
    args = p.parse_args()

    instruct_id, thinking_id, preset = resolve_model_ids(args)

    max_new_r = args.max_new_r if args.max_new_r is not None else args.max_new_tokens
    max_new_t = args.max_new_t if args.max_new_t is not None else 2048

    print("\n  Resolving snapshots ...")
    inst_dir = resolve_snapshot(instruct_id)
    think_dir = resolve_snapshot(thinking_id)
    print(f"  instruct dir: {inst_dir}")
    print(f"  thinking dir: {think_dir}")

    # Load arch and pick family / defaults after the snapshot exists in cache.
    arch = load_arch(instruct_id, cache_dir=CACHE_DIR)
    family = detect_family(arch)
    BASELINE = baseline_component(family)
    layer_chunk = (args.layer_chunk if args.layer_chunk is not None
                   else int(preset["default_layer_chunk"]))

    main_device = torch.device(args.device)
    reasoning_device = pick_reasoning_device(main_device, args.multi_gpu)
    split_decode = reasoning_device != main_device

    print("=" * 64)
    print("  CRANE auto S_reason (arch-adaptive): Pareto-gated Taylor")
    print(f"  preset: {args.model_preset}   family: {family}   baseline: {BASELINE}")
    print(f"  instruct: {instruct_id}")
    print(f"  thinking: {thinking_id}")
    print(f"  arch: {arch.num_layers}L, hidden={arch.hidden_dim}, "
          f"experts={arch.num_experts}, model_type={arch.model_type}")
    print(f"  layer_chunk: {layer_chunk}")
    print(f"  calibration_set: {args.calibration_set}")
    print(f"  main_device: {main_device}")
    print(f"  D_R decode device: {reasoning_device}")
    if split_decode:
        print("  multi-GPU split decode: ON")
    print(f"  output: {args.output}")
    print("=" * 64)

    os.makedirs(args.targets_cache, exist_ok=True)

    # Tokenizer — use instruct's; both variants share the same chat template.
    tok = AutoTokenizer.from_pretrained(instruct_id, cache_dir=CACHE_DIR)
    d_r_texts, d_t_texts, d_t_label = get_calibration_texts(
        tok, args.calibration_set
    )
    print(f"  |D_R| = {len(d_r_texts)}, |{d_t_label}| = {len(d_t_texts)}")

    dr_cache_meta = build_target_cache_meta(
        role="d_r_thinking",
        model_preset=args.model_preset,
        calibration_set=args.calibration_set,
        model_id=thinking_id,
        tokenizer_id=instruct_id,
        prompts=d_r_texts,
        max_new_tokens=max_new_r,
        max_seq_len=args.max_seq_len,
    )
    dt_cache_meta = build_target_cache_meta(
        role="d_t_instruct",
        model_preset=args.model_preset,
        calibration_set=args.calibration_set,
        model_id=instruct_id,
        tokenizer_id=instruct_id,
        prompts=d_t_texts,
        max_new_tokens=max_new_t,
        max_seq_len=args.max_seq_len,
    )
    dr_path = build_target_cache_path(args.targets_cache, dr_cache_meta)
    dt_path = build_target_cache_path(args.targets_cache, dt_cache_meta)
    print(f"  D_R cache: {dr_path}")
    print(f"  D_T cache: {dt_path}")

    triples_R = None
    dr_worker = None
    if os.path.exists(dr_path) and not args.force_decode:
        print(f"\n  Loading cached D_R targets: {dr_path}")
        triples_R = _load_targets(dr_path)
    elif split_decode:
        print(
            f"\n  Launching D_R target decode on {reasoning_device} "
            f"while {main_device} handles D_T/main work ..."
        )
        dr_worker = launch_decode_worker(
            model_id=thinking_id,
            tokenizer_id=instruct_id,
            prompts=d_r_texts,
            device=reasoning_device,
            cache_path=dr_path,
            cache_meta=dr_cache_meta,
            max_new_tokens=max_new_r,
            max_seq_len=args.max_seq_len,
            batch_size=args.decode_batch,
        )
    if triples_R is None and dr_worker is None:
        print("\n  Loading thinking model ...")
        t0 = time.time()
        think_model = AutoModelForCausalLM.from_pretrained(
            thinking_id, cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16,
            device_map=_device_map_for(reasoning_device, args.multi_gpu),
        )
        print(f"  loaded in {time.time()-t0:.1f}s, "
              f"GPU {torch.cuda.memory_allocated(reasoning_device)/1024**3:.1f} GB")
        print(f"  Decoding {len(d_r_texts)} D_R prompts (greedy, "
              f"max_new={max_new_r}) ...")
        t0 = time.time()
        triples_R = decode_targets(
            think_model, tok, d_r_texts, reasoning_device,
            max_new_tokens=max_new_r,
            max_seq_len=args.max_seq_len,
            batch_size=args.decode_batch,
        )
        print(f"  decoded in {time.time()-t0:.1f}s")
        _save_targets(dr_path, triples_R, meta=dr_cache_meta)
        print(f"  cached → {dr_path}")
        del think_model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n  Loading instruct model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        instruct_id, cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map=_device_map_for(main_device, args.multi_gpu),
    )
    print(f"  loaded in {time.time()-t0:.1f}s, "
          f"GPU {torch.cuda.memory_allocated(main_device)/1024**3:.1f} GB")

    triples_T = None
    if os.path.exists(dt_path) and not args.force_decode:
        print(f"  Loading cached D_T targets: {dt_path}")
        triples_T = _load_targets(dt_path)
    if triples_T is None:
        print(f"  Decoding {len(d_t_texts)} D_T prompts (greedy, "
              f"max_new={max_new_t}) ...")
        t0 = time.time()
        triples_T = decode_targets(
            model, tok, d_t_texts, main_device,
            max_new_tokens=max_new_t,
            max_seq_len=args.max_seq_len,
            batch_size=args.decode_batch,
        )
        print(f"  decoded in {time.time()-t0:.1f}s")
        _save_targets(dt_path, triples_T, meta=dt_cache_meta)
        print(f"  cached → {dt_path}")

    if triples_R is None and dr_worker is not None:
        print(f"\n  Waiting for D_R decode on {reasoning_device} ...")
        dr_worker.join()
        if dr_worker.exitcode != 0:
            raise RuntimeError(
                f"D_R decode worker failed on {reasoning_device} "
                f"(exitcode={dr_worker.exitcode})"
            )
        triples_R = _load_targets(dr_path)
        print(f"  Loaded D_R targets from worker cache: {dr_path}")

    toks_R = to_device(triples_R, main_device)
    toks_T = to_device(triples_T, main_device)

    think_loader = ThinkingWeightLoader(think_dir, num_experts=arch.num_experts)
    print(f"\n  thinking weight_map: {len(think_loader.weight_map)} keys "
          f"(num_experts={arch.num_experts})")
    print(f"\n  Accumulating per-element Pareto-Taylor")
    comp_agg, sub_numel = accumulate_pareto(
        model, toks_R, toks_T, main_device, think_loader,
        layer_chunk=layer_chunk,
    )

    del model, tok, toks_R, toks_T
    torch.cuda.empty_cache()
    gc.collect()

    # Per-layer S_raw and per-layer S_reason (normalized to BASELINE within each layer)
    s_raw_pl: Dict[Tuple[str, int], float] = {}
    for (comp, layer), v in comp_agg.items():
        if layer is None:
            continue
        tn = math.sqrt(v.get("theta_sq", 0.0))
        s_raw_pl[(comp, layer)] = (v["pareto_sum"] / tn) if tn > 0 else 0.0

    layers = sorted({l for (_, l) in s_raw_pl})
    components_seen = sorted({c for (c, _) in s_raw_pl})

    s_reason_pl: Dict[int, Dict[str, float]] = {}
    for l in layers:
        baseline_raw_l = s_raw_pl.get((BASELINE, l), 0.0)
        if baseline_raw_l <= 0:
            baseline_raw_l = 1e-12
        layer_dict = {}
        for c in components_seen:
            raw = s_raw_pl.get((c, l), 0.0)
            layer_dict[c] = raw / baseline_raw_l
        layer_dict[BASELINE] = 1.0
        s_reason_pl[l] = layer_dict

    # Aggregated per-component (across layers)
    agg_global = defaultdict(lambda: {"pareto_sum": 0.0, "theta_sq": 0.0})
    sub_numel_global = dict(sub_numel)
    for (comp, layer), v in comp_agg.items():
        agg_global[comp]["pareto_sum"] += v["pareto_sum"]
        agg_global[comp]["theta_sq"]   += v["theta_sq"]

    s_raw = {
        c: (v["pareto_sum"] / math.sqrt(v["theta_sq"])) if v["theta_sq"] > 0 else 0.0
        for c, v in agg_global.items()
    }
    baseline_raw = s_raw.get(BASELINE, 0.0) or 1e-12
    s_reason = {c: v / baseline_raw for c, v in s_raw.items()}
    s_reason[BASELINE] = 1.0

    print("\n  Per-component (aggregated across layers)")
    for c in sorted(s_reason):
        print(f"    {c:<14s} {s_reason[c]:>8.4f}")

    print(f"\n  Per-layer S_reason (normalized to {BASELINE}=1 within each layer)")
    header_cols = ["attention", "ffn", "expert", "shared_expert",
                   "linear_attn", "norm", "router"]
    col_keys = [c for c in header_cols if any(c in layer_dict for layer_dict in s_reason_pl.values())]
    if col_keys:
        header = "  " + "  ".join([f"{'layer':>5s}"] + [f"{k:>12s}" for k in col_keys])
        print(header)
        for l in layers:
            d = s_reason_pl[l]
            row = [f"{l:>5d}"] + [f"{d.get(k, 0):>12.4f}" for k in col_keys]
            print("  " + "  ".join(row))

    stats = backfill_phase2_stats(s_reason, sub_numel_global)
    stats["per_layer_S_reason"] = {
        str(l): {c: float(v) for c, v in d.items()}
        for l, d in s_reason_pl.items()
    }
    stats["per_component_S_reason"] = {c: float(v) for c, v in s_reason.items()}
    stats["metadata"] = {
        "model_preset": args.model_preset,
        "instruct_id": instruct_id,
        "thinking_id": thinking_id,
        "family": family,
        "baseline_component": BASELINE,
        "calibration_set": args.calibration_set,
        "num_d_r": len(d_r_texts),
        "num_d_t": len(d_t_texts),
        "main_device": str(main_device),
        "d_r_decode_device": str(reasoning_device),
        "layer_chunk": layer_chunk,
        "num_layers": arch.num_layers,
        "num_experts": arch.num_experts,
        "model_type": arch.model_type,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  wrote {args.output}")


if __name__ == "__main__":
    main()
