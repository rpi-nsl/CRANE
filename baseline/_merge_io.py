"""Shared IO helpers for baseline merge scripts: rerun-safe output prep
and index-driven streaming shard enumeration over two HF snapshots.
"""

import json
import os
import shutil
from collections import defaultdict
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


_DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.index.json",
    "config.json",
    "*.json",
    "*.model",
    "*.txt",
]


def validate_importance_sidecar(
    importance_path: str,
    expected_commit: str | None,
    expected_path: str | None,
    role: str,
    strict: bool = False,
) -> None:
    """Verify a precomputed importance .pt matches the model being merged.

    Reads the `<importance>.json` sidecar produced by `aim_importance.py` /
    `lewis_importance.py` and cross-checks the recorded HF commit / path.
    `strict=True` raises on mismatch; otherwise prints `[WARN]`.
    """
    sidecar = importance_path + ".json"
    if not os.path.isfile(sidecar):
        print(f"[{role}] WARN: no sidecar {sidecar!r}; cannot verify provenance "
              f"of {importance_path!r}")
        return
    try:
        with open(sidecar) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[{role}] WARN: failed to read {sidecar}: {e}")
        return

    sc = meta.get("resolved_commit")
    sp = meta.get("resolved_path")
    model_id = meta.get("model")

    def _fail(msg: str):
        full = (f"[{role}] importance file {importance_path!r} does not match "
                f"the current merge model: {msg}")
        if strict:
            raise ValueError(full)
        print(f"[{role}] WARN: {full}")

    if expected_commit and sc:
        if expected_commit != sc:
            _fail(f"expected commit {expected_commit!r}, sidecar has {sc!r} "
                  f"(model={model_id!r})")
            return
        print(f"[{role}] provenance ok: commit {sc[:12]} matches "
              f"(model={model_id!r})")
        return

    # Either no commit on one side — fall back to path comparison, which
    # is still meaningful for local directory models.
    if expected_path and sp:
        if os.path.realpath(expected_path) != os.path.realpath(sp):
            _fail(f"expected path {expected_path!r}, sidecar has {sp!r}")
            return
        print(f"[{role}] provenance ok: path matches (model={model_id!r})")
        return

    print(f"[{role}] WARN: could not verify provenance of {importance_path!r} "
          f"(no commit/path on one side); sidecar model={model_id!r}")


def pick_device(spec: str | None = None) -> torch.device:
    """Resolve a `--device` CLI value to a real `torch.device`.

    `spec=None` or `"auto"` picks `cuda:0` when CUDA is available, else CPU.
    Any explicit string (`"cpu"`, `"cuda"`, `"cuda:1"`) is passed through.
    Falls back to CPU with a printed warning if the user asks for CUDA but
    no GPU is visible — silently dropping to CPU on a 30B model would be
    a multi-hour foot-gun, so the warning is loud.
    """
    if spec is None or spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] requested device {spec!r} but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(spec)


def float_in_range(lo: float, hi: float, name: str = "value") -> Callable[[str], float]:
    """argparse `type=` validator for a float in a closed interval."""
    import argparse

    def _check(raw: str) -> float:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a float, got {raw!r}")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{lo}, {hi}], got {v}"
            )
        return v

    return _check


def resolve_model_snapshot(
    hf_id_or_path: str,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a HF id (or local path) to (local_dir, resolved_commit).

    `resolved_commit` is parsed from the snapshot path; None for local dirs.
    `revision` pins the commit when set; else snapshot_download picks latest.
    """
    # Local directory: use as-is, no commit info available.
    if os.path.isdir(hf_id_or_path) and not hf_id_or_path.startswith(("http://", "https://")):
        # If it's already a snapshot path, try to recover the hash from the path.
        commit = None
        parent = os.path.basename(os.path.dirname(hf_id_or_path.rstrip("/")))
        leaf = os.path.basename(hf_id_or_path.rstrip("/"))
        if parent == "snapshots" and len(leaf) >= 7 and all(c in "0123456789abcdef" for c in leaf):
            commit = leaf
        return hf_id_or_path, commit

    local = snapshot_download(
        hf_id_or_path,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=_DEFAULT_ALLOW_PATTERNS,
    )

    # snapshot_download returns .../snapshots/{commit_hash}/ — extract the hash.
    commit = None
    parts = local.rstrip("/").split(os.sep)
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if len(candidate) >= 7 and all(c in "0123456789abcdef" for c in candidate):
                commit = candidate
    return local, commit


def prepare_output_dir(output_dir: str, source_dir: str):
    """Wipe stale shard/metadata files in output_dir before a fresh merge."""
    if os.path.realpath(output_dir) == os.path.realpath(source_dir):
        raise ValueError(
            f"output_dir must differ from source_dir to avoid clobbering the "
            f"source snapshot: {output_dir}"
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return

    # Build the set of filenames we might write, so we can pre-remove them.
    to_remove: set[str] = set()
    for fname in os.listdir(output_dir):
        if fname.endswith(".safetensors") or fname.endswith(".index.json"):
            to_remove.add(fname)

    # Any file present in source_dir (other than shards themselves) will be
    # re-copied by copy_non_weight_files → remove stale copies first.
    if os.path.isdir(source_dir):
        for fname in os.listdir(source_dir):
            if fname.endswith(".safetensors"):
                continue
            to_remove.add(fname)

    # Previous merge_config.json (this script will rewrite it).
    to_remove.add("merge_config.json")

    for fname in to_remove:
        path = os.path.join(output_dir, fname)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def copy_non_weight_files(source_dir: str, output_dir: str):
    """Copy every non-weight file from source_dir to output_dir, overwriting."""
    for fname in os.listdir(source_dir):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(source_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


# ── Shard / weight-map driven IO ────────────────────────────────────────────


def load_weight_map(model_dir: str) -> tuple[dict[str, str], dict | None]:
    """Return (weight_map, index_metadata) for a model snapshot.

    `weight_map` is `{tensor_name: shard_filename}` from the snapshot's
    safetensors index (or synthesized for single-shard models).
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} has no weight_map")
        return weight_map, index

    # Single-shard fallback
    single = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(single):
        raise FileNotFoundError(
            f"Neither model.safetensors.index.json nor model.safetensors found in {model_dir}"
        )
    with safe_open(single, framework="pt") as f:
        weight_map = {k: "model.safetensors" for k in f.keys()}
    return weight_map, None


def _shard_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Invert weight_map → {shard_file: [tensor_name, ...]}."""
    out: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        out[shard].append(key)
    return out


class _ShardLoader:
    """Lazy safetensors loader that caches open file handles."""

    def __init__(self, model_dir: str, weight_map: dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}

    def get_tensor(self, name: str) -> torch.Tensor | None:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.model_dir, shard), framework="pt"
            )
        return self._handles[shard].get_tensor(name)

    def close(self):
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()


def iter_merge_by_instruct_shards(
    dir_i: str,
    dir_t: str,
    merge_fn: Callable[[str, torch.Tensor, "torch.Tensor | None"], "torch.Tensor | None"],
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """Drive the merge shard-by-shard, iterating instruct's index.

    For each instruct tensor, fetch the matching thinking tensor (across
    different shard layouts) and call `merge_fn(key, p_i, p_t) -> tensor|None`
    (None copies p_i unchanged). Returns a stats dict.
    """
    wm_i, _ = load_weight_map(dir_i)
    wm_t, _ = load_weight_map(dir_t)

    shard_to_keys_i = _shard_to_keys(wm_i)
    thinking_loader = _ShardLoader(dir_t, wm_t)

    new_weight_map: dict[str, str] = {}
    total_tensors = 0
    merged_tensors = 0
    shards_written = 0

    shard_names = sorted(shard_to_keys_i.keys())
    try:
        for shard_idx, shard_name in enumerate(shard_names):
            print(f"\n[{shard_idx+1}/{len(shard_names)}] Processing {shard_name} ...")
            shard_path = os.path.join(dir_i, shard_name)
            out_tensors: dict[str, torch.Tensor] = {}
            with safe_open(shard_path, framework="pt") as fh:
                for key in fh.keys():
                    p_i = fh.get_tensor(key)
                    p_t = thinking_loader.get_tensor(key)
                    total_tensors += 1
                    result = merge_fn(key, p_i, p_t)
                    if result is None:
                        out_tensors[key] = p_i
                    else:
                        out_tensors[key] = result
                        merged_tensors += 1
                    new_weight_map[key] = shard_name
            if not dry_run:
                save_file(out_tensors, os.path.join(output_dir, shard_name))
                print(f"  Saved {shard_name}")
            shards_written += 1
            del out_tensors
    finally:
        thinking_loader.close()

    # Thinking-only tensors: in wm_t but not in wm_i.
    thinking_only = sorted(set(wm_t.keys()) - set(wm_i.keys()))
    if thinking_only:
        print(
            f"\n  WARNING: {len(thinking_only)} tensors exist only in thinking model "
            f"and were NOT included in the merge:"
        )
        for k in thinking_only[:10]:
            print(f"    - {k}")
        if len(thinking_only) > 10:
            print(f"    ... and {len(thinking_only)-10} more")

    return {
        "total_tensors": total_tensors,
        "merged_tensors": merged_tensors,
        "thinking_only_tensors": thinking_only,
        "shards_written": shards_written,
        "weight_map": new_weight_map,
    }


def write_index_json(output_dir: str, weight_map: dict[str, str], total_size: int = 0):
    """Write a fresh model.safetensors.index.json matching the output shards."""
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
