#!/usr/bin/env python3
"""Public-source calibration loaders for CRANE Taylor-stability checks.

The builder writes deterministic JSONL files under
``${CRANE_DATA_DIR}/calibration_public``. This module only reads
those frozen files and applies the active tokenizer chat template.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from crane_calibration import _ROO_SYSTEM
except ModuleNotFoundError:
    from .crane_calibration import _ROO_SYSTEM


PUBLIC_CALIBRATION_DIR = Path(
    os.environ.get(
        "CRANE_PUBLIC_CALIBRATION_DIR",
        "${CRANE_DATA_DIR}/calibration_public",
    )
)

# Names ending in any of these suffixes are raw source pools, not directly
# Taylor-consumable subsets.
_POOL_SUFFIXES = ("_pool", "_lcb_bcb_reason", "_swe_tool")


def discover_public_calibration_sets() -> Tuple[str, ...]:
    """List every public calibration name with a JSONL on disk.

    Excludes raw source pools (``public_*_pool``, ``public_lcb_bcb_reason``,
    ``public_swe_tool``).
    """
    if not PUBLIC_CALIBRATION_DIR.exists():
        return ()
    names: List[str] = []
    for path in sorted(PUBLIC_CALIBRATION_DIR.glob("*.jsonl")):
        stem = path.stem
        if any(stem.endswith(s) for s in _POOL_SUFFIXES):
            continue
        names.append(stem)
    return tuple(names)


# Back-compat: the original five mix-seed names. Downstream code that wants the
# *complete* current list should call discover_public_calibration_sets().
PUBLIC_CALIBRATION_SETS = tuple(f"public_mix_seed{i}" for i in range(5))


def _jsonl_path(calibration_set: str) -> Path:
    return PUBLIC_CALIBRATION_DIR / f"{calibration_set}.jsonl"


def is_public_calibration_set(calibration_set: str) -> bool:
    """Return True if a JSONL file exists for this name (sources excluded)."""
    if not calibration_set.startswith("public_"):
        return False
    if any(calibration_set.endswith(s) for s in _POOL_SUFFIXES):
        return False
    return _jsonl_path(calibration_set).exists()


def _load_items(calibration_set: str) -> List[Dict[str, object]]:
    path = _jsonl_path(calibration_set)
    if not path.exists():
        raise FileNotFoundError(
            f"Public calibration file is missing: {path}. "
            "Run crane/build_public_calibration_subsets.py first."
        )
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("role") not in {"D_R", "D_T"}:
                raise ValueError(f"{path}:{line_no} has invalid role={item.get('role')!r}")
            if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
                raise ValueError(f"{path}:{line_no} has an empty prompt")
            items.append(item)
    return items


def get_public_calibration_items(calibration_set: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Return frozen public D_R/D_T item dictionaries for a calibration set."""
    if not is_public_calibration_set(calibration_set):
        known = list(discover_public_calibration_sets())
        raise ValueError(
            f"Unknown public calibration set {calibration_set!r}; "
            f"known on disk: {known}"
        )
    items = _load_items(calibration_set)
    d_r = [dict(item) for item in items if item["role"] == "D_R"]
    d_t = [dict(item) for item in items if item["role"] == "D_T"]
    if not d_r or not d_t:
        raise ValueError(
            f"{calibration_set} must contain at least one D_R and one D_T "
            f"prompt; got {len(d_r)} and {len(d_t)}"
        )
    return d_r, d_t


def _apply_user_template(tokenizer, prompts: Iterable[str]) -> List[str]:
    texts = []
    for prompt in prompts:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
        texts.append(text)
    return texts


def _apply_roo_template(tokenizer, prompts: Iterable[str]) -> List[str]:
    texts = []
    for prompt in prompts:
        msgs = [
            {"role": "system", "content": _ROO_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            merged = _ROO_SYSTEM + "\n\n" + prompt
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
        texts.append(text)
    return texts


def get_public_calibration_texts(tokenizer, calibration_set: str) -> Tuple[List[str], List[str], str]:
    """Return chat-templated public D_R and D_T prompts for Taylor scoring."""
    d_r, d_t = get_public_calibration_items(calibration_set)
    d_r_texts = _apply_user_template(tokenizer, (str(item["prompt"]) for item in d_r))
    d_t_texts = _apply_roo_template(tokenizer, (str(item["prompt"]) for item in d_t))
    return d_r_texts, d_t_texts, f"D_T_{calibration_set}"
