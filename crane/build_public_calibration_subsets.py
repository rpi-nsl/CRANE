#!/usr/bin/env python3
"""Build deterministic public calibration subsets for Taylor stability.

Outputs (default sweep):
  - public_lcb_bcb_reason.jsonl: prompt pool from LiveCodeBench + BigCodeBench.
  - public_swe_tool.jsonl: prompt pool from SWE-bench full, excluding Verified ids.
  - public_mix_seed{0..4}.jsonl: 36 D_R + 16 D_T prompts per seed.

Per-source robustness sweep (``--per-source``):
  - public_<source>_pool.jsonl: per-source D_R prompt pool (gsm8k/math/bbh/openorca).
  - public_<source>_seed<S>_n<N>.jsonl: N D_R from one source + 16 D_T from
    public_swe_tool (held fixed across sources to isolate the D_R signal).

Curated bootstrap (``--bootstrap``):
  - public_bootstrap_seed<S>.jsonl: D_R/D_T subsampled from the
    D_R / D_T from crane_calibration with given seed.

manifest.json is regenerated on every run with the union of all .jsonl files in
output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from datasets import load_dataset


DEFAULT_CACHE_DIR = "${HF_HOME}"
DEFAULT_OUTPUT_DIR = "${CRANE_DATA_DIR}/calibration_public"
PER_SOURCE_NAMES = ("gsm8k", "math", "bbh", "openorca")


def _clean(text: object, limit: int = 5000) -> str:
    value = "" if text is None else str(text)
    value = "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").splitlines())
    value = "\n".join(line for line in value.splitlines() if line.strip())
    return value.strip()[:limit]


def _hash_prompts(items: Iterable[Dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("role", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.get("source", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.get("source_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.get("prompt", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _write_jsonl(path: Path, items: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stream_dataset(name: str, split: str, cache_dir: str) -> Iterator[Dict[str, object]]:
    return iter(load_dataset(name, split=split, cache_dir=cache_dir, streaming=True))


def build_reason_pool(cache_dir: str, max_lcb: int, max_bcb: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []

    for row in _stream_dataset("livecodebench/code_generation", "test", cache_dir):
        content = _clean(row.get("question_content"))
        title = _clean(row.get("question_title"), limit=240)
        if len(content) < 200:
            continue
        prompt = (
            "Answer directly without chain-of-thought. /no_think\n"
            "Give exactly four short bullets for this programming problem: "
            "Algorithm, Invariant, Edge cases, Complexity. Do not write code. "
            "Keep the whole answer under 160 words.\n\n"
            f"Title: {title}\n"
            f"Platform: {row.get('platform', 'unknown')}\n\n"
            f"{content}"
        )
        items.append({
            "role": "D_R",
            "source": "LiveCodeBench/code_generation",
            "source_id": str(row.get("question_id")),
            "prompt": prompt,
        })
        if sum(item["source"].startswith("LiveCodeBench") for item in items) >= max_lcb:
            break

    for row in _stream_dataset("bigcode/bigcodebench", "v0.1.4", cache_dir):
        content = _clean(row.get("instruct_prompt"))
        if len(content) < 160:
            continue
        prompt = (
            "Answer directly without chain-of-thought. /no_think\n"
            "Give exactly four short bullets for the requested function: Behavior, "
            "Library calls, Edge cases, Tests. Do not write code. Keep the whole "
            "answer under 160 words.\n\n"
            f"{content}"
        )
        items.append({
            "role": "D_R",
            "source": "bigcode/bigcodebench:v0.1.4",
            "source_id": str(row.get("task_id")),
            "prompt": prompt,
        })
        if sum(item["source"].startswith("bigcode") for item in items) >= max_bcb:
            break

    items.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    return items


def _verified_ids(cache_dir: str) -> set:
    ds = load_dataset(
        "SWE-bench/SWE-bench_Verified",
        split="test",
        cache_dir=cache_dir,
        streaming=True,
    )
    return {str(row["instance_id"]) for row in islice(ds, 10000)}


def build_tool_pool(cache_dir: str, max_swe: int) -> List[Dict[str, object]]:
    skip_ids = _verified_ids(cache_dir)
    items: List[Dict[str, object]] = []
    for split in ("test", "dev", "train"):
        for row in _stream_dataset("SWE-bench/SWE-bench", split, cache_dir):
            instance_id = str(row.get("instance_id"))
            if instance_id in skip_ids:
                continue
            statement = _clean(row.get("problem_statement"), limit=3500)
            if not (200 <= len(statement) <= 3500):
                continue
            repo = str(row.get("repo"))
            prompt = (
                f"Fix the following issue in the {repo} repository.\n\n"
                "Read the relevant source files, identify the root cause, patch the "
                "smallest correct change, and run focused tests to verify. Start with "
                "one concrete tool-use step or a concise action plan. Keep the first "
                "response short and do not invent file contents before reading them.\n\n"
                "## Issue\n\n"
                f"{statement}"
            )
            items.append({
                "role": "D_T",
                "source": f"SWE-bench/SWE-bench:{split}",
                "source_id": instance_id,
                "repo": repo,
                "prompt": prompt,
            })
            if len(items) >= max_swe:
                items.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
                return items
    items.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    return items


def build_seed_mix(
    reason_pool: List[Dict[str, object]],
    tool_pool: List[Dict[str, object]],
    seed: int,
    d_r_size: int,
    d_t_size: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    lcb = [item for item in reason_pool if item["source"].startswith("LiveCodeBench")]
    bcb = [item for item in reason_pool if item["source"].startswith("bigcode")]
    if len(lcb) < d_r_size // 2 or len(bcb) < d_r_size - d_r_size // 2:
        raise ValueError("Reason pool is too small for stratified LiveCodeBench/BigCodeBench sampling")
    if len(tool_pool) < d_t_size:
        raise ValueError("SWE-bench tool pool is too small")

    d_r = rng.sample(lcb, d_r_size // 2) + rng.sample(bcb, d_r_size - d_r_size // 2)
    d_t = rng.sample(tool_pool, d_t_size)
    d_r.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    d_t.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))

    mixed = []
    for idx, item in enumerate(d_r):
        x = dict(item)
        x["subset_index"] = idx
        x["subset_seed"] = seed
        mixed.append(x)
    for idx, item in enumerate(d_t):
        x = dict(item)
        x["subset_index"] = idx
        x["subset_seed"] = seed
        mixed.append(x)
    return mixed


# ── Per-source D_R pools (cross-domain robustness) ─────────────────────────


def build_gsm8k_pool(cache_dir: str, max_n: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    ds = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    for row in ds:
        question = _clean(row.get("question"))
        if len(question) < 60:
            continue
        prompt = (
            "Answer directly without chain-of-thought. /no_think\n"
            "State the final numeric answer on its own line as `Answer: <n>`. "
            "Optionally precede it with at most two short bullets covering the "
            "key step and the unit; do not write paragraphs.\n\n"
            f"{question}"
        )
        items.append({
            "role": "D_R",
            "source": "openai/gsm8k:train",
            "source_id": f"gsm8k_train_{len(items):06d}",
            "prompt": prompt,
        })
        if len(items) >= max_n:
            break
    items.sort(key=lambda item: str(item["source_id"]))
    return items


def build_math_pool(cache_dir: str, max_n: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    ds = load_dataset(
        "HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir
    )
    for row in ds:
        problem = _clean(row.get("problem"))
        subject = _clean(row.get("subject"), limit=80) or "math"
        level = _clean(row.get("level"), limit=40) or "level-?"
        unique_id = (
            _clean(row.get("unique_id"), limit=200)
            or f"math500_{len(items):06d}"
        )
        if len(problem) < 40:
            continue
        prompt = (
            "Answer directly without chain-of-thought. /no_think\n"
            f"Solve the {subject} problem ({level}). Give the final answer "
            "in \\boxed{}; optionally precede it with one short bullet stating "
            "the key identity or theorem used.\n\n"
            f"{problem}"
        )
        items.append({
            "role": "D_R",
            "source": "HuggingFaceH4/MATH-500",
            "source_id": unique_id,
            "prompt": prompt,
        })
        if len(items) >= max_n:
            break
    items.sort(key=lambda item: str(item["source_id"]))
    return items


BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)


def build_bbh_pool(cache_dir: str, max_n: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    per_task_target = max(1, (max_n + len(BBH_TASKS) - 1) // len(BBH_TASKS))
    for task in BBH_TASKS:
        try:
            ds = load_dataset(
                "lukaemon/bbh", task, split="test", cache_dir=cache_dir
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [bbh] skipped {task}: {exc}", file=sys.stderr)
            continue
        taken = 0
        for idx, row in enumerate(ds):
            text = _clean(row.get("input"))
            if len(text) < 40:
                continue
            prompt = (
                "Answer directly without chain-of-thought. /no_think\n"
                f"This is a Big-Bench Hard ({task}) item. Give the final "
                "answer on its own line as `Answer: <choice>`; optionally "
                "precede it with one short bullet identifying the relevant "
                "rule.\n\n"
                f"{text}"
            )
            items.append({
                "role": "D_R",
                "source": f"lukaemon/bbh:{task}",
                "source_id": f"{task}_{idx:04d}",
                "prompt": prompt,
            })
            taken += 1
            if taken >= per_task_target:
                break
        if len(items) >= max_n:
            break
    items.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    return items[:max_n]


def build_openorca_pool(cache_dir: str, max_n: int) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    ds = load_dataset(
        "Open-Orca/OpenOrca", split="train", cache_dir=cache_dir, streaming=True
    )
    for row in ds:
        question = _clean(row.get("question"))
        if not (200 <= len(question) <= 2400):
            continue
        rid = str(row.get("id") or f"orca_{len(items):06d}")
        prompt = (
            "Answer directly without chain-of-thought. /no_think\n"
            "Give a concise answer in 3-5 short bullets; do not write a long "
            "essay.\n\n"
            f"{question}"
        )
        items.append({
            "role": "D_R",
            "source": "Open-Orca/OpenOrca",
            "source_id": rid,
            "prompt": prompt,
        })
        if len(items) >= max_n:
            break
    items.sort(key=lambda item: str(item["source_id"]))
    return items


SOURCE_POOL_BUILDERS = {
    "gsm8k": build_gsm8k_pool,
    "math": build_math_pool,
    "bbh": build_bbh_pool,
    "openorca": build_openorca_pool,
}


def build_per_source_subset(
    pool: List[Dict[str, object]],
    d_t_pool: List[Dict[str, object]],
    seed: int,
    d_r_size: int,
    d_t_size: int,
    source_name: str,
) -> List[Dict[str, object]]:
    """Sample N D_R from one source pool + fixed-size D_T from public_swe_tool."""
    if len(pool) < d_r_size:
        raise ValueError(
            f"{source_name} pool has {len(pool)} items, need >= {d_r_size}"
        )
    if len(d_t_pool) < d_t_size:
        raise ValueError(
            f"D_T pool has {len(d_t_pool)} items, need >= {d_t_size}"
        )
    rng = random.Random(seed)
    d_r = rng.sample(pool, d_r_size)
    d_t = rng.sample(d_t_pool, d_t_size)
    d_r.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    d_t.sort(key=lambda item: (str(item["source"]), str(item["source_id"])))
    out: List[Dict[str, object]] = []
    for idx, item in enumerate(d_r):
        x = dict(item)
        x["subset_index"] = idx
        x["subset_seed"] = seed
        x["subset_source"] = source_name
        out.append(x)
    for idx, item in enumerate(d_t):
        x = dict(item)
        x["subset_index"] = idx
        x["subset_seed"] = seed
        x["subset_source"] = "swe_tool"
        out.append(x)
    return out


def build_bootstrap_subset(
    seed: int,
    d_r_size: int,
    d_t_size: int,
) -> List[Dict[str, object]]:
    """Subsample the D_R / D_T lists by seed."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crane_calibration import (
        D_R_ITEMS,
        D_T_ITEMS,
    )
    if d_r_size > len(D_R_ITEMS):
        raise ValueError(
            f"D_R has {len(D_R_ITEMS)} items, "
            f"cannot bootstrap {d_r_size}"
        )
    if d_t_size > len(D_T_ITEMS):
        raise ValueError(
            f"D_T has {len(D_T_ITEMS)} items, "
            f"cannot bootstrap {d_t_size}"
        )
    rng = random.Random(seed)
    d_r_idx = sorted(rng.sample(range(len(D_R_ITEMS)), d_r_size))
    d_t_idx = sorted(rng.sample(range(len(D_T_ITEMS)), d_t_size))
    out: List[Dict[str, object]] = []
    for new_idx, src_idx in enumerate(d_r_idx):
        item = dict(D_R_ITEMS[src_idx])
        out.append({
            "role": "D_R",
            "source": "crane_calibration.D_R_ITEMS",
            "source_id": f"d_r_{src_idx:03d}",
            "prompt": item["prompt"],
            "source_style": item.get("source_style"),
            "subset_index": new_idx,
            "subset_seed": seed,
        })
    for new_idx, src_idx in enumerate(d_t_idx):
        item = dict(D_T_ITEMS[src_idx])
        out.append({
            "role": "D_T",
            "source": "crane_calibration.D_T_ITEMS",
            "source_id": f"d_t_{src_idx:03d}",
            "prompt": item["prompt"],
            "source_style": item.get("source_style"),
            "subset_index": new_idx,
            "subset_seed": seed,
        })
    return out


def summarize_file(path: Path) -> Dict[str, object]:
    items = _read_jsonl(path)
    return {
        "file": path.name,
        "total": len(items),
        "d_r": sum(item.get("role") == "D_R" for item in items),
        "d_t": sum(item.get("role") == "D_T" for item in items),
        "prompt_hash": _hash_prompts(items),
    }


def validate_outputs(output_dir: Path, seeds: List[int]) -> Dict[str, object]:
    files = [
        output_dir / "public_lcb_bcb_reason.jsonl",
        output_dir / "public_swe_tool.jsonl",
    ] + [output_dir / f"public_mix_seed{seed}.jsonl" for seed in seeds]
    files = [p for p in files if p.exists()]
    summary = {path.stem: summarize_file(path) for path in files}
    for seed in seeds:
        key = f"public_mix_seed{seed}"
        if key in summary and (summary[key]["d_r"] != 36 or summary[key]["d_t"] != 16):
            raise ValueError(
                f"{summary[key]['file']} expected 36/16 D_R/D_T, "
                f"got {summary[key]['d_r']}/{summary[key]['d_t']}"
            )
    return summary


def regenerate_manifest(output_dir: Path, seeds: List[int]) -> Dict[str, object]:
    """Walk every .jsonl in output_dir and regenerate manifest.json."""
    summary = {}
    for path in sorted(output_dir.glob("*.jsonl")):
        summary[path.stem] = summarize_file(path)
    manifest = {
        "source_datasets": {
            "D_R_mix": ["livecodebench/code_generation", "bigcode/bigcodebench:v0.1.4"],
            "D_R_per_source": list(SOURCE_POOL_BUILDERS.keys()),
            "D_T_mix_and_per_source": [
                "SWE-bench/SWE-bench excluding SWE-bench/SWE-bench_Verified instance_ids",
            ],
            "D_R_bootstrap": ["crane/crane_calibration.D_R_ITEMS"],
            "D_T_bootstrap": ["crane/crane_calibration.D_T_ITEMS"],
        },
        "mix_seeds": seeds,
        "per_source_names": list(SOURCE_POOL_BUILDERS.keys()),
        "summary": summary,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--max-lcb", type=int, default=120)
    ap.add_argument("--max-bcb", type=int, default=120)
    ap.add_argument("--max-swe", type=int, default=200)
    ap.add_argument("--d-r-size", type=int, default=36)
    ap.add_argument("--d-t-size", type=int, default=16)
    ap.add_argument(
        "--per-source",
        nargs="*",
        default=None,
        help="Build per-source D_R subsets. Provide source names "
             f"(any of {list(SOURCE_POOL_BUILDERS)}) or omit values to build "
             "all four. Combined with --per-source-sizes and --per-source-seeds.",
    )
    ap.add_argument(
        "--per-source-sizes", default="36,64,100",
        help="Comma-separated D_R sample sizes for per-source subsets.",
    )
    ap.add_argument(
        "--per-source-seeds", default="42",
        help="Comma-separated seeds for per-source subsets.",
    )
    ap.add_argument(
        "--per-source-pool-size", type=int, default=400,
        help="Maximum items in each per-source D_R pool.",
    )
    ap.add_argument(
        "--bootstrap", action="store_true",
        help="Build D_R/D_T bootstrap subsets from crane_calibration items.",
    )
    ap.add_argument(
        "--bootstrap-seeds", default="0,1,2,3,4",
        help="Comma-separated seeds for D_R/D_T bootstrap subsets.",
    )
    ap.add_argument(
        "--bootstrap-d-r-size", type=int, default=36,
    )
    ap.add_argument(
        "--bootstrap-d-t-size", type=int, default=16,
    )
    ap.add_argument(
        "--skip-mix", action="store_true",
        help="Skip rebuilding LCB/BCB pool and mix-seed subsets (use existing).",
    )
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.check_only and not args.skip_mix:
        reason_pool = build_reason_pool(args.cache_dir, args.max_lcb, args.max_bcb)
        tool_pool = build_tool_pool(args.cache_dir, args.max_swe)
        _write_jsonl(output_dir / "public_lcb_bcb_reason.jsonl", reason_pool)
        _write_jsonl(output_dir / "public_swe_tool.jsonl", tool_pool)
        for seed in seeds:
            mixed = build_seed_mix(reason_pool, tool_pool, seed, args.d_r_size, args.d_t_size)
            _write_jsonl(output_dir / f"public_mix_seed{seed}.jsonl", mixed)

    if not args.check_only and args.per_source is not None:
        sources: Sequence[str] = (
            args.per_source if args.per_source else list(SOURCE_POOL_BUILDERS)
        )
        sizes = [int(x) for x in args.per_source_sizes.split(",") if x.strip()]
        ps_seeds = [int(x) for x in args.per_source_seeds.split(",") if x.strip()]
        pool_size = max(args.per_source_pool_size, max(sizes))
        d_t_pool_path = output_dir / "public_swe_tool.jsonl"
        if not d_t_pool_path.exists():
            raise FileNotFoundError(
                f"{d_t_pool_path} is missing. Run without --skip-mix first or "
                "regenerate the SWE pool."
            )
        d_t_pool = _read_jsonl(d_t_pool_path)
        for source in sources:
            if source not in SOURCE_POOL_BUILDERS:
                raise SystemExit(
                    f"Unknown per-source name {source!r}; "
                    f"known: {list(SOURCE_POOL_BUILDERS)}"
                )
            pool_path = output_dir / f"public_{source}_pool.jsonl"
            if pool_path.exists():
                pool = _read_jsonl(pool_path)
                if len(pool) < max(sizes):
                    pool = SOURCE_POOL_BUILDERS[source](args.cache_dir, pool_size)
                    _write_jsonl(pool_path, pool)
            else:
                pool = SOURCE_POOL_BUILDERS[source](args.cache_dir, pool_size)
                _write_jsonl(pool_path, pool)
            for size in sizes:
                for seed in ps_seeds:
                    items = build_per_source_subset(
                        pool=pool,
                        d_t_pool=d_t_pool,
                        seed=seed,
                        d_r_size=size,
                        d_t_size=args.d_t_size,
                        source_name=source,
                    )
                    out_path = output_dir / f"public_{source}_seed{seed}_n{size}.jsonl"
                    _write_jsonl(out_path, items)
                    print(f"  [per-source] wrote {out_path}")

    if not args.check_only and args.bootstrap:
        bs_seeds = [int(x) for x in args.bootstrap_seeds.split(",") if x.strip()]
        for seed in bs_seeds:
            items = build_bootstrap_subset(
                seed=seed,
                d_r_size=args.bootstrap_d_r_size,
                d_t_size=args.bootstrap_d_t_size,
            )
            out_path = output_dir / f"public_bootstrap_seed{seed}.jsonl"
            _write_jsonl(out_path, items)
            print(f"  [bootstrap] wrote {out_path}")

    validate_outputs(output_dir, seeds)
    manifest = regenerate_manifest(output_dir, seeds)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
