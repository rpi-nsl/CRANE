#!/usr/bin/env python3
"""Aggregate per-model SWE-bench eval into a single results.{md,json} file.

Run after one model's harness has finished. Walks the model's output dir to:
  - tally resolved / unresolved from per-instance report.json files
  - sum prompt / completion / cache_read tokens from per-instance traj.json
  - read harness summary JSON for the official resolved_instances count
  - capture vLLM-side prefix cache stats if the server is still up

Writes:
  <out_dir>/results.json   — machine-readable single-source-of-truth
  <out_dir>/results.md     — human-friendly summary

Usage:
  python aggregate_model_results.py <out_dir> [--model SHORTNAME] [--vllm-port 18020]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


def harness_summary(out_dir: Path) -> dict:
    # The harness writes <model>.<run_tag>.json (with `resolved_instances`).
    # Skip preds.json and *.traj.json.
    candidates = [
        p for p in out_dir.glob("*.json")
        if p.name not in ("preds.json", "results.json")
        and not p.name.endswith(".traj.json")
    ]
    if not candidates:
        return {}
    # Pick the one with `resolved_instances` key, else the first.
    for p in candidates:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "resolved_instances" in d:
            return {"path": str(p.relative_to(out_dir)), **d}
    return {}


def per_instance_tally(out_dir: Path) -> dict:
    paths = list(out_dir.glob("logs/**/report.json"))
    resolved = unresolved = 0
    for p in paths:
        try:
            d = json.load(open(p))
            iid = next(iter(d))
            if d[iid].get("resolved", False):
                resolved += 1
            else:
                unresolved += 1
        except Exception:
            pass
    return {
        "graded": resolved + unresolved,
        "resolved": resolved,
        "unresolved": unresolved,
    }


def token_tally(out_dir: Path) -> dict:
    paths = list(out_dir.glob("*.traj.json"))
    sums = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }
    n_with_metrics = 0
    n_with_cache = 0
    for fp in paths:
        try:
            t = json.load(open(fp))
        except Exception:
            continue
        m = t.get("metrics") or {}
        u = m.get("accumulated_token_usage") or {}
        if not u:
            continue
        n_with_metrics += 1
        for k in sums:
            v = u.get(k, 0) or 0
            sums[k] += int(v)
        if (u.get("cache_read_tokens") or 0) > 0:
            n_with_cache += 1
    return {
        "trajectories": len(paths),
        "with_token_metrics": n_with_metrics,
        "with_cache_data": n_with_cache,
        "totals": sums,
    }


def vllm_cache_stats(port: int) -> dict:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/metrics", timeout=5
        ) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        if line.startswith(("vllm:prompt_tokens_total{",
                            "vllm:prompt_tokens_cached_total{")):
            key = line.split("{", 1)[0].split(":", 1)[1]
            try:
                val = float(line.rsplit(" ", 1)[1])
            except Exception:
                continue
            out[key] = int(val)
    if "prompt_tokens_total" in out and out["prompt_tokens_total"] > 0:
        out["cache_hit_rate"] = round(
            out.get("prompt_tokens_cached_total", 0) / out["prompt_tokens_total"], 4
        )
    return out


def render_md(model: str, summary: dict, tally: dict, tokens: dict, vllm: dict) -> str:
    res = summary.get("resolved_instances")
    tot = summary.get("total_instances")
    pr_official = f"{100 * res / tot:.1f}%" if (res is not None and tot) else "n/a"
    pr_recount_g = (
        f"{100 * tally['resolved'] / tally['graded']:.1f}%"
        if tally["graded"] else "n/a"
    )
    pr_recount_500 = (
        f"{100 * tally['resolved'] / 500:.1f}%"
        if tally["graded"] else "n/a"
    )
    sums = tokens["totals"]
    lines = [
        f"# {model} — SWE-bench Verified results",
        "",
        f"_Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Pass rate",
        "",
        f"| Source | Resolved | Total | Pass rate |",
        f"|---|---:|---:|---:|",
        f"| harness summary JSON | {res if res is not None else '—'} | {tot if tot is not None else '—'} | {pr_official} |",
        f"| per-instance report tally (graded subset) | {tally['resolved']} | {tally['graded']} | {pr_recount_g} |",
        f"| vs 500 (treats ungraded as failed) | {tally['resolved']} | 500 | {pr_recount_500} |",
        "",
        "## Token usage (sum across trajectories)",
        "",
        f"| Token type | Total |",
        f"|---|---:|",
        f"| prompt_tokens | {sums['prompt_tokens']:,} |",
        f"| completion_tokens | {sums['completion_tokens']:,} |",
        f"| cache_read_tokens | {sums['cache_read_tokens']:,} |",
        f"| cache_write_tokens | {sums['cache_write_tokens']:,} |",
        f"| reasoning_tokens | {sums['reasoning_tokens']:,} |",
        "",
        f"_Coverage: {tokens['with_token_metrics']} / {tokens['trajectories']} trajectories with metrics; "
        f"{tokens['with_cache_data']} reported cache_read>0._",
        "",
    ]
    if vllm:
        cached = vllm.get("prompt_tokens_cached_total", 0)
        total = vllm.get("prompt_tokens_total", 0)
        rate = vllm.get("cache_hit_rate", 0)
        lines += [
            "## vLLM-side prefix cache (server-wide, may include cross-model)",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| prompt_tokens_total | {total:,} |",
            f"| prompt_tokens_cached_total | {cached:,} |",
            f"| cache_hit_rate | {rate * 100:.1f}% |",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=None,
                    help="model shortname for the headline (default: derived from dir name)")
    ap.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "18020")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_dir():
        print(f"ERROR: not a directory: {out_dir}", file=sys.stderr)
        return 2

    model = args.model or out_dir.name.split("_verified_test")[0]

    summary = harness_summary(out_dir)
    tally = per_instance_tally(out_dir)
    tokens = token_tally(out_dir)
    vllm = vllm_cache_stats(args.vllm_port)

    payload = {
        "model": model,
        "out_dir": str(out_dir),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "harness_summary": summary,
        "tally_from_reports": tally,
        "token_usage": tokens,
        "vllm_cache_stats": vllm,
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_md(model, summary, tally, tokens, vllm))

    sums = tokens["totals"]
    print(f"[results] {model}: "
          f"resolved={tally['resolved']}/{tally['graded']} "
          f"({100 * tally['resolved'] / tally['graded']:.1f}% of graded, "
          f"{100 * tally['resolved'] / 500:.1f}% of 500); "
          f"prompt={sums['prompt_tokens']:,} "
          f"completion={sums['completion_tokens']:,} "
          f"cache_read={sums['cache_read_tokens']:,}")
    print(f"[results] wrote {json_path.relative_to(out_dir.parent)}, {md_path.relative_to(out_dir.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
