#!/usr/bin/env python3
"""Aggregate per-task pass/fail across multiple attempts of the same task.

Reads all result.json under a harbor job_dir, groups by task name (everything
before `__<uuid>` in the trial dir name), and reports:

  per-task    : pass_count / n_attempts (e.g. 2/3 means 2 of 3 attempts passed)
  pass_any@k  : task counts as pass if AT LEAST ONE attempt passed (best-of-n)
  pass_all@k  : task counts as pass only if ALL attempts passed (most strict)
  pass_mean   : average reward across all trials per task
  task_n_pass / task_n_fail : task-level summary at threshold (default 0.5 = majority)

Usage:
  python aggregate.py <job_dir>           # job_dir = e.g. .../baseline-ta_tb2-zai_<stamp>
  python aggregate.py <job_dir> --threshold 0.5
  python aggregate.py <job_dir> --json    # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def collect(job_dir: Path) -> dict[str, list[dict]]:
    """Return {task_name: [{reward, exc_type, trial_name}, ...]}."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    # job_dir layout: <job_dir>/<job_name>/<task__uuid>/result.json
    for r in job_dir.glob("*/*/result.json"):
        trial_dir = r.parent
        task_name = trial_dir.name.split("__")[0]
        try:
            d = json.loads(r.read_text())
        except Exception as e:
            by_task[task_name].append({"reward": 0.0, "exc_type": f"unreadable:{e}", "trial_name": trial_dir.name})
            continue
        ei = d.get("exception_info")
        rewards = (d.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", 0) or 0
        # Match wrapper's mutually-exclusive convention: exc trumps reward.
        # (A trial can have reward>0 AND exception_info — e.g. agent timed out
        # but verifier still passed; we count this as exc, not pass.)
        is_exc = ei is not None
        is_pass = (not is_exc) and float(reward) > 0
        by_task[task_name].append({
            "reward": 0.0 if is_exc else float(reward),
            "raw_reward": float(reward),
            "exc_type": (ei or {}).get("exception_type"),
            "is_pass": is_pass,
            "is_exc": is_exc,
            "trial_name": trial_dir.name,
        })
    return by_task


def aggregate(by_task: dict[str, list[dict]], threshold: float = 0.5):
    """
    Compute three pass@k metrics over n attempts per task.

    pass@1   : single-shot expected pass count = sum(n_pass / n_attempts for task)
               (= mean trial reward × n_tasks; what you'd get running each task once)
    pass@k   : best-of-k = at least 1 of k attempts passed (= pass_any)
    pass@all : strict consistency = all k attempts passed (= pass_all)

    Three are listed from least → most demanding. Gap between pass@1 and pass@k
    measures variance; gap between pass@k and pass@all measures stability.
    """
    pass1 = 0.0      # single-shot expected count (fractional)
    pass_k = 0       # best-of-k (any one passed)
    pass_all = 0     # all attempts passed
    pass_majority = 0  # >= threshold rate
    n_tasks = 0
    pertask = []
    for task, attempts in sorted(by_task.items()):
        n = len(attempts)
        n_pass = sum(1 for a in attempts if a["is_pass"])
        rate = n_pass / n if n else 0
        any_pass = n_pass > 0
        all_pass = n_pass == n
        maj_pass = rate >= threshold and rate > 0
        pertask.append({
            "task": task,
            "n_attempts": n,
            "n_pass": n_pass,
            "rate": rate,
            "any": any_pass,
            "all": all_pass,
            "majority": maj_pass,
        })
        n_tasks += 1
        pass1 += rate
        if any_pass: pass_k += 1
        if all_pass: pass_all += 1
        if maj_pass: pass_majority += 1
    return {
        "n_tasks": n_tasks,
        "pass@1": pass1,              # single-shot expected count
        "pass@k": pass_k,             # best-of-k
        "pass@all": pass_all,         # strict all-of-k
        "pass_majority": pass_majority,  # >=threshold rate (kept for compat)
        "pertask": pertask,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-tasks", action="store_true",
                    help="Print full per-task table (default: only stable pass / borderline / fails)")
    args = ap.parse_args()

    if not args.job_dir.exists():
        print(f"ERROR: {args.job_dir} not found", file=sys.stderr)
        sys.exit(1)

    by_task = collect(args.job_dir)
    if not by_task:
        print(f"ERROR: no result.json found under {args.job_dir}", file=sys.stderr)
        sys.exit(1)

    res = aggregate(by_task, args.threshold)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    # Detect varying n
    ns = sorted({pt["n_attempts"] for pt in res["pertask"]})
    n_label = ns[0] if len(ns) == 1 else f"varies {ns}"
    k = ns[0] if len(ns) == 1 else "k"

    print(f"\n=== Aggregate ({args.job_dir.name}) ===")
    print(f"  tasks: {res['n_tasks']}, attempts/task: {n_label}")
    print(f"  pass@1         (single-shot, expected)  : {res['pass@1']:>5.1f} / {res['n_tasks']}")
    print(f"  pass@{k:<3}      (best-of-{k}, any pass)      : {res['pass@k']:>5} / {res['n_tasks']}")
    print(f"  pass_majority  (rate >= {args.threshold:.2f})         : {res['pass_majority']:>5} / {res['n_tasks']}")
    print(f"  pass@all       (all-of-{k}, strict)        : {res['pass@all']:>5} / {res['n_tasks']}")

    # show ones with mixed results (most informative)
    mixed = [pt for pt in res["pertask"] if 0 < pt["n_pass"] < pt["n_attempts"]]
    stable_pass = [pt for pt in res["pertask"] if pt["all"]]
    if stable_pass:
        print(f"\n  Stable pass ({len(stable_pass)}):")
        for pt in stable_pass: print(f"    {pt['task']:40s} {pt['n_pass']}/{pt['n_attempts']}")
    if mixed:
        print(f"\n  Borderline / unstable ({len(mixed)}):")
        for pt in sorted(mixed, key=lambda x: -x["rate"]):
            print(f"    {pt['task']:40s} {pt['n_pass']}/{pt['n_attempts']} ({pt['rate']:.0%})")

    if args.show_tasks:
        print("\n  All tasks:")
        for pt in res["pertask"]:
            print(f"    {pt['task']:40s} {pt['n_pass']}/{pt['n_attempts']} ({pt['rate']:.0%})")


if __name__ == "__main__":
    main()
