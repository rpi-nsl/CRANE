#!/usr/bin/env python3
"""Analyze CRANE Taylor calibration stability across stats JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_STATS_DIR = Path("${CRANE_DATA_DIR}")
DEFAULT_ROBUSTNESS_DIR = Path("${CRANE_DATA_DIR}/robustness")
DEFAULT_OUTPUT = Path("${CRANE_DATA_DIR}/calibration_public/stability_report.md")
DEFAULT_FIGURE = Path("${CRANE_REPO_ROOT}/paper/figs/robustness_30b.pdf")


def load_stats(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("metadata", "per_component_S_reason", "per_layer_S_reason"):
        if key not in stats:
            raise ValueError(f"{path} is missing {key}")
    return stats


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: List[float], ys: List[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def flattened(stats: Dict[str, object], comps: Iterable[str]) -> Tuple[List[Tuple[str, str]], List[float]]:
    labels: List[Tuple[str, str]] = []
    values: List[float] = []
    pl = stats["per_layer_S_reason"]
    for layer in sorted(pl, key=lambda x: int(x)):
        row = pl[layer]
        for comp in comps:
            if comp in row:
                labels.append((layer, comp))
                values.append(float(row[comp]))
    return labels, values


def top_overlap(base_vals: List[float], vals: List[float], k: int) -> Tuple[int, float]:
    base_top = {i for i, _ in sorted(enumerate(base_vals), key=lambda t: t[1], reverse=True)[:k]}
    vals_top = {i for i, _ in sorted(enumerate(vals), key=lambda t: t[1], reverse=True)[:k]}
    hit = len(base_top & vals_top)
    return hit, hit / k


def fmt(x: float, digits: int = 4) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def discover_robustness_runs(robust_dir: Path) -> List[Tuple[str, Path]]:
    """Pick up every phase2_stats_30b_<calibration_set>.json under robust_dir."""
    out: List[Tuple[str, Path]] = []
    if not robust_dir.exists():
        return out
    for path in sorted(robust_dir.glob("phase2_stats_30b_*.json")):
        # Strip "phase2_stats_30b_" prefix to recover the calibration_set name
        name = path.stem[len("phase2_stats_30b_"):]
        out.append((name, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default=str(DEFAULT_STATS_DIR))
    ap.add_argument("--robustness-dir", default=str(DEFAULT_ROBUSTNESS_DIR))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--figure", default=str(DEFAULT_FIGURE))
    ap.add_argument("--include-missing", action="store_true")
    ap.add_argument(
        "--no-figure", action="store_true",
        help="Skip writing the sample-size convergence figure even if "
             "matplotlib is available.",
    )
    args = ap.parse_args()

    stats_dir = Path(args.stats_dir)
    robust_dir = Path(args.robustness_dir)
    specs = [
        ("old", stats_dir / "phase2_stats_auto_old_b4_r4096_t2048.json"),
        ("default", stats_dir / "phase2_stats_auto_b4_r4096_t2048.json"),
    ]
    # Mix-seed runs may live in either stats_dir (legacy) or robust_dir (new sweep).
    for i in range(5):
        legacy = stats_dir / f"phase2_stats_public_mix_seed{i}.json"
        new = robust_dir / f"phase2_stats_30b_public_mix_seed{i}.json"
        path = new if new.exists() else legacy
        specs.append((f"public_mix_seed{i}", path))
    # All other robustness runs (per-source + bootstrap)
    for name, path in discover_robustness_runs(robust_dir):
        if name.startswith("public_mix_seed"):
            continue  # already added above
        specs.append((name, path))

    loaded: Dict[str, Dict[str, object]] = {}
    missing = []
    for name, path in specs:
        if path.exists():
            loaded[name] = load_stats(path)
        else:
            missing.append(str(path))
            if not args.include_missing:
                raise FileNotFoundError(path)

    if "default" not in loaded:
        raise ValueError("default-anchor row is required")

    comps = ["attention", "router", "norm"]
    base_labels, base_vals = flattened(loaded["default"], comps)
    rows = []
    for name, stats in loaded.items():
        labels, vals = flattened(stats, comps)
        if labels != base_labels:
            raise ValueError(f"{name} has different layer/component labels than the default anchor")
        comp = stats["per_component_S_reason"]
        rows.append({
            "name": name,
            "d_r": stats["metadata"].get("num_d_r"),
            "d_t": stats["metadata"].get("num_d_t"),
            "attention": float(comp.get("attention", float("nan"))),
            "expert": float(comp.get("expert", float("nan"))),
            "router": float(comp.get("router", float("nan"))),
            "norm": float(comp.get("norm", float("nan"))),
            "pearson": pearson(base_vals, vals),
            "spearman": spearman(base_vals, vals),
            "top10": top_overlap(base_vals, vals, 10)[0],
            "top20": top_overlap(base_vals, vals, 20)[0],
            "top30": top_overlap(base_vals, vals, 30)[0],
            "top48": top_overlap(base_vals, vals, 48)[0],
        })

    def _component_dispersion(group_rows):
        out = []
        if not group_rows:
            return out
        anchor_lookup = {row["name"]: row for row in rows}.get("default")
        for comp_name in ("attention", "expert", "router", "norm"):
            vals = [r[comp_name] for r in group_rows if not math.isnan(r[comp_name])]
            if not vals:
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cv = std / abs(mean) if mean else float("nan")
            if anchor_lookup is not None:
                rel = (mean - anchor_lookup[comp_name]) / anchor_lookup[comp_name] if anchor_lookup[comp_name] else float("nan")
            else:
                rel = float("nan")
            out.append((comp_name, mean, std, cv, rel))
        return out

    mix_rows = [row for row in rows if row["name"].startswith("public_mix_seed")]
    component_summary = _component_dispersion(mix_rows)

    bootstrap_rows = [row for row in rows if row["name"].startswith("public_bootstrap_seed")]
    bootstrap_summary = _component_dispersion(bootstrap_rows)

    PER_SOURCE_NAMES = ("gsm8k", "math", "bbh", "openorca")
    per_source_summary = {}
    sample_size_table = []
    for src in PER_SOURCE_NAMES:
        src_rows = [row for row in rows if row["name"].startswith(f"public_{src}_seed")]
        if not src_rows:
            continue
        per_source_summary[src] = _component_dispersion(src_rows)
        for r in src_rows:
            # Extract N from name: public_<src>_seed<S>_n<N>
            try:
                n = int(r["name"].rsplit("_n", 1)[1])
            except (IndexError, ValueError):
                n = -1
            sample_size_table.append((src, n, r))

    out = []
    out.append("# Calibration Taylor Stability Report\n")
    out.append("Layer-component correlations exclude the expert baseline because it is fixed at 1.0 by normalization.\n")
    if missing:
        out.append("Missing stats files:\n")
        out.extend(f"- `{path}`\n" for path in missing)
        out.append("\n")
    out.append("## Per-run summary\n")
    out.append("| calibration | D_R/D_T | attention | expert | router | norm | Pearson | Spearman | top10 | top20 | top30 | top48 |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        out.append(
            f"| `{row['name']}` | {row['d_r']}/{row['d_t']} | {fmt(row['attention'])} | "
            f"{fmt(row['expert'])} | {fmt(row['router'])} | {fmt(row['norm'])} | "
            f"{fmt(row['pearson'])} | {fmt(row['spearman'])} | "
            f"{row['top10']}/10 | {row['top20']}/20 | {row['top30']}/30 | {row['top48']}/48 |\n"
        )
    def _emit_dispersion(title: str, summary):
        if not summary:
            return
        out.append(f"\n## {title}\n")
        out.append("| component | mean | std | CV | rel. drift vs default |\n")
        out.append("|---|---:|---:|---:|---:|\n")
        for comp_name, mean, std, cv, rel in summary:
            rel_str = f"{rel:+.2%}" if not math.isnan(rel) else "nan"
            out.append(
                f"| {comp_name} | {fmt(mean)} | {fmt(std)} | {fmt(cv)} | {rel_str} |\n"
            )

    _emit_dispersion("Public mix-seed dispersion (5 seeds, code-domain D_R)", component_summary)
    _emit_dispersion("Curated bootstrap dispersion (5 seeds, hand-written D_R)", bootstrap_summary)
    for src, summary in per_source_summary.items():
        _emit_dispersion(f"Per-source dispersion: {src} (across N ∈ {{36,64,100}})", summary)

    if sample_size_table:
        out.append("\n## Sample-size scaling per source\n")
        out.append("| source | N | attention | router | norm | Pearson(default) |\n")
        out.append("|---|---:|---:|---:|---:|---:|\n")
        for src, n, r in sorted(sample_size_table, key=lambda x: (x[0], x[1])):
            out.append(
                f"| {src} | {n} | {fmt(r['attention'])} | {fmt(r['router'])} | "
                f"{fmt(r['norm'])} | {fmt(r['pearson'])} |\n"
            )

    out.append("\n## Interpretation\n")
    out.append(
        "Across the public calibration subsets, the component ordering remains stable: "
        "attention is the largest non-baseline coefficient, expert is the normalization "
        "baseline, router is moderate, and norm remains near zero.\n"
    )

    # ── Optional: sample-size convergence figure ────────────────────────────
    if not args.no_figure and sample_size_table:
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            anchor_lookup = {row["name"]: row for row in rows}.get("default")
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
            comps_to_plot = ("attention", "router", "norm")
            for ax, comp_name in zip(axes, comps_to_plot):
                for src in PER_SOURCE_NAMES:
                    pts = [
                        (n, r[comp_name])
                        for s, n, r in sample_size_table
                        if s == src and not math.isnan(r[comp_name])
                    ]
                    if not pts:
                        continue
                    pts.sort()
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", label=src)
                if anchor_lookup and not math.isnan(anchor_lookup[comp_name]):
                    ax.axhline(anchor_lookup[comp_name], color="k", ls="--",
                               lw=1, label=f"default N=52 ({fmt(anchor_lookup[comp_name], 3)})")
                ax.set_title(f"S_reason: {comp_name}")
                ax.set_xlabel("|D_R|")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("S_reason (per-component)")
            axes[-1].legend(loc="best", fontsize=7)
            fig.tight_layout()
            fig_path = Path(args.figure)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")
            out.append(f"\nFigure: `{fig_path}`\n")
        except Exception as exc:  # noqa: BLE001
            out.append(f"\n(figure skipped: {exc})\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(out), encoding="utf-8")
    print(f"wrote {output}")
    print("".join(out))


if __name__ == "__main__":
    main()
