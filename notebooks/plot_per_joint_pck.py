"""
Generate per-joint PCK bar chart from an evaluation run.

Runs the full oracle evaluation pipeline on the DensePose val subset and
produces two figures saved to outputs/graphs/:

    per_joint_pck.png   — horizontal bar chart of PCK@0.2 per joint
    oks_summary.png     — gauge/text summary of OKS mean and MPJPE

Usage
-----
    python notebooks/plot_per_joint_pck.py
    python notebooks/plot_per_joint_pck.py --subset 100 --pck-threshold 0.2
    python notebooks/plot_per_joint_pck.py --results-json results/eval.json

If --results-json points to a previously saved evaluation JSON, the figures
are generated from that file without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-joint PCK bar chart from evaluation results."
    )
    parser.add_argument("--results-json", default=None,
                        help="Path to a JSON produced by eval.runner --out. "
                             "If omitted, runs evaluation first.")
    parser.add_argument("--root",          default="data/raw",    help="COCO data root")
    parser.add_argument("--split",         default="val2017",     help="Dataset split")
    parser.add_argument("--subset",        type=int, default=None,
                        help="Limit evaluation to N annotations (default: all)")
    parser.add_argument("--pck-threshold", type=float, default=0.2,
                        help="PCK threshold (default: 0.2)")
    parser.add_argument("--min-kp",        type=int, default=5,
                        help="Min visible keypoints per sample (default: 5)")
    parser.add_argument("--out",           default="outputs/graphs",
                        help="Output directory for figures (default: outputs/graphs)")
    return parser.parse_args()


# ── Run evaluation and return summary dict ────────────────────────────────────

def _run_evaluation(args: argparse.Namespace) -> dict:
    """Run the oracle evaluation pipeline and return a summary dict."""
    from eval.runner import _parse_args as runner_parse, evaluate

    runner_args = runner_parse([
        "--data-root",      args.root,
        "--split",          args.split,
        "--segmentor",      "oracle",
        "--pck-threshold",  str(args.pck_threshold),
        "--min-keypoints",  str(args.min_kp),
    ] + (["--subset", str(args.subset)] if args.subset else []))

    print("Running evaluation…")
    return evaluate(runner_args)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_per_joint_pck(summary: dict, out_dir: Path, threshold: float) -> None:
    """Horizontal bar chart — one bar per COCO joint, coloured by side."""
    names   = summary["joint_names"]
    pck_pj  = np.array(summary["pck_per_joint"], dtype=float)

    # Replace None (JSON-serialised NaN) with 0.0 for plotting
    pck_pj = np.where(np.isfinite(pck_pj), pck_pj, 0.0)

    # Colour by body side: centre = grey, left = blue, right = red
    LEFT_IDX  = {1, 3, 5, 7, 9, 11, 13, 15}
    RIGHT_IDX = {2, 4, 6, 8, 10, 12, 14, 16}
    colors = []
    for i in range(17):
        if i in LEFT_IDX:
            colors.append("#4C72B0")   # blue
        elif i in RIGHT_IDX:
            colors.append("#C44E52")   # red
        else:
            colors.append("#8C8C8C")   # grey (nose)

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(17)
    bars  = ax.barh(y_pos, pck_pj, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, pck_pj):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"PCK@{threshold}", fontsize=11)
    ax.set_title(
        f"Per-joint PCK@{threshold}  (n={summary['n_samples']} samples, "
        f"oracle segmentor)\n"
        f"Overall PCK={summary['pck_overall']:.3f}  |  "
        f"OKS={summary['oks_mean']:.3f}  |  "
        f"MPJPE={summary['mpjpe_overall']:.3f}",
        fontsize=10,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Left side"),
        mpatches.Patch(color="#C44E52", label="Right side"),
        mpatches.Patch(color="#8C8C8C", label="Centre"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.axvline(x=summary["pck_overall"], color="black",
               linestyle="--", linewidth=1.2, label="_nolegend_")
    ax.text(summary["pck_overall"] + 0.005, -0.8, "overall",
            fontsize=8, color="black", va="top")

    ax.invert_yaxis()   # nose at top
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = out_dir / "per_joint_pck.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_mpjpe(summary: dict, out_dir: Path) -> None:
    """Horizontal bar chart of normalised per-joint MPJPE."""
    names    = summary["joint_names"]
    mpjpe_pj = np.array(summary["mpjpe_per_joint"], dtype=float)
    mpjpe_pj = np.where(np.isfinite(mpjpe_pj), mpjpe_pj, np.nan)

    LEFT_IDX  = {1, 3, 5, 7, 9, 11, 13, 15}
    RIGHT_IDX = {2, 4, 6, 8, 10, 12, 14, 16}
    colors = []
    for i in range(17):
        if i in LEFT_IDX:
            colors.append("#4C72B0")
        elif i in RIGHT_IDX:
            colors.append("#C44E52")
        else:
            colors.append("#8C8C8C")

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(17)

    # Use 0.0 for joints with NaN (not evaluated / no lower-arm parts in SegFormer)
    plot_vals = np.where(np.isfinite(mpjpe_pj), mpjpe_pj, 0.0)
    bars = ax.barh(y_pos, plot_vals, color=colors, edgecolor="white", linewidth=0.5,
                   alpha=0.85)

    for bar, val, orig in zip(bars, plot_vals, mpjpe_pj):
        label = f"{val:.3f}" if np.isfinite(orig) else "n/a"
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("MPJPE / torso diagonal", fontsize=11)
    ax.set_title(
        f"Per-joint MPJPE (normalised)  "
        f"(n={summary['n_samples']} samples, oracle segmentor)\n"
        f"Overall MPJPE={summary['mpjpe_overall']:.3f}",
        fontsize=10,
    )

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Left side"),
        mpatches.Patch(color="#C44E52", label="Right side"),
        mpatches.Patch(color="#8C8C8C", label="Centre"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = out_dir / "per_joint_mpjpe.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate the summary dict
    if args.results_json:
        print(f"Loading results from {args.results_json}")
        with open(args.results_json) as f:
            summary = json.load(f)
        # JSON stores None for NaN; restore numpy expectations for plotting
        for k in ("pck_per_joint", "mpjpe_per_joint"):
            if k in summary:
                summary[k] = [float("nan") if v is None else v
                              for v in summary[k]]
        threshold = summary.get("pck_threshold", args.pck_threshold)
    else:
        summary   = _run_evaluation(args)
        threshold = args.pck_threshold

    print(f"\nSummary:")
    print(f"  n_samples    = {summary['n_samples']}")
    print(f"  skipped      = {summary.get('skipped', 'n/a')}")
    print(f"  PCK@{threshold}     = {summary['pck_overall']:.4f}")
    print(f"  OKS mean     = {summary['oks_mean']:.4f}")
    print(f"  MPJPE overall= {summary['mpjpe_overall']:.4f}")

    print("\nGenerating figures…")
    _plot_per_joint_pck(summary, out_dir, threshold)
    _plot_mpjpe(summary, out_dir)

    print(f"\nDone.  Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
