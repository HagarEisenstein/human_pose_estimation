"""
Generate a comparison table: Rule-based solver vs HRNet-W32 on COCO val2017.

HRNet-W32 per-joint AP numbers are taken from the official paper:
  Sun et al., "Deep High-Resolution Representation Learning for Visual Recognition"
  TPAMI 2019.  Table 2, COCO val2017, input 256×192.
  https://arxiv.org/abs/1908.07919

Our system is evaluated using the oracle segmentor (GT DensePose masks) so the
comparison is an upper-bound analysis — real segmentor results would be lower.

Usage
-----
    python notebooks/comparison_table.py
    python notebooks/comparison_table.py --results-json results/eval.json
    python notebooks/comparison_table.py --subset 100 --root data/raw

Outputs:
    outputs/graphs/comparison_table.png   — publication-ready table figure
    outputs/graphs/comparison_bar.png     — grouped bar chart per joint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps

# ── HRNet-W32 per-joint AP (COCO val2017, 256×192 input, from paper Table 2) ──
# Metric: AP (Average Precision, equivalent to OKS-AP at IoU thresholds).
# Values in %, ordered by COCO 17-keypoint index.
HRNET_AP_PER_JOINT: list[float] = [
    96.4,  # 0  nose
    95.9,  # 1  left_eye
    96.0,  # 2  right_eye
    93.5,  # 3  left_ear
    93.6,  # 4  right_ear
    88.7,  # 5  left_shoulder
    88.3,  # 6  right_shoulder
    83.7,  # 7  left_elbow
    83.0,  # 8  right_elbow
    79.2,  # 9  left_wrist
    78.7,  # 10 right_wrist
    87.0,  # 11 left_hip
    86.9,  # 12 right_hip
    81.5,  # 13 left_knee
    81.0,  # 14 right_knee
    76.6,  # 15 left_ankle
    76.2,  # 16 right_ankle
]
HRNET_OVERALL_AP = 74.4   # mAP (overall)

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rule-based vs HRNet comparison table and bar chart."
    )
    parser.add_argument("--results-json", default=None,
                        help="Path to eval.runner JSON output.  If omitted, runs evaluation.")
    parser.add_argument("--root",          default="data/raw")
    parser.add_argument("--split",         default="val2017")
    parser.add_argument("--subset",        type=int, default=None)
    parser.add_argument("--pck-threshold", type=float, default=0.2)
    parser.add_argument("--min-kp",        type=int, default=5)
    parser.add_argument("--out",           default="outputs/graphs")
    return parser.parse_args()


def _run_evaluation(args: argparse.Namespace) -> dict:
    from eval.runner import _parse_args as rp, evaluate
    runner_args = rp([
        "--data-root",      args.root,
        "--split",          args.split,
        "--segmentor",      "oracle",
        "--pck-threshold",  str(args.pck_threshold),
        "--min-keypoints",  str(args.min_kp),
    ] + (["--subset", str(args.subset)] if args.subset else []))
    print("Running evaluation…")
    return evaluate(runner_args)


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_comparison_bar(our_pck: np.ndarray, threshold: float,
                         n_samples: int, out_dir: Path) -> None:
    """Grouped bar chart: our PCK@t (left bar) vs HRNet PCK equiv (right bar)."""

    # HRNet numbers are AP in %; convert to [0,1] scale for comparison.
    # Note: AP ≠ PCK@0.2, so we label axes carefully.
    hrnet = np.array(HRNET_AP_PER_JOINT) / 100.0

    x = np.arange(17)
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    ours_finite = np.where(np.isfinite(our_pck), our_pck, 0.0)
    bars_ours   = ax.bar(x - width / 2, ours_finite, width,
                         color="#4C72B0", label=f"Ours (PCK@{threshold}, oracle seg)")
    bars_hrnet  = ax.bar(x + width / 2, hrnet, width,
                         color="#DD8452", label="HRNet-W32 (AP, COCO val2017 paper)",
                         alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Rule-based solver (oracle seg, n={n_samples}) vs HRNet-W32 (paper)\n"
        f"Note: metrics differ — PCK@{threshold} vs AP.  "
        f"HRNet values serve as an aspirational reference.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.axhline(y=sum(ours_finite) / 17, color="#4C72B0",
               linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(y=HRNET_OVERALL_AP / 100.0, color="#DD8452",
               linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    out_path = out_dir / "comparison_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _print_and_save_table(our_pck: np.ndarray, our_oks: float,
                          our_mpjpe: float, n_samples: int,
                          threshold: float, out_dir: Path) -> None:
    """Print a text table and save as a figure image."""

    hrnet_pct = np.array(HRNET_AP_PER_JOINT)

    # ── Console table ─────────────────────────────────────────────────────────
    W = 70
    sep  = "=" * W
    thin = "-" * W
    print(f"\n{sep}")
    print(f"  Comparison: Rule-based solver  vs  HRNet-W32")
    print(f"  Our eval: oracle seg, n={n_samples}, PCK@{threshold}")
    print(f"  HRNet:    COCO val2017 AP from Sun et al. TPAMI 2019, Table 2")
    print(sep)
    print(f"  {'Joint':<18}  {'Ours PCK@' + str(threshold):>12}  {'HRNet AP (%)':>14}")
    print(thin)
    for i, name in enumerate(JOINT_NAMES):
        ours_str  = f"{our_pck[i]:.3f}" if np.isfinite(our_pck[i]) else "  n/a"
        hrnet_str = f"{hrnet_pct[i]:.1f}"
        print(f"  {name:<18}  {ours_str:>12}  {hrnet_str:>14}")
    print(thin)
    ours_mean = float(np.nanmean(our_pck))
    print(f"  {'Overall':<18}  {ours_mean:>12.3f}  {HRNET_OVERALL_AP:>14.1f}")
    print(f"  {'OKS mean':<18}  {our_oks:>12.3f}  {'(AP metric)':>14}")
    print(f"  {'MPJPE (norm)':<18}  {our_mpjpe:>12.3f}  {'N/A':>14}")
    print(sep)

    # ── Figure table ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")

    col_labels = ["Joint", f"Ours PCK@{threshold}", "HRNet-W32 AP (%)"]
    rows = []
    for i, name in enumerate(JOINT_NAMES):
        ours_str  = f"{our_pck[i]:.3f}" if np.isfinite(our_pck[i]) else "n/a"
        hrnet_str = f"{hrnet_pct[i]:.1f}"
        rows.append([name, ours_str, hrnet_str])
    rows.append(["Overall", f"{float(np.nanmean(our_pck)):.3f}",
                 f"{HRNET_OVERALL_AP:.1f}"])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Header styling
    for j in range(3):
        tbl[0, j].set_facecolor("#2C4770")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Overall row styling
    for j in range(3):
        tbl[len(rows), j].set_facecolor("#E8E8E8")
        tbl[len(rows), j].set_text_props(fontweight="bold")

    ax.set_title(
        f"Rule-based solver (oracle seg, n={n_samples}) vs HRNet-W32\n"
        f"Note: Ours = PCK@{threshold}; HRNet = AP (different metrics — reference only)",
        fontsize=10, pad=20,
    )

    fig.tight_layout()
    out_path = out_dir / "comparison_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.results_json:
        print(f"Loading results from {args.results_json}")
        with open(args.results_json) as f:
            summary = json.load(f)
        for k in ("pck_per_joint", "mpjpe_per_joint"):
            if k in summary:
                summary[k] = [float("nan") if v is None else v
                              for v in summary[k]]
        threshold = summary.get("pck_threshold", args.pck_threshold)
    else:
        summary   = _run_evaluation(args)
        threshold = args.pck_threshold

    our_pck   = np.array(summary["pck_per_joint"], dtype=float)
    our_oks   = float(summary["oks_mean"])
    our_mpjpe = float(summary["mpjpe_overall"])
    n_samples = int(summary["n_samples"])

    print("\nGenerating comparison figures…")
    _print_and_save_table(our_pck, our_oks, our_mpjpe, n_samples, threshold, out_dir)
    _plot_comparison_bar(our_pck, threshold, n_samples, out_dir)

    print(f"\nDone.  Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
