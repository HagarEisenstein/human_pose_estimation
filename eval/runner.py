"""
Evaluation runner — wires the full pipeline into a CLI.

Runs the complete pose-estimation pipeline on a dataset subset and
reports PCK@0.2, OKS, and MPJPE — both overall and per joint.

Pipeline per sample
-------------------
    dataset[i]                 → PoseSample
    segmentor.predict(sample)  → (H, W) uint8 part mask
    joints.solve(part_mask)    → (17, 3) raw keypoints
    skeleton.assemble(...)     → (17, 3) refined keypoints
    metrics.pck / oks / mpjpe  → per-sample scores
    Accumulator.update()       → running totals

Segmentor modes
---------------
    oracle    — GTOracleSegmentor reads the ground-truth DensePose mask
                directly from the sample.  Requires DensePoseDataset.
                Use this to measure the upper-bound of the joint solver.
    segformer — SegFormerSegmentor runs the real HuggingFace model on
                sample.image.  Works with COCOPoseDataset (no GT masks).
                Requires:  pip install -e ".[segformer]"

Usage
-----
    # Quickest — oracle on 100 samples (same as  make eval)
    python -m eval.runner --data-root data/raw --split val2017 --subset 100

    # Real model on 500 samples, save results to JSON
    python -m eval.runner --data-root data/raw --split val2017 \\
        --subset 500 --segmentor segformer --out results/segformer.json

    make eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.metrics import Accumulator, mpjpe, oks, pck
from pose.joints import solve
from pose.parts import COCO_JOINT_NAMES
from pose.skeleton import assemble
from segmentation.base import GTOracleSegmentor, SegmentationModel


# ── CLI argument parsing ──────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="Run the pose-estimation pipeline and report metrics.",
    )
    parser.add_argument(
        "--data-root", default="data/raw",
        help="COCO data root directory  (default: data/raw)",
    )
    parser.add_argument(
        "--split", default="val2017",
        choices=["val2017", "train2017"],
        help="Dataset split to evaluate  (default: val2017)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to first N annotations  (default: all)",
    )
    parser.add_argument(
        "--segmentor", default="oracle",
        choices=["oracle", "segformer"],
        help=(
            "oracle   = GTOracleSegmentor using ground-truth DensePose masks\n"
            "segformer = SegFormerSegmentor (requires [segformer] extra)\n"
            "(default: oracle)"
        ),
    )
    parser.add_argument(
        "--pck-threshold", type=float, default=0.2,
        help="PCK distance threshold as fraction of torso diagonal  (default: 0.2)",
    )
    parser.add_argument(
        "--min-keypoints", type=int, default=5,
        help="Minimum visible GT keypoints required per sample  (default: 5)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to save the summary as a JSON file",
    )
    return parser.parse_args(argv)


# ── Dataset and segmentor factories ──────────────────────────────────────────

def _build_segmentor(name: str) -> SegmentationModel:
    """Instantiate the requested segmentor."""
    if name == "oracle":
        return GTOracleSegmentor()
    if name == "segformer":
        try:
            from segmentation.segformer import SegFormerSegmentor
        except ImportError as exc:
            sys.exit(
                "SegFormerSegmentor requires the optional [segformer] extra.\n"
                "Install with:  pip install -e '.[segformer]'\n"
                f"Original error: {exc}"
            )
        return SegFormerSegmentor()
    raise ValueError(f"Unknown segmentor: {name!r}")


def _build_dataset(data_root: str, split: str, subset: int | None,
                   segmentor_name: str, min_kp: int):
    """Build the dataset — DensePose for oracle, COCO for real models."""
    if segmentor_name == "oracle":
        # Oracle needs pre-computed DensePose masks
        from data.densepose_adapter import DensePoseDataset
        return DensePoseDataset(
            root=data_root, split=split,
            min_keypoints=min_kp, subset=subset,
        )
    else:
        # Real segmentor: plain COCO dataset (no GT masks needed)
        from data.coco_adapter import COCOPoseDataset
        return COCOPoseDataset(
            root=data_root, split=split,
            min_keypoints=min_kp, subset=subset,
        )


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> dict:
    """Run the full evaluation loop and return the summary dict."""

    print(f"\nBuilding dataset  root={args.data_root!r}  split={args.split!r}"
          f"  subset={args.subset}")
    dataset   = _build_dataset(args.data_root, args.split, args.subset,
                                args.segmentor, args.min_keypoints)
    segmentor = _build_segmentor(args.segmentor)

    print(f"Loaded {len(dataset)} annotations  |  segmentor={args.segmentor!r}\n")

    acc     = Accumulator()
    skipped = 0

    for i in tqdm(range(len(dataset)), desc="Evaluating", unit="sample"):

        # ── 1. Load sample ────────────────────────────────────────────────────
        try:
            sample = dataset[i]
        except FileNotFoundError:
            skipped += 1
            continue

        # ── 2. Get part mask from segmentor ───────────────────────────────────
        try:
            part_mask = segmentor.predict(sample)
        except ValueError:
            # GTOracleSegmentor raises ValueError when sample.part_mask is None
            # (sample has no DensePose annotation)
            skipped += 1
            continue

        # ── 3. Skip degenerate samples ────────────────────────────────────────
        torso_diagonal = sample.torso_diagonal
        if torso_diagonal < 1.0:
            skipped += 1
            continue

        # ── 4. Joint solver → skeleton assembly ───────────────────────────────
        raw_kp      = solve(part_mask)
        refined_kp  = assemble(raw_kp, part_mask, torso_diagonal)

        # ── 5. Compute per-sample metrics ─────────────────────────────────────
        pck_result  = pck(refined_kp, sample.keypoints, torso_diagonal,
                          threshold=args.pck_threshold)
        oks_score   = oks(refined_kp, sample.keypoints, sample.bbox)
        mpjpe_result = mpjpe(refined_kp, sample.keypoints, torso_diagonal)

        # ── 6. Accumulate ─────────────────────────────────────────────────────
        acc.update(pck_result, oks_score, mpjpe_result)

    # ── 7. Finalise summary ───────────────────────────────────────────────────
    summary = acc.summarise()
    summary["skipped"]       = skipped
    summary["segmentor"]     = args.segmentor
    summary["split"]         = args.split
    summary["pck_threshold"] = args.pck_threshold

    return summary


# ── Results printing ──────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    """Print a formatted results table to stdout."""
    W    = 62
    sep  = "=" * W
    thin = "-" * W

    pck_pj   = summary["pck_per_joint"]
    mpjpe_pj = summary["mpjpe_per_joint"]
    names    = summary["joint_names"]
    thresh   = summary["pck_threshold"]

    def _fmt(v: float, decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if np.isfinite(v) else "  n/a"

    print(f"\n{sep}")
    print(f"  Pose Estimation Evaluation Results")
    print(f"  {summary['n_samples']} samples  |  "
          f"{summary['skipped']} skipped  |  "
          f"segmentor={summary['segmentor']!r}  |  "
          f"split={summary['split']!r}")
    print(sep)
    print(f"  Overall PCK@{thresh}  :  {_fmt(summary['pck_overall'])}")
    print(f"  Overall OKS      :  {_fmt(summary['oks_mean'])}")
    print(f"  Overall MPJPE    :  {_fmt(summary['mpjpe_overall'])}")
    print(thin)
    print(f"  {'Joint':<22}  {'PCK@' + str(thresh):>9}  {'MPJPE':>9}")
    print(thin)
    for i, name in enumerate(names):
        print(f"  {name:<22}  {_fmt(pck_pj[i]):>9}  {_fmt(mpjpe_pj[i]):>9}")
    print(f"{sep}\n")


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _serialisable(summary: dict) -> dict:
    """Convert numpy arrays to lists so json.dump() works."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, np.ndarray):
            out[k] = [None if np.isnan(x) else float(x) for x in v]
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args    = _parse_args(argv)
    summary = evaluate(args)

    print_summary(summary)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(_serialisable(summary), f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
