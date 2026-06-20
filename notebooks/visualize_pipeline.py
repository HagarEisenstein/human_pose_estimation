"""
Generate 50 pipeline figures: image | part-mask | raw joints | assembled skeleton.

Each figure shows the full M2 pipeline in four panels for one sample, saved to
outputs/pipeline_figures/pipeline_<ann_id>.png.

Usage
-----
    python notebooks/visualize_pipeline.py
    python notebooks/visualize_pipeline.py --n 50 --root data/raw
    python notebooks/visualize_pipeline.py --out outputs/pipeline_figures

Requirements
------------
    Run  python -m data.download --densepose  first to get images + annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data.densepose_adapter import DensePoseDataset
from eval.viz import show_pipeline
from pose.joints import solve
from pose.skeleton import assemble
from segmentation.base import GTOracleSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save 50 pipeline figures (part mask → raw joints → assembled skeleton)."
    )
    parser.add_argument("--root",   default="data/raw",             help="COCO data root")
    parser.add_argument("--split",  default="val2017",              help="Dataset split")
    parser.add_argument("--n",      type=int, default=50,           help="Number of figures")
    parser.add_argument("--min-kp", type=int, default=5,            help="Min visible keypoints")
    parser.add_argument("--out",    default="outputs/pipeline_figures",
                        help="Output directory")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DensePoseDataset  root={args.root!r}  split={args.split!r}")
    # Use a large enough subset so we can collect args.n valid figures even with skips
    ds = DensePoseDataset(
        root=args.root,
        split=args.split,
        min_keypoints=args.min_kp,
        subset=None,          # load all available
    )
    print(f"  {len(ds)} annotations available")

    seg   = GTOracleSegmentor()
    saved = 0
    total = len(ds)

    bar = tqdm(range(total), unit="sample", desc="Rendering")
    for i in bar:
        if saved >= args.n:
            break

        # ── Load sample ───────────────────────────────────────────────────────
        try:
            sample = ds[i]
        except FileNotFoundError:
            continue

        # ── Get part mask (oracle = GT DensePose mask) ────────────────────────
        try:
            part_mask = seg.predict(sample)
        except ValueError:
            continue

        if sample.torso_diagonal < 1.0:
            continue

        # ── Run pipeline ──────────────────────────────────────────────────────
        raw_kp     = solve(part_mask)
        refined_kp = assemble(raw_kp, part_mask, sample.torso_diagonal)

        # ── Save figure ───────────────────────────────────────────────────────
        save_path = str(out_dir / f"pipeline_{sample.ann_id:08d}.png")
        show_pipeline(sample, raw_kp, refined_kp, save_path=save_path)
        saved += 1
        bar.set_postfix(saved=saved)

    print(f"\nDone.  {saved} pipeline figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
