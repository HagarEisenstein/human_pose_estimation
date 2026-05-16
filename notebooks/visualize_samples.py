"""
Generate a gallery of 50 sample figures: RGB image | part mask | GT skeleton.

Each figure is saved to outputs/figures/sample_<ann_id>.png.

Usage
-----
    python notebooks/visualize_samples.py
    python notebooks/visualize_samples.py --root data/raw --n 50 --split val2017
    python notebooks/visualize_samples.py --out outputs/figures --min-kp 5

Requirements
------------
    - COCO + DensePose annotations in data/raw/annotations/
    - Corresponding images in data/raw/val2017/
    Run  python -m data.download --densepose  to fetch everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from data.densepose_adapter import DensePoseDataset
from eval.viz import show_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save gallery figures of part masks + GT skeletons."
    )
    parser.add_argument(
        "--root", default="data/raw",
        help="COCO data root directory (default: data/raw)",
    )
    parser.add_argument(
        "--split", default="val2017",
        help="Dataset split to visualize (default: val2017)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of samples to visualize (default: 50)",
    )
    parser.add_argument(
        "--min-kp", type=int, default=5,
        help="Minimum visible keypoints required (default: 5)",
    )
    parser.add_argument(
        "--out", default="outputs/figures",
        help="Output directory for saved figures (default: outputs/figures)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DensePoseDataset  root={args.root}  split={args.split}")
    ds = DensePoseDataset(
        root=args.root,
        split=args.split,
        min_keypoints=args.min_kp,
        subset=args.n,
    )
    print(f"  {len(ds)} annotations loaded (subset={args.n})")

    saved = 0
    skipped = 0

    for i in tqdm(range(len(ds)), unit="sample", desc="Rendering"):
        try:
            sample = ds[i]
        except FileNotFoundError:
            skipped += 1
            continue

        if sample.part_mask is None:
            skipped += 1
            continue

        save_path = str(out_dir / f"sample_{sample.ann_id:08d}.png")
        show_sample(sample, save_path=save_path)
        saved += 1

    print(f"\nDone.  {saved} figures saved to {out_dir}/")
    if skipped:
        print(f"  {skipped} samples skipped (image not on disk or no DensePose mask).")


if __name__ == "__main__":
    main()
