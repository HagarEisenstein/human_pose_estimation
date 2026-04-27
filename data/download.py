"""
Download helper for COCO and DensePose annotation files.

Downloads only what's needed for development:
- COCO val2017 annotations (~240 MB)
- A mini subset of val2017 images (configurable, default 500)
- DensePose minival annotations (~60 MB, optional)

Usage
-----
    python -m data.download               # annotations + 500 images
    python -m data.download --subset 100  # annotations + 100 images
    python -m data.download --densepose   # also fetch DensePose annotations
    python -m data.download --images-only # skip re-downloading annotations
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from tqdm import tqdm

# ── URLs ──────────────────────────────────────────────────────────────────────

COCO_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_IMG_BASE = "http://images.cocodataset.org/val2017/"

DENSEPOSE_ANN_URL = (
    "https://dl.fbaipublicfiles.com/densepose/annotations/"
    "densepose_coco_2014_minival.json"
)

DATA_ROOT = Path(__file__).parent / "raw"


# ── Download utilities ────────────────────────────────────────────────────────

def _progress_hook(desc: str) -> Callable:
    """Returns a urllib reporthook that drives a tqdm bar."""
    pbar: list[tqdm] = []

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if not pbar:
            pbar.append(tqdm(total=total_size, unit="B", unit_scale=True, desc=desc))
        downloaded = block_num * block_size
        pbar[0].update(min(downloaded, total_size) - pbar[0].n)
        if downloaded >= total_size:
            pbar[0].close()

    return hook


def download_file(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download *url* to *dest*, skipping if already present."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(desc or dest.name))
    return dest


# ── COCO annotations ──────────────────────────────────────────────────────────

def download_coco_annotations(root: Path) -> None:
    ann_dir = root / "annotations"
    target = ann_dir / "person_keypoints_val2017.json"
    if target.exists():
        print("  [skip] COCO annotations already present")
        return

    zip_path = root / "annotations_trainval2017.zip"
    download_file(COCO_ANN_URL, zip_path, desc="COCO annotations")

    print("  Extracting annotations…")
    with zipfile.ZipFile(zip_path) as zf:
        # Only extract the files we need (val + train keypoints)
        members = [
            m for m in zf.namelist()
            if "person_keypoints" in m
        ]
        zf.extractall(root, members=members)
    zip_path.unlink()
    print(f"  Annotations saved to {ann_dir}")


# ── COCO images (subset) ──────────────────────────────────────────────────────

def download_coco_images(root: Path, subset: int) -> None:
    """Download the first *subset* val2017 images listed in the annotation."""
    ann_file = root / "annotations" / "person_keypoints_val2017.json"
    img_dir = root / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)

    with ann_file.open() as f:
        coco = json.load(f)

    # Collect unique image filenames needed by first `subset` valid annotations
    anns_sorted = sorted(coco["annotations"], key=lambda a: a["image_id"])
    img_id_set: set[int] = set()
    for ann in anns_sorted:
        if len(img_id_set) >= subset:
            break
        if ann.get("num_keypoints", 0) >= 1 and not ann.get("iscrowd"):
            img_id_set.add(ann["image_id"])

    id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}
    filenames = [id_to_fname[i] for i in img_id_set if i in id_to_fname]

    print(f"  Downloading {len(filenames)} COCO val2017 images…")
    for fname in tqdm(filenames, unit="img"):
        dest = img_dir / fname
        if dest.exists():
            continue
        url = COCO_IMG_BASE + fname
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed to download {fname}: {exc}")


# ── DensePose annotations ─────────────────────────────────────────────────────

def download_densepose_annotations(root: Path) -> None:
    dest = root / "annotations" / "densepose_coco_2014_minival.json"
    download_file(DENSEPOSE_ANN_URL, dest, desc="DensePose annotations")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO / DensePose data")
    parser.add_argument(
        "--root", default=str(DATA_ROOT),
        help="Destination directory (default: data/raw)"
    )
    parser.add_argument(
        "--subset", type=int, default=500,
        help="Number of val images to download (default: 500)"
    )
    parser.add_argument(
        "--densepose", action="store_true",
        help="Also download DensePose minival annotations"
    )
    parser.add_argument(
        "--images-only", action="store_true",
        help="Skip annotations, only download images"
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root.resolve()}")

    if not args.images_only:
        print("\n[1/3] COCO annotations")
        download_coco_annotations(root)

    print(f"\n[2/3] COCO val2017 images (subset={args.subset})")
    download_coco_images(root, subset=args.subset)

    if args.densepose:
        print("\n[3/3] DensePose annotations")
        download_densepose_annotations(root)

    print("\nDone.  Run  make eval  to verify the setup.")


if __name__ == "__main__":
    main()
