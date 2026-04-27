"""
COCO Keypoints dataset adapter.

Loads images and their 17-keypoint annotations from an on-disk COCO split.
Optionally loads DensePose part-mask annotations when available.

Usage
-----
    from data.coco_adapter import COCOPoseDataset

    ds = COCOPoseDataset(
        root="data/raw",
        split="val2017",
        min_keypoints=5,   # skip nearly invisible people
        subset=100,        # only load the first 100 annotations (dev mode)
    )
    sample = ds[0]
    print(sample.image.shape)     # (H, W, 3)
    print(sample.keypoints.shape) # (17, 3)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from data.base import PoseDataset, PoseSample

# COCO provides 17 keypoints in this order
COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KP = len(COCO_KP_NAMES)  # 17


class COCOPoseDataset(PoseDataset):
    """COCO 2017 person-keypoint dataset adapter.

    Args:
        root:           Path to the COCO root directory.
                        Expected layout::

                            root/
                            ├── annotations/
                            │   ├── person_keypoints_val2017.json
                            │   └── person_keypoints_train2017.json
                            ├── val2017/       ← images
                            └── train2017/

        split:          "val2017" or "train2017".
        min_keypoints:  Minimum number of *visible* keypoints required
                        to include an annotation (filters crowd/truncated).
        subset:         If set, only load the first N annotations.
                        Useful for fast development iterations.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val2017",
        min_keypoints: int = 5,
        subset: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split

        ann_file = self.root / "annotations" / f"person_keypoints_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                "Run  python -m data.download  to fetch the COCO annotations."
            )

        with ann_file.open() as f:
            coco = json.load(f)

        # Build id → image-info lookup
        self._images: dict[int, dict] = {img["id"]: img for img in coco["images"]}

        # Keep only person annotations with enough visible keypoints
        self._anns: list[dict] = [
            ann for ann in coco["annotations"]
            if ann.get("num_keypoints", 0) >= min_keypoints
            and not ann.get("iscrowd", False)
        ]

        if subset is not None:
            self._anns = self._anns[:subset]

    # ── PoseDataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._anns)

    def __getitem__(self, idx: int) -> PoseSample:
        ann = self._anns[idx]
        img_info = self._images[ann["image_id"]]

        # ── Load image ───────────────────────────────────────────────────────
        img_path = self.image_dir / img_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                "Run  python -m data.download  to fetch COCO images."
            )
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ── Keypoints ────────────────────────────────────────────────────────
        # COCO stores keypoints as flat [x1,y1,v1, x2,y2,v2, ...] (51 values)
        kp_flat = np.array(ann["keypoints"], dtype=np.float32)
        keypoints = kp_flat.reshape(NUM_KP, 3)  # (17, 3): x, y, visibility

        # ── Bounding box ─────────────────────────────────────────────────────
        bbox = np.array(ann["bbox"], dtype=np.float32)  # [x, y, w, h]

        return PoseSample(
            image=image_rgb,
            keypoints=keypoints,
            part_mask=None,  # populated by COCODensePoseDataset subclass
            bbox=bbox,
            image_id=ann["image_id"],
            ann_id=ann["id"],
            meta={
                "file_name": img_info["file_name"],
                "num_keypoints": ann["num_keypoints"],
            },
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def annotation_ids(self) -> list[int]:
        return [a["id"] for a in self._anns]

    def __repr__(self) -> str:
        return (
            f"COCOPoseDataset(split={self.split!r}, "
            f"n={len(self)}, root={self.root})"
        )
