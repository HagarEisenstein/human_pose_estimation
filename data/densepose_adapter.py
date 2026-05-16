"""
DensePose-COCO dataset adapter.

Extends the COCO adapter by also loading DensePose segmentation annotations
(body-part masks derived from the I channel of DensePose IUV maps).

The I channel encodes which of the 14 DensePose surface patches each pixel
belongs to (0 = background, 1–14 = body parts).  We remap these into our
canonical Part enum via pose.parts.DENSEPOSE_TO_PART.

Usage
-----
    from data.densepose_adapter import DensePoseDataset

    ds = DensePoseDataset(root="data/raw", split="val2017", subset=100)
    sample = ds[0]
    print(sample.part_mask.shape)  # (H, W)  — canonical part labels
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from data.base import PoseSample
from data.coco_adapter import COCOPoseDataset
from pose.parts import DENSEPOSE_TO_PART, remap_mask


class DensePoseDataset(COCOPoseDataset):
    """COCO val/train split with DensePose part-mask annotations.

    In addition to everything COCOPoseDataset loads, this adapter reads
    the ``densepose_coco_<split>.json`` annotation file (produced by the
    DensePose team) and reconstructs the per-pixel part mask for each
    annotated person.

    Args:
        root:        Same as COCOPoseDataset — COCO root directory.
        split:       "val2017" or "train2017".
        min_keypoints: Minimum visible keypoints to include an annotation.
        subset:      Limit to first N annotations (dev mode).

    Note:
        DensePose annotations are sparse — not every COCO person
        annotation has a DensePose counterpart.  Samples without a
        DensePose annotation return ``part_mask=None``.
    """

    DP_ANN_FILES = {
        "val2017":   "densepose_coco_2014_minival.json",
        "train2017": "densepose_coco_2014_train.json",
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "val2017",
        min_keypoints: int = 5,
        subset: int | None = None,
    ) -> None:
        super().__init__(root=root, split=split, min_keypoints=min_keypoints, subset=subset)

        # Try to load DensePose annotations (optional — fine if missing)
        dp_file = self.root / "annotations" / self.DP_ANN_FILES.get(split, "")
        self._dp_anns: dict[int, dict] = {}

        if dp_file.exists():
            with dp_file.open() as f:
                dp_data = json.load(f)
            for ann in dp_data.get("annotations", []):
                if "dp_masks" in ann or "dp_I" in ann:
                    self._dp_anns[ann["id"]] = ann
        else:
            import warnings
            warnings.warn(
                f"DensePose annotation file not found: {dp_file}\n"
                "part_mask will be None for all samples.  "
                "Run  python -m data.download --densepose  to fetch it.",
                stacklevel=2,
            )

    # ── Override __getitem__ to attach part_mask ──────────────────────────────

    def __getitem__(self, idx: int) -> PoseSample:
        sample = super().__getitem__(idx)

        dp_ann = self._dp_anns.get(sample.ann_id)
        if dp_ann is not None:
            sample.part_mask = self._build_part_mask(
                dp_ann, sample.bbox, sample.height, sample.width
            )

        return sample

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_part_mask(
        dp_ann: dict,
        bbox: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray:
        """Reconstruct a canonical part mask from a DensePose annotation.

        DensePose stores per-person segmentation as a set of COCO-RLE encoded
        masks inside the bounding box region.  We decode each mask, assign the
        corresponding part label, and paste it back into image coordinates.

        Returns:
            uint8 array of shape (img_h, img_w) with canonical Part labels.
        """
        from pycocotools import mask as mask_utils

        x, y, w, h = [int(v) for v in bbox]
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)

        dp_masks = dp_ann.get("dp_masks")
        if not dp_masks:
            return canvas

        for part_idx_1based, rle_or_none in enumerate(dp_masks, start=1):
            if rle_or_none is None:
                continue

            # RLE can be stored as a dict {counts, size}, a raw string, or a
            # list (polygon format in newer annotation files — skip those).
            if isinstance(rle_or_none, list):
                continue

            # DensePose masks are always encoded at 256×256 inside the bbox.
            if isinstance(rle_or_none, str):
                rle = {"counts": rle_or_none.encode(), "size": [256, 256]}
            else:
                rle = {"counts": rle_or_none["counts"], "size": [256, 256]}
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode()

            decoded256 = mask_utils.decode(rle).astype(np.uint8)  # (256, 256)
            # Resize to actual bbox dimensions
            decoded = cv2.resize(
                decoded256, (max(w, 1), max(h, 1)), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # Map to canonical label
            canonical = int(DENSEPOSE_TO_PART.get(part_idx_1based, 0))

            # Paste into full-image canvas (clip to image boundaries)
            y2 = min(y + h, img_h)
            x2 = min(x + w, img_w)
            patch_h = y2 - y
            patch_w = x2 - x
            region = canvas[y:y2, x:x2]
            region[decoded[:patch_h, :patch_w]] = canonical

        return canvas

    def __repr__(self) -> str:
        has_dp = len(self._dp_anns) > 0
        return (
            f"DensePoseDataset(split={self.split!r}, n={len(self)}, "
            f"densepose={'yes' if has_dp else 'missing'}, root={self.root})"
        )
