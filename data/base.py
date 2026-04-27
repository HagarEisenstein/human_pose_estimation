"""
Abstract base class for all dataset adapters in this project.

Every adapter returns a standardised sample dict so the rest of the
pipeline (joint solver, evaluator, visualiser) doesn't need to know
which dataset it's talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseSample:
    """One training / evaluation sample.

    Attributes:
        image:      RGB image as uint8 array of shape (H, W, 3).
        keypoints:  Float array of shape (K, 3) — columns are (x, y, visibility).
                    Visibility: 0 = not labelled, 1 = labelled but occluded,
                                2 = labelled and visible.
        part_mask:  Integer array of shape (H, W) with canonical Part labels
                    (see pose.parts.Part).  None when not available.
        bbox:       Tight bounding box [x, y, w, h] in pixel coords.
        image_id:   Source dataset image identifier.
        ann_id:     Source dataset annotation identifier.
        meta:       Any extra fields (e.g. crowd flag, num_keypoints).
    """

    image: np.ndarray                          # (H, W, 3) uint8
    keypoints: np.ndarray                      # (K, 3) float32
    part_mask: np.ndarray | None = None        # (H, W) uint8
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    image_id: int = -1
    ann_id: int = -1
    meta: dict = field(default_factory=dict)

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def num_keypoints_visible(self) -> int:
        return int((self.keypoints[:, 2] > 0).sum())

    @property
    def torso_diagonal(self) -> float:
        """Rough person scale: diagonal of the bounding box in pixels."""
        return float(np.hypot(self.bbox[2], self.bbox[3]))


class PoseDataset(ABC):
    """Base class for pose / part-segmentation dataset adapters."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> PoseSample: ...

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # Convenience: iterate over N samples only
    def take(self, n: int):
        for i, sample in enumerate(self):
            if i >= n:
                break
            yield sample
