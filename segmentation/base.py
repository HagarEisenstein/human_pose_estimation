"""
Segmentation model wrapper interface.

Defines the uniform contract that every segmentation backend must satisfy,
plus a GTOracleSegmentor that returns the pre-computed ground-truth part mask
from a PoseSample.  The oracle is used to test the joint solver and evaluator
in complete isolation — no GPU or external model required.

Usage
-----
    from segmentation.base import GTOracleSegmentor

    seg = GTOracleSegmentor()
    mask = seg.predict(sample)   # returns sample.part_mask unchanged
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SegmentationModel(ABC):
    """Abstract base class for all segmentation backends.

    Every backend receives a PoseSample and returns a canonical part mask of
    shape (H, W) with uint8 labels drawn from pose.parts.Part.

    Subclasses must implement predict().  They may optionally override
    batch_predict() for efficiency when processing many samples.
    """

    @abstractmethod
    def predict(self, sample) -> np.ndarray:
        """Run segmentation on one sample.

        Args:
            sample: A PoseSample instance.  The implementation may use
                    sample.image, sample.bbox, or any other field it needs.

        Returns:
            uint8 array of shape (H, W) with canonical Part labels.
            Background pixels must be 0 (Part.BACKGROUND).
        """
        ...

    def batch_predict(self, samples) -> list[np.ndarray]:
        """Run segmentation on a list of samples.

        Default implementation calls predict() in a loop.  Override for
        batched inference when using a GPU-backed model.

        Args:
            samples: Iterable of PoseSample instances.

        Returns:
            List of uint8 (H, W) arrays, one per sample.
        """
        return [self.predict(s) for s in samples]


class GTOracleSegmentor(SegmentationModel):
    """Perfect-oracle segmentor that returns the ground-truth DensePose mask.

    This is not a real model — it reads part_mask directly from the
    PoseSample.  Its purpose is to let every downstream component
    (joint solver, evaluator, visualizer) be tested and benchmarked
    against an upper-bound input before a real segmentation backbone
    is integrated.

    Raises:
        ValueError: if the sample has no part_mask (part_mask is None).
    """

    def predict(self, sample) -> np.ndarray:
        if sample.part_mask is None:
            raise ValueError(
                f"GTOracleSegmentor requires a sample with a pre-computed part_mask "
                f"(ann_id={sample.ann_id}).  Use DensePoseDataset instead of "
                f"COCOPoseDataset, or supply a sample that has part_mask set."
            )
        return sample.part_mask.copy()
