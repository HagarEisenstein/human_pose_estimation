"""Tests for segmentation/segformer.py — wrapper around HuggingFace SegFormer.

Tests use a mock model + processor injected through the constructor, so they
never download weights or require the ``transformers`` package to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from segmentation.base import SegmentationModel
from segmentation.segformer import SegFormerSegmentor
from pose.parts import Part, SEGFORMER_CLOTHES_TO_PART


class _FakeBatch(dict):
    """Mimics transformers' BatchEncoding — dict-like + has .to()."""

    def to(self, _device):
        return self


class _FakeProcessor:
    """Returns a tensor input the fake model can ignore."""

    def __call__(self, images, return_tensors="pt"):
        return _FakeBatch(pixel_values=torch.zeros(1, 3, 64, 64))


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Produces logits where channel ``target_class`` wins everywhere.

    Used so we can verify the remap step: the SegFormer source label
    ``target_class`` should appear in the output as
    ``SEGFORMER_CLOTHES_TO_PART[target_class]``.
    """

    def __init__(self, target_class: int, num_classes: int = 18):
        self.target_class = target_class
        self.num_classes = num_classes

    def __call__(self, **inputs):
        logits = torch.full((1, self.num_classes, 32, 32), -10.0)
        logits[:, self.target_class] = 10.0
        return _FakeOutput(logits)

    def to(self, _device):
        return self

    def eval(self):
        return self


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSegFormerSegmentor:
    def test_is_segmentation_model(self):
        assert issubclass(SegFormerSegmentor, SegmentationModel)

    def test_predict_returns_canonical_part_mask(self, dummy_sample):
        # Source class 11 = "Face" → Part.HEAD in the remap table
        seg = SegFormerSegmentor(
            model=_FakeModel(target_class=11),
            processor=_FakeProcessor(),
        )
        mask = seg.predict(dummy_sample)

        assert mask.shape == (dummy_sample.height, dummy_sample.width)
        assert mask.dtype == np.uint8
        assert np.all(mask == int(Part.HEAD))

    def test_remap_applied_for_each_source_label(self, dummy_sample):
        """Every source label should map to its canonical Part."""
        for src, expected in SEGFORMER_CLOTHES_TO_PART.items():
            seg = SegFormerSegmentor(
                model=_FakeModel(target_class=src),
                processor=_FakeProcessor(),
            )
            mask = seg.predict(dummy_sample)
            assert np.all(mask == int(expected)), (
                f"source label {src} produced mask values "
                f"{np.unique(mask).tolist()}, expected {int(expected)}"
            )

    def test_output_shape_matches_input(self, dummy_sample):
        """Logits are 32×32 internally — must be upsampled to the image size."""
        seg = SegFormerSegmentor(
            model=_FakeModel(target_class=0),
            processor=_FakeProcessor(),
        )
        mask = seg.predict(dummy_sample)
        assert mask.shape == dummy_sample.image.shape[:2]
