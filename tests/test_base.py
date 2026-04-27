"""Tests for data/base.py — PoseSample dataclass and PoseDataset ABC."""

from __future__ import annotations

import numpy as np
import pytest

from data.base import PoseSample


class TestPoseSample:
    def test_height_width(self, dummy_sample):
        assert dummy_sample.height == 480
        assert dummy_sample.width == 640

    def test_num_keypoints_visible(self, dummy_sample):
        # All 17 keypoints have visibility=2 in the fixture
        assert dummy_sample.num_keypoints_visible == 17

    def test_torso_diagonal(self, dummy_sample):
        # bbox = [50, 60, 540, 360] → diagonal = hypot(540, 360) ≈ 648.8
        diag = dummy_sample.torso_diagonal
        expected = float(np.hypot(540, 360))
        assert abs(diag - expected) < 1.0

    def test_part_mask_present(self, dummy_sample):
        assert dummy_sample.part_mask is not None
        assert dummy_sample.part_mask.shape == (480, 640)

    def test_image_dtype(self, dummy_sample):
        assert dummy_sample.image.dtype == np.uint8
        assert dummy_sample.image.ndim == 3
