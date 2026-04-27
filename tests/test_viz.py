"""Tests for eval/viz.py — visualization functions (no display required)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.viz import draw_keypoints, draw_part_mask, draw_skeleton


class TestDrawPartMask:
    def test_output_shape_matches_input(self, dummy_sample):
        result = draw_part_mask(dummy_sample.image, dummy_sample.part_mask)
        assert result.shape == dummy_sample.image.shape

    def test_output_dtype_uint8(self, dummy_sample):
        result = draw_part_mask(dummy_sample.image, dummy_sample.part_mask)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_original(self, dummy_sample):
        result = draw_part_mask(dummy_sample.image, dummy_sample.part_mask, alpha=0.0)
        np.testing.assert_array_equal(result, dummy_sample.image)

    def test_alpha_one_differs_from_original(self, dummy_sample):
        result = draw_part_mask(dummy_sample.image, dummy_sample.part_mask, alpha=1.0)
        # With alpha=1, the color layer should replace the image — they differ
        assert not np.array_equal(result, dummy_sample.image)


class TestDrawKeypoints:
    def test_output_shape(self, dummy_sample):
        result = draw_keypoints(dummy_sample.image, dummy_sample.keypoints)
        assert result.shape == dummy_sample.image.shape

    def test_invisible_keypoints_skipped(self, dummy_sample):
        kp = dummy_sample.keypoints.copy()
        kp[:, 2] = 0  # mark all invisible
        result = draw_keypoints(dummy_sample.image, kp, only_visible=True)
        # Should be identical to original (no dots drawn)
        np.testing.assert_array_equal(result, dummy_sample.image)


class TestDrawSkeleton:
    def test_output_shape(self, dummy_sample):
        result = draw_skeleton(dummy_sample.image, dummy_sample.keypoints)
        assert result.shape == dummy_sample.image.shape

    def test_returns_different_from_input(self, dummy_sample):
        result = draw_skeleton(dummy_sample.image, dummy_sample.keypoints)
        assert not np.array_equal(result, dummy_sample.image)
