"""Tests for eval/viz.py — visualization functions (no display required)."""

from __future__ import annotations

import numpy as np
import pytest

from eval.viz import draw_keypoints, draw_part_mask, draw_skeleton, show_pipeline
from pose.joints import solve
from pose.skeleton import assemble


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


class TestShowPipeline:
    """show_pipeline must not raise and must close the figure (no display)."""

    def _make_kp(self, sample):
        raw_kp     = solve(sample.part_mask)
        refined_kp = assemble(raw_kp, sample.part_mask, sample.torso_diagonal)
        return raw_kp, refined_kp

    def test_runs_without_error(self, dummy_sample, tmp_path):
        raw_kp, refined_kp = self._make_kp(dummy_sample)
        save_path = str(tmp_path / "pipeline.png")
        show_pipeline(dummy_sample, raw_kp, refined_kp, save_path=save_path)
        assert (tmp_path / "pipeline.png").exists()

    def test_output_file_is_non_empty(self, dummy_sample, tmp_path):
        raw_kp, refined_kp = self._make_kp(dummy_sample)
        save_path = str(tmp_path / "pipeline.png")
        show_pipeline(dummy_sample, raw_kp, refined_kp, save_path=save_path)
        assert (tmp_path / "pipeline.png").stat().st_size > 1000  # at least 1 KB

    def test_no_part_mask_does_not_crash(self, dummy_sample, tmp_path):
        """show_pipeline should handle sample.part_mask = None gracefully."""
        dummy_sample.part_mask = None
        raw_kp     = np.zeros((17, 3), dtype=np.float32)
        refined_kp = np.zeros((17, 3), dtype=np.float32)
        save_path  = str(tmp_path / "pipeline_no_mask.png")
        show_pipeline(dummy_sample, raw_kp, refined_kp, save_path=save_path)
        assert (tmp_path / "pipeline_no_mask.png").exists()

    def test_all_zero_keypoints(self, dummy_sample, tmp_path):
        """All-zero (undetected) keypoints should still produce a valid figure."""
        raw_kp     = np.zeros((17, 3), dtype=np.float32)
        refined_kp = np.zeros((17, 3), dtype=np.float32)
        save_path  = str(tmp_path / "pipeline_zero.png")
        show_pipeline(dummy_sample, raw_kp, refined_kp, save_path=save_path)
        assert (tmp_path / "pipeline_zero.png").exists()
