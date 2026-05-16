"""Tests for segmentation/base.py — SegmentationModel interface and GTOracleSegmentor."""

from __future__ import annotations

import numpy as np
import pytest

from data.base import PoseSample
from segmentation.base import GTOracleSegmentor, SegmentationModel


class TestSegmentationModelInterface:
    def test_is_abstract(self):
        """SegmentationModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SegmentationModel()  # type: ignore[abstract]

    def test_gt_oracle_is_subclass(self):
        assert issubclass(GTOracleSegmentor, SegmentationModel)


class TestGTOracleSegmentor:
    def test_returns_array(self, dummy_sample):
        seg = GTOracleSegmentor()
        result = seg.predict(dummy_sample)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_image(self, dummy_sample):
        seg = GTOracleSegmentor()
        result = seg.predict(dummy_sample)
        assert result.shape == (dummy_sample.height, dummy_sample.width)

    def test_output_dtype_uint8(self, dummy_sample):
        seg = GTOracleSegmentor()
        result = seg.predict(dummy_sample)
        assert result.dtype == np.uint8

    def test_output_matches_part_mask(self, dummy_sample):
        seg = GTOracleSegmentor()
        result = seg.predict(dummy_sample)
        np.testing.assert_array_equal(result, dummy_sample.part_mask)

    def test_returns_copy_not_same_object(self, dummy_sample):
        """predict() must return a copy so callers cannot mutate the sample."""
        seg = GTOracleSegmentor()
        result = seg.predict(dummy_sample)
        assert result is not dummy_sample.part_mask

    def test_raises_when_no_part_mask(self, dummy_sample):
        seg = GTOracleSegmentor()
        dummy_sample.part_mask = None
        with pytest.raises(ValueError, match="part_mask"):
            seg.predict(dummy_sample)

    def test_batch_predict_returns_list(self, dummy_sample):
        seg = GTOracleSegmentor()
        results = seg.batch_predict([dummy_sample, dummy_sample])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_predict_shapes(self, dummy_sample):
        seg = GTOracleSegmentor()
        results = seg.batch_predict([dummy_sample, dummy_sample])
        for r in results:
            assert r.shape == (dummy_sample.height, dummy_sample.width)
