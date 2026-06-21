"""Tests for pose/parts.py — part taxonomy and label remapping."""

from __future__ import annotations

import numpy as np
import pytest

from pose.parts import (
    DENSEPOSE_TO_PART,
    JOINT_DEFINITIONS,
    Part,
    remap_mask,
)


class TestPartEnum:
    def test_background_is_zero(self):
        assert int(Part.BACKGROUND) == 0

    def test_all_parts_unique(self):
        values = [int(p) for p in Part]
        assert len(values) == len(set(values))

    def test_expected_count(self):
        # 15 parts including background
        assert len(Part) == 15


class TestJointDefinitions:
    def test_17_joints(self):
        assert len(JOINT_DEFINITIONS) == 17

    def test_coco_indices_unique(self):
        indices = [j.coco_idx for j in JOINT_DEFINITIONS]
        assert len(indices) == len(set(indices))

    def test_coco_indices_range(self):
        for j in JOINT_DEFINITIONS:
            assert 0 <= j.coco_idx <= 16

    def test_limb_joints_have_two_different_parts(self):
        """Joints defined by two parts should actually have two different parts."""
        limb_joints = [
            j for j in JOINT_DEFINITIONS
            if j.name in ("left_elbow", "right_elbow", "left_knee", "right_knee")
        ]
        for j in limb_joints:
            assert j.part_a != j.part_b, f"{j.name} should span two different parts"


class TestRemapMask:
    def test_basic_remap(self):
        mask = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
        remap = {0: Part.BACKGROUND, 1: Part.HEAD, 2: Part.TORSO, 3: Part.UPPER_ARM_L}
        result = remap_mask(mask, remap)
        assert result[0, 0] == int(Part.BACKGROUND)
        assert result[0, 1] == int(Part.HEAD)
        assert result[0, 2] == int(Part.TORSO)
        assert result[1, 0] == int(Part.UPPER_ARM_L)

    def test_output_dtype(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        result = remap_mask(mask, {0: Part.BACKGROUND})
        assert result.dtype == np.uint8

    def test_densepose_remap_all_keys(self):
        """Every DensePose label should map to a valid Part."""
        for src, dst in DENSEPOSE_TO_PART.items():
            assert isinstance(dst, Part), f"label {src} maps to {dst!r}, expected Part"

