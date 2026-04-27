"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from data.base import PoseSample
from pose.parts import Part


@pytest.fixture
def dummy_sample() -> PoseSample:
    """A minimal PoseSample with synthetic data — no disk I/O required."""
    H, W = 480, 640
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # 17 keypoints, all visible
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        keypoints[i] = [W * 0.3 + i * 10, H * 0.4 + i * 5, 2]

    # Simple part mask: upper half = torso, lower half = upper legs
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[: H // 2, :] = int(Part.TORSO)
    mask[H // 2 :, : W // 2] = int(Part.UPPER_LEG_L)
    mask[H // 2 :, W // 2 :] = int(Part.UPPER_LEG_R)

    return PoseSample(
        image=image,
        keypoints=keypoints,
        part_mask=mask,
        bbox=np.array([50.0, 60.0, W - 100.0, H - 120.0], dtype=np.float32),
        image_id=999,
        ann_id=1,
    )
