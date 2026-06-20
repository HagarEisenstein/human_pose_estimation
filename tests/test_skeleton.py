"""Tests for pose/skeleton.py — skeleton assembly and plausibility checks."""

from __future__ import annotations

import numpy as np
import pytest

from pose.parts import Part
from pose.skeleton import (
    _interior_angle_deg,
    _detect_swap_from_joints,
    assemble,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kp(n: int = 17) -> np.ndarray:
    """Return a zeroed (17, 3) float32 keypoints array."""
    return np.zeros((n, 3), dtype=np.float32)


def _set(kp: np.ndarray, idx: int, x: float, y: float, conf: float = 1.0) -> None:
    """Set one keypoint by COCO index."""
    kp[idx] = (x, y, conf)


def _blank_mask(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _torso_mask(h: int = 200, w: int = 200) -> np.ndarray:
    """Mask with TORSO filling the central vertical strip."""
    mask = _blank_mask(h, w)
    mask[50:150, 75:125] = int(Part.TORSO)
    return mask


# ── Output contract ───────────────────────────────────────────────────────────

class TestAssembleContract:
    def test_returns_float32(self):
        kp = _kp()
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert out.dtype == np.float32

    def test_returns_17x3(self):
        kp = _kp()
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert out.shape == (17, 3)

    def test_does_not_mutate_input(self):
        kp = _kp()
        _set(kp, 5,  150.0, 80.0)
        _set(kp, 6,   50.0, 80.0)
        original = kp.copy()
        assemble(kp, _blank_mask(), torso_diagonal=300.0)
        np.testing.assert_array_equal(kp, original)

    def test_all_zero_input_stays_zero(self):
        kp = _kp()
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert np.all(out == 0.0)


# ── Step 1: confidence threshold ──────────────────────────────────────────────

class TestConfidenceThreshold:
    def test_low_confidence_joint_zeroed(self):
        kp = _kp()
        _set(kp, 5, 100.0, 80.0, conf=0.03)   # below default min_conf=0.05
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert out[5, 2] == 0.0

    def test_sufficient_confidence_kept(self):
        kp = _kp()
        _set(kp, 5, 100.0, 80.0, conf=0.5)
        _set(kp, 11, 100.0, 150.0, conf=0.5)  # left_hip — trunk limb needs parent
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert out[5, 2] > 0.0

    def test_custom_min_conf(self):
        kp = _kp()
        _set(kp, 5, 100.0, 80.0, conf=0.1)
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0, min_conf=0.15)
        assert out[5, 2] == 0.0


# ── Step 2: left/right correction ────────────────────────────────────────────

class TestDetectSwapFromJoints:
    def test_no_swap_when_left_shoulder_right_of_right(self):
        """left_shoulder.x > right_shoulder.x → normal, no swap."""
        kp = _kp()
        _set(kp, 5, 150.0, 80.0)   # left_shoulder  at x=150 (right side of image)
        _set(kp, 6,  50.0, 80.0)   # right_shoulder at x=50  (left  side of image)
        assert _detect_swap_from_joints(kp) == False

    def test_swap_when_left_shoulder_left_of_right(self):
        """left_shoulder.x < right_shoulder.x → swapped labels."""
        kp = _kp()
        _set(kp, 5,  50.0, 80.0)   # left_shoulder on LEFT of image → swap
        _set(kp, 6, 150.0, 80.0)
        assert _detect_swap_from_joints(kp) == True

    def test_falls_back_to_hips_when_shoulders_missing(self):
        kp = _kp()
        _set(kp, 11, 50.0, 150.0)   # left_hip on left → swap
        _set(kp, 12, 150.0, 150.0)
        assert _detect_swap_from_joints(kp) == True

    def test_returns_none_when_neither_pair_visible(self):
        kp = _kp()
        assert _detect_swap_from_joints(kp) is None


class TestCorrectLR:
    def test_swapped_labels_are_corrected(self):
        """After swap: left_shoulder should be at x=150 (right of image)."""
        kp = _kp()
        _set(kp, 5,  50.0, 80.0)    # left_shoulder incorrectly on left
        _set(kp, 6, 150.0, 80.0)    # right_shoulder incorrectly on right
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        # After correction, left_shoulder should be at x=150
        assert out[5, 0] == pytest.approx(150.0)
        assert out[6, 0] == pytest.approx(50.0)

    def test_correct_labels_not_modified(self):
        kp = _kp()
        _set(kp, 5, 150.0, 80.0)   # left_shoulder correctly on right
        _set(kp, 6,  50.0, 80.0)   # right_shoulder correctly on left
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        assert out[5, 0] == pytest.approx(150.0)
        assert out[6, 0] == pytest.approx(50.0)

    def test_all_lr_pairs_swapped_together(self):
        """When labels are swapped, ALL left/right pairs must be fixed."""
        kp = _kp()
        _set(kp, 5,  50.0, 80.0)    # left_shoulder on wrong side → triggers swap
        _set(kp, 6, 150.0, 80.0)
        _set(kp, 11,  50.0, 150.0)  # left_hip also on wrong side
        _set(kp, 12, 150.0, 150.0)
        out = assemble(kp, _blank_mask(), torso_diagonal=300.0)
        # After swap: left_hip should have moved to x=150
        assert out[11, 0] == pytest.approx(150.0)
        assert out[12, 0] == pytest.approx(50.0)


# ── Step 3: limb-length filter ────────────────────────────────────────────────

class TestLimbLengthFilter:
    def test_normal_limb_length_kept(self):
        """Upper arm = 0.3 × torso_diagonal → within [0.05, 1.2] → kept."""
        kp = _kp()
        td = 300.0
        _set(kp, 5, 100.0,  80.0)   # left_shoulder
        _set(kp, 7, 100.0, 170.0)   # left_elbow: distance = 90px = 0.3 × td
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] > 0.0

    def test_too_long_limb_zeroed(self):
        """Upper arm = 4 × torso_diagonal → exceeds max_ratio (1.2) → zeroed."""
        kp = _kp()
        td = 100.0
        _set(kp, 5, 100.0,  80.0)   # left_shoulder
        _set(kp, 7, 100.0, 480.0)   # distance = 400px = 4.0 × td → too long
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] == 0.0

    def test_too_short_limb_zeroed(self):
        """Upper arm = 0.01 × torso_diagonal → below min_ratio (0.05) → zeroed."""
        kp = _kp()
        td = 300.0
        _set(kp, 5, 100.0, 80.0)
        _set(kp, 7, 100.0, 82.0)   # distance = 2px = 0.007 × td → too short
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] == 0.0

    def test_parent_missing_skips_check(self):
        """If parent joint is invisible, child should not be zeroed by limb filter."""
        kp = _kp()
        _set(kp, 5, 0.0, 0.0, 0.0)  # left_shoulder invisible
        _set(kp, 7, 100.0, 170.0)   # left_elbow visible
        td = 100.0
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        # No parent → limb check skipped → elbow stays (unless other filters hit)
        assert out[7, 2] > 0.0


# ── Step 4: joint-angle filter ────────────────────────────────────────────────

class TestAngleFilter:
    def test_valid_elbow_angle_kept(self):
        """90° elbow (right angle bend) is valid — should pass."""
        kp = _kp()
        # shoulder at (0, 0), elbow at (0, 100), wrist at (100, 100)
        # angle at elbow: vectors (0,-100) and (100,0) → 90°
        _set(kp, 5,   0.0,   0.0)   # left_shoulder
        _set(kp, 7,   0.0, 100.0)   # left_elbow
        _set(kp, 9, 100.0, 100.0)   # left_wrist
        td = 500.0
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] > 0.0

    def test_near_zero_elbow_angle_zeroed(self):
        """Angle ≈ 2° at elbow (nearly folded back on itself) → zeroed."""
        kp = _kp()
        # shoulder at (0,0), elbow at (0,100), wrist almost at (0,0)
        # vectors: (0,-100) and (0,-99) → almost parallel → near-0° angle
        _set(kp, 5, 0.0,   0.0)
        _set(kp, 7, 0.0, 100.0)
        _set(kp, 9, 0.0,   1.0)   # wrist near shoulder → tiny interior angle
        td = 500.0
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] == 0.0

    def test_straight_arm_not_rejected(self):
        """A straight arm (≈ 180°) is valid — should NOT be rejected."""
        kp = _kp()
        # shoulder → elbow → wrist in a straight vertical line
        _set(kp, 5,  0.0,   0.0)   # shoulder
        _set(kp, 7,  0.0, 100.0)   # elbow
        _set(kp, 9,  0.0, 200.0)   # wrist → 180° interior angle
        td = 500.0
        out = assemble(kp, _blank_mask(), torso_diagonal=td)
        assert out[7, 2] > 0.0

    def test_missing_endpoint_skips_angle_check(self):
        """If wrist is missing, elbow angle can't be checked → elbow kept."""
        kp = _kp()
        _set(kp, 5, 0.0,   0.0)
        _set(kp, 7, 0.0, 100.0)
        # kp[9] (left_wrist) stays zero
        out = assemble(kp, _blank_mask(), torso_diagonal=500.0)
        assert out[7, 2] > 0.0


# ── Interior angle helper ─────────────────────────────────────────────────────

class TestInteriorAngleDeg:
    def test_right_angle(self):
        p = np.array([0.0, 0.0])
        j = np.array([0.0, 1.0])
        c = np.array([1.0, 1.0])
        assert _interior_angle_deg(p, j, c) == pytest.approx(90.0, abs=0.5)

    def test_straight_line_180(self):
        p = np.array([0.0, 0.0])
        j = np.array([0.0, 1.0])
        c = np.array([0.0, 2.0])
        assert _interior_angle_deg(p, j, c) == pytest.approx(180.0, abs=0.5)

    def test_degenerate_returns_zero(self):
        """All three points coincide → degenerate, return 0."""
        p = np.array([1.0, 1.0])
        j = np.array([1.0, 1.0])
        c = np.array([1.0, 1.0])
        assert _interior_angle_deg(p, j, c) == 0.0
