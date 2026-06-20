"""Tests for pose/joints.py — rule-based joint solver."""

from __future__ import annotations

import numpy as np
import pytest

from pose.joints import DILATE_RADIUS, _EXTREMITY_JOINTS, solve
from pose.parts import Part


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank(h: int = 200, w: int = 200) -> np.ndarray:
    """All-background mask."""
    return np.zeros((h, w), dtype=np.uint8)


def _fill(mask: np.ndarray, part: Part, rows: slice, cols: slice) -> None:
    """Fill a rectangular region of the mask with a part label."""
    mask[rows, cols] = int(part)


# ── Output contract ───────────────────────────────────────────────────────────

class TestSolveOutputContract:
    def test_returns_float32(self):
        mask = _blank()
        kp = solve(mask)
        assert kp.dtype == np.float32

    def test_shape_is_17x3(self):
        mask = _blank()
        kp = solve(mask)
        assert kp.shape == (17, 3)

    def test_blank_mask_all_zero(self):
        """No parts → no joints → everything zero."""
        kp = solve(_blank())
        assert np.all(kp == 0.0)

    def test_confidence_in_0_1(self):
        """Confidence column must always be in [0, 1]."""
        mask = _blank()
        _fill(mask, Part.TORSO,       np.s_[:100], np.s_[:])
        _fill(mask, Part.UPPER_LEG_L, np.s_[100:], np.s_[:100])
        kp = solve(mask)
        conf = kp[:, 2]
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_x_within_image_width(self):
        mask = _blank(200, 300)
        _fill(mask, Part.TORSO,       np.s_[:100], np.s_[:])
        _fill(mask, Part.UPPER_ARM_L, np.s_[100:], np.s_[:])
        kp = solve(mask)
        visible = kp[kp[:, 2] > 0]
        assert np.all(visible[:, 0] >= 0) and np.all(visible[:, 0] < 300)

    def test_y_within_image_height(self):
        mask = _blank(200, 300)
        _fill(mask, Part.TORSO,       np.s_[:100], np.s_[:])
        _fill(mask, Part.UPPER_ARM_L, np.s_[100:], np.s_[:])
        kp = solve(mask)
        visible = kp[kp[:, 2] > 0]
        assert np.all(visible[:, 1] >= 0) and np.all(visible[:, 1] < 200)


# ── Boundary joints ───────────────────────────────────────────────────────────

class TestBoundaryJoints:
    def test_left_hip_found_at_boundary(self):
        """TORSO on top half, UPPER_LEG_L on bottom half → left hip at y≈100."""
        mask = _blank(200, 200)
        _fill(mask, Part.TORSO,       np.s_[:100], np.s_[:])
        _fill(mask, Part.UPPER_LEG_L, np.s_[100:], np.s_[:])

        kp = solve(mask)
        # left_hip = coco_idx 11
        x, y, conf = kp[11]
        assert conf > 0.0, "left hip should be found"
        # Joint should be near the boundary row (row 100)
        assert abs(y - 100) <= DILATE_RADIUS + 5

    def test_missing_part_gives_zero_confidence(self):
        """If one of the two parts is absent, confidence must be 0."""
        mask = _blank()
        _fill(mask, Part.TORSO, np.s_[:100], np.s_[:])
        # UPPER_ARM_L is absent → left shoulder (coco_idx 5) not found
        kp = solve(mask)
        assert kp[5, 2] == 0.0

    def test_both_parts_present_gives_positive_confidence(self):
        mask = _blank()
        _fill(mask, Part.UPPER_ARM_L, np.s_[:100], np.s_[:])
        _fill(mask, Part.LOWER_ARM_L, np.s_[100:], np.s_[:])
        # left_elbow = coco_idx 7
        kp = solve(mask)
        assert kp[7, 2] > 0.0

    def test_joint_between_two_parts_is_near_boundary(self):
        """Left knee: UPPER_LEG_L top half, LOWER_LEG_L bottom half."""
        mask = _blank(200, 200)
        _fill(mask, Part.UPPER_LEG_L, np.s_[:100], np.s_[:])
        _fill(mask, Part.LOWER_LEG_L, np.s_[100:], np.s_[:])
        # left_knee = coco_idx 13
        kp = solve(mask)
        _, y, conf = kp[13]
        assert conf > 0.0
        assert abs(y - 100) <= DILATE_RADIUS + 5

    def test_parts_far_apart_gives_zero_confidence(self):
        """Parts with a big gap between them — band should be empty."""
        mask = _blank(300, 200)
        _fill(mask, Part.UPPER_LEG_L, np.s_[:50],   np.s_[:])
        _fill(mask, Part.LOWER_LEG_L, np.s_[250:],  np.s_[:])
        kp = solve(mask)
        # Gap is 200 rows — much larger than DILATE_RADIUS
        assert kp[13, 2] == 0.0


# ── Endpoint joints (HEAD) ────────────────────────────────────────────────────

class TestEndpointJoints:
    """HEAD joints: nose (0), left_eye (1), right_eye (2),
                    left_ear (3), right_ear (4)."""

    @pytest.fixture
    def head_mask(self) -> np.ndarray:
        """200×200 mask with HEAD filling rows 20–120, cols 60–140."""
        mask = _blank(200, 200)
        _fill(mask, Part.HEAD, np.s_[20:120], np.s_[60:140])
        return mask

    def test_no_head_gives_zero_confidence(self):
        mask = _blank()
        kp = solve(mask)
        for idx in (0, 1, 2, 3, 4):
            assert kp[idx, 2] == 0.0

    def test_head_present_gives_positive_confidence(self, head_mask):
        kp = solve(head_mask)
        for idx in (0, 1, 2, 3, 4):
            assert kp[idx, 2] == 1.0

    def test_nose_is_at_top_of_head(self, head_mask):
        """Nose (coco_idx 0) should be in the upper portion of the HEAD box.

        The DensePose HEAD mask covers the face surface, not the full skull,
        so the top of the HEAD mask corresponds to the forehead/nose area.
        HEAD rows 20–120 (h=100) → top 20 % is rows 20–40.
        """
        kp = solve(head_mask)
        _, y, conf = kp[0]
        assert conf == 1.0
        assert y <= 50   # within upper region of HEAD mask

    def test_left_ear_is_rightmost(self, head_mask):
        """left_ear (coco_idx 3) should be near the right edge of HEAD."""
        kp = solve(head_mask)
        x, _, conf = kp[3]
        assert conf == 1.0
        # HEAD cols 60–140 → right 20% is ~cols 124–140
        assert x >= 110

    def test_right_ear_is_leftmost(self, head_mask):
        """right_ear (coco_idx 4) should be near the left edge of HEAD."""
        kp = solve(head_mask)
        x, _, conf = kp[4]
        assert conf == 1.0
        # HEAD cols 60–140 → left 20% is cols 60–76
        assert x <= 90

    def test_left_eye_right_of_right_eye(self, head_mask):
        """left_eye (person's left) should have larger x than right_eye."""
        kp = solve(head_mask)
        x_left_eye  = kp[1, 0]
        x_right_eye = kp[2, 0]
        assert x_left_eye > x_right_eye


# ── Extremity joints ──────────────────────────────────────────────────────────

class TestExtremityJoints:
    """Shoulder, hip, elbow, wrist, ankle use the extremity strategy."""

    def test_shoulder_at_top_of_upper_arm(self):
        """left_shoulder (coco_idx 5) → topmost pixels of UPPER_ARM_L."""
        mask = _blank(200, 200)
        # UPPER_ARM_L occupies bottom half (rows 100–200)
        _fill(mask, Part.UPPER_ARM_L, np.s_[100:], np.s_[:])
        kp = solve(mask)
        _, y, conf = kp[5]
        assert conf > 0.0, "left_shoulder should be found"
        # Topmost 15 % of rows 100–200 → centroid ≈ 107
        assert y < 120, f"shoulder y={y:.1f} should be near the top of UPPER_ARM_L"

    def test_shoulder_absent_when_arm_missing(self):
        """No UPPER_ARM_L → left_shoulder confidence = 0."""
        mask = _blank()
        _fill(mask, Part.TORSO, np.s_[:], np.s_[:])  # only torso
        kp = solve(mask)
        assert kp[5, 2] == 0.0

    def test_hip_at_top_of_upper_leg(self):
        """left_hip (coco_idx 11) → topmost pixels of UPPER_LEG_L."""
        mask = _blank(200, 200)
        _fill(mask, Part.UPPER_LEG_L, np.s_[80:], np.s_[:])
        kp = solve(mask)
        _, y, conf = kp[11]
        assert conf > 0.0
        assert y < 100, f"hip y={y:.1f} should be near top of UPPER_LEG_L (row 80)"

    def test_elbow_at_bottom_of_upper_arm(self):
        """left_elbow (coco_idx 7) → bottommost pixels of UPPER_ARM_L."""
        mask = _blank(200, 200)
        # UPPER_ARM_L at rows 20–120
        _fill(mask, Part.UPPER_ARM_L, np.s_[20:120], np.s_[:])
        kp = solve(mask)
        _, y, conf = kp[7]
        assert conf > 0.0
        # Bottom 15 % of rows 20–120 (h=100) → centroid ≈ 113
        assert y > 100, f"elbow y={y:.1f} should be near bottom of UPPER_ARM_L"

    def test_ankle_at_top_of_foot(self):
        """left_ankle (coco_idx 15) → topmost pixels of FOOT_L."""
        mask = _blank(200, 200)
        _fill(mask, Part.FOOT_L, np.s_[150:], np.s_[:])
        kp = solve(mask)
        _, y, conf = kp[15]
        assert conf > 0.0
        assert y < 170, f"ankle y={y:.1f} should be near top of FOOT_L (row 150)"

    def test_all_extremity_joints_in_dispatch_table(self):
        """All 10 extremity joint names must be in _EXTREMITY_JOINTS."""
        expected = {
            "left_shoulder", "right_shoulder",
            "left_hip",      "right_hip",
            "left_elbow",    "right_elbow",
            "left_wrist",    "right_wrist",
            "left_ankle",    "right_ankle",
        }
        assert expected == set(_EXTREMITY_JOINTS.keys())
