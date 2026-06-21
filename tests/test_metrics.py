"""Tests for eval/metrics.py — PCK, OKS, MPJPE, and Accumulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import OKS_AP_THRESHOLDS, Accumulator, MPJPEResult, PCKResult, mpjpe, oks, pck


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pred(n: int = 17) -> np.ndarray:
    """Zeroed (17, 3) predicted keypoints."""
    return np.zeros((n, 3), dtype=np.float32)


def _gt(n: int = 17) -> np.ndarray:
    """Zeroed (17, 3) GT keypoints (all invisible)."""
    return np.zeros((n, 3), dtype=np.float32)


def _set_pred(kp: np.ndarray, idx: int, x: float, y: float, conf: float = 1.0) -> None:
    kp[idx] = (x, y, conf)


def _set_gt(kp: np.ndarray, idx: int, x: float, y: float, vis: float = 2.0) -> None:
    kp[idx] = (x, y, vis)


def _bbox(w: float = 200.0, h: float = 300.0) -> np.ndarray:
    return np.array([0.0, 0.0, w, h], dtype=np.float32)


# ── PCK tests ─────────────────────────────────────────────────────────────────

class TestPCK:
    def test_perfect_prediction_score_one(self):
        """Exact match on all joints → score = 1.0."""
        pr, gt = _pred(), _gt()
        for i in range(17):
            _set_pred(pr, i, float(i * 10), float(i * 10))
            _set_gt(gt,   i, float(i * 10), float(i * 10))
        result = pck(pr, gt, torso_diagonal=300.0)
        assert result.score == pytest.approx(1.0)

    def test_prediction_within_threshold_correct(self):
        """Joint 5 predicted 10px off; threshold = 0.2 × 300 = 60px → correct."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 110.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        result = pck(pr, gt, torso_diagonal=300.0)
        assert result.per_joint[5] == 1.0

    def test_prediction_outside_threshold_wrong(self):
        """Joint 5 predicted 100px off; threshold = 60px → wrong."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 200.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        result = pck(pr, gt, torso_diagonal=300.0)
        assert result.per_joint[5] == 0.0

    def test_invisible_gt_excluded_from_score(self):
        """GT invisible (vis=0) → not counted in score."""
        pr, gt = _pred(), _gt()
        # All GT invisible → score should be NaN (nothing to evaluate)
        result = pck(pr, gt, torso_diagonal=300.0)
        assert math.isnan(result.score)

    def test_unpredicted_joint_counts_as_wrong(self):
        """GT visible but pred conf=0 → joint is wrong."""
        pr, gt = _pred(), _gt()
        _set_gt(gt, 5, 100.0, 100.0)
        # pr[5] stays at (0, 0, 0) → confidence = 0
        result = pck(pr, gt, torso_diagonal=300.0)
        assert result.per_joint[5] == 0.0
        assert result.score == pytest.approx(0.0)

    def test_returns_pck_result(self):
        result = pck(_pred(), _gt(), 300.0)
        assert isinstance(result, PCKResult)
        assert result.per_joint.shape == (17,)
        assert result.gt_visible.shape == (17,)

    def test_custom_threshold(self):
        """threshold=0.5 × 100 = 50px; joint at 40px off → correct."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 0, 140.0, 100.0)
        _set_gt(gt,   0, 100.0, 100.0)   # dist = 40px
        r_strict = pck(pr, gt, torso_diagonal=100.0, threshold=0.2)  # thresh=20 → wrong
        r_loose  = pck(pr, gt, torso_diagonal=100.0, threshold=0.5)  # thresh=50 → correct
        assert r_strict.per_joint[0] == 0.0
        assert r_loose.per_joint[0]  == 1.0

    def test_gt_visible_flag(self):
        """gt_visible should be True only where GT visibility > 0."""
        pr, gt = _pred(), _gt()
        _set_gt(gt, 3, 50.0, 50.0, vis=1.0)   # occluded but labelled → visible
        _set_gt(gt, 5, 50.0, 50.0, vis=2.0)   # visible
        # joints 3 and 5 should be flagged as visible
        result = pck(pr, gt, 300.0)
        assert result.gt_visible[3]
        assert result.gt_visible[5]
        assert not result.gt_visible[0]


# ── OKS tests ─────────────────────────────────────────────────────────────────

class TestOKS:
    def test_perfect_prediction_returns_one(self):
        """d=0 for all joints → exp(0) = 1 for each → OKS = 1.0."""
        pr, gt = _pred(), _gt()
        for i in range(17):
            _set_pred(pr, i, 100.0, 100.0)
            _set_gt(gt,   i, 100.0, 100.0)
        score = oks(pr, gt, _bbox())
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_no_visible_gt_returns_zero(self):
        """All GT invisible → denominator = 0 → OKS = 0."""
        score = oks(_pred(), _gt(), _bbox())
        assert score == pytest.approx(0.0)

    def test_unpredicted_joint_contributes_zero(self):
        """GT visible but pred conf=0 → that joint contributes 0 to numerator."""
        pr, gt = _pred(), _gt()
        _set_gt(gt, 5, 100.0, 100.0)
        # pr[5] stays zero (not predicted)
        score = oks(pr, gt, _bbox())
        assert score == pytest.approx(0.0)

    def test_close_prediction_high_oks(self):
        """Small distance → high OKS (close to 1)."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 101.0, 100.0)   # 1px off
        _set_gt(gt,   5, 100.0, 100.0)
        score = oks(pr, gt, _bbox(200, 300))
        assert score > 0.9

    def test_far_prediction_low_oks(self):
        """Large distance → low OKS (close to 0)."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 500.0, 500.0)   # far off
        _set_gt(gt,   5, 100.0, 100.0)
        score = oks(pr, gt, _bbox(200, 300))
        assert score < 0.1

    def test_oks_in_zero_one(self):
        pr, gt = _pred(), _gt()
        for i in range(17):
            _set_pred(pr, i, float(i * 5), 50.0)
            _set_gt(gt,   i, float(i * 5 + 3), 50.0)
        score = oks(pr, gt, _bbox())
        assert 0.0 <= score <= 1.0

    def test_zero_scale_bbox_returns_zero(self):
        """Degenerate bbox (zero size) → returns 0 safely."""
        pr, gt = _pred(), _gt()
        _set_gt(gt, 5, 100.0, 100.0)
        score = oks(pr, gt, np.array([0., 0., 0., 0.]))
        assert score == 0.0


# ── MPJPE tests ───────────────────────────────────────────────────────────────

class TestMPJPE:
    def test_perfect_prediction_zero_error(self):
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 100.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        result = mpjpe(pr, gt, torso_diagonal=300.0)
        assert result.per_joint[5] == pytest.approx(0.0)
        assert result.overall == pytest.approx(0.0)

    def test_known_error_value(self):
        """Joint 5: pred=(130,100), gt=(100,100) → dist=30, td=300 → error=0.1."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 130.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        result = mpjpe(pr, gt, torso_diagonal=300.0)
        assert result.per_joint[5] == pytest.approx(0.1, abs=1e-6)

    def test_invisible_gt_gives_nan(self):
        """GT invisible → per_joint should be NaN for that joint."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 100.0, 100.0)
        # gt[5] stays at visibility=0
        result = mpjpe(pr, gt, torso_diagonal=300.0)
        assert math.isnan(result.per_joint[5])

    def test_unpredicted_joint_gives_nan(self):
        """GT visible but pred conf=0 → NaN (not included in average)."""
        pr, gt = _pred(), _gt()
        _set_gt(gt, 5, 100.0, 100.0)
        # pr[5] stays zero
        result = mpjpe(pr, gt, torso_diagonal=300.0)
        assert math.isnan(result.per_joint[5])

    def test_overall_is_nanmean_of_per_joint(self):
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 130.0, 100.0)   # error = 30/300 = 0.1
        _set_gt(gt,   5, 100.0, 100.0)
        _set_pred(pr, 7, 100.0, 150.0)   # error = 50/300 ≈ 0.167
        _set_gt(gt,   7, 100.0, 100.0)
        result = mpjpe(pr, gt, torso_diagonal=300.0)
        expected = np.nanmean(result.per_joint)
        assert result.overall == pytest.approx(expected, abs=1e-6)

    def test_returns_mpjpe_result(self):
        result = mpjpe(_pred(), _gt(), 300.0)
        assert isinstance(result, MPJPEResult)
        assert result.per_joint.shape == (17,)

    def test_zero_torso_diagonal_returns_nan(self):
        result = mpjpe(_pred(), _gt(), torso_diagonal=0.0)
        assert math.isnan(result.overall)
        assert np.all(np.isnan(result.per_joint))


# ── Accumulator tests ─────────────────────────────────────────────────────────

class TestAccumulator:
    def _perfect_sample(self) -> tuple[PCKResult, float, MPJPEResult]:
        """A sample where joint 5 is perfectly predicted."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 100.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        td = 300.0
        return (
            pck(pr, gt, td),
            oks(pr, gt, _bbox()),
            mpjpe(pr, gt, td),
        )

    def test_n_samples_increments(self):
        acc = Accumulator()
        for _ in range(3):
            acc.update(*self._perfect_sample())
        summary = acc.summarise()
        assert summary["n_samples"] == 3

    def test_perfect_samples_pck_one(self):
        acc = Accumulator()
        for _ in range(5):
            acc.update(*self._perfect_sample())
        summary = acc.summarise()
        assert summary["pck_overall"] == pytest.approx(1.0)

    def test_perfect_samples_oks_one(self):
        acc = Accumulator()
        for _ in range(5):
            acc.update(*self._perfect_sample())
        summary = acc.summarise()
        assert summary["oks_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_samples_mpjpe_zero(self):
        acc = Accumulator()
        for _ in range(5):
            acc.update(*self._perfect_sample())
        summary = acc.summarise()
        assert summary["mpjpe_overall"] == pytest.approx(0.0, abs=1e-6)

    def test_summary_contains_required_keys(self):
        acc = Accumulator()
        acc.update(*self._perfect_sample())
        summary = acc.summarise()
        for key in ("pck_overall", "pck_per_joint", "oks_mean",
                    "mpjpe_overall", "mpjpe_per_joint",
                    "n_samples", "joint_names"):
            assert key in summary

    def test_pck_per_joint_shape(self):
        acc = Accumulator()
        acc.update(*self._perfect_sample())
        assert acc.summarise()["pck_per_joint"].shape == (17,)

    def test_mpjpe_per_joint_shape(self):
        acc = Accumulator()
        acc.update(*self._perfect_sample())
        assert acc.summarise()["mpjpe_per_joint"].shape == (17,)

    def test_joint_never_appearing_is_nan(self):
        """Joint 0 never evaluated → pck_per_joint[0] should be NaN."""
        acc = Accumulator()
        acc.update(*self._perfect_sample())   # only joint 5 is set
        summary = acc.summarise()
        assert math.isnan(summary["pck_per_joint"][0])

    def test_weighted_pck_accumulation(self):
        """Sample A: joint 5 correct. Sample B: joint 5 wrong.
        Overall joint-5 PCK should be 0.5."""
        pr_a, gt_a = _pred(), _gt()
        _set_pred(pr_a, 5, 100.0, 100.0)
        _set_gt(gt_a,   5, 100.0, 100.0)

        pr_b, gt_b = _pred(), _gt()
        _set_pred(pr_b, 5, 999.0, 999.0)   # far off → wrong
        _set_gt(gt_b,   5, 100.0, 100.0)

        td = 300.0
        acc = Accumulator()
        acc.update(pck(pr_a, gt_a, td), oks(pr_a, gt_a, _bbox()), mpjpe(pr_a, gt_a, td))
        acc.update(pck(pr_b, gt_b, td), oks(pr_b, gt_b, _bbox()), mpjpe(pr_b, gt_b, td))

        summary = acc.summarise()
        assert summary["pck_per_joint"][5] == pytest.approx(0.5)

    def test_empty_accumulator_summary(self):
        """Summarising with zero samples → NaN/0 values, no crash."""
        acc = Accumulator()
        summary = acc.summarise()
        assert summary["n_samples"] == 0
        assert math.isnan(summary["pck_overall"])
        assert math.isnan(summary["oks_mean"])
        assert math.isnan(summary["mpjpe_overall"])
        assert math.isnan(summary["oks_ap"])


# ── OKS-AP tests ──────────────────────────────────────────────────────────────

class TestOKSAP:
    def test_thresholds_are_coco_standard(self):
        """10 thresholds from 0.50 to 0.95 in steps of 0.05."""
        assert len(OKS_AP_THRESHOLDS) == 10
        assert OKS_AP_THRESHOLDS[0] == pytest.approx(0.50)
        assert OKS_AP_THRESHOLDS[-1] == pytest.approx(0.95)

    def test_perfect_samples_ap_one(self):
        """OKS=1.0 for every sample → clears every threshold → AP = 1.0."""
        pr, gt = _pred(), _gt()
        _set_pred(pr, 5, 100.0, 100.0)
        _set_gt(gt,   5, 100.0, 100.0)
        td = 300.0
        acc = Accumulator()
        for _ in range(5):
            acc.update(pck(pr, gt, td), oks(pr, gt, _bbox()), mpjpe(pr, gt, td))
        assert acc.summarise()["oks_ap"] == pytest.approx(1.0, abs=1e-6)

    def test_mixed_scores_give_partial_ap(self):
        """One perfect sample (OKS=1, clears all 10 thresholds) and one
        zero-OKS sample (clears none) → AP averages to 0.5."""
        pr_good, gt_good = _pred(), _gt()
        _set_pred(pr_good, 5, 100.0, 100.0)
        _set_gt(gt_good,   5, 100.0, 100.0)

        pr_bad, gt_bad = _pred(), _gt()
        _set_gt(gt_bad, 5, 100.0, 100.0)   # pred stays unset → OKS = 0

        td = 300.0
        acc = Accumulator()
        acc.update(pck(pr_good, gt_good, td), oks(pr_good, gt_good, _bbox()),
                   mpjpe(pr_good, gt_good, td))
        acc.update(pck(pr_bad, gt_bad, td), oks(pr_bad, gt_bad, _bbox()),
                   mpjpe(pr_bad, gt_bad, td))
        assert acc.summarise()["oks_ap"] == pytest.approx(0.5, abs=1e-6)
