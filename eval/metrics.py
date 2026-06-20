"""
Evaluation metrics for pose estimation — M2.

Three metrics, each operating on one (pred, GT) sample pair:

  PCK@t  (Percentage of Correct Keypoints)
    A predicted joint is "correct" when its Euclidean distance from the
    GT joint is ≤ t × torso_diagonal.  Standard threshold t = 0.2.
    Joints where the GT is invisible are excluded from the count.
    Joints where the prediction has confidence = 0 are counted as wrong.

  OKS    (Object Keypoint Similarity)
    COCO's official metric.  Each visible GT joint contributes an
    exponential decay term:  exp(−d² / (2 s² σ²))
    where d = pixel distance, s = sqrt(bbox area), σ = per-joint constant.
    Result ∈ [0, 1].  Unpredicted joints contribute 0.

  MPJPE  (Mean Per-Joint Position Error)
    Mean pixel distance between predicted and GT joints, normalised by
    torso_diagonal.  Only joints where BOTH GT is visible AND prediction
    confidence > 0 are included.  Per-joint values use NaN for excluded
    joints so nanmean gives the correct overall average.

Use Accumulator to collect per-sample results and summarise a dataset.

Usage
-----
    from eval.metrics import pck, oks, mpjpe, Accumulator

    acc = Accumulator()
    for sample in dataset:
        pred = assemble(solve(sample.part_mask), sample.part_mask,
                        sample.torso_diagonal)
        acc.update(
            pck(pred, sample.keypoints, sample.torso_diagonal),
            oks(pred, sample.keypoints, sample.bbox),
            mpjpe(pred, sample.keypoints, sample.torso_diagonal),
        )
    summary = acc.summarise()
    print(summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose.parts import COCO_JOINT_NAMES

# ── OKS per-joint sigma constants (from the COCO benchmark paper) ─────────────
# Larger σ = more positional tolerance for that joint.
# Order matches COCO 17-keypoint index (nose=0 … right_ankle=16).
_OKS_SIGMAS = np.array([
    0.026,   # 0  nose
    0.025,   # 1  left_eye
    0.025,   # 2  right_eye
    0.035,   # 3  left_ear
    0.035,   # 4  right_ear
    0.079,   # 5  left_shoulder
    0.079,   # 6  right_shoulder
    0.072,   # 7  left_elbow
    0.072,   # 8  right_elbow
    0.062,   # 9  left_wrist
    0.062,   # 10 right_wrist
    0.107,   # 11 left_hip
    0.107,   # 12 right_hip
    0.087,   # 13 left_knee
    0.087,   # 14 right_knee
    0.089,   # 15 left_ankle
    0.089,   # 16 right_ankle
], dtype=np.float64)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PCKResult:
    """PCK outcome for one sample.

    Attributes:
        per_joint:  (17,) float32.  1.0 = correct, 0.0 = wrong or GT invisible.
        gt_visible: (17,) bool.  True where GT keypoint was labelled.
        score:      Overall fraction of visible GT joints that are correct.
                    NaN when the sample has no visible GT joints.
    """
    per_joint:  np.ndarray   # (17,) float32
    gt_visible: np.ndarray   # (17,) bool
    score:      float        # scalar


@dataclass
class MPJPEResult:
    """MPJPE outcome for one sample.

    Attributes:
        per_joint:  (17,) float64.  Normalised pixel error per joint.
                    NaN when the joint was not evaluated (GT invisible or
                    prediction missing).
        overall:    Mean of per_joint, ignoring NaNs.
                    NaN when no joint was evaluated at all.
    """
    per_joint:  np.ndarray   # (17,) float64, NaN where not evaluated
    overall:    float        # scalar (NaN if nothing evaluated)


# ── Per-sample metrics ────────────────────────────────────────────────────────

def pck(
    pred:           np.ndarray,
    gt_keypoints:   np.ndarray,
    torso_diagonal: float,
    threshold:      float = 0.2,
) -> PCKResult:
    """Compute PCK@threshold for one sample.

    Args:
        pred:            (17, 3) float32 from skeleton.assemble().
                         Columns: x, y, confidence.
        gt_keypoints:    (17, 3) float32 from PoseSample.keypoints.
                         Column 2 is COCO visibility: 0=unlabelled,
                         1=occluded, 2=visible.
        torso_diagonal:  Pixel length of the bounding-box diagonal
                         (PoseSample.torso_diagonal).  The distance
                         threshold is  threshold × torso_diagonal.
        threshold:       PCK threshold as a fraction of torso_diagonal.
                         Default 0.2 (standard PCK@0.2).

    Returns:
        PCKResult with per-joint correctness and overall score.

    How it works
    ------------
    For each joint i where GT is labelled (gt_visibility > 0):
      - If pred[i, 2] == 0 (not predicted)  → wrong
      - Else compute d_i = ||pred[i,:2] − gt[i,:2]||
            if d_i ≤ threshold × torso_diagonal  → correct (1.0)
            else                                  → wrong   (0.0)

    Joints where GT is invisible are excluded from the score entirely.
    The overall score is:  correct_count / visible_gt_count.
    """
    gt_xy   = gt_keypoints[:, :2].astype(np.float64)
    gt_vis  = gt_keypoints[:, 2] > 0          # (17,) bool
    pr_xy   = pred[:, :2].astype(np.float64)
    pr_conf = pred[:, 2]

    dist_threshold = threshold * torso_diagonal

    per_joint = np.zeros(17, dtype=np.float32)

    for i in range(17):
        if not gt_vis[i]:
            continue                          # GT invisible → skip
        if pr_conf[i] == 0:
            per_joint[i] = 0.0               # not predicted → wrong
            continue
        dist = float(np.linalg.norm(pr_xy[i] - gt_xy[i]))
        per_joint[i] = 1.0 if dist <= dist_threshold else 0.0

    visible_count = int(gt_vis.sum())
    if visible_count == 0:
        score = float("nan")
    else:
        score = float(per_joint[gt_vis].sum() / visible_count)

    return PCKResult(per_joint=per_joint, gt_visible=gt_vis, score=score)


def oks(
    pred:         np.ndarray,
    gt_keypoints: np.ndarray,
    bbox:         np.ndarray,
) -> float:
    """Compute OKS (Object Keypoint Similarity) for one sample.

    Args:
        pred:          (17, 3) float32 — predicted keypoints.
        gt_keypoints:  (17, 3) float32 — GT keypoints with visibility.
        bbox:          (4,) float32 — bounding box [x, y, w, h] in pixels.

    Returns:
        OKS score ∈ [0, 1].  Returns 0.0 if no GT joints are visible.

    Formula (per COCO benchmark)
    ----------------------------
    For each joint i where GT is visible (v_i > 0):

        e_i = exp( −d_i² / (2 · s² · σ_i²) )

        where  d_i = Euclidean distance (predicted vs GT)
               s   = sqrt(bbox_w × bbox_h)   ← object scale
               σ_i = per-joint constant from _OKS_SIGMAS

    If the joint is not predicted (pred conf = 0), d_i → ∞ → e_i = 0.

    OKS = Σ e_i  /  Σ δ(v_i > 0)

    Interpretation: OKS = 1.0 means perfect prediction on all joints.
    OKS = 0.5 means predictions are about (s·σ) pixels off on average.
    """
    gt_xy   = gt_keypoints[:, :2].astype(np.float64)
    gt_vis  = gt_keypoints[:, 2] > 0
    pr_xy   = pred[:, :2].astype(np.float64)
    pr_conf = pred[:, 2]

    # Object scale: square root of bounding-box area
    bbox_w, bbox_h = float(bbox[2]), float(bbox[3])
    scale = float(np.sqrt(bbox_w * bbox_h))
    if scale < 1e-9:
        return 0.0

    numerator   = 0.0
    denominator = 0.0

    for i in range(17):
        if not gt_vis[i]:
            continue                           # skip invisible GT joints

        denominator += 1.0

        if pr_conf[i] == 0:
            continue                           # unpredicted → contributes 0

        d_sq      = float(np.sum((pr_xy[i] - gt_xy[i]) ** 2))
        denom_sq  = 2.0 * (scale ** 2) * (_OKS_SIGMAS[i] ** 2)
        numerator += float(np.exp(-d_sq / denom_sq))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def mpjpe(
    pred:           np.ndarray,
    gt_keypoints:   np.ndarray,
    torso_diagonal: float,
) -> MPJPEResult:
    """Compute normalised MPJPE for one sample.

    Args:
        pred:            (17, 3) float32 — predicted keypoints.
        gt_keypoints:    (17, 3) float32 — GT keypoints with visibility.
        torso_diagonal:  Scale reference for normalisation.

    Returns:
        MPJPEResult with per-joint errors (NaN where not evaluated)
        and overall mean error.

    How it works
    ------------
    For each joint i:
      - Evaluated ONLY when GT is visible AND pred conf > 0.
      - error_i = ||pred[i,:2] − gt[i,:2]|| / torso_diagonal
      - Joints that are not evaluated → NaN in per_joint array.

    overall = nanmean(per_joint)   ← ignores NaN entries.

    Normalising by torso_diagonal makes errors comparable across
    images where the person occupies different amounts of the frame.
    """
    gt_xy   = gt_keypoints[:, :2].astype(np.float64)
    gt_vis  = gt_keypoints[:, 2] > 0
    pr_xy   = pred[:, :2].astype(np.float64)
    pr_conf = pred[:, 2]

    per_joint = np.full(17, np.nan, dtype=np.float64)

    if torso_diagonal < 1e-9:
        return MPJPEResult(per_joint=per_joint, overall=float("nan"))

    for i in range(17):
        if not gt_vis[i]:
            continue                          # GT invisible → not evaluated
        if pr_conf[i] == 0:
            continue                          # not predicted → not evaluated
        dist = float(np.linalg.norm(pr_xy[i] - gt_xy[i]))
        per_joint[i] = dist / torso_diagonal

    overall = float(np.nanmean(per_joint)) \
              if not np.all(np.isnan(per_joint)) else float("nan")

    return MPJPEResult(per_joint=per_joint, overall=overall)


# ── Dataset-level accumulator ─────────────────────────────────────────────────

@dataclass
class Accumulator:
    """Collects per-sample metric results and computes dataset-level summaries.

    Usage
    -----
        acc = Accumulator()
        for sample in dataset:
            ...
            acc.update(pck_result, oks_score, mpjpe_result)
        summary = acc.summarise()

    The summary dict contains:
        pck_overall        — scalar: correct / visible across all samples
        pck_per_joint      — (17,) float: per-joint PCK
        oks_mean           — scalar: mean OKS across samples
        mpjpe_overall      — scalar: mean normalised error across all evaluated joints
        mpjpe_per_joint    — (17,) float: per-joint mean normalised error
        n_samples          — int: number of samples accumulated
        joint_names        — list[str]: COCO joint names in index order
    """

    # PCK accumulators: sum of correct and sum of visible per joint
    _pck_correct: np.ndarray = field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )
    _pck_visible: np.ndarray = field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )

    # OKS accumulator
    _oks_sum:   float = 0.0
    _oks_count: int   = 0

    # MPJPE accumulators: running sum and count of valid errors per joint
    _mpjpe_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )
    _mpjpe_count: np.ndarray = field(
        default_factory=lambda: np.zeros(17, dtype=np.float64)
    )

    _n_samples: int = 0

    def update(
        self,
        pck_result:   PCKResult,
        oks_score:    float,
        mpjpe_result: MPJPEResult,
    ) -> None:
        """Add one sample's results to the running totals.

        PCK aggregation strategy
        ------------------------
        We accumulate correct counts and visible counts separately per
        joint rather than averaging scores.  This means a sample with
        one visible joint and a sample with 17 visible joints contribute
        proportionally — the sample with more labelled joints carries
        more weight.  The formula is:

            per_joint_PCK[j] = Σ correct[j]  /  Σ visible[j]

        This is the same strategy used by the COCO benchmark.

        MPJPE aggregation
        -----------------
        For each joint, we accumulate the sum of errors and count of
        evaluated joints across samples.  NaN entries in MPJPEResult
        are excluded.  Final per-joint MPJPE = sum / count.
        """
        # PCK
        self._pck_correct += pck_result.per_joint.astype(np.float64)
        self._pck_visible += pck_result.gt_visible.astype(np.float64)

        # OKS
        if np.isfinite(oks_score):
            self._oks_sum   += oks_score
            self._oks_count += 1

        # MPJPE — only accumulate valid (non-NaN) per-joint values
        valid = ~np.isnan(mpjpe_result.per_joint)
        self._mpjpe_sum[valid]   += mpjpe_result.per_joint[valid]
        self._mpjpe_count[valid] += 1

        self._n_samples += 1

    def summarise(self) -> dict:
        """Compute and return the dataset-level summary dictionary."""

        # Per-joint PCK: correct / visible (NaN where joint never appeared)
        with np.errstate(invalid="ignore", divide="ignore"):
            pck_per_joint = np.where(
                self._pck_visible > 0,
                self._pck_correct / self._pck_visible,
                np.nan,
            )

        # Overall PCK: total correct / total visible across all joints & samples
        total_correct = float(self._pck_correct.sum())
        total_visible = float(self._pck_visible.sum())
        pck_overall = total_correct / total_visible \
                      if total_visible > 0 else float("nan")

        # Mean OKS
        oks_mean = self._oks_sum / self._oks_count \
                   if self._oks_count > 0 else float("nan")

        # Per-joint MPJPE
        with np.errstate(invalid="ignore", divide="ignore"):
            mpjpe_per_joint = np.where(
                self._mpjpe_count > 0,
                self._mpjpe_sum / self._mpjpe_count,
                np.nan,
            )

        # Overall MPJPE: mean over all joints that had at least one evaluation
        mpjpe_overall = float(np.nanmean(mpjpe_per_joint)) \
                        if not np.all(np.isnan(mpjpe_per_joint)) \
                        else float("nan")

        return {
            "pck_overall":     pck_overall,
            "pck_per_joint":   pck_per_joint,
            "oks_mean":        oks_mean,
            "mpjpe_overall":   mpjpe_overall,
            "mpjpe_per_joint": mpjpe_per_joint,
            "n_samples":       self._n_samples,
            "joint_names":     COCO_JOINT_NAMES,
        }
