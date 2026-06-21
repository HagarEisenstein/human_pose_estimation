"""
Skeleton assembly and plausibility checks — M2.

Takes the raw (17, 3) keypoints produced by the joint solver and applies
three sequential post-processing steps:

  1. Confidence threshold  — joints below min_conf are zeroed immediately.

  2. Limb-length filter    — each limb's pixel length is compared to the
     torso diagonal (the scale reference).  Limbs that are impossibly short
     or impossibly long have their endpoint's confidence zeroed.

  3. Joint-angle filter    — the interior angle at elbow and knee joints is
     computed.  Angles below a minimum threshold indicate the solver placed
     a joint in an anatomically impossible position; confidence is zeroed.

Left/right correction (_correct_lr / _detect_swap_from_joints /
_detect_swap_from_pca below) is implemented but NOT applied by default.
It assumes the person directly faces the camera ("anatomical left always
renders at higher image x"), which doesn't hold for back-facing, profile,
or otherwise non-frontal poses.  Measured on DensePose-oracle masks (whose
part labels are already anatomically correct regardless of camera facing),
this correction fired on 27% of samples and made the result worse 90% of
the time it fired (0% improved) — so it is disabled here.  The functions
remain available/tested for segmentation backends that genuinely produce
ambiguous left/right labels (e.g. from a symmetric source mask).

Usage
-----
    from pose.skeleton import assemble

    refined = assemble(raw_keypoints, part_mask, torso_diagonal)
    # refined: (17, 3) float32, implausible joints zeroed out
"""

from __future__ import annotations

import numpy as np

from pose.parts import Part

# ── Limb constraints ──────────────────────────────────────────────────────────
# Each tuple: (parent_coco_idx, child_coco_idx, name, min_ratio, max_ratio)
# Limb length must lie in [min_ratio * torso_diagonal, max_ratio * torso_diagonal].
# Bounds are intentionally generous to avoid false rejections.
_LIMB_CONSTRAINTS: list[tuple[int, int, str, float, float]] = [
    (5,  7,  "left_upper_arm",  0.05, 1.2),
    (7,  9,  "left_lower_arm",  0.05, 1.1),
    (6,  8,  "right_upper_arm", 0.05, 1.2),
    (8,  10, "right_lower_arm", 0.05, 1.1),
    (11, 13, "left_upper_leg",  0.05, 1.5),
    (13, 15, "left_lower_leg",  0.05, 1.5),
    (12, 14, "right_upper_leg", 0.05, 1.5),
    (14, 16, "right_lower_leg", 0.05, 1.5),
    (5,  11, "left_trunk",      0.10, 2.0),
    (6,  12, "right_trunk",     0.10, 2.0),
]

# ── Angle constraints ─────────────────────────────────────────────────────────
# Each tuple: (parent_idx, joint_idx, child_idx, min_interior_deg)
# Interior angle at joint_idx between the two incoming limbs.
# We only enforce a minimum (≈ no hyperextension or impossible fold-back).
_ANGLE_CONSTRAINTS: list[tuple[int, int, int, float]] = [
    (5,  7,  9,  10.0),   # left elbow:  shoulder → elbow → wrist
    (6,  8,  10, 10.0),   # right elbow: shoulder → elbow → wrist
    (11, 13, 15, 10.0),   # left knee:   hip → knee → ankle
    (12, 14, 16, 10.0),   # right knee:  hip → knee → ankle
]

# ── Left/right joint pairs (swapped together if labels are flipped) ───────────
_LR_PAIRS: list[tuple[int, int]] = [
    (1,  2),   # left_eye   ↔ right_eye
    (3,  4),   # left_ear   ↔ right_ear
    (5,  6),   # left_shoulder ↔ right_shoulder
    (7,  8),   # left_elbow    ↔ right_elbow
    (9,  10),  # left_wrist    ↔ right_wrist
    (11, 12),  # left_hip      ↔ right_hip
    (13, 14),  # left_knee     ↔ right_knee
    (15, 16),  # left_ankle    ↔ right_ankle
]


# ── Public API ────────────────────────────────────────────────────────────────

def assemble(
    keypoints: np.ndarray,
    part_mask: np.ndarray,
    torso_diagonal: float,
    *,
    min_conf: float = 0.05,
) -> np.ndarray:
    """Refine raw joint-solver output into an anatomically plausible skeleton.

    Args:
        keypoints:       (17, 3) float32 from joints.solve().
                         Columns: x, y, confidence.
        part_mask:       (H, W) uint8 canonical Part labels — used for
                         torso PCA when shoulders/hips are not both visible.
        torso_diagonal:  Pixel length of the bounding-box diagonal
                         (sample.torso_diagonal).  Used as the scale
                         reference for limb-length checks.
        min_conf:        Joints with confidence below this value are zeroed
                         before any other check is applied.

    Returns:
        (17, 3) float32 with the same layout as keypoints.
        Implausible joints have their confidence (and x, y) set to 0.
    """
    kp = keypoints.copy()

    # Step 1 — zero joints that the solver couldn't locate reliably
    _zero_low_confidence(kp, min_conf)

    # Step 2 — limb-length plausibility
    if torso_diagonal > 0:
        _apply_limb_filter(kp, torso_diagonal)

    # Step 3 — joint-angle plausibility
    _apply_angle_filter(kp)

    return kp


# ── Step 1: confidence threshold ──────────────────────────────────────────────

def _zero_low_confidence(kp: np.ndarray, min_conf: float) -> None:
    """Zero out joints whose confidence is below min_conf (in-place).

    Why: the joint solver assigns confidence=0 when a part is absent, and
    low-but-nonzero confidence when the intersection band is very small.
    Both cases are unreliable; zeroing them prevents them from influencing
    downstream metrics.
    """
    low = kp[:, 2] < min_conf
    kp[low] = 0.0


# ── Step 2: left/right correction ────────────────────────────────────────────

def _correct_lr(kp: np.ndarray, part_mask: np.ndarray) -> np.ndarray:
    """Detect and fix a global left/right label swap.

    The COCO convention: "left" refers to the person's anatomical left,
    which in a standard frontal photo appears on the RIGHT side of the image
    (higher image x-coordinate).  So in normal cases:
        left_shoulder.x  >  right_shoulder.x
        left_hip.x       >  right_hip.x

    Strategy (tried in order until one succeeds):
      1. Both shoulders visible → compare their x-coordinates.
      2. Both hips visible      → compare their x-coordinates.
      3. Fallback: torso PCA    → project available left/right joints
                                  onto the lateral body axis.

    If a swap is detected, ALL left/right pairs are swapped simultaneously.
    """
    kp = kp.copy()

    swap_needed = _detect_swap_from_joints(kp)

    if swap_needed is None:
        # Fallback: use torso PCA
        swap_needed = _detect_swap_from_pca(kp, part_mask)

    if swap_needed:
        for l_idx, r_idx in _LR_PAIRS:
            kp[[l_idx, r_idx]] = kp[[r_idx, l_idx]]

    return kp


def _detect_swap_from_joints(kp: np.ndarray) -> bool | None:
    """Check shoulder then hip x-positions.  Returns True/False/None.

    Returns None when neither pair has both joints visible (can't decide).

    For a frontal-facing person:
      - person's left  → camera's right → higher image x
      - person's right → camera's left  → lower  image x
    So left_shoulder.x SHOULD be greater than right_shoulder.x.
    If it is smaller, the labels are swapped.
    """
    l_sh, r_sh = kp[5], kp[6]
    if l_sh[2] > 0 and r_sh[2] > 0:
        # left_shoulder.x < right_shoulder.x  →  labels are swapped
        return bool(l_sh[0] < r_sh[0])

    l_hi, r_hi = kp[11], kp[12]
    if l_hi[2] > 0 and r_hi[2] > 0:
        return bool(l_hi[0] < r_hi[0])

    return None   # cannot decide from joints alone


def _detect_swap_from_pca(kp: np.ndarray, part_mask: np.ndarray) -> bool:
    """Use PCA on the TORSO mask to find the lateral body axis.

    Steps:
    1. Collect all TORSO pixel (x, y) coordinates.
    2. Centre them and run SVD — the second right singular vector is
       the lateral axis (perpendicular to the main body axis).
    3. Orient the lateral axis so it points toward higher image x
       (i.e., the direction of "anatomical left" in a frontal view).
    4. Project the centroids of visible left-side and right-side joints
       onto that axis.  The mean projection of left joints should be
       POSITIVE (they're on the left of the torso = higher x).
       If it is negative, labels are swapped.

    Falls back to False (no swap) when the torso is absent or too small
    for reliable PCA, or when there are no visible left/right pairs.
    """
    torso = part_mask == int(Part.TORSO)
    if not torso.any():
        return False

    rows, cols = np.where(torso)
    if len(rows) < 10:      # too few pixels for PCA
        return False

    # (x, y) coordinates of all TORSO pixels
    coords = np.stack([cols, rows], axis=1).astype(float)
    centroid = coords.mean(axis=0)
    centred = coords - centroid

    # SVD: rows of Vt are principal directions
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    lateral = Vt[1]  # second PC = lateral axis, unit vector

    # Orient: positive direction = higher image x (anatomical left, frontal)
    if lateral[0] < 0:
        lateral = -lateral

    # Project visible left and right joints onto the lateral axis
    left_indices  = [l for l, _ in _LR_PAIRS]
    right_indices = [r for _, r in _LR_PAIRS]

    def _mean_proj(indices: list[int]) -> float | None:
        pts = [kp[i, :2] for i in indices if kp[i, 2] > 0]
        if not pts:
            return None
        arr = np.array(pts) - centroid
        return float(np.mean(arr @ lateral))

    left_proj  = _mean_proj(left_indices)
    right_proj = _mean_proj(right_indices)

    if left_proj is None or right_proj is None:
        return False   # not enough data to decide

    # Left joints should project higher (positive) than right joints
    return bool(left_proj < right_proj)


# ── Step 3: limb-length filter ────────────────────────────────────────────────

def _apply_limb_filter(kp: np.ndarray, torso_diagonal: float) -> None:
    """Zero the child joint when limb length is outside plausible bounds.

    For each defined limb (parent → child):
    1. Both parent and child must be visible (conf > 0).
    2. Compute Euclidean distance between them in pixels.
    3. Express it as a ratio:  ratio = distance / torso_diagonal.
    4. If ratio < min_ratio or ratio > max_ratio → the joint is unreliable
       → zero the child joint's confidence (and x, y).

    Why zero the child and not the parent?
    The parent is shared by multiple limbs.  Zeroing the child avoids
    cascading failures up the kinematic tree while still flagging the
    problem joint.
    """
    for parent_idx, child_idx, _, min_r, max_r in _LIMB_CONSTRAINTS:
        parent = kp[parent_idx]
        child  = kp[child_idx]

        if parent[2] == 0 or child[2] == 0:
            continue   # can't check — one end is already invisible

        dist  = float(np.linalg.norm(child[:2] - parent[:2]))
        ratio = dist / torso_diagonal

        if ratio < min_r or ratio > max_r:
            kp[child_idx] = 0.0


# ── Step 4: joint-angle filter ────────────────────────────────────────────────

def _apply_angle_filter(kp: np.ndarray) -> None:
    """Zero a joint when its interior angle is anatomically impossible.

    For each angle triplet (parent → joint → child):
    1. All three joints must be visible.
    2. Compute the interior angle at the middle joint (in degrees).
    3. If the angle < min_deg → zero the middle joint's confidence.

    An interior angle near 0° means the two limb segments fold almost
    completely back on each other — geometrically impossible for elbows
    and knees.  This catches solver errors where a joint is placed on
    the wrong side of the body.

    A nearly straight limb (angle close to 180°) is perfectly valid and
    is NOT rejected.
    """
    for parent_idx, joint_idx, child_idx, min_deg in _ANGLE_CONSTRAINTS:
        p = kp[parent_idx]
        j = kp[joint_idx]
        c = kp[child_idx]

        if p[2] == 0 or j[2] == 0 or c[2] == 0:
            continue   # can't check — an endpoint is missing

        angle = _interior_angle_deg(p[:2], j[:2], c[:2])

        if angle < min_deg:
            kp[joint_idx] = 0.0


def _interior_angle_deg(
    p: np.ndarray,
    j: np.ndarray,
    c: np.ndarray,
) -> float:
    """Interior angle at joint j, formed by segments p→j and c→j.

    Vectors from j to each neighbour:
        v1 = p - j   (from joint toward parent)
        v2 = c - j   (from joint toward child)

    The angle between v1 and v2 is the interior angle at j.
    Uses the dot-product formula: cos(θ) = (v1·v2) / (|v1| |v2|).

    Returns angle in degrees, in [0°, 180°].
    """
    v1 = p - j
    v2 = c - j
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product < 1e-9:
        return 0.0   # degenerate — joint coincides with a neighbour
    cos_a = float(np.dot(v1, v2) / norm_product)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))
