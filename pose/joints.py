"""
Rule-based joint solver — M2.

For each of the 17 COCO joints defined in JOINT_DEFINITIONS, derives a
2D location from the canonical part mask using three geometric strategies:

  Boundary joints  (used for: knees)
    Dilate both adjacent part masks → find intersection band → center-of-mass.
    Confidence = intersection band area / smaller-part area (clipped 0–1).

  Extremity joints  (used for: shoulders, hips, elbows, wrists, ankles)
    The anatomical joint sits at the END of one body segment, not at the
    segment boundary.  Take the topmost or bottommost pixels of the relevant
    part and return their centroid.
      shoulder → topmost pixels of UPPER_ARM   (top of arm ≈ shoulder joint)
      hip      → topmost pixels of UPPER_LEG   (top of thigh ≈ hip joint)
      elbow    → bottommost pixels of UPPER_ARM (bottom ≈ elbow joint)
      wrist    → bottommost pixels of LOWER_ARM
      ankle    → topmost pixels of FOOT

  Endpoint joints  (used for: nose, eyes, ears — all HEAD region)
    Locate face landmarks from spatial sub-regions of the HEAD mask.
    Confidence = 1.0 when HEAD is present, 0.0 otherwise.

Usage
-----
    from pose.joints import solve

    keypoints = solve(sample.part_mask)   # (17, 3) float32: x, y, confidence
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass

from pose.parts import JOINT_DEFINITIONS, JointDef, Part

# ── Dilation parameters ───────────────────────────────────────────────────────

# How many pixels each mask is expanded before intersecting.
# Larger radius = wider band = easier to find the joint but less precise.
DILATE_RADIUS: int = 7

# Square structuring element used for dilation.
_STRUCT = np.ones((2 * DILATE_RADIUS + 1, 2 * DILATE_RADIUS + 1), dtype=bool)

# ── Extremity joint dispatch table ────────────────────────────────────────────
# Maps joint name → (Part to examine, direction)
# "top"    = use topmost pixels of the part  (joint is at START of segment)
# "bottom" = use bottommost pixels            (joint is at END of segment)
#
# Rationale: the boundary-center strategy works well for joints that sit
# exactly WHERE two segments meet (e.g. knee = UPPER_LEG ∩ LOWER_LEG).
# But many joints are anatomically at the END of a segment, not at the
# clothing boundary:
#   shoulder → not at shirt armhole (TORSO∩UPPER_ARM center)
#              but at the TOP of the upper arm
#   hip      → not at waistband (TORSO∩UPPER_LEG center)
#              but at the TOP of the upper leg
#   elbow    → BOTTOM of the upper arm (where forearm begins)
#   wrist    → BOTTOM of the lower arm
#   ankle    → TOP of the foot

_EXTREMITY_JOINTS: dict[str, tuple[Part, str]] = {
    "left_shoulder":  (Part.UPPER_ARM_L, "top"),
    "right_shoulder": (Part.UPPER_ARM_R, "top"),
    "left_hip":       (Part.UPPER_LEG_L, "top"),
    "right_hip":      (Part.UPPER_LEG_R, "top"),
    "left_elbow":     (Part.UPPER_ARM_L, "bottom"),
    "right_elbow":    (Part.UPPER_ARM_R, "bottom"),
    "left_wrist":     (Part.LOWER_ARM_L, "bottom"),
    "right_wrist":    (Part.LOWER_ARM_R, "bottom"),
    "left_ankle":     (Part.FOOT_L,      "top"),
    "right_ankle":    (Part.FOOT_R,      "top"),
}


# ── Public API ────────────────────────────────────────────────────────────────

def solve(part_mask: np.ndarray) -> np.ndarray:
    """Derive 17 COCO joint locations from a canonical part mask.

    Args:
        part_mask: (H, W) uint8 array with canonical Part labels.

    Returns:
        Float32 array of shape (17, 3).
        Columns: x (col), y (row), confidence ∈ [0, 1].
        Row index matches JointDef.coco_idx (COCO ordering).
        confidence == 0.0 means the joint could not be located.
    """
    keypoints = np.zeros((17, 3), dtype=np.float32)

    for jdef in JOINT_DEFINITIONS:
        x, y, conf = _solve_one(part_mask, jdef)
        keypoints[jdef.coco_idx] = (x, y, conf)

    return keypoints


# ── Internal dispatch ─────────────────────────────────────────────────────────

def _solve_one(
    part_mask: np.ndarray,
    jdef: JointDef,
) -> tuple[float, float, float]:
    """Route to the correct solver based on joint type.

    Priority order:
      1. HEAD endpoint joints (nose, eyes, ears)   — part_a == part_b
      2. Extremity joints (shoulder, hip, elbow, wrist, ankle) — from table
      3. Boundary joints (knee)  — dilated mask intersection center-of-mass
    """
    if jdef.part_a == jdef.part_b:
        return _endpoint_joint(part_mask, jdef.name)
    if jdef.name in _EXTREMITY_JOINTS:
        part, direction = _EXTREMITY_JOINTS[jdef.name]
        return _extremity_joint(part_mask, part, direction)
    return _boundary_joint(part_mask, jdef.part_a, jdef.part_b)


# ── Extremity joints ──────────────────────────────────────────────────────────

def _extremity_joint(
    part_mask: np.ndarray,
    part: Part,
    direction: str,
    *,
    fraction: float = 0.15,
) -> tuple[float, float, float]:
    """Locate a joint at the extreme end of a single part region.

    Many anatomical joints sit at the TIP of a body segment rather than
    at the boundary between two segments.  This function finds the
    topmost or bottommost cluster of pixels within the part region.

    Args:
        part_mask:  (H, W) uint8 canonical part labels.
        part:       Which canonical Part to examine.
        direction:  "top"    → joint is at the START (topmost pixels).
                    "bottom" → joint is at the END (bottommost pixels).
        fraction:   Fraction of the part's height to average over.
                    Default 0.15 = use the top/bottom 15 % of pixels.

    Returns:
        (x, y, confidence).  confidence = 0 when the part is absent;
        proportional to part area otherwise (saturates at 1.0).

    Examples
    --------
    Shoulder (UPPER_ARM_L, "top"):
        The shoulder joint is where the arm attaches to the torso —
        i.e., the topmost point of the upper arm region.

    Hip (UPPER_LEG_L, "top"):
        The hip joint is at the top of the thigh.

    Elbow (UPPER_ARM_L, "bottom"):
        The elbow is at the bottom of the upper arm where the forearm starts.

    Ankle (FOOT_L, "top"):
        The ankle joint is at the top of the foot.
    """
    mask = part_mask == int(part)
    if not mask.any():
        return 0.0, 0.0, 0.0

    rows, cols = np.where(mask)
    area       = int(mask.sum())
    min_r      = int(rows.min())
    max_r      = int(rows.max())
    h          = max(max_r - min_r + 1, 1)

    if direction == "top":
        threshold = min_r + h * fraction
        sel = rows <= threshold
        if not sel.any():
            sel = rows == min_r       # fallback: topmost row only
    else:  # "bottom"
        threshold = max_r - h * fraction
        sel = rows >= threshold
        if not sel.any():
            sel = rows == max_r       # fallback: bottommost row only

    cy = float(rows[sel].mean())
    cx = float(cols[sel].mean())

    # Confidence grows with part area; saturates around 300 pixels
    confidence = float(np.clip(area / 300.0, 0.05, 1.0))

    return cx, cy, confidence


# ── Boundary joints ───────────────────────────────────────────────────────────

def _boundary_joint(
    part_mask: np.ndarray,
    part_a: Part,
    part_b: Part,
) -> tuple[float, float, float]:
    """Locate a joint at the boundary between two adjacent part regions.

    The geometric idea: the joint lives where two body segments meet.
    By dilating (expanding) each segment's mask outward, we create a
    band of overlap — the intersection of the two dilated masks.
    The center-of-mass of that band is our joint estimate.

    Steps
    -----
    1. Extract a binary mask for part_a (True where pixels belong to part_a).
    2. Extract a binary mask for part_b.
    3. If either mask is empty → joint not visible, return confidence 0.
    4. Dilate both masks by DILATE_RADIUS pixels using a square kernel.
       Dilation expands each region outward so they overlap at the boundary.
    5. Compute intersection: pixels that are in BOTH dilated masks.
       This is the "intersection band" along the boundary between the two parts.
    6. If the band is empty (parts too far apart) → confidence 0.
    7. Center-of-mass of the band → (x, y) joint location.
    8. Confidence = band area / area of the smaller part, clipped to [0, 1].
       Larger band relative to part size = more reliable joint.
    """
    # Step 1–2: binary masks
    mask_a = part_mask == int(part_a)
    mask_b = part_mask == int(part_b)

    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())

    # Step 3: if either part is absent, joint is invisible
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0, 0.0

    # Step 4: dilate both masks
    dilated_a = binary_dilation(mask_a, structure=_STRUCT)
    dilated_b = binary_dilation(mask_b, structure=_STRUCT)

    # Step 5: intersection band
    band = dilated_a & dilated_b
    band_area = int(band.sum())

    # Step 6: empty band
    if band_area == 0:
        return 0.0, 0.0, 0.0

    # Step 7: center-of-mass — scipy returns (row, col) = (y, x)
    cy, cx = center_of_mass(band)

    # Step 8: confidence
    confidence = float(np.clip(band_area / (min(area_a, area_b) + 1e-6), 0.0, 1.0))

    return float(cx), float(cy), confidence


# ── Endpoint joints ───────────────────────────────────────────────────────────

def _endpoint_joint(
    part_mask: np.ndarray,
    joint_name: str,
) -> tuple[float, float, float]:
    """Locate nose, eyes, and ears from spatial extremities of the HEAD mask.

    The HEAD region is a blob in the image. Each face landmark is found by
    selecting a spatial sub-region of that blob and taking its centroid:

      nose      → centroid of the topmost 20 % of HEAD pixels
                  (the top of the face, closest to the top of the image)
      left_eye  → centroid of upper-right quadrant of the HEAD bounding box
                  (COCO "left" = person's anatomical left = camera's right)
      right_eye → centroid of upper-left quadrant
      left_ear  → centroid of the rightmost 20 % of HEAD pixels
      right_ear → centroid of the leftmost 20 % of HEAD pixels

    Confidence = 1.0 if HEAD region exists, 0.0 otherwise.
    These joints are inherently less reliable than boundary joints because
    we only have a coarse head blob, not individual facial landmarks.
    """
    head = part_mask == int(Part.HEAD)

    if not head.any():
        return 0.0, 0.0, 0.0

    # All pixel coordinates belonging to HEAD
    rows, cols = np.where(head)

    # Bounding box of the HEAD region
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    h = max_r - min_r + 1   # height of HEAD bounding box
    w = max_c - min_c + 1   # width  of HEAD bounding box

    # Select a sub-region of HEAD pixels based on joint name
    if joint_name == "nose":
        # The DensePose HEAD mask covers the face surface (not the full skull
        # with hair), so the topmost pixels of HEAD correspond to the forehead
        # and upper nose area — close to the COCO nose keypoint location.
        sel = rows <= min_r + h * 0.20
        if not sel.any():
            sel = rows == rows.min()   # fallback: topmost row only

    elif joint_name == "left_eye":
        # Upper 40 % of rows AND right 50 % of cols
        # (person's left eye appears on the right side of image, frontal view)
        sel = (rows <= min_r + h * 0.40) & (cols >= min_c + w * 0.50)
        if not sel.any():
            sel = rows <= min_r + h * 0.40   # fallback: upper half only

    elif joint_name == "right_eye":
        # Upper 40 % of rows AND left 50 % of cols
        sel = (rows <= min_r + h * 0.40) & (cols < min_c + w * 0.50)
        if not sel.any():
            sel = rows <= min_r + h * 0.40

    elif joint_name == "left_ear":
        # Rightmost 20 % of cols (person's left ear, camera's right)
        sel = cols >= max_c - w * 0.20
        if not sel.any():
            sel = cols == cols.max()   # fallback: rightmost column only

    elif joint_name == "right_ear":
        # Leftmost 20 % of cols
        sel = cols <= min_c + w * 0.20
        if not sel.any():
            sel = cols == cols.min()

    else:
        # Unknown head joint — fall back to full HEAD centroid
        cy, cx = center_of_mass(head)
        return float(cx), float(cy), 1.0

    cy = float(rows[sel].mean())
    cx = float(cols[sel].mean())
    return cx, cy, 1.0
