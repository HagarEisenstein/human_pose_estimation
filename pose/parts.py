"""
Part taxonomy and label remapping.

Defines the canonical set of body parts used throughout this project and
provides lookup tables to remap labels from external segmentation models into this shared schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# ── Canonical part IDs ────────────────────────────────────────────────────────

class Part(IntEnum):
    """Canonical body-part labels used by the joint solver."""
    BACKGROUND = 0
    HEAD = 1
    TORSO = 2
    UPPER_ARM_L = 3
    UPPER_ARM_R = 4
    LOWER_ARM_L = 5
    LOWER_ARM_R = 6
    HAND_L = 7
    HAND_R = 8
    UPPER_LEG_L = 9
    UPPER_LEG_R = 10
    LOWER_LEG_L = 11
    LOWER_LEG_R = 12
    FOOT_L = 13
    FOOT_R = 14


NUM_PARTS = len(Part)

# Human-readable names for plotting
PART_NAMES: dict[Part, str] = {
    Part.BACKGROUND:   "background",
    Part.HEAD:         "head",
    Part.TORSO:        "torso",
    Part.UPPER_ARM_L:  "upper-arm-L",
    Part.UPPER_ARM_R:  "upper-arm-R",
    Part.LOWER_ARM_L:  "lower-arm-L",
    Part.LOWER_ARM_R:  "lower-arm-R",
    Part.HAND_L:       "hand-L",
    Part.HAND_R:       "hand-R",
    Part.UPPER_LEG_L:  "upper-leg-L",
    Part.UPPER_LEG_R:  "upper-leg-R",
    Part.LOWER_LEG_L:  "lower-leg-L",
    Part.LOWER_LEG_R:  "lower-leg-R",
    Part.FOOT_L:       "foot-L",
    Part.FOOT_R:       "foot-R",
}

# Distinct BGR colours for visualization (one per part)
PART_COLORS: dict[Part, tuple[int, int, int]] = {
    Part.BACKGROUND:   (0,   0,   0),
    Part.HEAD:         (255, 200, 100),
    Part.TORSO:        (100, 200, 255),
    Part.UPPER_ARM_L:  (255, 100, 100),
    Part.UPPER_ARM_R:  (100, 100, 255),
    Part.LOWER_ARM_L:  (255, 160, 160),
    Part.LOWER_ARM_R:  (160, 160, 255),
    Part.HAND_L:       (200, 80,  80),
    Part.HAND_R:       (80,  80,  200),
    Part.UPPER_LEG_L:  (100, 255, 100),
    Part.UPPER_LEG_R:  (50,  180, 50),
    Part.LOWER_LEG_L:  (180, 255, 180),
    Part.LOWER_LEG_R:  (130, 200, 130),
    Part.FOOT_L:       (150, 220, 150),
    Part.FOOT_R:       (80,  160, 80),
}


# ── Adjacent part pairs that define each joint ────────────────────────────────

@dataclass(frozen=True)
class JointDef:
    """Describes how a joint is derived from two adjacent part masks."""
    name: str
    part_a: Part
    part_b: Part
    coco_idx: int  # index in the COCO 17-keypoint array (-1 = not in COCO)


# Each entry: the boundary between part_a and part_b defines this joint.
JOINT_DEFINITIONS: list[JointDef] = [
    JointDef("nose",           Part.HEAD,        Part.HEAD,        0),   # extremity
    JointDef("left_eye",       Part.HEAD,        Part.HEAD,        1),   # extremity
    JointDef("right_eye",      Part.HEAD,        Part.HEAD,        2),   # extremity
    JointDef("left_ear",       Part.HEAD,        Part.HEAD,        3),   # extremity
    JointDef("right_ear",      Part.HEAD,        Part.HEAD,        4),   # extremity
    JointDef("left_shoulder",  Part.TORSO,       Part.UPPER_ARM_L, 5),
    JointDef("right_shoulder", Part.TORSO,       Part.UPPER_ARM_R, 6),
    JointDef("left_elbow",     Part.UPPER_ARM_L, Part.LOWER_ARM_L, 7),
    JointDef("right_elbow",    Part.UPPER_ARM_R, Part.LOWER_ARM_R, 8),
    JointDef("left_wrist",     Part.LOWER_ARM_L, Part.HAND_L,      9),
    JointDef("right_wrist",    Part.LOWER_ARM_R, Part.HAND_R,      10),
    JointDef("left_hip",       Part.TORSO,       Part.UPPER_LEG_L, 11),
    JointDef("right_hip",      Part.TORSO,       Part.UPPER_LEG_R, 12),
    JointDef("left_knee",      Part.UPPER_LEG_L, Part.LOWER_LEG_L, 13),
    JointDef("right_knee",     Part.UPPER_LEG_R, Part.LOWER_LEG_R, 14),
    JointDef("left_ankle",     Part.LOWER_LEG_L, Part.FOOT_L,      15),
    JointDef("right_ankle",    Part.LOWER_LEG_R, Part.FOOT_R,      16),
]

COCO_JOINT_NAMES = [j.name for j in JOINT_DEFINITIONS]  # ordered by coco_idx


# ── Label remapping tables ────────────────────────────────────────────────────

# DensePose dp_masks 14-channel labels → canonical Part
#
# dp_masks is a 14-element list (channels 1-14); channel order verified
# empirically against COCO GT keypoints (each part's pixel centroid was
# checked against its expected nearest keypoint across 37 val2017 samples,
# see eval diagnostics) — the previous table's channel order did not match
# the actual dp_masks layout, scrambling every limb label.
# https://github.com/facebookresearch/DensePose/blob/main/doc/DENSEPOSE_IUV.md
DENSEPOSE_TO_PART: dict[int, Part] = {
    0:  Part.BACKGROUND,
    1:  Part.TORSO,
    2:  Part.HAND_R,
    3:  Part.HAND_L,
    4:  Part.FOOT_L,
    5:  Part.FOOT_R,
    6:  Part.UPPER_LEG_R,
    7:  Part.UPPER_LEG_L,
    8:  Part.LOWER_LEG_R,
    9:  Part.LOWER_LEG_L,
    10: Part.UPPER_ARM_L,
    11: Part.UPPER_ARM_R,
    12: Part.LOWER_ARM_L,
    13: Part.LOWER_ARM_R,
    14: Part.HEAD,
}


# SegFormer-clothes (mattmdjaga/segformer_b2_clothes) 18-class labels → canonical Part
# Trained on the ATR clothes-parsing dataset.  No separate left/right for arms or legs
# in the source labels — they get the same Part on both sides.
# https://huggingface.co/mattmdjaga/segformer_b2_clothes
SEGFORMER_CLOTHES_TO_PART: dict[int, Part] = {
    0:  Part.BACKGROUND,
    1:  Part.HEAD,         # Hat
    2:  Part.HEAD,         # Hair
    3:  Part.HEAD,         # Sunglasses
    4:  Part.TORSO,        # Upper-clothes
    5:  Part.BACKGROUND,   # Skirt — no canonical mapping
    6:  Part.UPPER_LEG_L,  # Pants → both legs share label (no L/R in source)
    7:  Part.TORSO,        # Dress
    8:  Part.TORSO,        # Belt
    9:  Part.FOOT_L,       # Left-shoe
    10: Part.FOOT_R,       # Right-shoe
    11: Part.HEAD,         # Face
    12: Part.LOWER_LEG_L,  # Left-leg
    13: Part.LOWER_LEG_R,  # Right-leg
    14: Part.UPPER_ARM_L,  # Left-arm  (no upper/lower distinction in source)
    15: Part.UPPER_ARM_R,  # Right-arm
    16: Part.BACKGROUND,   # Bag
    17: Part.HEAD,         # Scarf
}


def remap_mask(mask_hw: "np.ndarray", remap: dict[int, Part]) -> "np.ndarray":  # noqa: F821
    """Apply a label remap table to a (H, W) integer mask.

    Args:
        mask_hw: Integer array of shape (H, W) with source labels.
        remap: Mapping from source label integer → canonical Part.

    Returns:
        Integer array of shape (H, W) with canonical Part values.
    """
    import numpy as np

    out = np.zeros_like(mask_hw, dtype=np.uint8)
    for src, dst in remap.items():
        out[mask_hw == src] = int(dst)
    return out
