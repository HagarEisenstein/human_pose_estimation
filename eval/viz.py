"""
Visualization utilities — overlay part masks and skeletons on images.

Main entry points:
    draw_part_mask(image, mask)         → coloured part overlay
    draw_keypoints(image, keypoints)    → joints as dots
    draw_skeleton(image, keypoints)     → joints + limb lines
    show_sample(sample)                 → all-in-one panel
    show_pipeline(sample, raw_kp, refined_kp)  → pipeline 4-panel figure
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pose.parts import PART_COLORS, PART_NAMES, Part, JOINT_DEFINITIONS

# ── COCO skeleton connectivity (pairs of keypoint indices) ────────────────────
COCO_SKELETON = [
    (0, 1), (0, 2),              # nose → eyes
    (1, 3), (2, 4),              # eyes → ears
    (5, 7), (7, 9),              # left  arm
    (6, 8), (8, 10),             # right arm
    (5, 6),                      # shoulders
    (5, 11), (6, 12),            # shoulders → hips
    (11, 12),                    # hips
    (11, 13), (13, 15),          # left  leg
    (12, 14), (14, 16),          # right leg
]

LEFT_JOINTS  = {1, 3, 5, 7, 9, 11, 13, 15}   # COCO indices for left side
RIGHT_JOINTS = {2, 4, 6, 8, 10, 12, 14, 16}  # COCO indices for right side

LIMB_COLOR_LEFT  = (255, 100, 100)  # BGR blue-ish
LIMB_COLOR_RIGHT = (100, 100, 255)  # BGR red-ish
LIMB_COLOR_MID   = (100, 255, 100)  # BGR green-ish


# ── Part mask overlay ─────────────────────────────────────────────────────────

def draw_part_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a coloured part mask on *image*.

    Args:
        image:  RGB uint8 (H, W, 3).
        mask:   Integer (H, W) with canonical Part labels.
        alpha:  Blend weight for the mask layer (0 = invisible, 1 = opaque).

    Returns:
        RGB uint8 (H, W, 3) with mask blended in.
    """
    out_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    color_layer = np.zeros_like(out_bgr)

    for part in Part:
        if part == Part.BACKGROUND:
            continue
        color_bgr = PART_COLORS[part]
        color_layer[mask == int(part)] = color_bgr

    blended = cv2.addWeighted(out_bgr, 1.0 - alpha, color_layer, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# ── Keypoint / skeleton drawing ───────────────────────────────────────────────

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    radius: int = 5,
    only_visible: bool = True,
) -> np.ndarray:
    """Draw keypoint dots on *image*.

    Args:
        image:        RGB uint8 (H, W, 3).
        keypoints:    (17, 3) float array — x, y, visibility.
        radius:       Dot radius in pixels.
        only_visible: If True, skip keypoints with visibility == 0.

    Returns:
        RGB uint8 copy with dots drawn.
    """
    out_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

    for i, (x, y, v) in enumerate(keypoints):
        if only_visible and v == 0:
            continue
        color = (0, 255, 255)  # yellow in BGR
        cv2.circle(out_bgr, (int(x), int(y)), radius, color, -1)
        cv2.circle(out_bgr, (int(x), int(y)), radius, (0, 0, 0), 1)  # outline

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    thickness: int = 2,
    radius: int = 5,
    only_visible: bool = True,
) -> np.ndarray:
    """Draw skeleton (joints + limb lines) on *image*.

    Args:
        image:        RGB uint8 (H, W, 3).
        keypoints:    (17, 3) float array — x, y, visibility.
        thickness:    Line thickness in pixels.
        radius:       Joint dot radius in pixels.
        only_visible: Skip invisible joints.

    Returns:
        RGB uint8 copy with skeleton drawn.
    """
    out_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

    # Draw limbs
    for (i, j) in COCO_SKELETON:
        xi, yi, vi = keypoints[i]
        xj, yj, vj = keypoints[j]
        if only_visible and (vi == 0 or vj == 0):
            continue
        # Pick colour based on which side the limb belongs to
        if i in LEFT_JOINTS or j in LEFT_JOINTS:
            color = LIMB_COLOR_LEFT
        elif i in RIGHT_JOINTS or j in RIGHT_JOINTS:
            color = LIMB_COLOR_RIGHT
        else:
            color = LIMB_COLOR_MID
        cv2.line(out_bgr, (int(xi), int(yi)), (int(xj), int(yj)), color, thickness)

    # Draw joints on top
    for i, (x, y, v) in enumerate(keypoints):
        if only_visible and v == 0:
            continue
        cv2.circle(out_bgr, (int(x), int(y)), radius, (0, 255, 255), -1)
        cv2.circle(out_bgr, (int(x), int(y)), radius, (0, 0, 0), 1)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ── Compound panel ────────────────────────────────────────────────────────────

def show_sample(
    sample,   # PoseSample
    *,
    figsize: tuple[int, int] = (18, 6),
    save_path: str | None = None,
) -> None:
    """Display a 3-panel figure: raw image | part mask | skeleton.

    Args:
        sample:     A PoseSample instance.
        figsize:    Matplotlib figure size.
        save_path:  If set, save the figure to this path instead of showing.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"image_id={sample.image_id}  ann_id={sample.ann_id}  "
        f"visible_kp={sample.num_keypoints_visible}/17",
        fontsize=11,
    )

    # Panel 1 — raw image
    axes[0].imshow(sample.image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Panel 2 — part mask (or blank placeholder)
    if sample.part_mask is not None:
        masked = draw_part_mask(sample.image, sample.part_mask, alpha=0.6)
        axes[1].imshow(masked)
        # Legend patches
        patches = [
            mpatches.Patch(
                color=np.array(PART_COLORS[p][::-1]) / 255,  # BGR→RGB, normalise
                label=PART_NAMES[p],
            )
            for p in Part
            if p != Part.BACKGROUND and np.any(sample.part_mask == int(p))
        ]
        if patches:
            axes[1].legend(handles=patches, loc="lower right", fontsize=6, ncol=2)
    else:
        axes[1].imshow(sample.image)
        axes[1].text(
            0.5, 0.5, "No part mask available",
            ha="center", va="center", transform=axes[1].transAxes, color="red",
        )
    axes[1].set_title("Part mask")
    axes[1].axis("off")

    # Panel 3 — ground-truth skeleton
    skel_img = draw_skeleton(sample.image, sample.keypoints)
    axes[2].imshow(skel_img)
    axes[2].set_title("GT skeleton")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Pipeline panel ────────────────────────────────────────────────────────────

def show_pipeline(
    sample,              # PoseSample — provides .image, .part_mask, .keypoints
    raw_kp: np.ndarray,  # (17, 3) output of pose.joints.solve()
    refined_kp: np.ndarray,  # (17, 3) output of pose.skeleton.assemble()
    *,
    figsize: tuple[int, int] = (24, 6),
    save_path: str | None = None,
) -> None:
    """Display a 4-panel pipeline figure.

    Panels (left to right):
        1. Original image
        2. Part-mask overlay on image  (segmentor output)
        3. Raw joint predictions on part-mask  (joint solver output)
        4. Assembled + refined skeleton on original image  (skeleton assembly output)

    An optional 5th column can be toggled by passing a GT skeleton separately,
    but the default keeps it to 4 panels to stay readable at A4 width.

    Args:
        sample:     PoseSample — provides .image, .part_mask, .keypoints.
        raw_kp:     (17, 3) float32 — raw keypoints from solve().
        refined_kp: (17, 3) float32 — refined keypoints from assemble().
        figsize:    Matplotlib figure size.
        save_path:  If set, save to this path instead of calling plt.show().
    """
    image = sample.image  # RGB uint8

    # Panel 2 — part-mask overlay
    if sample.part_mask is not None:
        mask_overlay = draw_part_mask(image, sample.part_mask, alpha=0.55)
    else:
        mask_overlay = image.copy()

    # Panel 3 — raw joints drawn on top of part-mask overlay
    raw_on_mask = mask_overlay.copy()
    raw_bgr = cv2.cvtColor(raw_on_mask, cv2.COLOR_RGB2BGR)
    for i, (x, y, conf) in enumerate(raw_kp):
        if conf == 0:
            continue
        cv2.circle(raw_bgr, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.circle(raw_bgr, (int(x), int(y)), 6, (0, 0, 0), 1)
    raw_on_mask = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Panel 4 — refined skeleton on original image
    refined_img = draw_skeleton(image, refined_kp, thickness=2, radius=5)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    vis_kp  = int(np.sum(sample.keypoints[:, 2] > 0))
    raw_kp_count  = int(np.sum(raw_kp[:, 2] > 0))
    ref_kp_count  = int(np.sum(refined_kp[:, 2] > 0))

    fig.suptitle(
        f"image_id={sample.image_id}  |  "
        f"GT visible={vis_kp}/17  |  "
        f"raw joints={raw_kp_count}/17  |  "
        f"assembled={ref_kp_count}/17",
        fontsize=10,
    )

    axes[0].imshow(image)
    axes[0].set_title("Original image", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask_overlay)
    axes[1].set_title("Part mask (segmentor)", fontsize=9)
    axes[1].axis("off")
    # Add a compact legend for parts that actually appear
    if sample.part_mask is not None:
        patches = [
            mpatches.Patch(
                color=np.array(PART_COLORS[p][::-1]) / 255,
                label=PART_NAMES[p],
            )
            for p in Part
            if p != Part.BACKGROUND and np.any(sample.part_mask == int(p))
        ]
        if patches:
            axes[1].legend(handles=patches, loc="lower right",
                           fontsize=5, ncol=2)

    axes[2].imshow(raw_on_mask)
    axes[2].set_title(f"Raw joints ({raw_kp_count}/17)", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(refined_img)
    axes[3].set_title(f"Assembled skeleton ({ref_kp_count}/17)", fontsize=9)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)
