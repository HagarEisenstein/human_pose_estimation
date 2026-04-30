# Project Plan: Human Pose Estimation from Body-Part Segmented Images

## 1. Overview

Build a human pose estimation system that derives skeletal joint locations from
**body-part segmentation masks** rather than directly regressing keypoints from
RGB pixels. The core idea: when two adjacent segments (e.g. upper arm and
forearm) meet, the boundary between them — combined with each segment's
geometry — defines the joint that connects them. Joints further from any
intersection (head top, fingertips, foot tips) are derived from segment
extremities and skeletal priors.

The system is benchmarked against ground-truth keypoint datasets to quantify
fidelity for downstream computer vision tasks (action recognition, AR/VR,
biomechanics, sports analytics).

## 2. Goals & Non-Goals

**Goals**
- Recover 2D (and optionally 2.5D) joint locations from per-pixel body-part
  labels.
- Produce a standard skeleton (e.g. COCO 17-keypoint or extended 25-keypoint)
  per detected person.
- Match or beat a direct keypoint-regression baseline on PCK@0.2 / OKS for
  scenes where segmentation is available.
- Provide a reproducible evaluation harness (CLI + notebooks).

**Non-Goals (v1)**
- Training a segmentation model from scratch — we consume an existing one
  (DensePose / Self-Correction Human Parsing / Mask2Former-Human).
- Real-time deployment (>30 FPS); v1 targets correctness first.
- Multi-view 3D reconstruction.

## 3. Inputs & Outputs

| Stage          | Input                                  | Output                                         |
|----------------|----------------------------------------|------------------------------------------------|
| Segmentation   | RGB image                              | Per-pixel part labels (N parts) + person mask  |
| Joint solver   | Part-label map for one person          | K keypoints with confidences                   |
| Post-process   | Raw keypoints                          | Topology-consistent skeleton, occlusion flags  |
| Evaluation     | Predicted skeleton + GT keypoints      | PCK / OKS / per-joint MPJPE                    |

## 4. Datasets

- **MS COCO Keypoints** — bounding boxes + 17 keypoints; large, diverse.
- **MPII Human Pose** — 16 keypoints; activity-rich.
- **LIP / ATR / Pascal-Person-Part** — semantic part segmentation labels.
- **DensePose-COCO** — dense UV correspondences; doubles as fine-grained parts.
- **CrowdPose** — stress test for occlusion and overlapping people.

A small **paired** subset (segmentation + keypoints on the same image) is the
primary evaluation set; COCO + DensePose-COCO covers this.

## 5. Method

### 5.1 Part taxonomy

Define a parts schema aligned with target skeleton:
`head, torso, upper-arm-L/R, forearm-L/R, hand-L/R, upper-leg-L/R,
lower-leg-L/R, foot-L/R`. Map source segmentation labels → schema via a
lookup table.

### 5.2 Joint inference rules

For each joint J connecting parts A and B:
1. **Intersection band**: dilate mask(A) and mask(B) by k px, intersect.
2. **Center of mass** of the intersection band → candidate joint.
3. **Skeletal prior refinement**: project the candidate onto the medial axis
   of A (and B) and average; this stabilizes against irregular boundaries.
4. **Confidence** = f(intersection area, mask quality, distance to medial axis).

Endpoint joints (head-top, wrists if hand absent, ankles → toes):
- Use the farthest point of the part's medial axis from the parent joint.

Symmetry handling: left/right disambiguation via torso orientation (PCA on
torso mask) and consistency with previous frames if temporal input.

### 5.3 Skeleton assembly

- Build the skeleton tree rooted at the pelvis (or torso COM).
- Enforce limb-length plausibility against per-subject anthropometric ratios
  estimated from torso size.
- Reject implausible angles via joint-angle priors (e.g. elbow ∈ [0°, 170°]).

### 5.4 Multi-person

Per-person instance masks → independent solve. For overlap, use depth-order
heuristics (mask area, occlusion completeness) before refinement.

## 6. System Architecture

```
┌────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────┐
│  RGB image │ →  │ Part segmentation│ →  │ Joint inference│ →  │ Skeleton │
└────────────┘    │ (frozen / API)   │    │ (rule-based +  │    │ assembly │
                  └──────────────────┘    │  optional MLP) │    └────┬─────┘
                                          └────────────────┘         │
                                                                     ▼
                                                           ┌──────────────────┐
                                                           │ Evaluation suite │
                                                           └──────────────────┘
```

Code layout:

```
human_pose_estimation/
├── data/                  # dataset adapters, download scripts
├── segmentation/          # wrapper around chosen seg model
├── pose/
│   ├── parts.py           # part taxonomy + label remap
│   ├── joints.py          # rule-based joint solver
│   ├── refine.py          # optional learned residual head
│   └── skeleton.py        # tree assembly + plausibility
├── eval/
│   ├── metrics.py         # PCK, OKS, MPJPE
│   ├── runner.py          # evaluation harness CLI
│   └── viz.py             # overlay rendering
├── notebooks/             # exploration + ablations
├── tests/
└── PROJECT_PLAN.md
```

## 7. Milestones

| # | Milestone                                  | Exit criteria                                          | Est. |
|---|--------------------------------------------|--------------------------------------------------------|------|
| 1 | Repo scaffold + dataset adapters           | COCO + DensePose loaders; CI green                     | 1 wk |
| 2 | Segmentation wrapper                       | Frozen model produces part masks for 100 images        | 1 wk |
| 3 | Rule-based joint solver (v0)               | PCK@0.2 ≥ 0.55 on val subset                           | 2 wk |
| 4 | Skeleton assembly + plausibility           | No NaNs, all skeletons topologically valid             | 1 wk |
| 5 | Evaluation harness + baselines             | Reports PCK/OKS vs. HRNet / OpenPose baseline          | 1 wk |
| 6 | Learned residual refinement (optional)     | +5 PCK over rule-based                                 | 2 wk |
| 7 | Multi-person + occlusion handling          | CrowdPose AP ≥ 0.40                                    | 2 wk |
| 8 | Ablations + final report                   | Plots, tables, reproducible script                     | 1 wk |

Total: ~10–11 weeks.

## 8. Evaluation Plan

- **Primary**: PCK@0.2 and OKS-AP on COCO val2017 (paired with DensePose).
- **Secondary**: per-joint MPJPE in pixels, normalized by torso diagonal.
- **Robustness**: degrade input segmentation (mask noise σ, dropped parts)
  and plot metric vs. degradation.
- **Comparative**: HRNet-W32 (top-down regression) and OpenPose (bottom-up
  PAFs) as baselines on the same split.
- **Qualitative**: 50-image gallery with overlay; failure-mode taxonomy
  (occlusion, self-touching limbs, unusual poses).

## 9. Risks & Mitigations

| Risk                                        | Mitigation                                            |
|---------------------------------------------|-------------------------------------------------------|
| Segmentation errors propagate to joints     | Confidence weighting; learned residual; mask cleanup  |
| Left/right ambiguity                        | Torso PCA + temporal smoothing                        |
| Datasets disagree on part taxonomy          | Explicit remap tables; report per-dataset numbers     |
| Self-occlusion (arm against torso)          | Skeletal priors + plausibility filter                 |
| Compute for segmentation backbone           | Cache masks to disk; eval on fixed subset             |

## 10. Tooling & Stack

- Python 3.11, PyTorch, OpenCV, NumPy, SciPy (medial-axis transform).
- `pycocotools`, `densepose` reference impl.
- `pytest` + `ruff` + `mypy`; pre-commit hooks.
- Experiment tracking via `mlflow` or W&B (decide in milestone 1).
- Optional: ONNX export for the residual model.

## 11. Deliverables

1. Reproducible repo with `make eval` running end-to-end on a small split.
2. Pretrained residual-refinement weights (if M6 is reached).
3. Evaluation report (PDF + notebook) with tables, plots, gallery.
4. CLI: `pose-from-seg --image X.jpg --seg X_parts.png → keypoints.json`.

## 12. Open Questions

- Which segmentation backbone to standardize on (Mask2Former vs. SCHP vs.
  DensePose)? Decision in M2 after a short bake-off.
- 2D only, or extend to 2.5D using DensePose UV? Defer to M5 review.
- Temporal extension for video — separate follow-up project.
