# Project Plan: Human Pose Estimation from Body-Part Segmented Images

**Student:** Hagar Eisenstein
**Course:** Advanced topics in image processing
**Programming Language:** Python

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
- Recover 2D joint locations from per-pixel body-part labels.
- Produce a standard COCO 17-keypoint skeleton per detected person.
- Match a direct keypoint-regression baseline (HRNet-W32) on PCK@0.2 / OKS for
  scenes where segmentation is available.
- Provide a reproducible evaluation harness (CLI + notebooks).

**Non-Goals (v1)**
- Training a segmentation model from scratch — we consume an existing one
  (DensePose / Self-Correction Human Parsing / Mask2Former-Human).
- Real-time deployment (>30 FPS); v1 targets correctness first.
- Multi-view 3D reconstruction.
- Crowded / overlapping multi-person scenes and CrowdPose evaluation
  (deferred to future work).
- Temporal / video extension (deferred to future work).

## 3. Inputs & Outputs

| Stage          | Input                                  | Output                                         |
|----------------|----------------------------------------|------------------------------------------------|
| Segmentation   | RGB image                              | Per-pixel part labels (15 parts) + person mask |
| Joint solver   | Part-label map for one person          | 17 keypoints with confidences                  |
| Post-process   | Raw keypoints                          | Topology-consistent skeleton, occlusion flags  |
| Evaluation     | Predicted skeleton + GT keypoints      | PCK / OKS / per-joint MPJPE                    |

## 4. Datasets

- **MS COCO Keypoints** — bounding boxes + 17 keypoints; large, diverse.
- **DensePose-COCO** — dense UV correspondences; doubles as fine-grained parts.
- **MPII / LIP / Pascal-Person-Part** — optional secondary references.

A small **paired** subset (segmentation + keypoints on the same image) is the
primary evaluation set; COCO val2017 + DensePose-COCO covers this.

## 5. Method

### 5.1 Part taxonomy

Define a 15-part schema:
`background, head, torso, upper-arm-L/R, lower-arm-L/R, hand-L/R,
upper-leg-L/R, lower-leg-L/R, foot-L/R`. Map source segmentation labels →
schema via a lookup table.

### 5.2 Joint inference rules

For each joint J connecting parts A and B:
1. **Intersection band**: dilate mask(A) and mask(B) by k px, intersect.
2. **Center of mass** of the intersection band → candidate joint.
3. **Skeletal prior refinement**: project the candidate onto the medial axis
   of A (and B) and average; this stabilizes against irregular boundaries.
4. **Confidence** = f(intersection area, mask quality, distance to medial axis).

Endpoint joints (nose, eyes, ears, ankle tips):
- Use the farthest point of the part's medial axis from the parent joint.

Symmetry handling: left/right disambiguation via torso orientation (PCA on
torso mask).

### 5.3 Skeleton assembly

- Build the skeleton tree rooted at the pelvis (or torso COM).
- Enforce limb-length plausibility against per-subject anthropometric ratios
  estimated from torso size.
- Reject implausible angles via joint-angle priors (e.g. elbow ∈ [0°, 170°]).

### 5.4 Multi-person

- **Stretch goal — non-overlapping case**: process each COCO person
  annotation independently by cropping to its bounding box, running the
  single-person pipeline, and re-projecting joints to image coordinates.
  Handles scenes with multiple separated people at low cost (~1–2 days).
- **Out of scope — crowded / overlapping case**: full handling (instance
  segmentation, depth ordering, occlusion reasoning) deferred to future work.

## 6. System Architecture

```
┌────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────┐
│  RGB image │ →  │ Part segmentation│ →  │ Joint inference│ →  │ Skeleton │
└────────────┘    │ (frozen / API)   │    │ (rule-based)   │    │ assembly │
                  └──────────────────┘    └────────────────┘    └────┬─────┘
                                                                     │
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
│   ├── refine.py          # (stretch) learned residual head
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

| # | Milestone                                          | Due Date | Exit criteria                                              |
|---|----------------------------------------------------|----------|------------------------------------------------------------|
| 1 | Foundation & Segmentation Pipeline                 | 14.5     | COCO + DensePose loaders, segmentation wrapper, CI green, ≥20 tests passing |
| 2 | Joint Inference, Skeleton Assembly & Evaluation    | 28.5     | Rule-based solver + plausibility filter; PCK/OKS reported on COCO val2017 |
| 3 | Refinement, Multi-Person Handling & Final Report   | 11.6     | HRNet-W32 baseline comparison, failure-mode gallery, final report, demo CLI |

## 8. Evaluation Plan

- **Primary**: PCK@0.2 and OKS-AP on COCO val2017 (paired with DensePose).
- **Secondary**: per-joint MPJPE in pixels, normalized by torso diagonal.
- **Comparative**: HRNet-W32 (top-down regression) as the baseline on the same split.
- **Qualitative**: ~10 successes + ~10 characterized failures gallery; brief
  3–4-mode failure characterization (occlusion, thin-limb noise, atypical poses).
- **Stretch (only if time permits)**:
  - Non-overlapping multi-person evaluation via per-annotation cropping
    (report PCK/OKS on COCO val2017 images containing ≥2 annotated people).
  - Robustness sweep under degraded segmentation (mask noise σ, dropped parts).

## 9. Risks & Mitigations

| Risk                                        | Mitigation                                            |
|---------------------------------------------|-------------------------------------------------------|
| Segmentation errors propagate to joints     | Confidence weighting; mask cleanup; plausibility filter |
| Left/right ambiguity                        | Torso PCA disambiguation                              |
| Datasets disagree on part taxonomy          | Explicit remap tables; report per-dataset numbers     |
| Self-occlusion (arm against torso)          | Skeletal priors + plausibility filter                 |
| Compute for segmentation backbone           | Cache masks to disk; eval on fixed subset             |
| M3 over-scope (multi-baseline, multi-person)| Trim deliverables; defer crowded multi-person to future work |

## 10. Tooling & Stack

- Python 3.11, PyTorch, OpenCV, NumPy, SciPy (medial-axis transform).
- `pycocotools`, `densepose` reference impl.
- `pytest` + `ruff` + `mypy`; pre-commit hooks; GitHub Actions CI.
- Optional: `mlflow` for run tracking.

## 11. Deliverables

1. Reproducible repo with `make eval` running end-to-end on a small split.
2. Final graphs: per-joint PCK breakdown, error histogram, OKS-AP vs. HRNet.
3. Failure-mode gallery (~20 images, captioned).
4. Final report: Motivation, Methodology, Joint Inference Operator, Skeleton
   Assembly, Evaluation Metrics, Validation Results, Conclusions.
5. CLI: `pose-from-seg --image X.jpg --seg X_parts.png → keypoints.json`.
6. *(Stretch)* Non-overlapping multi-person evaluation results on COCO
   val2017 images with ≥2 annotated people.

## 12. Open Questions

- Which segmentation backbone to standardize on (Mask2Former vs. SCHP vs.
  DensePose)? Decision in M1 after a short bake-off.
- Does the teacher's brief require video tracking, or is per-image inference
  sufficient? Confirm before M3.
- Whether to invest in the stretch goal (learned residual refinement) — gated
  on M2 results.
