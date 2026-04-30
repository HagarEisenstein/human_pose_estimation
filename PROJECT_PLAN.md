# Project Work Plan

**Project Overview:** Human Pose Estimation from Body-Part Segmented Images

## Table of contents
1. Project Overview
2. Milestone 1: Foundation & Segmentation Pipeline
3. Milestone 2: Joint Inference, Skeleton Assembly & Evaluation
4. Milestone 3: Validation & Final Report

---

**Project Overview:** This project focuses on building a human pose estimation
system that derives skeletal joint locations from body-part segmentation masks
rather than directly regressing keypoints from RGB pixels. The core idea is
that boundaries between adjacent body segments (e.g., the intersection of the
upper arm and forearm) define the joints that connect them, while extremity
joints (head landmarks, fingertips, foot tips) are derived from segment
endpoints and skeletal priors. By benchmarking the system's generated
skeletons against ground-truth keypoint datasets, we quantitatively evaluate
its fidelity, accuracy, and reliability for downstream computer vision
applications.

**Students:** [TBD]
**Course:** [TBD]
**Project title:** Human Pose Estimation from Body-Part Segmented Images
**Milestones:** 3
**Programming Language:** Python

---

## Milestone 1
**Foundation & Segmentation Pipeline**
**Due Date:** TBD

### Objectives
- Define a canonical body-part taxonomy (15 parts: head, torso, upper/lower
  arms L/R, hands L/R, upper/lower legs L/R, feet L/R) and encode joint
  definitions as adjacent-part pairs.
- Implement a unified `PoseSample` data contract and `PoseDataset` abstract
  base class that every adapter conforms to.
- Implement dataset adapters for COCO Keypoints (17-keypoint ground truth) and
  DensePose-COCO (paired part-segmentation masks).
- Wrap a frozen pretrained segmentation backbone (Mask2Former / SCHP /
  DensePose) behind a uniform interface, with label-remap tables that
  translate source labels to the canonical taxonomy.
- Implement visualization tools to render part-mask overlays, keypoint dots,
  and skeleton lines on RGB images.
- Set up the project scaffold: `pyproject.toml`, dependency files, Makefile,
  pre-commit hooks, and CI workflow.

### Deliverables
- Source code for the canonical part taxonomy, `PoseSample` dataclass,
  `PoseDataset` ABC, COCO and DensePose adapters, segmentation wrapper, and
  download helper CLI.
- Visualizations of paired RGB images with overlaid part masks and
  ground-truth skeletons.
- Justification of the rule-based, segmentation-driven approach (explaining
  why mask-boundary geometry was chosen over direct keypoint regression).
- Test suite covering data structures, label remapping, and visualization
  (≥20 passing tests with CI green).
- Initial figures showing part masks aligned with COCO ground-truth keypoints
  on 50 sample images.

### Comments
This milestone focuses on the fundamental data structures and pipeline
plumbing for the joint-inference experiment. By isolating the segmentation
stage behind a fixed contract, downstream stages (joint solver, evaluator)
are insulated from the choice of segmentation backbone, enabling clean
ablations later.

---

## Milestone 2
**Joint Inference, Skeleton Assembly & Evaluation**
**Due Date:** TBD

### Objectives
- Define and implement the core **Joint Inference Operator** that derives a 2D
  joint location from each pair of adjacent part masks via:
  - Dilated mask intersection band → center-of-mass candidate.
  - Medial-axis projection of each part for refinement.
  - Confidence score from intersection area and mask quality.
- Handle endpoint joints (nose, eyes, ears, foot tips) via medial-axis
  extremities of single parts.
- Implement skeleton assembly: build a tree rooted at the pelvis, enforce
  limb-length plausibility from torso-derived anthropometric ratios, and
  reject implausible joint angles (e.g., elbow ∈ [0°, 170°]).
- Resolve left/right ambiguity using torso PCA orientation.
- Implement raster-based evaluation metrics: PCK@0.2 (Percentage of Correct
  Keypoints), OKS-AP (Object Keypoint Similarity), and per-joint MPJPE
  (pixel error normalized by torso diagonal).
- Compute and store distances against the ground-truth keypoints on a
  paired COCO + DensePose validation subset.

### Deliverables
- Code for the rule-based joint solver, skeleton-assembly module, and
  plausibility filters.
- Implementation of PCK, OKS, and MPJPE metrics plus a CLI evaluation runner.
- Side-by-side visualizations of part mask → raw joints → assembled skeleton
  on 50 images.
- Preliminary graphs showing per-joint PCK and overall OKS-AP across the
  COCO val2017 paired subset.

### Comments
This milestone focuses on the core algorithmic contribution. The joint solver
is purely geometric and rule-based — interpretable, debuggable, and traceable
to a specific mask or rule when failures occur. The plausibility filter
ensures topological validity even when segmentation is noisy.

---

## Milestone 3
**Validation & Final Report**
**Due Date:** TBD

### Objectives
- Analyze and visualize joint-inference behavior: per-joint PCK curves and
  error distribution histograms across the COCO validation subset.
- Build a qualitative gallery: ~10 successful predictions and ~10
  characterized failure cases.
- Compare the segmentation-driven solver against **one** standard baseline
  (HRNet-W32 — the most widely cited top-down regressor) on the same split.
- Briefly characterize 3–4 dominant failure modes (e.g., self-occlusion,
  segmentation noise on thin limbs, atypical poses).
- Prepare the final report and a runnable end-to-end demo.

### Deliverables
- Final graphs: per-joint PCK breakdown, error histogram, and overall
  OKS-AP comparison vs. the chosen baseline.
- Failure-mode gallery with short captions.
- Final report describing: Motivation, Methodology (segmentation-driven
  joint inference), Joint Inference Operator design, Skeleton Assembly &
  Plausibility, Evaluation Metrics, Validation Results, and Conclusions.
- Final demo CLI: `pose-from-seg --image X.jpg --seg X_parts.png →
  keypoints.json` running end-to-end.
- Reproducibility script `make eval` running the full pipeline on a held-out
  split.

### Stretch goals (only if time permits)
- Robustness sweep under mild synthetic mask noise.
- Lightweight learned residual-refinement head over the rule-based output.

### Out of scope (future work)
- Multi-person handling and CrowdPose evaluation.
- Temporal extension to video.
- Multiple competing baselines (OpenPose, ViTPose, etc.).

### Comments
This final milestone focuses on cleanly closing the research story rather
than expanding scope. The report explicitly answers the core question — *can
we recover accurate joint locations from part-segmentation masks alone?* —
and quantifies when the geometric approach matches or trails direct
regression. Ambitious extensions (multi-person, learned refinement) are
deferred to future work to keep M3 deliverables tractable and the analysis
focused.
