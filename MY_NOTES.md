# My Project Notes
## Human Pose Estimation from Body-Part Segmented Images

---

## What is this project about?

The goal is to look at a photo of a person and figure out **where their body joints are**
(nose, shoulders, elbows, wrists, hips, knees, ankles — 17 joints total).

The twist: instead of detecting joints directly from the photo, we first label every pixel
with a **body part** (torso, left arm, right leg, etc.), and then use the **boundaries
between those regions** to find the joints.

For example: the elbow is where the upper-arm region meets the lower-arm region.

```
Photo  →  body-part labels per pixel  →  joint locations
```

---

## Key concepts explained simply

### What is a pixel label / part mask?
Every photo is made of tiny pixels. A **part mask** is a second image the same size as
the photo, where instead of colours, each pixel holds a number that says which body part
it belongs to.

```
Photo pixel [50, 100] = colour (210, 180, 140)   ← what colour it is
Mask  pixel [50, 100] = 2                         ← 2 means "torso"
```

Background pixels (not part of any person) get label 0.

### What is a keypoint?
A **keypoint** is a single x,y coordinate marking an exact joint location — for example,
"the left elbow is at pixel (320, 145)". Each keypoint also has a **visibility** flag:
- 0 = not labelled
- 1 = labelled but hidden (occluded)
- 2 = labelled and visible

COCO has 17 keypoints per person: nose, eyes, ears, shoulders, elbows, wrists, hips,
knees, ankles.

### What is a bounding box (bbox)?
A rectangle `[x, y, width, height]` that tightly wraps around one person in the image.
Used to isolate which part of the image belongs to this specific person.

### What is a numpy array (np.ndarray)?
A grid of numbers stored efficiently in memory. Much faster than Python lists for
mathematical operations. A 3D array of shape `(H, W, 3)` stores an image — H rows of
pixels, W columns, 3 colour values per pixel (Red, Green, Blue).

### What is shape?
Describes the size of an array: `(H, W, 3)` means height × width × 3 colour channels.
- `image.shape[0]` → height
- `image.shape[1]` → width
- `image.shape[2]` → 3 (always, for RGB)

### What is dtype?
The type of each number in the array.
- `uint8` → whole numbers 0–255 (used for pixel colours and part labels)
- `float32` → decimal numbers (used for joint coordinates)

### What is a dataclass?
A Python class whose main job is to hold data. You declare the fields and their types,
and Python automatically gives you a constructor, equality checks, etc. `PoseSample` is
a dataclass.

### What is an abstract base class (ABC)?
A class that defines what methods subclasses **must** have, without implementing them.
It is a contract. `PoseDataset` is an ABC — it says every dataset must have `__len__`
and `__getitem__`, but leaves the actual loading to each subclass.

### What is RLE (Run-Length Encoding)?
A compressed way to store a binary mask. Instead of saving every pixel, it stores
"N pixels of value 0, then M pixels of value 1, ...". DensePose stores its masks in
RLE format to save space. We decode it back into a pixel array before using it.

---

## Project structure

```
human_pose_estimation/
│
├── data/                    ← loading data from datasets
│   ├── base.py              ← PoseSample (data container) + PoseDataset (ABC)
│   ├── coco_adapter.py      ← loads images + 17 keypoints from COCO
│   ├── densepose_adapter.py ← extends COCO adapter, also loads part masks
│   └── download.py          ← CLI to download COCO + DensePose files
│
├── pose/                    ← body-part and joint logic
│   ├── parts.py             ← canonical part labels + joint definitions + remap tables
│   ├── joints.py            ← (M2) rule-based joint solver — not yet implemented
│   ├── skeleton.py          ← (M2) skeleton assembly — not yet implemented
│   └── refine.py            ← (stretch) learned refinement — not yet implemented
│
├── segmentation/            ← segmentation model wrappers
│   └── base.py              ← SegmentationModel ABC + GTOracleSegmentor
│
├── eval/                    ← evaluation and visualization
│   ├── viz.py               ← draw part masks, keypoints, and skeletons on images
│   ├── metrics.py           ← (M2) PCK, OKS, MPJPE — not yet implemented
│   └── runner.py            ← (M2) evaluation CLI — not yet implemented
│
├── notebooks/
│   └── visualize_samples.py ← script to generate 50 sample figures
│
├── tests/                   ← automated tests (34 passing)
│
└── PROJECT_PLAN.md          ← milestones, objectives, due dates
```

---

## The data pipeline explained

```
COCO JSON file
    ↓  (loaded by COCOPoseDataset)
_anns — filtered list of person annotations
    ↓  (on each ds[i] call)
image loaded from disk → RGB pixel array (H, W, 3)
keypoints parsed       → (17, 3) float array: x, y, visibility per joint
bbox                   → [x, y, w, h] bounding box
    ↓  (DensePoseDataset adds this)
DensePose JSON → dp_masks (14 RLE-encoded binary masks)
    → decoded into (H, W) uint8 part mask with canonical labels
    ↓
PoseSample(image, keypoints, part_mask, bbox, image_id, ann_id)
```

---

## What each important class does

### `PoseSample` (data/base.py)
A passive data container — holds all information about one person in one image.
Fields: `image`, `keypoints`, `part_mask`, `bbox`, `image_id`, `ann_id`, `meta`.
Also has helper properties: `height`, `width`, `num_keypoints_visible`, `torso_diagonal`.

### `PoseDataset` (data/base.py)
Abstract base class that every dataset adapter inherits from.
Forces every adapter to implement `__len__` and `__getitem__`.
Provides `__iter__` and `take(n)` for free.

### `COCOPoseDataset` (data/coco_adapter.py)
Loads images and 17 keypoints from the COCO dataset.
Filters out crowd annotations and persons with too few visible keypoints.
Returns a `PoseSample` with `part_mask=None`.

### `DensePoseDataset` (data/densepose_adapter.py)
Extends `COCOPoseDataset`. Also loads DensePose part masks.
For each sample, decodes 14 RLE binary masks and pastes them onto a full-image canvas.
Returns a `PoseSample` with `part_mask` filled in.

### `Part` enum (pose/parts.py)
The 15 canonical body-part labels used everywhere in the project:
BACKGROUND(0), HEAD(1), TORSO(2), UPPER_ARM_L/R(3/4), LOWER_ARM_L/R(5/6),
HAND_L/R(7/8), UPPER_LEG_L/R(9/10), LOWER_LEG_L/R(11/12), FOOT_L/R(13/14).

### `JointDef` + `JOINT_DEFINITIONS` (pose/parts.py)
Defines the rule for finding each joint: "joint X is at the boundary between part A
and part B". For example: left elbow = boundary between UPPER_ARM_L and LOWER_ARM_L.
Head joints (nose, eyes, ears) are found from extremities of the HEAD region.

### `DENSEPOSE_TO_PART` / `SCHP_TO_PART` (pose/parts.py)
Translation tables that convert source dataset labels into canonical Part labels.
DensePose has 14 labels, SCHP has 20 — both get mapped to the same 15 canonical labels
so the joint solver always receives the same format.

### `SegmentationModel` (segmentation/base.py)
Abstract base class for all segmentation backends.
Every backend must implement `predict(sample) → part_mask (H, W) uint8`.

### `GTOracleSegmentor` (segmentation/base.py)
A fake "perfect" segmentation model that just returns the ground-truth part mask
already stored in a PoseSample. Used for testing the joint solver without needing
a real GPU model.

---

## Milestone progress

### Milestone 1 — Foundation & Segmentation Pipeline (due 14.5) ✅ DONE
- [x] Canonical part taxonomy (`Part`, `JointDef`, remap tables)
- [x] `PoseSample` dataclass + `PoseDataset` ABC
- [x] COCO adapter
- [x] DensePose adapter
- [x] Segmentation model wrapper (`SegmentationModel` + `GTOracleSegmentor`)
- [x] Visualization tools (`draw_part_mask`, `draw_keypoints`, `draw_skeleton`)
- [x] Download helper CLI
- [x] Project scaffold (`pyproject.toml`, Makefile, CI, pre-commit)
- [x] 34 passing tests (requirement: ≥20)
- [x] Figure-generation script (`notebooks/visualize_samples.py`)

### Milestone 2 — Joint Inference, Skeleton Assembly & Evaluation (due 28.5) 🔲 NEXT
- [ ] Joint solver (`pose/joints.py`) — find joint locations from part mask boundaries
- [ ] Skeleton assembly (`pose/skeleton.py`) — connect joints into a valid skeleton
- [ ] Evaluation metrics (`eval/metrics.py`) — PCK, OKS, MPJPE
- [ ] Evaluation CLI (`eval/runner.py`)
- [ ] Side-by-side visualizations on 50 images
- [ ] Comparison table vs. HRNet baseline

### Milestone 3 — Refinement & Final Report (due 11.6) 🔲 FUTURE
- [ ] Per-joint PCK breakdown graphs
- [ ] Failure-mode gallery
- [ ] Final report
- [ ] End-to-end demo CLI









Human Pose Estimation Pipeline — Interview Guide
The CV Description
"Designing an end-to-end pipeline that predicts body joint positions from images using semantic part segmentation, built on the COCO and DensePose datasets."

"end-to-end pipeline" The system takes a raw photo as input and outputs joint positions (x,y coordinates) as output. You built every step in between — loading data, segmenting body parts, solving joint locations, and visualizing results.

"predicts body joint positions" The goal is to find where joints are (elbow, knee, shoulder, etc.) in an image — their exact pixel coordinates. This is called pose estimation.

"semantic part segmentation" The approach: instead of detecting joints directly, first label every pixel as a body part (head, upper arm, lower leg, etc.), then find joints at the boundaries between neighboring parts. The elbow is where "upper arm" pixels meet "lower arm" pixels.

"COCO and DensePose datasets"

COCO: industry-standard dataset with 17 manually labeled keypoints per person
DensePose: COCO extension with per-pixel body part labels
Both are used by major research labs (Meta, Google). Using them shows familiarity with real-world CV data.
The CV Description
"Implemented data abstractions, dataset adapters, and visualization tools following software engineering best practices (CI/CD, type checking, 25+ tests)."

"data abstractions" Designed a PoseSample dataclass — a standardized container that holds the image, keypoints, part mask, and bounding box. Every part of the pipeline speaks the same data format, regardless of where the data came from.

Also designed PoseDataset — an abstract interface that any dataset must follow, so the rest of the pipeline doesn't care whether it's reading from COCO, DensePose, or a future dataset.

"dataset adapters" Two concrete implementations of that interface:

COCOPoseDataset — loads images and keypoints from COCO annotation JSON files
DensePoseDataset — extends COCO, additionally loads and decodes per-pixel part masks, remapping external labels into the project's shared vocabulary
"visualization tools" viz.py — functions to draw colored part masks, keypoint dots, and skeleton lines on images. Used to visually verify the pipeline is working correctly at each stage.

"CI/CD" A GitHub Actions workflow that automatically runs tests and checks on every push. If any code breaks, you know immediately. This is standard in professional software teams.

"type checking" Using mypy to statically verify that functions receive and return the correct data types — catching bugs before the code even runs.

"25+ tests" Automated test suite using pytest with synthetic data (no real images needed). Tests cover data structures, label remapping, and visualization functions. Ensures nothing breaks as the codebase grows.

How to talk about it in an interview
If asked "tell me about this project":

"I'm building a pose estimation pipeline — given a photo of a person, it finds where their joints are. The novel approach is using body part segmentation as an intermediate step: first label each pixel as a body part, then find joints at the boundaries between parts. I built the full data pipeline on COCO and DensePose, and I'm currently integrating the segmentation model."

If asked "what did you learn?":

"Mostly how to design clean abstractions. I built a data contract that separates the dataset format from the rest of the pipeline — so swapping datasets or adding new ones doesn't touch any other code. And I learned to work with real CV datasets and industry tooling."

If asked about the software engineering practices:

"I set up CI/CD with GitHub Actions, type checking with mypy, and wrote 25+ automated tests. The goal was to make the codebase maintainable — not just code that runs once, but code you can confidently change."
