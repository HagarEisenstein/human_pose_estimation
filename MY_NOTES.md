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

### `SegFormerSegmentor` (segmentation/segformer.py)
A real segmentation backend wrapping `mattmdjaga/segformer_b2_clothes` from HuggingFace.
Loads lazily on first `predict()` call (~100 MB download). Converts ATR 18-class output
to canonical `Part` labels via `SEGFORMER_CLOTHES_TO_PART`. Requires optional dependency:
`pip install -e ".[segformer]"`.

### `SEGFORMER_CLOTHES_TO_PART` (pose/parts.py)
Remap table from the ATR 18-class clothing-parsing scheme to the canonical 15-part
taxonomy. For example: Hat/Hair/Face/Sunglasses → `Part.HEAD`, Left-arm → `Part.UPPER_ARM_L`.
Some source labels (Bag, Skirt) have no meaningful body-part mapping and go to `Part.BACKGROUND`.

---

## How the pieces connect — Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATASET LAYER                            │
│                                                                 │
│  COCOPoseDataset          DensePoseDataset                      │
│  (coco_adapter.py)        (densepose_adapter.py)                │
│                                                                 │
│  Loads image +            Loads image + keypoints               │
│  keypoints from COCO      + fills part_mask from                │
│  part_mask = None         GT DensePose annotations              │
│                           using DENSEPOSE_TO_PART remap         │
└────────────────┬──────────────────────┬────────────────────────┘
                 │                      │
                 ▼                      ▼
         PoseSample              PoseSample
       (no part_mask)          (with part_mask)
                 │                      │
┌────────────────▼──────────────────────▼────────────────────────┐
│                     SEGMENTATION LAYER                          │
│                                                                 │
│  SegFormerSegmentor       GTOracleSegmentor                     │
│  (segformer.py)           (base.py)                             │
│                                                                 │
│  Runs real model on       Just returns sample.part_mask         │
│  sample.image →           unchanged — only works on             │
│  predicts part_mask       DensePoseDataset samples              │
│  using SEGFORMER_         (GT already there)                    │
│  CLOTHES_TO_PART remap                                          │
└────────────────────────────┬───────────────────────────────────┘
                             │
                    (H, W) uint8 part_mask
                    with canonical Part labels
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                      POSE LAYER  (M2 — TODO)                   │
│                                                                 │
│  joints.py          skeleton.py         refine.py              │
│  Joint solver       Skeleton assembly   Plausibility filter     │
│  reads part_mask,   builds tree from    rejects bad angles,     │
│  finds boundaries   joints              limb lengths            │
│  between adjacent                                               │
│  Parts → joint XY                                               │
└────────────────────────────┬───────────────────────────────────┘
                             │
                    predicted keypoints
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                      EVAL LAYER  (M2 — TODO)                   │
│                                                                 │
│  metrics.py         runner.py           viz.py                  │
│  PCK, OKS, MPJPE    CLI evaluation      Overlays part mask,     │
│  compare predicted  loop over dataset   skeleton, keypoints     │
│  vs. GT keypoints                       on RGB image            │
└─────────────────────────────────────────────────────────────────┘
```

### Two paths to fill `part_mask`

Both paths produce the same format — downstream layers don't care which was used:

| Path | Dataset | Segmentor | When to use |
|---|---|---|---|
| **Ground truth** | `DensePoseDataset` | `GTOracleSegmentor` | Testing M2/M3 with perfect masks |
| **Real model** | `COCOPoseDataset` | `SegFormerSegmentor` | Real end-to-end pipeline |

---

## SegFormer implementation — detailed notes

### What was added and why

The M1 plan required wrapping a pretrained segmentation backbone behind the
`SegmentationModel` interface. `GTOracleSegmentor` satisfied the interface contract
but is not a real model. The following files implement an actual backbone:

| File | What changed |
|---|---|
| `pose/parts.py` | Added `SEGFORMER_CLOTHES_TO_PART` remap table (18 ATR labels → 15 canonical Parts) |
| `segmentation/segformer.py` | New file — `SegFormerSegmentor` class |
| `segmentation/__init__.py` | Exports `SegFormerSegmentor` |
| `pyproject.toml` | Added optional dependency group `[segformer]` for `transformers>=4.36` |
| `tests/test_segformer.py` | New file — 4 tests using a mocked model (no download needed) |

Install the optional dependency to use it:
```
pip install -e ".[segformer]"
```

---

### The remap table: `SEGFORMER_CLOTHES_TO_PART`

The model (`mattmdjaga/segformer_b2_clothes`) is trained on the **ATR
clothes-parsing dataset** which has 18 clothing-oriented classes (Hat, Hair,
Dress, etc.). These need to be translated to the project's 15 anatomical `Part`
labels.

Key design decisions in the table:
- Clothing labels that cover a body region map to the relevant Part (e.g.
  Hat/Hair/Face → `Part.HEAD`, Upper-clothes → `Part.TORSO`).
- Labels with no anatomical meaning (Bag, Skirt, Belt) → `Part.BACKGROUND`.
- The source has no separate upper/lower arm — one "Left-arm" label covers both.
  It maps to `UPPER_ARM_L/R` (the larger region). This means wrists will be harder
  to recover with this model than with DensePose ground truth.
- No left/right leg distinction in source either — "Pants" maps to `UPPER_LEG_L`
  as a compromise.

---

### How `SegFormerSegmentor.predict()` works step by step

1. **`_ensure_loaded()`** — loads the HuggingFace processor and model on first call
   only. Importing the module is cheap; the ~200 MB download happens once and is
   cached by HuggingFace in `~/.cache/huggingface/`.

2. **PIL conversion** — `sample.image` is a numpy RGB uint8 array. It gets wrapped
   in a `PIL.Image` because HuggingFace processors expect PIL.

3. **Preprocessing** — `AutoImageProcessor` handles resize (to 512×512),
   normalization (ImageNet mean/std), and conversion to a `(1, 3, 512, 512)`
   float tensor.

4. **Inference** — model outputs `logits` of shape `(1, 18, h, w)` where `h, w`
   are about 1/4 of the input (128×128). Each pixel has a score for all 18 classes.
   Wrapped in `torch.no_grad()` so no gradients are computed.

5. **Upsample back to original size** — `F.interpolate` with bilinear mode resizes
   the logit tensor to `(sample.height, sample.width)`. Done *before* argmax so
   class boundaries are smoother.

6. **Argmax** — each pixel takes the class with the highest score → `(H, W) int32`
   with values 0–17 (ATR source labels).

7. **Remap** — `remap_mask(source_mask, SEGFORMER_CLOTHES_TO_PART)` converts every
   source label to its canonical `Part` value → `(H, W) uint8` with values 0–14,
   exactly matching the format `GTOracleSegmentor` produces.

---

### Test strategy: injecting a mock model

Tests never download weights. The constructor accepts `model=` and `processor=`
keyword arguments for injection:

```python
seg = SegFormerSegmentor(
    model=_FakeModel(target_class=11),   # source label 11 = "Face"
    processor=_FakeProcessor(),
)
mask = seg.predict(dummy_sample)
assert np.all(mask == int(Part.HEAD))    # SEGFORMER_CLOTHES_TO_PART[11] = HEAD
```

`_FakeModel` produces logits where one channel is always 10.0 and all others
are -10.0, so argmax always picks the target class. `_FakeProcessor` returns
a zero tensor — the model ignores it anyway. This verifies:
- The remap is applied correctly for every source label (18 separate assertions).
- The upsample step works (fake logits are 32×32, image is 480×640).
- The output dtype and shape contract are met.

`pytest.importorskip("torch")` at the top skips the whole file cleanly if torch
is not installed.

---

### How to use it on a real image

```python
from data.coco_adapter import COCOPoseDataset
from segmentation import SegFormerSegmentor

ds  = COCOPoseDataset(root="data/raw", split="val2017")
sample = ds[0]               # part_mask is None — COCO has no segmentation data

seg = SegFormerSegmentor()   # first call downloads ~200 MB to HF cache
mask = seg.predict(sample)   # (H, W) uint8, canonical Part labels

sample.part_mask = mask      # slot it in — now identical format to DensePose GT
```

From here every M2 stage (joint solver, evaluator, visualizer) works unchanged —
they only see a `PoseSample` with a filled `part_mask`, and don't care where it
came from.

---

### The role of `pose/parts.py` throughout

`parts.py` is the shared vocabulary used by every layer:

| What | Used by |
|---|---|
| `Part` enum (15 labels) | Every layer — the common language for part_mask values |
| `DENSEPOSE_TO_PART` | `DensePoseDataset` — remaps GT annotations to canonical labels |
| `SEGFORMER_CLOTHES_TO_PART` | `SegFormerSegmentor` — remaps model output to canonical labels |
| `SCHP_TO_PART` | Ready for a future SCHP/LIP backbone wrapper |
| `remap_mask()` | Used by both adapters to apply remap tables pixel-by-pixel |
| `JOINT_DEFINITIONS` | Will be used by `joints.py` (M2) — defines which Part pairs produce which joint |

---

## M1 — Full Implementation Explanation

### What M1 is

Milestone 1 is the **foundation layer** of the project. It does not yet predict any
joints — its job is to build the data plumbing and vocabulary that every later stage
(joint solver, evaluator) will depend on. Everything in M1 is a contract, a loader,
or a translator.

---

### File map & responsibilities

```
data/
  base.py              ← PoseSample (data container) + PoseDataset (abstract interface)
  coco_adapter.py      ← concrete loader: COCO JSON → PoseSample (no part mask)
  densepose_adapter.py ← extends coco_adapter: decodes RLE part masks → PoseSample
  download.py          ← CLI that fetches raw dataset files from the internet

pose/
  parts.py             ← canonical vocabulary: Part enum, JointDef, remap tables

segmentation/
  base.py              ← SegmentationModel ABC + GTOracleSegmentor (testing stand-in)

eval/
  viz.py               ← draw_part_mask / draw_keypoints / draw_skeleton / show_sample

notebooks/
  visualize_samples.py ← end-to-end script: load 50 samples → save 3-panel figures

tests/
  conftest.py          ← shared dummy_sample fixture (synthetic data, no disk I/O)
  test_base.py         ← tests for PoseSample properties
  test_parts.py        ← tests for Part enum, JointDef, remap_mask
  test_segmentation.py ← tests for GTOracleSegmentor
  test_viz.py          ← tests for all three draw_* functions
```

---

### The single shared data contract: `PoseSample` and `PoseDataset`

**`data/base.py`** is the foundation of everything. It defines two things.

**`PoseSample`** — a dataclass that holds one annotated person. Every file that
touches data speaks this type:

| Field | Type | What it holds |
|---|---|---|
| `image` | `(H, W, 3) uint8` | RGB pixel array |
| `keypoints` | `(17, 3) float32` | x, y, visibility per joint |
| `part_mask` | `(H, W) uint8` or `None` | canonical Part label per pixel |
| `bbox` | `(4,) float32` | `[x, y, w, h]` bounding box |
| `image_id` | `int` | COCO image id |
| `ann_id` | `int` | COCO annotation id |
| `meta` | `dict` | extra fields (filename, num_keypoints) |

Helper properties: `height`, `width`, `num_keypoints_visible`, `torso_diagonal`.

**`PoseDataset`** — an ABC that forces every loader to implement `__len__` and
`__getitem__`. It provides `__iter__` and `take(n)` for free. The downstream joint
solver and evaluator never need to know which dataset they are reading from.

---

### The vocabulary layer: `pose/parts.py`

Defines the **shared language** used by every other module.

**`Part` (IntEnum)** — 15 canonical labels:

```
BACKGROUND=0, HEAD=1, TORSO=2,
UPPER_ARM_L=3, UPPER_ARM_R=4, LOWER_ARM_L=5, LOWER_ARM_R=6,
HAND_L=7, HAND_R=8,
UPPER_LEG_L=9, UPPER_LEG_R=10, LOWER_LEG_L=11, LOWER_LEG_R=12,
FOOT_L=13, FOOT_R=14
```

Every pixel in every `part_mask` carries one of these values.

**`JointDef` + `JOINT_DEFINITIONS`** — encodes the geometric rule for finding each
of the 17 COCO joints. Each entry says: *"joint X lives at the boundary between
part A and part B."*

```python
JointDef("left_elbow",    Part.UPPER_ARM_L, Part.LOWER_ARM_L, coco_idx=7)
JointDef("left_shoulder", Part.TORSO,       Part.UPPER_ARM_L, coco_idx=5)
```

Head joints (nose, eyes, ears) are a special case — both `part_a` and `part_b` are
`Part.HEAD` because they are found from the extremities of the head region, not a
boundary between two different parts.

**`DENSEPOSE_TO_PART` and `SCHP_TO_PART`** — translation dictionaries. DensePose
has its own 14-label scheme; SCHP has a 20-label clothing-aware scheme. Both get
remapped into the canonical 15 `Part` labels via `remap_mask()`, so the joint solver
always receives the same format regardless of where the mask came from.

---

### The data loaders: `coco_adapter.py` and `densepose_adapter.py`

Both inherit from `PoseDataset` in a chain:

```
PoseDataset (ABC)                 ← data/base.py
    └── COCOPoseDataset           ← data/coco_adapter.py
            └── DensePoseDataset  ← data/densepose_adapter.py
```

**`COCOPoseDataset`**:
- At construction: reads `person_keypoints_val2017.json`, builds `_images`
  (id→info dict) and `_anns` (filtered list — only annotations with
  ≥ `min_keypoints` visible joints and no crowd flag).
- At `ds[i]`: loads the image with OpenCV (BGR→RGB), reshapes the flat
  51-value COCO keypoint array into `(17, 3)`, returns a `PoseSample`
  with `part_mask=None`.

**`DensePoseDataset`**:
- At construction: calls `super().__init__()` (all COCO loading), then
  additionally reads `densepose_coco_2014_minival.json` and builds
  `_dp_anns` (ann_id → DensePose annotation dict).
- At `ds[i]`: calls `super().__getitem__(idx)` to get the COCO sample,
  then calls `_build_part_mask()` to fill in `sample.part_mask`.

**`_build_part_mask()` step by step** (the most complex piece of M1):
1. Creates a zero canvas `(img_h, img_w) uint8` — all background.
2. Iterates over the 14 RLE masks in `dp_ann["dp_masks"]` (one per DensePose part).
3. Each RLE mask is encoded at 256×256 inside the bounding box.
4. Decodes via `pycocotools.mask.decode()` → binary `(256, 256)` array.
5. Resizes to actual bbox dimensions with `cv2.INTER_NEAREST` (no label blurring).
6. Looks up the canonical label: `DENSEPOSE_TO_PART[part_idx]`.
7. Pastes the binary patch onto the canvas at `canvas[y:y+h, x:x+w]`.
8. Returns the full-image canvas with canonical `Part` label values.

---

### The segmentation interface: `segmentation/base.py`

**`SegmentationModel`** (ABC) — defines the contract: any backend must implement
`predict(sample) → (H, W) uint8`. A real model (Mask2Former, SCHP, etc.) plugs in
here without touching anything else. `batch_predict()` defaults to a loop over
`predict()`.

**`GTOracleSegmentor`** — the testing stand-in. Its `predict()` returns
`sample.part_mask.copy()`. Used in two ways:
- Tests M2's joint solver against a perfect upper-bound mask (no GPU needed).
- Raises `ValueError` if called on a sample where `part_mask is None` — a
  deliberate safety net.

**`predict()` does NOT call the joint solver.** They are two separate stages.
An external runner calls them in sequence:

```python
mask   = seg_model.predict(sample)    # step 1: produce part mask
joints = joint_solver.solve(mask)     # step 2: find joints from mask (M2)
```

`SegmentationModel` is only needed when `sample.part_mask is None`:

| Dataset | `part_mask` in sample | Need `SegmentationModel`? |
|---|---|---|
| `COCOPoseDataset` | always `None` | Yes — must call `predict()` |
| `DensePoseDataset` (annotation exists) | filled in | No — already there |
| `DensePoseDataset` (no DensePose entry) | `None` | Yes — must call `predict()` |

---

### The visualization layer: `eval/viz.py`

Imports `PART_COLORS`, `PART_NAMES`, `Part`, `JOINT_DEFINITIONS` from `pose/parts.py`
and operates on `PoseSample` fields.

- **`draw_part_mask(image, mask, alpha)`** — for each `Part`, finds pixels where
  `mask == int(part)`, fills them with `PART_COLORS[part]` (BGR), then
  `cv2.addWeighted` blends the color layer with the original image.
- **`draw_keypoints(image, keypoints)`** — iterates over the 17 keypoints, draws a
  yellow dot at each `(x, y)` if visibility > 0.
- **`draw_skeleton(image, keypoints)`** — draws lines for each of the 16 COCO limb
  pairs, colored by left (blue) / right (red) / midline (green), then draws dots on
  top.
- **`show_sample(sample)`** — combines all three into a 3-panel matplotlib figure:
  raw image | part mask overlay | GT skeleton.

---

### The end-to-end script: `notebooks/visualize_samples.py`

The deliverable that proves the full M1 pipeline works. It connects every module:

```
DensePoseDataset(root, split)   [densepose_adapter.py]
        ↓  ds[i] → PoseSample
show_sample(sample)             [eval/viz.py]
        ↓  draw_part_mask + draw_skeleton
outputs/figures/sample_<ann_id>.png
```

Takes CLI args (`--root`, `--split`, `--n`, `--out`), iterates the first N samples,
skips any missing from disk or with no DensePose mask.

---

### The download helper: `data/download.py`

A standalone CLI that bootstraps the project from scratch. Nothing else imports it.

1. Downloads `annotations_trainval2017.zip`, extracts only `person_keypoints_*.json`.
2. Reads the annotation file, collects image filenames for the first N valid
   annotations, downloads them one by one.
3. Optionally downloads `densepose_coco_2014_minival.json` from Facebook's servers.

Run once as `python -m data.download`.

---

### Test suite

| File | What it covers |
|---|---|
| `conftest.py` | `dummy_sample` fixture: 480×640 synthetic image, 17 visible keypoints, 3-region part mask. No disk I/O needed for any test. |
| `test_base.py` | `PoseSample` properties: `height`, `width`, `num_keypoints_visible`, `torso_diagonal`, dtype |
| `test_parts.py` | `Part` enum uniqueness and count; 17 `JointDef` entries with unique COCO indices; `remap_mask` output; both remap tables map to valid `Part` values |
| `test_segmentation.py` | `SegmentationModel` is abstract; `GTOracleSegmentor` correct shape/dtype/values; returns a copy not the original; raises `ValueError` on `part_mask=None` |
| `test_viz.py` | All three `draw_*` functions return correct shape/dtype; `alpha=0` leaves image unchanged; invisible keypoints skipped |

---

### How files import each other

```
data/base.py              (no project imports — root)
pose/parts.py             (no project imports — root)
data/coco_adapter.py      → data/base.py
data/densepose_adapter.py → data/coco_adapter.py + pose/parts.py
segmentation/base.py      (no project imports)
eval/viz.py               → pose/parts.py
notebooks/visualize_samples.py → data/densepose_adapter.py + eval/viz.py
tests/conftest.py         → data/base.py + pose/parts.py
```

`data/base.py` and `pose/parts.py` are the two roots that everything else builds on.
No import cycles anywhere.

---

### Full data flow (end to end)

```
python -m data.download
        ↓
data/raw/annotations/person_keypoints_val2017.json
data/raw/annotations/densepose_coco_2014_minival.json
data/raw/val2017/*.jpg
        ↓
DensePoseDataset(root, split)       [densepose_adapter.py]
  inherits COCOPoseDataset          [coco_adapter.py]
  which inherits PoseDataset        [base.py]
        ↓  ds[i]
PoseSample(
  image       (H,W,3) uint8    ← loaded by cv2, BGR→RGB
  keypoints   (17,3) float32   ← COCO flat array, reshaped
  part_mask   (H,W) uint8      ← RLE decoded → resized → DENSEPOSE_TO_PART remapped
  bbox        (4,) float32
  image_id / ann_id / meta
)
        ↓
GTOracleSegmentor.predict()         [segmentation/base.py]
  returns part_mask as-is (copy)
        ↓
(H, W) uint8 canonical part mask  ← ready for M2 joint solver
        ↓  (also passed to viz)
show_sample(sample)                 [eval/viz.py]
  draw_part_mask  → uses PART_COLORS  from pose/parts.py
  draw_skeleton   → uses COCO_SKELETON connectivity
        ↓
outputs/figures/sample_<ann_id>.png
```

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

### Milestone 2 — Joint Inference, Skeleton Assembly & Evaluation (due 28.5) ✅ DONE
- [x] Joint solver (`pose/joints.py`) — boundary + extremity + endpoint strategies
- [x] Skeleton assembly (`pose/skeleton.py`) — confidence filter, L/R correction, limb-length filter, angle filter
- [x] Evaluation metrics (`eval/metrics.py`) — PCK@0.2, OKS, MPJPE + Accumulator
- [x] Evaluation CLI (`eval/runner.py`) — oracle & segformer modes, JSON output
- [x] Side-by-side pipeline figures (`notebooks/visualize_pipeline.py`) — 50 images
- [x] Per-joint PCK bar chart (`notebooks/plot_per_joint_pck.py`)
- [x] Comparison table vs. HRNet (`notebooks/comparison_table.py`)
- [x] Overall PCK@0.2 = 0.197, OKS = 0.041 (oracle segmentor, 37 samples)
- [x] 137 passing tests

### Milestone 3 — Refinement & Final Report (due 11.6) 🔲 NEXT
- [ ] Per-joint PCK breakdown graphs (generate with `notebooks/plot_per_joint_pck.py`)
- [ ] Failure-mode gallery (~10 successes + ~10 failures with captions)
- [ ] Final report: Motivation, Methodology, Results, Conclusions
- [ ] End-to-end demo CLI (`pose-from-seg --image X.jpg`)
- [ ] Comparison vs. HRNet-W32 table in report









## Human Pose Estimation Pipeline — Interview Guide
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

pytest — automated testing

You write small functions that check your code works correctly. Then you run pytest and it runs all of them automatically and tells you what passed or failed.

Example from your project: a test that creates a fake PoseSample and checks that num_keypoints_visible returns the right number.

Without it: you manually run the code and eyeball if it looks right.
With it: one command tells you if anything broke.

mypy — type checking

Python normally doesn't care about types at all — you can pass a string where a number is expected and it won't complain until it crashes at runtime. mypy reads your code before running it and warns you: "this function expects an integer but you're passing a string."

Your project uses type hints like def predict(self, sample) -> np.ndarray: — mypy verifies those are respected everywhere.

CI/CD — automatic checks on every push

CI = Continuous Integration. Every time you push code to GitHub, it automatically runs mypy and pytest in the cloud. If anything fails, GitHub shows a red ✗ on your commit.

You set this up in .github/workflows/ci.yml.

Without it: you have to remember to run tests manually.
With it: broken code can never slip through unnoticed.

One sentence summary for an interview:

"I set up automated testing with pytest, static type checking with mypy, and wired both into GitHub Actions so every push is validated automatically

---

## Flow between Joint Solver and Skeleton Assembly

### The big picture

```
part_mask  ──→  joints.solve()  ──→  skeleton.assemble()  ──→  final keypoints
  (image)       (raw locations)       (cleaned up)
```

---

### Stage 1 — Input: `part_mask`

The starting point is a segmentation mask — a 2D image where every pixel is labeled
with a body part (HEAD, TORSO, LEFT_ARM, etc.). This is assumed to come from an
earlier stage (e.g. a segmentation model).

```
┌─────────────────┐
│  H H H          │   H = HEAD
│  H H H          │
│  T T T T        │   T = TORSO
│  T T T T        │
│ LA T T RA       │   LA = LEFT_ARM, RA = RIGHT_ARM
└─────────────────┘
```

---

### Stage 2 — `joints.solve(part_mask)` → raw keypoints

For each of the 17 COCO joints, it asks: *"where does this joint live in the mask?"*
using two strategies:

- **Boundary joints** (e.g. shoulder = where TORSO meets ARM):
  - Dilate both part masks outward
  - Find the overlap band between them
  - Take the center-of-mass of that band → that's the joint location
  - Confidence = how big the band is relative to the smaller part

- **Endpoint joints** (nose, eyes, ears):
  - Look at spatial extremities of the HEAD blob
  - e.g. left ear = rightmost 20% of HEAD pixels

Output: a `(17, 3)` array — 17 joints, each with `(x, y, confidence)`. At this point
the locations can be noisy or wrong.

---

### Stage 3 — `skeleton.assemble(keypoints, part_mask, torso_diagonal)` → clean keypoints

Takes the raw output and runs 4 correction steps in order:

```
raw keypoints
     │
     ▼
Step 1: _zero_low_confidence()   — discard joints the solver wasn't sure about
     │
     ▼
Step 2: _correct_lr()            — fix left/right label swaps
     │
     ▼
Step 3: _apply_limb_filter()     — discard joints where limb lengths are anatomically impossible
     │
     ▼
Step 4: _apply_angle_filter()    — discard joints that create impossible bend angles (e.g. elbow bending backwards)
     │
     ▼
final keypoints (17, 3)
```

---

### Why the separation?

`joints.solve` is purely geometric — it just finds *where* parts meet in the mask,
with no knowledge of anatomy. It can produce confident but wrong answers (e.g. swapped
labels, impossibly long limbs).

`skeleton.assemble` is purely a cleanup/validation layer — it doesn't look at the image
at all, only at the numbers coming out of the solver. This separation keeps each stage
simple and independently testable.

---

## M2 — Key Changes Made and Why

### The Problem We Discovered

After running the first evaluation, the results were:
- **PCK@0.2 = 0.164**
- Shoulders: 0.000, Elbows: 0.000, Wrists: 0.000, Ankles: 0.000

Six joints were always exactly zero. This looked like a bug, but it was actually
two separate problems — one structural, one algorithmic.

---

### Problem 1 — Wrong Geometric Strategy for Most Joints

**What the original solver did:**

For every boundary joint, it used the same strategy:
1. Dilate both part masks outward
2. Find the overlap band between them
3. Take the **center-of-mass of the band** as the joint location

This makes perfect sense for the **knee** (where UPPER_LEG physically meets
LOWER_LEG in the mask). But it's wrong for joints where the anatomical position
is at the **END** of a body segment, not at the clothing boundary.

**The insight:**

The DensePose/SegFormer mask labels clothing surfaces, not anatomy. So:

| Joint | What boundary gives you | Where the joint actually is |
|-------|------------------------|----------------------------|
| Shoulder | Shirt armhole (TORSO ∩ UPPER_ARM) | Top of the upper arm |
| Hip | Waistband (TORSO ∩ UPPER_LEG) | Top of the thigh |
| Elbow | Sleeve crease (UPPER_ARM ∩ LOWER_ARM) | Bottom of the upper arm |
| Wrist | Cuff (LOWER_ARM ∩ HAND) | Bottom of the forearm |
| Ankle | Top of shoe (LOWER_LEG ∩ FOOT) | Top of the foot |

The boundary center for these joints falls at a seam in the clothing, which
can be several centimeters away from the actual anatomical joint.

---

### The Fix — Extremity Joint Strategy

We added a third solver strategy: **"extremity of a single part"**.

Instead of looking at where two parts meet, we look at where one part ends:

```python
_EXTREMITY_JOINTS = {
    "left_shoulder":  (Part.UPPER_ARM_L, "top"),     # topmost pixels of upper arm
    "right_shoulder": (Part.UPPER_ARM_R, "top"),
    "left_hip":       (Part.UPPER_LEG_L, "top"),     # topmost pixels of upper leg
    "right_hip":      (Part.UPPER_LEG_R, "top"),
    "left_elbow":     (Part.UPPER_ARM_L, "bottom"),  # bottommost pixels of upper arm
    "right_elbow":    (Part.UPPER_ARM_R, "bottom"),
    "left_wrist":     (Part.LOWER_ARM_L, "bottom"),  # bottommost pixels of forearm
    "right_wrist":    (Part.LOWER_ARM_R, "bottom"),
    "left_ankle":     (Part.FOOT_L,      "top"),     # topmost pixels of foot
    "right_ankle":    (Part.FOOT_R,      "top"),
}
```

The function `_extremity_joint()` takes the top or bottom **15% of pixels** of
the relevant part and returns their centroid. This avoids the clothing-seam offset.

**New routing logic in `_solve_one()`:**
```
1. HEAD endpoint joints (nose/eyes/ears)  → endpoint_joint()
2. Joints in _EXTREMITY_JOINTS           → extremity_joint()   ← NEW
3. Everything else (knees)                → boundary_joint()
```

---

### Problem 2 — Ankles Still Zero

Even after the fix, ankles remained 0.000. The reason is different:

The DensePose `densepose_coco_2014_minival.json` annotations **rarely include FOOT
labels**. DensePose was mainly annotated on the torso and limbs — feet were either
occluded, cut off by the image frame, or simply not annotated.

So `_extremity_joint(mask, Part.FOOT_L, "top")` correctly returns confidence=0
when there are no FOOT pixels in the mask. This is not a bug — it is a data gap.

**If you were asked in an interview:** *"Why are ankles always zero?"*
> *"The DensePose annotations we're using don't reliably label the foot surface —
> most samples in our subset have no FOOT pixels at all. Our solver correctly returns
> zero confidence when it can't find the relevant part. This is a data limitation,
> not an algorithmic one. Using a full DensePose training split, or a model
> specifically trained to segment feet, would resolve it."*

---

### Results Before and After

| Joint | Before | After | Change |
|-------|--------|-------|--------|
| nose | 0.464 | 0.464 | — same |
| left_shoulder | 0.000 | **0.088** | ✅ unlocked |
| right_shoulder | 0.000 | **0.029** | ✅ unlocked |
| left_elbow | 0.000 | **0.091** | ✅ unlocked |
| right_elbow | 0.000 | **0.065** | ✅ unlocked |
| left_wrist | 0.000 | **0.074** | ✅ unlocked |
| right_wrist | 0.000 | **0.107** | ✅ unlocked |
| left_hip | 0.121 | **0.303** | ✅ big gain |
| right_hip | 0.265 | **0.412** | ✅ big gain |
| left_knee | 0.240 | 0.240 | — same (boundary still correct) |
| **Overall PCK** | **0.164** | **0.197** | **✅ +20%** |
| **Overall OKS** | **0.029** | **0.041** | **✅ +41%** |
| Overall MPJPE | 0.236 | 0.312 | ⚠️ see below |

---

### Why MPJPE Got Worse Despite PCK Improving

PCK and MPJPE measure different things:
- **PCK** = binary: was the prediction close enough? (within 20% of torso size)
- **MPJPE** = continuous: what was the exact pixel error?

Before the fix, joints like elbows and wrists had **confidence = 0**, so they were
**excluded from MPJPE entirely** (NaN in the per-joint array, excluded from mean).

After the fix, those joints are **detected but imprecisely** — they are in roughly
the right region (PCK says "correct") but still not pinpoint accurate (MPJPE shows
a large distance). Adding imprecise joints to the mean raises MPJPE.

**If you were asked in an interview:** *"Your MPJPE got worse — is that a problem?"*
> *"Not necessarily. Before the change, MPJPE was artificially low because six joints
> were excluded from the calculation entirely — they contributed NaN, not a large error.
> After the change, those joints are detected and contribute a real error. The fact that
> PCK improved means the detections are genuinely useful — they land within the acceptance
> threshold. MPJPE getting worse just tells me the new detections are imprecise, not that
> they are wrong in direction."*

---

### The Nose Fix That Failed

We also tried changing the nose detection: instead of the **top 20% of HEAD pixels**,
we used the **middle 45–72% of HEAD height** (reasoning: the nose tip is not at the
top of the skull — it is lower down the face).

This made the nose **worse** (0.464 → 0.286). Why?

The DensePose HEAD mask is not the full skull — it covers only the **visible face
surface**. So the "top" of the HEAD mask is actually the **forehead**, which is
very close to the nose tip in COCO's keypoint scheme. Our "correction" moved the
prediction too far down.

**Lesson:** intuition about what a mask covers must be validated against real data.
We reverted this change.

---

### Interview Summary — What Changed and Why

> *"After the first evaluation, I noticed six joints — shoulders, elbows, wrists,
> and ankles — had PCK of exactly 0.000. I diagnosed two root causes.*
>
> *For shoulders, hips, elbows, and wrists: the solver was finding the center of the
> clothing boundary between parts, which corresponds to a seam in the shirt or
> trousers — not the anatomical joint. I redesigned the solver strategy for these
> joints to instead use the extreme endpoint of one body segment (e.g. the topmost
> pixel of the upper arm for the shoulder joint). This improved overall PCK by 20%
> and unlocked six previously undetectable joints.*
>
> *For ankles: the DensePose annotations in our subset rarely include foot labels,
> so there simply aren't any pixels to detect from. This is a data gap, not a bug.*
>
> *I also tried to fix the nose detection by moving it lower in the face, but
> empirical testing showed the original was actually better — a reminder that
> geometric intuition about segmentation masks must be validated against real data."*
