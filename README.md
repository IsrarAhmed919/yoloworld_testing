# Is there a single confidence threshold for open-vocabulary detection?

Open-vocabulary detectors like YOLO-World let you detect a class by naming it, with no
training. Every deployment guide then tells you to pick a confidence threshold and ship.

This repo is an attempt to check whether that advice actually holds — because early testing
suggests the usable threshold is **different for every prompt**, which would mean a single
global `conf` silently decides which of your classes work and which don't.

**Status:** early. The COCO evaluation harness works and has produced a zero-shot baseline on
100 images. The threshold sweep itself is being redone with the correct metric — see
[Correction](#correction-map-cannot-answer-this-question) below.

---

## Why this matters in production

In a deployed video system, the confidence threshold is the dial that trades false positives
against misses. Set it too low and operators get flooded with false alarms, stop trusting the
system, and eventually turn it off. Set it too high and you miss the events the system exists
to catch.

With a closed-vocabulary detector you tune that dial once against a labelled validation set.
With an open-vocabulary detector the class list is decided at runtime by whoever types the
prompt — so if the correct threshold moves with the prompt, there is no validation set to tune
against and no safe default to ship.

That is the question this repo is trying to answer with numbers.

---

## Setup

| | |
|---|---|
| Model | `yolov8s-world.pt` (YOLO-World S) via Ultralytics |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB, Ada Lovelace |
| Eval data | COCO val2017 (5,000 images, 80 classes) |
| Framework | Ultralytics, PyTorch, pycocotools, torchmetrics |

The 6 GB card is deliberate, not a limitation. Jetson Orin Nano ships with 8 GB shared memory,
so results from constrained hardware are more representative of real edge deployments than
results from a datacentre GPU.

---

## Preliminary observation — webcam

**Method.** `yoloworld.py` runs YOLO-World on a live webcam feed with four prompts set at
once: `person`, `ring`, `headphones`, `hand`. All four objects were physically present in
frame. Confidence threshold was varied by hand between runs.

**What happened.**

| Confidence | person | ring | headphones | hand | Box count |
|---|---|---|---|---|---|
| below 0.25 | detected | detected | detected | detected | far too many, heavy false positives |
| above 0.25 | detected | detected | **missed** | **missed** | few, and accurate |

Two things follow, if this survives proper measurement:

1. **No single threshold worked for all four prompts.** Any value that kept `headphones` and
   `hand` visible also produced an unusable number of false positives; any value that cleaned
   up the output silently dropped two of the four classes entirely.
2. **The failure was calibration, not capability.** An earlier guess was that `hand` might be
   undetectable because it is a body *part* rather than a whole object, and grounding datasets
   rarely annotate parts. That was wrong — the model detects it fine at low confidence. It
   simply is not confident about it.

**Why this is not a result yet.** One scene, one lighting condition, one camera, no ground
truth, and "too many boxes" is an impression rather than a measurement. It is enough to
justify the experiment below. It is not enough to claim anything.

---

## Baseline — zero-shot `person` on COCO val2017

First quantitative run. Harness: `pycocotools` for ground truth, `torchmetrics` for mAP,
`iscrowd` regions excluded.

**Scale:** 100 images · 283 ground-truth people · 3,661 raw detections at `conf=0.001`

That ratio is worth stating plainly — at near-zero confidence the model emits roughly **13
boxes for every real person**. Whatever else is true, an open-vocabulary detector run without a
threshold is not a detector, it is a proposal generator.

| conf | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| 0.01 | 0.713 | 0.506 |
| 0.05 | 0.692 | 0.493 |
| 0.10 | 0.651 | 0.469 |
| 0.25 | 0.552 | 0.420 |
| 0.40 | 0.483 | 0.372 |
| 0.60 | 0.373 | 0.308 |

**What is valid here:** YOLO-World S, prompted with the single word `person` and never trained
on this task, reaches **mAP@0.5 ≈ 0.71** on COCO person detection. For zero-shot that is
respectable, and it is the number the "prompt versus train" comparison will be measured
against once a trained baseline is run on the identical 100 images.

### Correction: mAP cannot answer this question

**What is *not* valid is reading an optimal threshold off that table.**

The column decreases monotonically, which looks like "lower is better" but is an artifact of
the metric. Average precision already integrates the full precision-recall curve and ranks
detections by score internally. Filtering by confidence *before* computing AP simply truncates
that curve — recall is lost and nothing is gained. The AP-optimal threshold is therefore always
≈ 0, for every class, in every experiment. It measures the ranking, not the operating point.

The threshold question needs **precision, recall and F1 at each threshold**, with the optimum
at the F1 peak. That sweep is being rerun; the tables below stay empty until it is done.

This mistake is left in the README on purpose. The whole claim of this repo is that people
choose thresholds badly, so quietly deleting a run that chose a metric badly would be a poor
way to make the argument.

---

## Planned experiment — COCO val2017

### Protocol

1. Select 12–15 COCO classes spanning frequency: common (`person`, `car`, `dog`, `chair`),
   mid (`backpack`, `umbrella`, `laptop`, `sink`), rare (`toothbrush`, `hair drier`,
   `scissors`, `toaster`).
2. For each class, write several prompt variants — e.g. `person` / `a person` / `human` /
   `pedestrian`, or `hair drier` / `hair dryer` / `blow dryer`.
3. Run inference **once per prompt at `conf=0.001`** and store every detection with its score.
   Thresholds are then swept offline in post-processing. This turns *(prompts × thresholds)*
   model runs into *prompts* model runs.
4. Compute precision, recall and AP against COCO ground truth at each threshold.

Ground-truth handling: `iscrowd` regions are excluded, since crowd annotations are unlabelled
groups and counting them as misses would distort recall.

### Metrics

- **Precision, recall and F1 at each swept threshold** — greedy IoU≥0.5 matching, each ground
  truth box claimed at most once. This is what selects the operating point.
- **Optimal threshold** per (class, prompt) = the value maximising F1
- `mAP@0.5` and `mAP@0.5:0.95` reported alongside, as a threshold-independent measure of
  overall model quality — *not* used for threshold selection, for the reason given above

### Results — to be filled

**Table 1 — does the optimal threshold vary by class?**

| class | frequency | best conf | AP@0.5 at best | AP@0.5 at global 0.25 | loss |
|---|---|---|---|---|---|
| person | common | | | | |
| chair | common | | | | |
| backpack | mid | | | | |
| toothbrush | rare | | | | |
| hair drier | rare | | | | |

**Table 2 — does prompt wording change accuracy and calibration?**

| class | prompt variant | AP@0.5 | best conf |
|---|---|---|---|
| person | `person` | | |
| person | `a person` | | |
| person | `human` | | |
| person | `pedestrian` | | |
| hair drier | `hair drier` | | |
| hair drier | `hair dryer` | | |

Identical ground truth, identical images — only the wording differs. `hair drier` is COCO's
own spelling, which makes it a free natural experiment against the more common `hair dryer`.

---

## Repo contents

| File | Purpose |
|---|---|
| `yolo_on_video.py` | Baseline: closed-vocabulary YOLOv8n on webcam, for comparison |
| `yoloworld.py` | Open-vocabulary YOLO-World on webcam with runtime-set classes |
| `yolo_world_vs_coco.ipynb` | COCO evaluation harness (in progress) |
| `coco/` | val2017 images and annotations (gitignored) |

Model weights are gitignored. They download automatically on first run.

---

## Reproducing

```bash
pip install ultralytics pycocotools torchmetrics opencv-python

# webcam observation
python yoloworld.py

# COCO evaluation data
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip -d coco/ && unzip annotations_trainval2017.zip -d coco/
```

---

## Roadmap

- [x] Webcam observation across confidence thresholds
- [x] COCO val2017 downloaded
- [x] Evaluation harness working — 100-image zero-shot `person` baseline, mAP@0.5 0.713
- [ ] Redo the sweep with precision / recall / F1 instead of mAP
- [ ] Trained-detector baseline on the same 100 images, for the prompt-vs-train comparison
- [ ] Full sweep across classes and prompt variants
- [ ] Threshold stability: do per-class optima transfer between datasets and scenes?
- [ ] Latency and throughput on constrained hardware — FP16, INT8, FP8 (Ada) via TensorRT
- [ ] Multi-stream: where does the throughput ceiling sit, and is it inference or NVDEC decode?

The stability question is the one that matters most. If per-class thresholds have to be
recalibrated for every new scene, then open-vocabulary detection is considerably harder to
deploy than the benchmark numbers in the papers suggest.

---

## License

MIT
