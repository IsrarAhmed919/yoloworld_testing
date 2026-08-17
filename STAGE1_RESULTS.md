# Stage 1: YOLO-World on TensorRT, and why your benchmark decides the answer

RTX 4050 Laptop (6 GB), Ubuntu 24.04, driver 570.172.08, CUDA 12.4, TensorRT 10.16.
`yolov8s-worldv2`, FP16, 640px, five-class fixed vocabulary, 150 measured frames
after 20 warmup.

## The numbers

| | webcam-capped (30 FPS source) | uncapped |
|---|---|---|
| PyTorch FP16 | 27.94 ms, p95 34.28, **30.1 FPS** | 31.61 ms, p95 48.95, **31.3 FPS** |
| TensorRT FP16 | 11.70 ms, p95 14.20, **30.0 FPS** | 10.80 ms, p95 13.82, **89.5 FPS** |

## 1. On a webcam, TensorRT looks worthless

Both backends deliver 30 FPS end to end. Identical. A 2.9x reduction in inference
time is completely invisible, because `cap.read()` blocks until the camera produces
the next frame and the camera runs at 30 Hz no matter how fast the model is.

This is the trap. Benchmark your optimisation against a webcam and you will measure
your webcam.

## 2. Uncapped, the speedup is 2.93x

Remove the rate limit and PyTorch's 31.61 ms becomes TensorRT's 10.80 ms. End to end,
31.3 FPS becomes 89.5.

That headroom is the thing you are actually buying. It is not a faster single camera,
it is the difference between one stream and three on the same GPU.

## 3. The tail matters more than the mean

| | mean | p95 | tail inflation |
|---|---|---|---|
| PyTorch, uncapped | 31.61 ms | 48.95 ms | +55% |
| TensorRT, uncapped | 10.80 ms | 13.82 ms | +28% |

Under saturation PyTorch's worst frames degrade almost twice as badly. The direction
of change differs too: saturating PyTorch made it **slower** (27.94 to 31.61 ms),
while saturating TensorRT made it **faster** (11.70 to 10.80 ms), since back to back
execution keeps clocks warm and avoids per call setup.

For a deployment judged on dropped frames rather than average throughput, that
predictability is worth more than the 2.9x.

## A box-count discrepancy that turned out to be my own bug

The first run showed 1.00 detections per frame on PyTorch and 1.92 on TensorRT, which
looked like FP16 changing the model's behaviour and would have invalidated a pure
speed comparison.

It was neither. The uncapped mode grabbed a fresh seed frame on every invocation, so
the two backends were never scored on the same scene. Given one fixed frame they agree
exactly:

| conf | PyTorch | TensorRT |
|---|---|---|
| 0.10 | 1 box, person 0.845 | 1 box, person 0.927 |
| 0.25 | 1 box, person 0.845 | 1 box, person 0.927 |
| 0.40 | 1 box, person 0.845 | 1 box, person 0.927 |

Confidence does shift, which is expected from FP16, but not enough to move any
detection across a threshold. The benchmark now takes a fixed image path so both
backends are guaranteed identical input.

Worth stating plainly because it is the general lesson of this whole exercise: two of
the three surprises here came from the measurement setup, not the system being
measured.

## Setup note that cost an hour

`pip install tensorrt` currently resolves to the 11.x CUDA 13 build. On driver 570
(the Ubuntu 24.04 default) that fails at engine build time as:

```
cudaError 35: CUDA driver version is insufficient for CUDA runtime version
TypeError: pybind11::init(): factory function returned nullptr
```

TensorRT 11 needs r580 or newer. On a 570 driver, pin the CUDA 12 line:

```bash
pip install "tensorrt-cu12>=10.0,<11"
```

## Reproducing

```bash
python bench_live.py --export
python bench_live.py --backend torch  --source 0 --no-show --synthetic
python bench_live.py --backend engine --source 0 --no-show --synthetic
python bench_live.py --backend engine --source 0 --no-show
```

## Next

Stage 1 bakes the vocabulary in via `set_classes()` before export, which is what makes
the engine fast and what makes the classes fixed. Stage 2 is changing what you detect
without rebuilding the engine, which means exporting the class embeddings as a runtime
input tensor rather than a baked constant, and measuring what that flexibility costs.

## What open vocabulary actually covers

Same weights, no retraining, only the prompt changes. Tested on a street scene:

| prompt | detections | note |
|---|---|---|
| `bus` | 1 @ 0.91 | countable object, works |
| `person` | 4 @ 0.91 | countable object, works |
| `man in white shirt` | 3 @ 0.63 | compositional phrase, works but see below |
| `red bus` | 1 @ 0.30 | attribute costs two thirds of the confidence |
| `wheel`, `window`, `door` | 0 | object **parts** are not found |
| `road`, `building`, `tree` | 0 | amorphous **stuff** is not found |
| `backpack`, `handbag`, `luggage` | 0 | not present in the scene |

Two things worth knowing before promising this to anyone:

**It detects things, not stuff.** Countable objects work. Regions like road, sky and
building return nothing at any threshold. That is a detector trained on object
detection behaving exactly as it should, but it surprises people who expect a
vision-language model to segment anything they can name.

**Attribute phrases bias ranking, they do not filter.** `man in white shirt` scored the
correct person at 0.63, but still fired on two other people at 0.33 and 0.17. Useful
for ordering candidates, not reliable as a filter without a threshold you tune per
prompt. This is the same conclusion the earlier benchmark in this repo reached: the
usable confidence threshold is a property of the prompt, not of the model.

## Benchmark variance on a laptop GPU

Three runs of the same comparison gave speedups of 2.93x, 2.39x and 2.06x. PyTorch's
mean on identical input moved from 40.0 ms to 29.0 ms between consecutive runs, a 27%
swing driven by clock and thermal state rather than anything in the code.

Quote a range, or a median across runs. A single laptop figure is not reproducible,
including by the person who published it.
