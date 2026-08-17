"""Stage 1: YOLO-World on a live stream, PyTorch vs TensorRT.

Measures the two numbers that are not the same thing:

  inference time  how long the model itself takes on one frame
  end-to-end FPS  what the stream actually delivers, including capture,
                  letterbox, NMS and draw

The gap between them is the whole point. A model that infers in 6 ms does not
give you 166 FPS, and quoting the first number as if it were the second is the
most common lie in edge-CV demos.

Usage:
    python bench_live.py --backend torch  --source 0
    python bench_live.py --backend engine --source 0
    python bench_live.py --export                 # build the .engine first
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

# Stage 1 is a FIXED vocabulary. set_classes() bakes the CLIP text embeddings
# into the head, which is what makes the TensorRT export fast. It is also why
# these classes cannot change at runtime. Stage 2 is about undoing exactly this.
CLASSES = ["person", "backpack", "cell phone", "bottle", "chair"]

WEIGHTS = "yolov8s-worldv2.pt"
ENGINE = "yolov8s-worldv2.engine"


def export_engine(imgsz: int) -> None:
    """PyTorch -> ONNX -> TensorRT FP16, with the vocabulary baked in."""
    model = YOLOWorld(WEIGHTS)
    model.set_classes(CLASSES)
    print(f"exporting {WEIGHTS} -> TensorRT FP16 at {imgsz}px, classes={CLASSES}")
    print("this takes a few minutes, TensorRT is profiling kernels for your 4050")
    model.export(format="engine", half=True, imgsz=imgsz, device=0)
    print(f"wrote {ENGINE}")


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source!r}")
    return cap


class SyntheticCapture:
    """Replays one grabbed frame with no rate limit.

    A webcam blocks in read() until the next frame is ready, so end-to-end FPS
    saturates at the capture rate (usually 30) and every backend looks
    identical once the model is faster than the camera. This isolates what the
    pipeline can actually sustain when nothing is throttling it.
    """

    def __init__(self, source: str):
        # a path means a FIXED frame, which is the only way two backends get
        # compared on identical input. grabbing a fresh frame per run means
        # each backend sees a different scene and box counts diverge for
        # reasons that have nothing to do with the model.
        if not source.isdigit():
            frame = cv2.imread(source)
            if frame is None:
                raise SystemExit(f"could not read image {source!r}")
            self.frame = frame
            return
        cap = open_source(source)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not grab a seed frame")
        self.frame = frame

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


def run(backend, source, imgsz, n_frames, warmup, show, synthetic=False):
    if backend == "torch":
        model = YOLOWorld(WEIGHTS)
        model.set_classes(CLASSES)
        model.to("cuda")
        predict_kw = dict(device=0, half=True)
    else:
        # YOLOWorld() expects a torch checkpoint and cannot load a serialised
        # engine. The generic loader handles exported formats and picks the
        # class names out of the metadata TensorRT export embeds in the file,
        # which is where set_classes() ended up after reparameterisation.
        model = YOLO(ENGINE, task="detect")
        predict_kw = dict()

    cap = SyntheticCapture(source) if synthetic else open_source(source)
    infer_ms: list[float] = []
    loop_ms: list[float] = []
    n_boxes = 0

    mode = "synthetic (uncapped)" if synthetic else f"live source {source}"
    print(f"\nbackend={backend}  {mode}  imgsz={imgsz}  measure={n_frames}")
    print("q to quit early\n")

    i = 0
    while i < warmup + n_frames:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("stream ended")
            break

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=imgsz, conf=0.10, verbose=False, **predict_kw)
        # the model call is async on GPU; ultralytics syncs before returning results,
        # so this timing is honest without an explicit torch.cuda.synchronize()
        t1 = time.perf_counter()

        r = results[0]
        if show:
            cv2.imshow(f"yolo-world [{backend}]", r.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        loop_end = time.perf_counter()

        if i >= warmup:                     # discard warmup, TRT is slow on frame 1
            infer_ms.append((t1 - t0) * 1000)
            loop_ms.append((loop_end - loop_start) * 1000)
            n_boxes += len(r.boxes)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    if not infer_ms:
        raise SystemExit("no frames measured")

    inf = np.array(infer_ms)
    loop = np.array(loop_ms)
    return {
        "backend": backend,
        "frames": len(inf),
        "infer_mean_ms": inf.mean(),
        "infer_p50_ms": np.percentile(inf, 50),
        "infer_p95_ms": np.percentile(inf, 95),
        "model_only_fps": 1000 / inf.mean(),
        "end_to_end_fps": 1000 / loop.mean(),
        "boxes_per_frame": n_boxes / len(inf),
    }


def report(row: dict) -> None:
    print(f"\n{'='*58}\n  {row['backend'].upper()}  ({row['frames']} frames)\n{'='*58}")
    print(f"  inference   mean {row['infer_mean_ms']:6.2f} ms   "
          f"p50 {row['infer_p50_ms']:6.2f}   p95 {row['infer_p95_ms']:6.2f}")
    print(f"  model-only FPS   {row['model_only_fps']:6.1f}   <- the number people quote")
    print(f"  end-to-end FPS   {row['end_to_end_fps']:6.1f}   <- the number you actually get")
    print(f"  boxes/frame      {row['boxes_per_frame']:6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "engine"], default="torch")
    ap.add_argument("--source", default="0", help="0 for webcam, or an rtsp:// url")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--export", action="store_true", help="build the TensorRT engine and exit")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="replay one frame uncapped, so the camera rate stops hiding the difference")
    a = ap.parse_args()

    if a.export:
        export_engine(a.imgsz)
    else:
        report(run(a.backend, a.source, a.imgsz, a.frames, a.warmup, not a.no_show, a.synthetic))
