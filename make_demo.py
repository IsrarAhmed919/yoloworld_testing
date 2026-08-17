"""Produce portfolio assets from the stage 1 benchmark.

Records a short clip once, then replays the SAME frames through both backends
so the comparison is honest, and writes:

    demo_side_by_side.mp4   PyTorch left, TensorRT right, live per-frame timing
    stills/still_*.png      annotated frames for a portfolio or a proposal

Recording once and processing twice matters. Running both models live would
halve the throughput of each and the FPS overlay would be a number neither
backend actually achieves.

Usage:
    python make_demo.py                 # 15 seconds from the webcam
    python make_demo.py --seconds 20
"""

from __future__ import annotations

import argparse
import pathlib
import time

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

CLASSES = ["person", "backpack", "cell phone", "bottle", "chair"]
WEIGHTS = "yolov8s-worldv2.pt"
ENGINE = "yolov8s-worldv2.engine"

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (120, 255, 120)
WHITE = (255, 255, 255)
DARK = (28, 28, 28)


def record(seconds: int, source: str, max_frames: int) -> list[np.ndarray]:
    """Grab frames from a file or a camera.

    A file is preferred for anything public: the footage is reproducible, the
    licence is known, and a fixed surveillance camera is a far better match for
    what this model is actually for than a laptop webcam.
    """
    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source!r}")

    if is_camera:
        print(f"recording {seconds}s from camera. hold up a phone or a bottle.")
        for n in (3, 2, 1):
            print(f"  {n}...", flush=True)
            time.sleep(1)
        print("  recording")
        frames, end = [], time.time() + seconds
        while time.time() < end:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    else:
        print(f"reading up to {max_frames} frames from {source}")
        frames = []
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)

    cap.release()
    print(f"  got {len(frames)} frames")
    return frames


def process(frames, backend: str):
    """Run every frame through one backend, returning annotated frames + timings."""
    if backend == "torch":
        model = YOLOWorld(WEIGHTS)
        model.set_classes(CLASSES)
        model.to("cuda")
        kw = dict(device=0, half=True)
    else:
        model = YOLO(ENGINE, task="detect")
        kw = {}

    for _ in range(10):                      # warmup, never counted
        model.predict(frames[0], imgsz=640, conf=0.25, verbose=False, **kw)

    out, times = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=640, conf=0.25, verbose=False, **kw)[0]
        times.append((time.perf_counter() - t0) * 1000)
        out.append(r.plot())
    print(f"  {backend:6s} mean {np.mean(times):5.2f} ms  ({1000/np.mean(times):5.1f} FPS)")
    return out, np.array(times)


def banner(img, title: str, ms: float, fps: float, mean_ms: float):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 78), DARK, -1)
    cv2.putText(img, title, (14, 30), FONT, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"{ms:5.1f} ms   {fps:5.1f} FPS", (14, 62), FONT, 0.7, GREEN, 2, cv2.LINE_AA)
    cv2.putText(img, f"mean {mean_ms:.1f} ms", (w - 165, 62), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    return img


def main(seconds: int, source: str, out_fps: int, max_frames: int):
    frames = record(seconds, source, max_frames)
    if len(frames) < 10:
        raise SystemExit("not enough frames captured")

    print("processing with each backend on the identical frames")
    torch_frames, torch_ms = process(frames, "torch")
    trt_frames, trt_ms = process(frames, "engine")

    speedup = torch_ms.mean() / trt_ms.mean()
    print(f"\n  speedup {speedup:.2f}x  ({torch_ms.mean():.1f} ms -> {trt_ms.mean():.1f} ms)")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter("demo_side_by_side.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w * 2, h + 48))
    stills = pathlib.Path("stills"); stills.mkdir(exist_ok=True)
    still_at = {int(len(frames) * p) for p in (0.2, 0.45, 0.7, 0.9)}

    for i, (a, b) in enumerate(zip(torch_frames, trt_frames)):
        a = banner(a.copy(), "PyTorch FP16", torch_ms[i], 1000 / torch_ms[i], torch_ms.mean())
        b = banner(b.copy(), "TensorRT FP16", trt_ms[i], 1000 / trt_ms[i], trt_ms.mean())
        pair = np.hstack([a, b])
        cv2.line(pair, (w, 0), (w, pair.shape[0]), WHITE, 2)   # seam, so it reads as one comparison

        # caption centred across the FULL width, otherwise the panel seam cuts it
        # in half and it looks like two unrelated captions
        strip = np.full((48, w * 2, 3), DARK, np.uint8)
        caption = "YOLO-World open-vocabulary  |  RTX 4050 laptop  |  identical frames both sides"
        (tw, _), _ = cv2.getTextSize(caption, FONT, 0.62, 1)
        cv2.putText(strip, caption, ((w * 2 - tw) // 2, 31), FONT, 0.62, WHITE, 1, cv2.LINE_AA)

        badge = f"{speedup:.1f}x"
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.9, 2)
        cv2.putText(strip, badge, (w * 2 - bw - 20, 34), FONT, 0.9, GREEN, 2, cv2.LINE_AA)
        composed = np.vstack([pair, strip])

        writer.write(composed)
        if i in still_at:
            cv2.imwrite(str(stills / f"still_{i:04d}.png"), composed)
    writer.release()

    print(f"\nwrote demo_side_by_side.mp4  ({len(frames)} frames at {out_fps} fps)")
    print(f"wrote {len(still_at)} stills to stills/")
    print("\nnote: the video plays at a constant rate so it is watchable. the honest")
    print("comparison is the per-frame timing burned into each panel, not playback speed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--source", default="0")
    ap.add_argument("--out-fps", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=250,
                    help="cap on frames read from a video file")
    a = ap.parse_args()
    main(a.seconds, a.source, a.out_fps, a.max_frames)
