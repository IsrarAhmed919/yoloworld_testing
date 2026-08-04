"""Open-vocabulary detection on a live feed, with the confidence threshold adjustable
at runtime.

The point of this script is the threshold sweep: instead of editing `conf` and
restarting, press +/- to move it while the same objects stay in frame. With --log,
per-class detection counts are written to CSV so "too many boxes" becomes a number.

    python yoloworld.py --classes person ring headphones hand --conf 0.25 --log sweep.csv

Keys:  q quit   + / - adjust threshold   s save frame
"""

import argparse
import csv
import time
from collections import deque

import cv2
from ultralytics import YOLOWorld

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="webcam index or video path")
parser.add_argument("--classes", nargs="+", default=["person", "ring", "headphones", "hand"])
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--weights", default="yolov8s-world.pt")
parser.add_argument("--log", help="write per-frame class counts to this CSV")
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise SystemExit(f"could not open source: {args.source}")

model = YOLOWorld(args.weights)
model.set_classes(args.classes)

conf = args.conf
frame_times = deque(maxlen=30)
log_file = open(args.log, "w", newline="") if args.log else None
log = csv.writer(log_file) if log_file else None
if log:
    log.writerow(["frame", "conf", *args.classes, "total"])

cv2.namedWindow("YOLO-World", cv2.WINDOW_NORMAL)

try:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        result = model(frame, conf=conf, verbose=False)[0]
        frame_times.append(time.perf_counter() - start)

        counts = [0] * len(args.classes)
        for c in result.boxes.cls.int().tolist():
            counts[c] += 1

        annotated = result.plot()
        fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
        summary = "  ".join(f"{n}:{c}" for n, c in zip(args.classes, counts))
        cv2.putText(annotated, f"conf {conf:.2f}   {fps:5.1f} fps   {summary}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if log:
            log.writerow([frame_idx, f"{conf:.2f}", *counts, sum(counts)])

        cv2.imshow("YOLO-World", annotated)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):
            conf = min(0.95, round(conf + 0.05, 2))
            print(f"conf -> {conf:.2f}")
        if key == ord("-"):
            conf = max(0.01, round(conf - 0.05, 2))
            print(f"conf -> {conf:.2f}")
        if key == ord("s"):
            cv2.imwrite(f"frame_{frame_idx}_conf{conf:.2f}.jpg", annotated)
            print(f"saved frame_{frame_idx}_conf{conf:.2f}.jpg")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
