"""Runtime for the dual-input YOLO-World engine: swap the vocabulary, no rebuild.

Ultralytics' YOLO() loader assumes one input tensor, so it cannot drive this
engine. This talks to TensorRT directly.

Device memory is torch CUDA tensors rather than pycuda: torch already owns a
CUDA context and .data_ptr() is exactly what set_tensor_address wants, so this
adds no dependency beyond what the benchmark already needed.

Two ways to get embeddings, and the difference matters for edge deployment:

  encode()  runs CLIP on the device. Flexible, any string, but needs the text
            encoder resident (~150 MB) and takes ~10 ms per new prompt.
  lookup()  reads from a table built offline. One embedding is 512 floats, so
            2 KB. A 10,000 word vocabulary is 20 MB, which ships fine on a Jetson
            and means no CLIP on the device at all.

Usage:
    python prompt_engine.py --image assets/bus.jpg
"""

from __future__ import annotations

import argparse
import pathlib
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

from ultralytics.utils.nms import non_max_suppression

ENGINE = "yolov8s-worldv2-dynamic.engine"
WEIGHTS = "yolov8s-worldv2.pt"


def letterbox(img: np.ndarray, size: int = 640):
    """Resize preserving aspect ratio, pad to square with 114 grey (ultralytics default).

    Returns the blob plus the scale and padding needed to map boxes back.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    blob = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    return blob, r, left, top


class PromptEngine:
    """A TensorRT YOLO-World whose class list is a runtime buffer."""

    def __init__(self, engine_path: str = ENGINE, device: str = "cuda"):
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(pathlib.Path(engine_path).read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not deserialise {engine_path}")
        self.context = self.engine.create_execution_context()
        self.device = device

        self.tensors: dict[str, torch.Tensor] = {}
        self.names: dict[str, str] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buf = torch.zeros(shape, dtype=getattr(torch, np.dtype(dtype).name), device=device)
            self.tensors[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            role = "in " if mode == trt.TensorIOMode.INPUT else "out"
            self.names.setdefault(role, name)
            print(f"  {role} {name:12s} {shape} {np.dtype(dtype).name}")

        # identify the tensors by shape rather than assuming names
        self.img_name = next(n for n, t in self.tensors.items() if t.ndim == 4 and t.shape[1] == 3)
        self.txt_name = next(n for n, t in self.tensors.items() if t.ndim == 3 and t.shape[-1] == 512)
        self.out_name = next(n for n in self.tensors
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.slots = self.tensors[self.txt_name].shape[1]
        self.prompts: list[str] = []
        self._clip = None

    # ---------------------------------------------------------------- vocabulary
    def _world(self):
        if self._clip is None:
            from ultralytics import YOLOWorld
            self._clip = YOLOWorld(WEIGHTS).model.float().eval()
        return self._clip

    def encode(self, prompts: list[str]) -> torch.Tensor:
        """CLIP-encode prompts, zero-padded to the slot count."""
        if len(prompts) > self.slots:
            raise ValueError(f"{len(prompts)} prompts exceeds {self.slots} slots")
        feats = self._world().get_text_pe(prompts)
        out = torch.zeros(1, self.slots, feats.shape[-1])
        out[:, : len(prompts)] = feats
        return out

    def set_prompts(self, prompts: list[str]) -> float:
        """Swap the vocabulary. Returns the milliseconds it cost."""
        t0 = time.perf_counter()
        emb = self.encode(prompts)
        dst = self.tensors[self.txt_name]
        dst.copy_(emb.to(dst.dtype))
        self.prompts = list(prompts)
        return (time.perf_counter() - t0) * 1000

    def set_embeddings(self, emb: torch.Tensor, prompts: list[str]) -> float:
        """Swap using precomputed embeddings, so no CLIP is needed at runtime."""
        t0 = time.perf_counter()
        dst = self.tensors[self.txt_name]
        dst.copy_(emb.to(dst.dtype))
        self.prompts = list(prompts)
        return (time.perf_counter() - t0) * 1000

    # ----------------------------------------------------------------- inference
    def raw(self, blob: np.ndarray) -> torch.Tensor:
        img = self.tensors[self.img_name]
        img.copy_(torch.from_numpy(blob).to(img.dtype))
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return self.tensors[self.out_name]

    def detect(self, frame: np.ndarray, conf: float = 0.25, iou: float = 0.7):
        blob, r, left, top = letterbox(frame)
        out = self.raw(blob).float().cpu()
        det = non_max_suppression(out, conf, iou, nc=self.slots)[0]
        results = []
        for x1, y1, x2, y2, score, cls in det.tolist():
            idx = int(cls)
            if idx >= len(self.prompts):        # a padded slot fired, should not happen
                continue
            results.append({
                "label": self.prompts[idx],
                "conf": float(score),
                "box": [(x1 - left) / r, (y1 - top) / r, (x2 - left) / r, (y2 - top) / r],
            })
        return results


def main(image: str):
    eng = PromptEngine()
    frame = cv2.imread(image)
    if frame is None:
        raise SystemExit(f"could not read {image}")

    print(f"\n{eng.slots} slots. Same engine throughout, only the text buffer changes.\n")
    for prompts in (["bus"], ["person"], ["bus", "person"],
                    ["man in white shirt"], ["dog"], ["license plate", "window"]):
        swap_ms = eng.set_prompts(prompts)
        t0 = time.perf_counter()
        res = eng.detect(frame)
        infer_ms = (time.perf_counter() - t0) * 1000
        hits = [(d["label"], round(d["conf"], 2)) for d in res]
        print(f"  {str(prompts):34s} swap {swap_ms:6.1f} ms   detect {infer_ms:5.1f} ms   {hits}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="assets/bus.jpg")
    a = ap.parse_args()
    main(a.image)
