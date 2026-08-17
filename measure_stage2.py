"""What does a runtime-swappable vocabulary actually cost?

Three engines, so the two variables separate:

  baked-5   stage 1. set_classes() before export, embeddings folded into weights.
  dyn-5     same 5 classes, same [1,9,8400] output, but txt_feats is an input.
            baked-5 vs dyn-5 isolates the cost of swappability.
  dyn-32    32 slots. dyn-5 vs dyn-32 isolates the cost of extra slots.

Also measures the two ways to change vocabulary, because the difference decides
whether a text encoder has to live on the device at all.

Usage:
    python measure_stage2.py
"""

from __future__ import annotations

import pathlib
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

from prompt_engine import PromptEngine, letterbox

CLASSES5 = ["person", "backpack", "cell phone", "bottle", "chair"]
REPEATS = 200
WARMUP = 30


def strip_ultralytics_header(path: str) -> bytes:
    """Return the raw serialised engine from a file ultralytics wrote.

    ultralytics prepends its own metadata to .engine files: a 4-byte
    little-endian length, then a JSON blob holding imgsz, task, and the class
    NAMES that set_classes() baked in. That is how YOLO(engine) recovers the
    vocabulary. A raw trt.Runtime does not know about it and rejects the file
    with a magic-tag mismatch (the length reads as a bogus tag), so skip past it.

    Engines built by build_engine.py have no such header and start with 'ftrt'.
    """
    data = pathlib.Path(path).read_bytes()
    if data[:4] == b"ftrt":
        return data
    n = int.from_bytes(data[:4], "little")
    if 0 < n < 100_000 and data[4:5] == b"{":
        import json
        meta = json.loads(data[4:4 + n].decode("utf-8", "ignore"))
        names = meta.get("names")
        if isinstance(names, dict):
            names = list(names.values())
        print(f"    ultralytics header: {n} bytes, baked classes = {names}")
        return data[4 + n:]
    return data


class BakedEngine:
    """Stage 1's single-input engine, driven through the same raw() path."""

    def __init__(self, path: str):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(strip_ultralytics_header(path))
        if self.engine is None:
            raise SystemExit(f"could not deserialise {path}")
        self.context = self.engine.create_execution_context()
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = trt.nptype(self.engine.get_tensor_dtype(n))
            buf = torch.zeros(shape, dtype=getattr(torch, np.dtype(dt).name), device="cuda")
            self.tensors[n] = buf
            self.context.set_tensor_address(n, buf.data_ptr())
        self.img_name = next(n for n, t in self.tensors.items() if t.ndim == 4 and t.shape[1] == 3)
        self.out_name = next(n for n in self.tensors
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    def raw(self, blob):
        img = self.tensors[self.img_name]
        img.copy_(torch.from_numpy(blob).to(img.dtype))
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return self.tensors[self.out_name]


def time_raw(fn, blob, label):
    """GPU inference only. No letterbox, no NMS, no host transfer."""
    for _ in range(WARMUP):
        fn(blob)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(blob)
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    print(f"  {label:34s} mean {a.mean():6.2f} ms   p50 {np.percentile(a,50):6.2f}   "
          f"p95 {np.percentile(a,95):6.2f}   {1000/a.mean():6.1f} FPS")
    return a.mean()


def main():
    frame = cv2.imread("assets/bus.jpg")
    blob, *_ = letterbox(frame)

    print("=" * 74)
    print("  RAW GPU INFERENCE ONLY (no preprocess, no NMS, no host copy)")
    print("=" * 74)

    baked = BakedEngine("yolov8s-worldv2.engine")
    m_baked = time_raw(baked.raw, blob, "baked 5 classes  (stage 1)")
    del baked
    torch.cuda.empty_cache()

    if pathlib.Path("yolov8s-worldv2-dyn5.engine").exists():
        d5 = PromptEngine("yolov8s-worldv2-dyn5.engine")
        d5.set_prompts(CLASSES5)
        m_d5 = time_raw(d5.raw, blob, "runtime 5 slots")
        del d5
        torch.cuda.empty_cache()
    else:
        m_d5 = None
        print("  (5-slot engine not built yet)")

    d32 = PromptEngine("yolov8s-worldv2-dynamic.engine")
    d32.set_prompts(CLASSES5)
    m_d32 = time_raw(d32.raw, blob, "runtime 32 slots")

    print()
    if m_d5:
        print(f"  cost of swappability   {m_baked:.2f} -> {m_d5:.2f} ms   "
              f"({(m_d5/m_baked - 1) * 100:+.1f}%)")
        print(f"  cost of 32 vs 5 slots  {m_d5:.2f} -> {m_d32:.2f} ms   "
              f"({(m_d32/m_d5 - 1) * 100:+.1f}%)")
    print(f"  baked-5 vs runtime-32  {m_baked:.2f} -> {m_d32:.2f} ms   "
          f"({(m_d32/m_baked - 1) * 100:+.1f}%)")

    print("\n" + "=" * 74)
    print("  CHANGING THE VOCABULARY: live CLIP vs a precomputed table")
    print("=" * 74)

    # live CLIP, warmed up so the model load is not counted
    d32.set_prompts(["warmup"])
    live = []
    for p in (["person"], ["dog"], ["bicycle"], ["traffic cone"], ["forklift"]):
        live.append(d32.set_prompts(p))
    print(f"  live CLIP encode, 1 prompt         mean {np.mean(live):7.2f} ms")

    live2 = [d32.set_prompts(["person", "dog"]), d32.set_prompts(["car", "truck"])]
    print(f"  live CLIP encode, 2 prompts        mean {np.mean(live2):7.2f} ms")

    # precomputed: build the table once offline, then it is a memcpy
    vocab = ["person", "dog", "bicycle", "traffic cone", "forklift", "car", "truck",
             "hard hat", "safety vest", "backpack", "bottle", "chair"]
    table = {w: d32.encode([w]) for w in vocab}
    kb = 512 * 4 / 1024
    look = []
    for w in vocab:
        look.append(d32.set_embeddings(table[w], [w]))
    print(f"  precomputed lookup + memcpy        mean {np.mean(look):7.2f} ms")
    print(f"\n  speedup {np.mean(live)/np.mean(look):.0f}x. one embedding is {kb:.0f} KB, "
          f"so a 10,000 word table is {512*4*10000/1e6:.0f} MB on disk")
    print("  which means no text encoder has to run on the device at all")


if __name__ == "__main__":
    main()
