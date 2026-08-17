"""Build a TensorRT engine from the dual-input ONNX.

Ultralytics' exporter assumes one input, so it cannot build this. Straight
TensorRT API instead. Both inputs are static ([1,3,640,640] and [1,slots,512]),
so no optimization profile is needed: the slot count is fixed at export time
because WorldDetect splits on a Python int.

Usage:
    python build_engine.py --onnx yolov8s-worldv2-dynamic.onnx --fp16
"""

from __future__ import annotations

import argparse
import pathlib

import tensorrt as trt


def build(onnx_path: str, engine_path: str, fp16: bool, workspace_gb: int) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    data = pathlib.Path(onnx_path).read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print("  parse error:", parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    print(f"parsed {onnx_path}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  input  {t.name:12s} {t.shape}  {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"  output {t.name:12s} {t.shape}  {t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("  warning: platform reports no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    print("building. TensorRT is timing kernel candidates per layer, expect minutes.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")

    pathlib.Path(engine_path).write_bytes(serialized)
    # TensorRT 10 returns IHostMemory, which has .nbytes but no len()
    print(f"wrote {engine_path} ({serialized.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="yolov8s-worldv2-dynamic.onnx")
    ap.add_argument("--engine", default="yolov8s-worldv2-dynamic.engine")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--workspace", type=int, default=4, help="GB")
    a = ap.parse_args()
    build(a.onnx, a.engine, a.fp16, a.workspace)
