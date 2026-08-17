"""Stage 2: export YOLO-World with the vocabulary as a RUNTIME INPUT.

Stage 1 called set_classes() before export. That runs CLIP once and folds the
text embeddings into the graph as constants, which is why the engine is fast and
why the classes are frozen. Changing one word meant a 393 second rebuild.

This exports the same network with txt_feats as a second graph input, so the
vocabulary becomes 32x512 floats you write into a buffer at inference time.

Why it works without graph surgery: WorldModel.predict() already accepts
txt_feats and threads it to C2fAttn, ImagePoolingAttn and WorldDetect. The
embeddings only became constants because torch.onnx.export traced a single
image input. Wrapping the call in a Module with two inputs keeps every internal
text projection inside the graph.

Why the class count is FIXED rather than dynamic: WorldDetect.forward does

    boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)

and self.nc is a Python int read at trace time. So the split size is baked. The
workaround is a fixed maximum slot count, padding unused slots with zeros.
Padding is safe because BNContrastiveHead computes

    F.normalize(w) -> einsum -> x * logit_scale.exp() + bias

and a zero embedding gives score = bias, which is -8.9 to -12.0 on the trained
checkpoint, so sigmoid(bias) is about 1e-4. Padded slots cannot fire.

Usage:
    python export_dynamic.py --slots 32
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLOWorld

WEIGHTS = "yolov8s-worldv2.pt"
EMBED_DIM = 512


class DualInputWorld(torch.nn.Module):
    """Wraps WorldModel so ONNX sees (images, txt_feats) instead of (images,)."""

    def __init__(self, world_model: torch.nn.Module, slots: int):
        super().__init__()
        self.m = world_model
        head = self.m.model[-1]
        head.nc = slots          # decides the split size, and the output channels
        head.export = True       # return the inference tensor, not the training dict
        head.format = "onnx"

    def forward(self, images: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        return self.m.predict(images, txt_feats=txt_feats)


def build_embeddings(prompts: list[str], slots: int, world_model) -> torch.Tensor:
    """CLIP-encode prompts and zero-pad to the fixed slot count."""
    if len(prompts) > slots:
        raise ValueError(f"{len(prompts)} prompts exceeds {slots} slots")
    feats = world_model.get_text_pe(prompts)            # [1, len(prompts), 512]
    out = torch.zeros(1, slots, feats.shape[-1], dtype=feats.dtype)
    out[:, : len(prompts)] = feats
    return out


def main(slots: int, imgsz: int, out_path: str) -> None:
    yw = YOLOWorld(WEIGHTS)
    world = yw.model.float().eval()

    # populate the head's anchor/stride cache at the target size. Detect._inference
    # rebuilds these only when the input shape changes, so they must exist before
    # tracing or they get traced as whatever shape ran first.
    world.model[-1].nc = slots
    dummy_txt = torch.zeros(1, slots, EMBED_DIM)
    dummy_img = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        world.predict(dummy_img, txt_feats=dummy_txt)

    wrapper = DualInputWorld(world, slots).eval()

    with torch.no_grad():
        out = wrapper(dummy_img, dummy_txt)
    print(f"torch forward OK, output {tuple(out.shape)}  "
          f"(expect [1, {4 + slots}, {(imgsz//8)**2 + (imgsz//16)**2 + (imgsz//32)**2}])")

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_txt),
        out_path,
        input_names=["images", "txt_feats"],
        output_names=["output0"],
        opset_version=17,
        do_constant_folding=True,   # folds image-path constants, leaves txt_feats alone
        dynamo=False,
    )
    print(f"wrote {out_path}")

    import onnx
    m = onnx.load(out_path)
    print("\ngraph inputs:")
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"  {i.name:12s} {dims}")
    print("graph outputs:")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name:12s} {dims}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=32, help="max simultaneous classes")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="yolov8s-worldv2-dynamic.onnx")
    a = ap.parse_args()
    main(a.slots, a.imgsz, a.out)
