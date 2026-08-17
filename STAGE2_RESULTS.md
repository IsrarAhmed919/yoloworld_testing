# Stage 2: a TensorRT engine whose vocabulary is a runtime buffer

Stage 1 baked the class list into the engine, so changing one word meant a 393
second rebuild. This exports the same network with the text embeddings as a
second graph input, so the vocabulary becomes 32x512 floats you write into a
buffer between frames.

RTX 4050 Laptop (6 GB), TensorRT 10.16, FP16, `yolov8s-worldv2`, 640px.

## It works

Same engine, same session, no rebuild. Only the text buffer changes:

| prompt | detections |
|---|---|
| `bus` | bus 0.90 |
| `person` | 4 people: 0.91, 0.89, 0.88, 0.74 |
| `bus`, `person` | 5 boxes, both classes at once |
| `man in white shirt` | 0.60, 0.46 |
| `dog` | nothing, correctly |

## Finding 1: swappability is free

Three engines so the variables separate. Interleaved round-robin over 8 rounds of
60 frames each, so thermal drift hits all three equally. Raw GPU inference only,
no preprocessing, no NMS, no host copy.

| engine | median | min | max | own spread |
|---|---|---|---|---|
| baked 5 classes (stage 1) | 3.73 ms | 3.69 | 3.99 | 8.1% |
| runtime 5 slots | 3.75 ms | 3.59 | 3.92 | 9.0% |
| runtime 32 slots | 3.67 ms | 3.53 | 3.74 | 6.0% |

**Cost of making the vocabulary swappable: +0.6%. Cost of 32 slots instead of 5:
-2.2%.** Both are far inside the 9% each engine varies by on its own, so the
honest statement is that neither effect is measurable on this hardware.

The first, non-interleaved run measured the 32-slot engine as 12% *faster* than
the 5-slot one. That is physically impossible, since 32 slots computes strictly
more, and it is what proved the spread was noise rather than signal. A single
sequential pass would have produced a confident and wrong number in either
direction.

Physically this makes sense. The image backbone pushes 640x640x3 through 68
convolutions. The text path is 32x512 through a handful of linear layers and
seven Einsums. It disappears into the noise.

**So there is no performance reason to bake the vocabulary.** Stage 1's design
buys nothing and costs a 393 second rebuild per vocabulary change. The caveat is
hardware: on a Jetson Nano the text path is a larger share of a smaller budget,
so this should be re-measured there rather than assumed.

## Finding 2: do not run CLIP on the device

Two ways to fill the text buffer:

| method | cost |
|---|---|
| live CLIP encode, 1 prompt | 115.41 ms |
| live CLIP encode, 2 prompts | 200.61 ms |
| precomputed table + memcpy | **0.08 ms** |

**1,395x.** Live encoding also scales with prompt count, so a five-word
vocabulary change stalls a 30 FPS pipeline for the better part of a second.

One embedding is 512 float32 = 2 KB. A 10,000 word vocabulary is a 20 MB lookup
table you build offline and ship with the model. That is nothing on a Jetson, and
it means **no text encoder has to exist on the device at all**.

This retires the hardest part of the original stage 3 plan. There is no need to
convert CLIP or an LLM to TensorRT and wire it into DeepStream. The language model
runs wherever it likes, emits words, and the pipeline does a dictionary lookup and
a 2 KB memcpy.

## Finding 3: two thirds of "inference time" is not inference

Raw GPU execution is 3.7 ms, about 270 FPS. The same engine through
ultralytics' `predict()` measured 10.8 ms in stage 1.

The missing ~7 ms is letterboxing, the host transfer, and NMS on a
[1, 36, 8400] tensor in Python. So when a benchmark quotes 10 ms for TensorRT
YOLO, roughly two thirds of that is pre and postprocessing, not the model. If you
need more throughput, the model is not where the time is.

## How it works

`WorldModel.predict()` already accepts `txt_feats` and threads it to `C2fAttn`,
`ImagePoolingAttn` and `WorldDetect`. The embeddings only became constants in
stage 1 because `torch.onnx.export` traced a single image input. Wrapping the call
in a Module that takes both tensors keeps every internal text projection inside
the graph, so no ONNX surgery is needed:

```python
class DualInputWorld(torch.nn.Module):
    def __init__(self, world_model, slots):
        super().__init__()
        self.m = world_model
        head = self.m.model[-1]
        head.nc = slots
        head.export = True
    def forward(self, images, txt_feats):
        return self.m.predict(images, txt_feats=txt_feats)
```

Verification that it actually worked, rather than trusting the export: in stage 1
all seven Einsum nodes read `CONSTANT[1, 5, 512]`. Here all seven read
`live x live`, and there are zero baked 512-dim constants left.

### Why the slot count is fixed rather than dynamic

`WorldDetect.forward` ends with

```python
boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
```

`self.nc` is a Python int read at trace time, so the split size is baked into the
graph. A truly dynamic class axis would need that split to be dynamic.

The workaround is a fixed slot count with zero padding. That is safe because
`BNContrastiveHead` computes `normalize(w) -> einsum -> x * logit_scale.exp() + bias`,
and on the trained checkpoint those biases are -11.95, -10.41 and -8.87. A zero
embedding therefore scores `sigmoid(bias)`, about 1e-4. In every test run no
padded slot ever fired.

### Loading an ultralytics-exported engine with raw TensorRT

Ultralytics prepends its own metadata: a 4-byte little-endian length then a JSON
blob. On the stage 1 engine that is 614 bytes and contains, among other things,
the class names `set_classes()` baked in, which is how `YOLO(engine)` recovers the
vocabulary.

A bare `trt.Runtime` does not know about it, reads the length as a magic tag, and
fails with

```
Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed
(614 != 1953657958)
```

Engines from `build_engine.py` start with `ftrt` and load directly. Skip the
header when loading an ultralytics file.

## Reproducing

```bash
python export_dynamic.py --slots 32                 # dual-input ONNX
python build_engine.py                              # TensorRT, ~7 min
python prompt_engine.py --image assets/bus.jpg      # swap prompts, no rebuild
python measure_stage2.py                            # the numbers above
```

`prompt_engine.py` uses torch CUDA tensors for device memory, so `.data_ptr()`
feeds `set_tensor_address` directly and pycuda is not needed.

## Next

Stage 3 is now much smaller than planned. The vocabulary path is a 2 KB memcpy, so
what remains is producing the word list: a language model outside the video
pipeline emitting class names, and a precomputed embedding table to look them up.
The pipeline never pauses and never loads a text encoder.
