# Models and datasets

Nomikos separates training and data preparation from production inference.
Training tools live in `src/`; the deployable runtime reads the verified
catalog in [`nomikos_inference/registry.yaml`](../../nomikos_inference/registry.yaml).

## Runtime models

### BLLA segmentation

`blla-segment` runs an inference-owned BLLA topology on ONNX Runtime to identify
line candidates on a page. Nomikos converts those candidates into editable
geometry, preserves the legacy `kraken_ceiling` field, and simplifies polygons.

### PP-OCRv6 segmentation

`ppocr-segment` runs PP-OCRv6 medium text detection on ONNX Runtime, followed
by a refinement stage (merges same-row fragments, cuts overlapping stacked
lines, flags suspect detections and reads them last). See
[ppocrv6-segmenter.md](ppocrv6-segmenter.md) for the detector, the refinement
rules, and the validation.

In the platform pickers the segmenters appear as seg-v1 (blla-segment) and
seg-v2 (ppocr-segment); each picker starts on the project's default model,
else seg-v1, and there is no Default entry.

### Calamari HTR

`greek-calamari-v1`, `armenian-calamari-v1`, `syriac-calamari-v2` and `coptic-calamari-v1` run the
Calamari graph on ONNX Runtime for line transcription. All four are the same
topology, retrained per script:

```text
CNN → max pooling → CNN → max pooling
   → 200-unit bidirectional LSTM → dropout
   → 200-unit bidirectional LSTM → linear CTC logits
   → greedy text decoding + character confidences
```

The second recurrent layer is the only structural change from the first
generation of these models, and it is invisible to the runtime: the codec, line
height (48) and blank index travel in the graph's own `metadata_props`, so a
deeper stack is a different set of weights and not a different adapter. What
separates the four is the codec they were trained over: 259 characters for
polytonic Greek, 96 for Armenian, 71 for Syriac, 39 for Coptic. Coptic's structure is
verified, not assumed: its checkpoint carries `lstm_layers: 2` with BiLSTM weights at
`layers.4` and `layers.6`, 40/60-filter 3x3 convolutions, and 200-unit LSTM hidden
states, matching the Syriac checkpoint on every structural field, and the republished
ONNX records `lstm_layers: 2` in its metadata.

The loader validates the graph's own `calamari-onnx-v1` metadata - codec, line
height, blank index - and verifies the configured artifact digest before opening
the file. The trained checkpoint is converted to the run artifact by
`nomikos_inference/export/`, and both are published at the same **Hub
revision** (ADR 0006, which supersedes 0004). `tests/export/` runs the graph and
the artifact on real
weights and compares them, because a conversion step is exactly where a model
can drift from what was trained.

All three siblings were trained by the in-house PyTorch trainer
(`src/models/calamari/trainer.py`), not by the vendored TensorFlow Calamari
tree, which is a research artifact and is not shipped in the inference image.
That decides what the runtime may feed them: **the serving input must be exactly
the training input.** Grayscale, an aspect-preserving bilinear resize to the line
height the graph declares, raw `uint8` values (the graph divides by 255 itself),
no inversion, no centre-line dewarping, no padding. Nothing else. Until
2026-09-07 the runtime ran the legacy TensorFlow Calamari processors instead,
which no registry model was ever trained on; on Armenian pages that are in the
training data that produced 1 exact line in 30 where the training recipe
produces 30 in 30 on the same `best.onnx`.

The line crop is part of that input. One crop function serves every model, the
training exporter's `crop_polygon`: the polygon's bounding box, widened by a
padding, with every pixel outside the polygon painted white on the crop. What is
**per model** is that padding, because the Greek finetuning crops were exported
with 0 px while the Armenian and Syriac ones used 12. Every `task: transcribe`
entry states both in required `line_crop` and `line_crop_padding` fields. Using
the wrong padding costs about as much as the wrong preprocessing did: over the
whole Grec1360 corpus Greek reads 125 of 204 lines exactly at padding 0 (CER
0.050) and none of them at 12 (CER 0.304). The crop is then handed to the model
re-encoded the way the exporter wrote it, a grayscale JPEG at quality 82, so the
lossy round trip the training pixels went through is reproduced rather than
skipped. Polygon coordinates are rounded to integers the way the exporter's
`parse_points` rounds them, and the CTC decoder returns exactly what the
trainer's codec returns, edge whitespace included: neither side trims.

The benchmark for all of this is the trainer, not the ground truth. Run over
the three whole training documents held in Supabase, with the real `src/`
exporter, dataset and `best.pt` on one side and the runner's `run_model` with
`best.onnx` on the other, serving reproduces the trainer's text on every line:
819 of 819 Armenian, 204 of 204 Greek, 504 of 504 Syriac, with the trainer at
batch size 1. (At its default batch of 16 the trainer disagrees with itself,
because its collate pads lines to the batch's widest and the network can see
the pad; that is a trainer property and is described in the ADR.)

See [ADR 0007](../adr/0007-serving-preprocessing-reproduces-the-training-loader.md)
for the measurements and for the three files that have to move together;
`tests/inference/unit/test_calamari_training_parity.py` compares the serving
functions against the training functions and is what keeps the two in step. The
`preprocessing` string in a published graph's ONNX metadata is a stale label on
the existing artifacts, not a description of what the runtime does.

Calamari, PP-OCR, and BLLA are capable enough for the current manuscript workflow while
remaining practical for CPU-first local execution. The helper does not require
CUDA, a GPU, or a training environment.

### Syriac PP-OCR recognition

`syriac-ppocr-v1` runs PP-OCRv6 recognition for Syriac on ONNX Runtime for line
transcription. It takes the kraken line polygon without a margin at line height
96, with 16 px white horizontal padding applied inside the adapter. See the
[release note](syriac-ppocr-v1-release-2026-09-20.md) and the
[adapter note](ppocr-rec-adapter-2026-09-20.md) for the publication, the serving
recipe, and the verification.

TrOCR appears in research experiments only. It has no runtime adapter, registry
entry, packaged checkpoint, or platform catalog path and is not supported by
the product.

## Current catalog

| ID                   | Task       | Architecture     | Artifact                                                                                         |
| -------------------- | ---------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `blla-segment`     | Segment    | BLLA (ONNX Runtime) | `blla.onnx` from [segmentation repo](https://huggingface.co/nomikos-project/segmentation-blla) |
| `ppocr-segment` | Segment | PP-OCRv6 detection (ONNX Runtime) | `ppocrv6-det.onnx` from [segmentation repo](https://huggingface.co/nomikos-project/segmentation-ppocrv6-det) |
| `greek-calamari-v1` | Transcribe | Calamari (ONNX Runtime) | `best.onnx` from the [Hugging Face checkpoint](https://huggingface.co/nomikos-project/greek-htr-calamari), pinned revision |
| `armenian-calamari-v1` | Transcribe | Calamari (ONNX Runtime) | `best.onnx` from the [Hugging Face checkpoint](https://huggingface.co/nomikos-project/armenian-htr-calamari), pinned revision |
| `syriac-calamari-v2` | Transcribe | Calamari (ONNX Runtime) | `best.onnx` from the [Hugging Face checkpoint](https://huggingface.co/nomikos-project/syriac-htr-calamari), pinned revision |
| `syriac-ppocr-v1` | Transcribe | PP-OCR recognition (ONNX Runtime) | `model.onnx` from the [Hugging Face checkpoint](https://huggingface.co/nomikos-project/syriac-htr-ppocr_rec), pinned revision |
| `coptic-calamari-v1` | Transcribe | Calamari (ONNX Runtime) | `best.onnx` from the [Hugging Face checkpoint](https://huggingface.co/nomikos-project/coptic-htr-calamari), pinned revision |

Coptic is registered at a measured crop padding of 12 px: CER 0.0177 at 12 against 0.1890
at 0 on 2823 platform Coptic project lines through `best.pt` at batch size 1, minimum flat
across 10 to 12 (measured 2026-09-18; full curve below). The first pin (`bdaa22d3`) was
unservable: its `best.onnx` carried no `temperature` metadata key so the adapter refused
to open it, and its time axis was frozen at the traced example length 8 so every real
line failed at the LSTM node. The pin now points at `b103b562`, which replaces only
`best.onnx` with a fresh export of the same `best.pt` through `nomikos_inference/export`,
verified 2823 of 2823 platform lines byte-identical to `best.pt` at batch size 1,
CER 0.0177. No
training-data claim is made for it here: the card gives a held-out test CER of 0.0823 over
283 lines, and nothing else quoted above was measured for it.

The padding measurement behind the Coptic pin, 2026-09-18: serving crop bytes through the
real trainer loader into `best.pt` at batch size 1 with the real trainer decoder, no
normalisation, over the platform's Coptic project lines (2823 lines, 34491 characters).
CER falls 10x from padding 0 to the minimum and rises again after it, so the curve is a
padding signal rather than noise. Paddings 10 and 12 tie within noise (6 edits), and 12 is
registered because it is the exporter's `PADDING` constant, the value the Armenian and
Syriac crops used whenever they were not Greek's 0.

| Padding px | CER | Exact / 2823 |
| --- | ---: | ---: |
| 0 | 0.1890 | 763 |
| 2 | 0.1025 | 1297 |
| 4 | 0.0500 | 1869 |
| 6 | 0.0274 | 2236 |
| 8 | 0.0189 | 2441 |
| 10 | 0.0175 | 2489 |
| **12** | **0.0177** | **2485** |
| 16 | 0.0219 | 2434 |
| 20 | 0.0407 | 2039 |

Seven runtime models are registered. An eighth needs a compatible adapter, immutable Hub revision, SHA-256
digest, registry entry, platform catalog metadata, tests, and a declared host
eligibility.

## From correction to training data

The intended expert-in-the-loop pipeline is:

1. Annotate page geometry.
2. Pair each segment with a transcription.
3. Correct and review the text as ground truth.
4. Export processed line images and transcription files.
5. Stage labelled crops under `nomikos_inference/publish/artifacts/staging/datasets/`.
6. Validate and publish a separate Hugging Face dataset repository.
7. Train or fine-tune a script-specific model.
8. Publish verified weights and add the model to the registry.

The staging layout is:

```text
nomikos_inference/publish/artifacts/staging/datasets/<dataset-slug>/
  images/
    manuscript-001/line-0001.png
  labels.csv
```

`labels.csv` is UTF-8 CSV with `image,transcription` columns. Each crop needs
one non-empty transcription. Source crops remain independent of
model-specific resizing and normalization.

Preparation tools support PAGE XML and eScriptorium/Transkribus-style inputs.
This repository provides the preparation and publishing workflow, but it does
not establish the size, rights, license, provenance, or publication status of
every planned corpus. Call a dataset expert-curated only when its release
documents the experts, review process, provenance, rights, and license.

## Publish a model

Dry-run validation:

```bash
PYTHONPATH=. python scripts/hf/publish_model.py \
  --script greek \
  --architecture calamari \
  --model-version v1 \
  --registry-tag stable \
  --task transcribe
```

After setting a write-capable `HF_TOKEN`, add `--upload`. Then pin the
immutable revision and SHA-256 in `nomikos_inference/registry.yaml`, run tests,
prefetch the weights, update platform catalog metadata, and deploy the
matching runtime.

Dataset validation:

```bash
PYTHONPATH=. python scripts/hf/publish_dataset.py \
  greek-manuscript-lines --script greek
```

Use `--upload` only for a reviewed release. See
[`adding-inference-models.md`](adding-inference-models.md) and the
[Hub publishing reference](../../scripts/hf/README.md).
