# PP-OCR segmentation tooling

One-off operator tooling for the PP-OCRv6 detection parity check of
2026-09-19. These scripts run on the inference box, not in this repo's
environments, and they add no dependencies to any `pyproject.toml`.

## Scripts

| Script | Purpose |
|--------|---------|
| `export_onnx.py` | PaddleX medium-det directory → `pp-ocrv6-medium-det.onnx` with dynamic batch, height and width |
| `verify_parity.py` | Paddle static oracle vs exported ONNX over the same tensors, plus fixtures and timing |

`export_onnx.py` converts with paddle2onnx at a pinned opset and then asserts
the input axes are still symbolic: a frozen axis is refused rather than
written, because that exact failure has shipped before. It prints opset, input
and output names and shapes, file size and SHA-256, and saves the same facts
beside the artifact as `.meta.json`.

`verify_parity.py` is the measurement the export stands on. Per page at
max-side caps 1920 and 960 it builds the tensor with PaddleX 3.7.0's own
preprocessing classes, runs the same tensor through the Paddle static
predictor (mkldnn) and onnxruntime, compares the raw probability maps, runs
PaddleX's own DB postprocess on both maps, and compares the boxes. It also
runs the full `paddleocr.TextDetection` pipeline from the raw image (proving
the hand-built tensor path is faithful), checks five extra input shapes, saves
reference fixtures for the serving adapter, and times both runtimes. It exits
0 only on the full pass gate and non-zero otherwise.

## Runbook (on the box)

The reference environment is the benchmark tree; the parity venv lives beside
it. `system-libs` supplies the libgomp the Paddle wheels need:

```bash
cd /root/ppocrv6-onnx-parity-20260919
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
LD_LIBRARY_PATH=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu \
./venv/bin/python export_onnx.py \
  --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --output /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx
BASE="./venv/bin/python verify_parity.py --model-dir /root/ppocrv6-bench-20260919/models/medium \
  --onnx /root/ppocrv6-onnx-parity-20260919/pp-ocrv6-medium-det.onnx \
  --images /root/orli-bench/images \
  --grec-images /root/ppocrv6-bench-20260919/images \
  --workdir /root/ppocrv6-onnx-parity-20260919/run"
RUN="OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
LD_LIBRARY_PATH=/root/ppocrv6-bench-20260919/system-libs/sysroot/usr/lib/x86_64-linux-gnu"
$RUN $BASE maps > run/maps.log 2>&1
$RUN $BASE shapes > run/shapes.log 2>&1
$RUN $BASE pipeline > run/pipeline.log 2>&1
$RUN $BASE timing > run/timing.log 2>&1
$RUN $BASE report --report-json /root/ppocrv6-onnx-parity-20260919/parity-report.json > run/report.log 2>&1
```

Run the stages in order and check each log before moving on. `maps`
exits non-zero when its part of the pass gate fails; in that case stop and do
not run the later stages.

The ONNX itself is never committed to git; it stays in the artifact
directories. Full results, including the per-page parity table, the
shape-coverage table and timing, are in
`docs/inference/ppocrv6-onnx-parity-2026-09-19.md`.

## Serving adapter check (`verify_adapter.py`)

The parity proof above covers the exported graph; `verify_adapter.py`
measures the serving adapter (`run_ppocr_det_segment`: PIL decode,
preprocessing, pyclipper DB postprocess) on real weights. For each of the 14
reference fixtures it reads the page BYTES, calls the production entry
point with refinement switched off (`merge_fragments` false,
`resolve_overlaps` false, `noise_policy` off, so the gate measures the
detector and nothing else) in quad mode (`box_type` quad, pinning the quad
path now that the serving default is polygons), and compares the returned line quads
with the fixture boxes: counts, then greedy one-to-one matching by quad
centre with corner-SET distances (each corner counts only its nearest
corner in the matched box, since corner order may differ). It writes
`results.json` and one `<page>.overlay.jpg` per page (adapter quads in
reading order in green with numbers beside each quad's left edge,
synthetic baselines in yellow, unmatched fixture boxes in red) and exits
non-zero unless every page has identical counts, nothing unmatched, and
max corner distance at most 0.5 px.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /Users/krishuagarwal/Desktop/Programming/python/greekOCR/.venv/bin/python \
  scripts/segmentation/ppocr/verify_adapter.py \
  --fixtures /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/fixtures \
  --images /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_orli-env/server/images \
  --grec-images /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/adapter-e2e \
  --onnx /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/pp-ocrv6-medium-det.onnx \
  --output-dir /Users/krishuagarwal/Desktop/Programming/python/greekOCR-wt/_ppocr-parity/adapter-e2e
```

Each image is matched to its fixture by SHA-256 before use. The 12 Coptic
pages come from the server image directory; `grec-p1.webp` and `grec-p4.jpg`
live on the reference box (`nomikos:/root/ppocrv6-bench-20260919/images/`)
and are fetched with scp into the output directory, whose SHA-256 match is
then verified the same way. The ONNX digest defaults to the pinned artifact
(`--artifact-sha256` overrides it). Measured 2026-09-20: identical counts
on all 14 pages with nothing unmatched and 0.000 px corner distance
against the 0.5 px gate. (The first version used a shapely unclip and
measured 2.2 to 3.2 px max; replacing it with PaddleX's own pyclipper
unclip gave exact parity.) Full table and reading order notes are
in `docs/inference/ppocrv6-segmenter.md`.

## Refinement evaluation (`evaluate_refinement.py`)

`evaluate_refinement.py` runs the adapter on the 12 Coptic benchmark pages
with refinement off, defaults and `drop`, scoring each response against the
benchmark target lines with the benchmark's matching (a port of `axis`,
`assign` and `finish` from
`server/ppocrv6-quality_eval_20260919.py`; the port recovers 1002 of 1018
care targets where the benchmark tool recovers 1008). It prints detections,
precision, recall, F1, targets covered by exactly one detection, pairs
sharing at least 20% area, suspects flagged and real lines wrongly flagged,
and writes reading-order overlays for vat-1r and c13 with suspects in red
into the output directory. Needs `--fixtures` and `--images` as above plus
`--gt` (the ground truth recon directory) and `--onnx` with
`--artifact-sha256`. With `--merge-gap-ratio` (repeatable or comma
separated) one run scores the defaults and drop variants at each gap
ratio, prints a sweep table (recall, precision, F1, lines lost against
off with page and line, lines fixed, merged count) with a per-ratio sweep
summary in the JSON, and skips the overlays.
`--merge-max-overlap-ratio` sweeps the same way at gap 0.25.
`--box-type quad|poly|both` (default `quad`) selects the served geometry;
`both` runs off, defaults and drop in each mode and additionally prints a
per-page and overall quad-vs-poly comparison (mean polygon points, mean
polygon over quad area, mean overlap between consecutive lines per mode,
seconds per page per mode) into `poly_stats` in the JSON.

## `profile_onnx.py`

Follow-up performance tooling for the 2026-09-20 slowdown analysis (the
accepted export ran about 4x slower than Paddle mkldnn). Same box, same
venv, same arguments as `verify_parity.py`, plus `--perfdir` for its
outputs. Subcommands: `counts` (node type census), `profile` (onnxruntime
profiler, top nodes), `sessions` (cheap session options, split with
`--only` so no command runs over 3 minutes), `candidate` (checker, shape
cases and c13 timing for one file), `compare` (numerics of EXTENDED and
denormal_as_zero against the default on 3 pages), `final` (accepted file
at EXTENDED on 3 pages plus the big tensor, at 8 and 2 threads, and Paddle
on the big tensor). Findings and the serving recommendation are in
`docs/inference/ppocrv6-onnx-performance-2026-09-20.md`: use
`ORT_ENABLE_EXTENDED` with 2 intra-op threads per process.
