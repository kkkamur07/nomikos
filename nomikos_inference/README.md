# nomikos-inference

The manuscript **segment** and **transcribe** runtime. It lives at the repository
root in `nomikos_inference/`, separate from the Nomikos platform API in
`nomikos/backend/`, and it is the one distribution published to PyPI: a
researcher's laptop and a hosted worker install the same wheel
(ADR 0002).

For the public product overview, setup, model availability, and architecture,
see the [root README](../README.md), [use and hosting guide](../docs/guides/using-and-hosting.md),
[models and datasets guide](../docs/inference/models-and-datasets.md), and
[technical architecture](../docs/architecture.md).

## Install

```bash
uv tool install nomikos-inference
nomikos --version
```

No flags, no uv version floor. That is a deliberate outcome of ADR 0006 rather
than a simplification of the instructions: `onnxruntime` publishes one CPU wheel
per platform, so there is no accelerator variant to resolve by accident and
nothing for the install command to pin.

The previous instruction was `uv tool install nomikos-inference
--torch-backend=cpu`, and it was load-bearing — without it a Linux resolve pulled
sixteen `nvidia-*`/`triton` wheels, 4801 MB installed against 969 MB with the
flag. No packaging metadata can express "resolve this dependency from that
index", so the flag was the pin and nothing in the package could enforce it;
worse, uv 0.7.x accepted `UV_TORCH_BACKEND=cpu` and silently ignored it.

Measured closure on macOS arm64: **372 MB**, down from 817 MB. Every supported
platform resolves the same way, Intel macOS included — it was excluded only
because PyTorch ships no wheel for it.

## Status

| Piece | State |
|-------|--------|
| Console entry point (`nomikos_inference/cli/`) | `nomikos pair`, `nomikos run`, `nomikos version`, `nomikos upgrade` |
| Hub integration (`nomikos_inference/hub/`) | `hf://` resolution, cache manifest, artifact SHA-256 |
| Request/response contracts (`nomikos_inference/contracts/`) | Defined for segment, transcribe, jobs, and callbacks |
| Model registry (`nomikos_inference/registry.yaml`) | Transcribe (Calamari, PP-OCR) + segmentation (BLLA, PP-OCR) entries |
| Model runner (`nomikos_inference/jobs/runner.py`) | Registry lookup, weight resolution, and model execution |

## Pairing a machine

```bash
nomikos pair
```

It prints a link and a confirmation code, then waits. **Open the link and check
that the page shows the same code the terminal did.** That comparison is the
only check available: every other field on the consent screen — the machine
name, the platform, the version — is supplied by whoever started the pairing, so
a convincing one is not evidence. A different code means the request came from
somewhere else. Close the page and approve nothing
(ADR 0001, decision 13).

The URL is printed before any browser is opened, and no browser is opened at all
over SSH. `--no-browser` forces that everywhere.

The device token lands in `~/.nomikos/device.json`, mode `0600` in a `0700`
directory. Running `pair` again on a paired machine reports the existing pairing
rather than creating a second device; `--force` overrides that. If the device has
been removed from the account, `pair` says so and exits non-zero — revoking is a
decision made in the browser, and only `--force` reverses it.

| Variable | Default | Purpose |
|----------|---------|---------|
| `NOMIKOS_API_URL` | `https://api.nomikos.app` | Platform to pair against (`--api-url` wins) |
| `NOMIKOS_HOME` | `~/.nomikos` | Where the device credential is kept |

```bash
nomikos version
```

Reports the installed distribution version — the exact string sent as
`X-Nomikos-Agent-Version` on every claim, which the platform's **version floor**
refuses on. It contacts nothing: the floor is served only on a claim response,
and reporting a version should not cost a page of work.

## Building the wheel

```bash
uv build            # -> dist/nomikos_inference-<version>-py3-none-any.whl
```

The repository root is the project, because a build backend cannot reach
outside its own root and the package it publishes is `nomikos_inference/`. Development
still runs from the tree (`tool.uv.package = false`), so `uv sync` does not
install the wheel into every dev environment.

## Releasing, and what that changed about security patching

A release is a PyPI publish. Tag `v<version>` matching `[project].version`,
publish the GitHub release, and `.github/workflows/release.yml` builds the
wheel and sdist on one runner, exports the published dependency closure
(`uv export --no-default-groups`), gates it on a CRITICAL/HIGH Trivy scan,
attests build provenance, and uploads to PyPI through Trusted Publishing. There
is no signing secret: the upload token is minted per run from the workflow's
OIDC identity.

Until 0.1.6 a release was instead four per-OS PyInstaller builds, a Developer ID
notarization pass, an Authenticode signing pass, and a GPG-signed checksum
manifest - roughly 1,137 lines of packaging around 516 lines of program. Its
real cost was not the line count. **A frozen installer made this project the
distributor of a vendored dependency tree with no update channel.** Every CVE in
`protobuf`, `Pillow`, or `scipy` meant rebuilding on four runners, re-signing,
re-notarizing, and then having no way to make anyone install the result: the
installed base decided when, or whether, to be patched.

Patching is now two moves:

1. Bump the dependency floor here and re-lock.
2. Raise the platform's **version floor** (`INFERENCE_AGENT_MIN_VERSION`).

The second is what makes the first land. An **inference agent** below the floor
is refused at the claim endpoint and told to upgrade
(ADR 0002, #55),
so a vulnerable agent stops taking work whether or not its operator noticed.
The floor is served by the platform rather than read from PyPI, which means it
turns without a release.

To upgrade an installed agent by hand:

```bash
uv tool upgrade nomikos-inference
```

## One queue, owned by the platform

This package holds no job queue, no database, and no claim loop. A queued page is
a row in the platform's `jobs` table; an inference agent claims it, runs it
through the same `run_model()` the sync path uses, and reports the outcome
through the platform's existing job callback contract. See ADR 0003.

There is consequently no `inference-api` container: the registry endpoint an
agent syncs from is served by the platform on port 8000.

## Weights layout

Registry models resolve weights at runtime from:

| Source | Example | Cache / path |
|--------|---------|----------------|
| Hub | `hf://nomikos-project/greek-htr-calamari@stable` | `~/.nomikos/hf/cache/<registry_model_id>/<registry_tag>/` |
| Local bundled (offline) | `file://local/syriac/calamari/v1/stable/best.pt` | `publish/artifacts/local/...` |
| BLLA segmentation | `hf://nomikos-project/segmentation-blla@stable` | `blla.onnx` in the Hub cache |

No local weight checkout is required for the default Hub models; they download from their public repos on first use into `HF_CACHE_ROOT`.

### Runtime

All four architectures run on ONNX Runtime, CPU only (ADR 0006).
Transcribe loads `best.onnx` through `nomikos_inference/architectures/calamari/` or
`model.onnx` through `nomikos_inference/architectures/ppocr_rec/`, and
segment loads `blla.onnx` through `nomikos_inference/architectures/blla/` or
`ppocrv6-det.onnx` through `nomikos_inference/architectures/ppocr_det/`. The graph
carries its own codec, line height and blank index in `metadata_props`, which is
why one file is the whole model - there is no sidecar and no `.pt` to pair it
with.

Calamari preprocessing is the **training recipe** and nothing else: the runtime
cuts and prepares a line exactly the way the trainer that produced the weights
did, because a serving input that differs from the training input is a wrong
answer from a correct model
([ADR 0007](../docs/adr/0007-serving-preprocessing-reproduces-the-training-loader.md)).

The **artifact SHA-256** is verified in `architectures/artifact.py` *before* the
loader opens the file. That check used to be what kept `torch.load` off an
unverified pickle; there is no unpickling left to reach, because a `.pt` is
refused by suffix, so what it defends now is integrity.

Both native checkpoints are still published beside the graphs at the same **Hub
revision**, so every cache directory holds a file this runtime must not open.
`find_hub_artifact` names only `.onnx` for exactly that reason.

PyTorch is not in the published closure. It builds the artifacts and lives in
`nomikos_inference/export/` with the exporters, behind the `export` dependency
group; `tests/inference/unit/test_architecture_contract.py` asserts nothing under
`nomikos_inference/` imports it.

Training and vendored TensorFlow Calamari: [`docs/guides/learnings.md`](../docs/guides/learnings.md#calamari-training).

**Hub integration:** `hf://` weight-source resolution, Hub cache, and digest verification live *inside* this package at `nomikos_inference/hub/` - they are on the runtime path, so they ship in the published wheel. Publish-side tooling (staging tree, model cards, collection sync) is `nomikos_inference/publish/` and `scripts/hf/`, which never ship: the first sits in this directory so that it imports like everything else here, and is named in the `exclude` list of both build targets in `pyproject.toml` so that being in the directory does not make it part of the wheel. See `nomikos_inference/CONTEXT.md` for domain terminology and [`scripts/hf/README.md`](../scripts/hf/README.md) for the Hub publish runbook.

## Run from a source checkout

From the repository root, with the `inference` dependency group installed:

```bash
uv sync --group inference
PYTHONPATH=. python -m nomikos_inference.cli run --api-url http://localhost:8000
```

That is the same claim loop the installed `nomikos run` executes; there is no
server to start, because since ADR 0002 nothing here listens.

Environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERENCE_REGISTRY_PATH` | `nomikos_inference/registry.yaml` | Model catalog file |
| `HF_CACHE_ROOT` | `~/.nomikos/hf/cache` | Hub weight download cache |
| `HF_TOKEN` | unset | Required only for **private** or gated Hub repos; all nomikos inference repos are public |

Prefetch Hub weights without running inference:

```bash
PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v2 --registry-tag stable
```

## Contracts

Shared Pydantic schemas in `nomikos_inference/contracts/` define the wire format for inference endpoints:

- **Segment** - `SegmentRunResponse` (`nomikos_inference/contracts/segment.py`): page image in, blocks and line polygons out.
- **Transcribe** - `TranscribeRunResponse` / `TranscribeBatchRunResponse` (`nomikos_inference/contracts/transcribe.py`): line image(s) in, text and per-character confidence out.

Both tasks reference models by `registry_model_id` and optional `registry_tag` (default `stable`).

Job callbacks use a tagged output union: `output.kind` is either `segment` or `transcribe`, and `output.data` contains the matching result schema. Invalid callback shapes, such as a `done` callback with an `error` field, missing output, or a `task`/`output.kind` mismatch, are request-body validation failures. When an endpoint accepts `JobCallbackRequest` directly, FastAPI should return **422 Unprocessable Entity** for those cases. Use **404 Not Found** only for runtime lookups such as an unknown job id or unknown `registry_model_id`.

## Registry

`nomikos_inference/registry.yaml` lists available models and weight locations. Example entries:

- `greek-calamari-v1`, `armenian-calamari-v1`, `syriac-calamari-v2`, `coptic-calamari-v1` - transcribe,
  Calamari architecture, one Hub repo and pinned revision + digest each. Same
  architecture (40/60-filter CNN, two 200-unit BiLSTMs, line height 48; read off the
  Coptic and Syriac checkpoints, which agree on every structural field), one codec
  per script, so the entries differ only in codec, `weights_source`, and the pins.
- `blla-segment` - segment, BLLA `blla.onnx` weights
- `ppocr-segment` - segment, PP-OCRv6 detection `ppocrv6-det.onnx` weights from [segmentation-ppocrv6-det](https://huggingface.co/nomikos-project/segmentation-ppocrv6-det)
- `syriac-ppocr-v1` - transcribe, Syriac PP-OCR recognition `model.onnx` weights from [syriac-htr-ppocr_rec](https://huggingface.co/nomikos-project/syriac-htr-ppocr_rec)

In the platform pickers the segmenters appear as kraken (blla-segment) and ppocr (ppocr-segment); kraken is preselected and there is no Default entry.

Weights are resolved at runtime from the Hub cache (`~/.nomikos/hf/cache/`) or, in a source
checkout only, local bundled paths (`nomikos_inference/publish/artifacts/local/`, which an
install does not have).
New `hf://` entries should include both `hub_revision` and `artifact_sha256`; see
the migration note in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

**Adding a model:** step-by-step checklist in [`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).
Export through the one supported path (`scripts/hf/export_calamari_onnx.py`, the single exporter every
entry point delegates to) and prove the pinned artifact serves before publishing: the nightly `ml` lane
opens it through the real adapter and runs representative widths.

## Admission control

`run_model` enforces the `INFERENCE_*` limits before base64 decoding or model
loading, and `JobSubmitRequest` enforces the same ones at submission - so a page
is bounded on the way into the queue and again on the machine that runs it.
Defaults are: 160 MiB encoded image, 100 MiB decoded image, 128 MiB job payload,
200 million pixels, 64 MiB parameters, depth 8, 8,000,000 parameter items,
10,000 transcription lines, and 256 geometry points per line. Allowed image
formats default to `JPEG,PNG,TIFF,WEBP`. Operators may lower or raise these with
the corresponding `INFERENCE_MAX_*` environment variables; only trusted
deployment configuration should do so.

These are per-process input controls only. Queue admission, rate limiting across
replicas, and stale-lease recovery belong to the platform's queue, which is now
the only one.

There is nothing here to expose. ADR 0002 deleted the loopback service, its CORS
allowlist, its request-body middleware, and its per-process rate limiter,
because all four existed to make a listening socket on a researcher's machine
safe - and the socket is gone. An **inference agent** reaches the platform
outbound, authenticated as itself with a **device credential**, and accepts no
connections at all.

## Tests

```bash
uv run --group inference --group export pytest tests/inference tests/hf
```

Full-suite layout, `DATABASE_URL` caveats, and failure analysis: [`docs/guides/testing.md`](../docs/guides/testing.md).

## Related docs

- Nomikos platform API and job integration: [`nomikos/backend/README.md`](../nomikos/backend/README.md)
- Compose stack and env vars: [`infrastructure/docker-compose.yml`](../infrastructure/docker-compose.yml) and [`nomikos/README.md`](../nomikos/README.md)
