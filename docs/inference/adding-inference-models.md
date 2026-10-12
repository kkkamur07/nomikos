# Adding an inference model

End-to-end checklist for shipping a new **segment** or **transcribe** model to production. Three stores must stay aligned:

| Store | Purpose | Updated by |
|-------|---------|------------|
| **`nomikos_inference/registry.yaml`** | Runtime catalog: task, architecture, weights URI, `host_eligibility` | Git + API deploy |
| **Postgres `inference_models`** | Editor model picker (`GET /inference/models`) | Seed script or DB insert |
| **Hub (or bundled weights)** | Checkpoint bytes | `publish_model.py` or `nomikos_inference/publish/artifacts/local/` |

Vocabulary: [`nomikos_inference/CONTEXT.md`](../../nomikos_inference/CONTEXT.md).

## Overview

```
Train / export checkpoint
    → stage under nomikos_inference/publish/artifacts/staging/…
    → publish to Hub (hf://…)
    → add entry to nomikos_inference/registry.yaml
    → add InferenceModel row in Postgres
    → deploy platform API
    → release the published package so agents carry the new entry
    → first run downloads weights into ~/.nomikos/hf/cache/
```

**Where the Registry lives at runtime:** an **inference agent** reads the
`registry.yaml` that ships inside its own **published package**
(`INFERENCE_REGISTRY_PATH` overrides it, which is how a source checkout runs an
unreleased entry). It does not sync one from the platform - the claim it
receives names a **registry model id** and **registry tag**, and the agent
resolves both locally. The deployed API still serves the file at
`GET /inference/v1/registry` (public YAML + `ETag`), which is a convenient way
to confirm a deploy shipped the entry you expect.

**Weights:** resolved lazily from `weights_source` on the first run that needs
them - the same path on a researcher's laptop and on a hosted worker, because
both run the same package. See [`scripts/hf/README.md`](../../scripts/hf/README.md).

---

## 1. Choose identifiers

Follow the **registry model id** convention:

```
{script}-{architecture}-{model_version}
```

Examples: `greek-calamari-v1`, `armenian-calamari-v1`, `syriac-calamari-v2`, `blla-segment`.

| Field | Meaning | Example |
|-------|---------|---------|
| **script** | Writing system / language family | `greek`, `armenian`, `syriac` |
| **architecture** | Runtime adapter | `calamari`, `blla-segment` |
| **model_version** | Family generation | `v1`, `v2` |
| **registry tag** | Named weight snapshot | `stable` (default) |

**Hub repo slug** (separate from registry model id): `{script}-htr-{architecture}` → `syriac-htr-calamari`.

**Platform `artifact_ref`** (Postgres): `registry://<registry_model_id>?tag=<registry_tag>`
Example: `registry://greek-calamari-v1?tag=stable`.

---

## 2. Publish weights

### Calamari transcribe (Hub)

1. Copy the converted `.pt` checkpoint into the **Hub staging tree**, then
   export the graph beside it. The `.onnx` is what the runtime loads (ADR 0006);
   the `.pt` is published with it as the export input, and `publish_model.py`
   refuses a staging directory that has no `.onnx` in it:

   ```
   nomikos_inference/publish/artifacts/staging/models/{script}/calamari/{model_version}/{registry_tag}/
     best.pt
     best.onnx
   ```

   ```bash
   uv run --group export python scripts/hf/export_calamari_onnx.py \
     --checkpoint nomikos_inference/publish/artifacts/staging/models/{script}/calamari/{model_version}/{registry_tag}/best.pt \
     --destination nomikos_inference/publish/artifacts/staging/models/{script}/calamari/{model_version}/{registry_tag}/best.onnx
   ```

   The script prints the SHA-256 of both files plus the classes, line height,
   recurrent depth and opset, which is what step 4 below copies into the
   Registry. Pass `--repo-id` and `--revision` instead of `--checkpoint` to
   export straight from a published Hub commit.

   Then check the artifact against the graph it was traced from, on real line
   crops, before publishing it. ADR 0006 exists because a published artifact had
   silently drifted from its exporter, and this is the measurement that would
   have caught it:

   ```bash
   uv run --group export python scripts/hf/verify_calamari_parity.py \
     --checkpoint .../best.pt --onnx .../best.onnx --lines path/to/line/crops
   ```

   It exits non-zero if any greedy decode differs between the two.

   This script is the one supported export path: it calls the single exporter in
   `nomikos_inference/export/calamari`, which the trainer-side entry point
   (`src/models/calamari/export.py`) delegates to, so all three spellings produce
   the same graph. A published ONNX must also pass the real-weights check: the
   `ml` lane opens the pinned artifact through the real adapter and runs
   representative widths (`tests/inference/integration/test_published_package.py`), which runs nightly
   because it downloads real weights, so run it before every publish.

2. Dry-run, then upload:

   ```bash
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script syriac \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe

   export HF_TOKEN=hf_...
   PYTHONPATH=. python scripts/hf/publish_model.py \
     --script syriac \
     --architecture calamari \
     --model-version v1 \
     --registry-tag stable \
     --task transcribe \
     --upload
   ```

3. Optional: warm cache locally:

   ```bash
   PYTHONPATH=. python scripts/hf/fetch_model.py syriac-calamari-v2 --registry-tag stable
   ```

4. Optional: add the Hub repo to [`nomikos_inference/publish/collection.yaml`](../../nomikos_inference/publish/collection.yaml) and run `sync_collection.py`.

Full publish runbook: [`scripts/hf/README.md`](../../scripts/hf/README.md).

### Record immutable provenance for new Hub entries

Before adding an `hf://` source to the Registry, resolve the public Hub tag to
its 40-character commit and read the SHA-256 **of the published `.onnx`**, which
is the file the runtime opens. It is not the digest of `best.pt`: ADR 0006 moved
the pin from the native checkpoint to the exported graph, and the checkpoint is
still published beside it at the same revision, so a repo listing offers you two
plausible digests and only one of them is checked. Record both values in the
same Registry version entry. The runtime uses `hub_revision` when present and
verifies `artifact_sha256` before loading or reusing cache contents.

Existing `hf://` entries without both provenance fields are accepted for
migration compatibility and resolve from their `weights_source` tag. Do not add
new unpinned entries; update legacy entries with both fields when their
published artifact provenance is available.

For a public model repo, `huggingface_hub` can report both values without
`HF_TOKEN`:

```bash
uv run --group inference python -c \
  "from huggingface_hub import HfApi; print(HfApi().model_info('<namespace>/<repo>', revision='stable', files_metadata=True))"
```

`HF_TOKEN` is only required by the publish command, not by model download or
metadata lookup for public repositories.

### BLLA segment

The BLLA runtime loads `blla.onnx` from the registry-pinned
`segmentation-blla` Hub artifact, exported from `blla.safetensors` the same way:

```bash
uv run --group export python -c "
from pathlib import Path
from nomikos_inference.export.blla import export_blla_onnx
d = Path('nomikos_inference/publish/artifacts/staging/models/segmentation/blla/v1/stable')
export_blla_onnx(d / 'blla.safetensors', d / 'blla.onnx', example_width=64)"
```

The inference image does not install the Kraken Python package and nothing in
the repository does: the parity harness Kraken was the oracle for was not
restored with the rest of the ONNX path, because the Torch graph in
`nomikos_inference/export/` is a closer oracle and is already here.

**Verify the export before publishing, not after.** `blla.onnx` was published
once from a pre-fix exporter and the mistake survived a whole ADR - see ADR 0006.
`uv run --group export pytest tests/export -q` compares graph against artifact on
real weights.

### Local / offline dev

Point `weights_source` at bundled files under `nomikos_inference/publish/artifacts/local/`:

```yaml
weights_source: file://local/syriac/calamari/v1/stable/best.pt
```

(`file://local/…` paths are relative to `nomikos_inference/publish/artifacts/`, which
is in the package directory but excluded from the wheel, so an installed package
resolves none of them and says so.)

---

## 3. Register in `nomikos_inference/registry.yaml`

Add a model block under `models:`:

```yaml
models:
  syriac-calamari-v2:
    task: transcribe          # transcribe | segment
    architecture: calamari    # calamari | blla-segment
    device: cpu               # compute hint (cpu | cuda)
    host_eligibility: local   # local | remote | any
    line_crop: polygon-white  # required on transcribe: the crop function this model was trained on (ADR 0007)
    line_crop_padding: 12     # required on transcribe: px the crop was widened by before masking (Greek uses 0)
    versions:
      stable:
        weights_source: hf://<namespace>/syriac-htr-calamari@stable
        hub_revision: <40-character-resolved-Hub-commit>
        artifact_sha256: <sha256-of-best.onnx>
```

When present, the two provenance fields must be supplied together for an
`hf://` **weights source**. `weights_source` retains the **registry tag** for
readability; a pinned `hub_revision` prevents a changed Hub tag from changing
the bytes inference loads until a reviewed Registry update replaces the
revision and digest. For packaged Kraken assets, record `artifact_sha256`
without `hub_revision` so the packaged `.mlmodel` is also verified before
Kraken loads it.

**`host_eligibility`**

| Value | Behaviour |
|-------|-----------|
| `local` | May run on an **inference agent** on the researcher's machine |
| `remote` | Hosted worker only (GPU / large models) |
| `any` | Either host; **host preference** and **capacity** fix one at submission |

Run unit tests:

```bash
uv run --group inference --group export pytest tests/inference/unit/test_registry.py -q
```

---

## 4. Register in the platform catalog (Postgres)

The editor lists models from **`inference_models`**, not directly from `registry.yaml`. Each row needs:

| Column | Value |
|--------|-------|
| `name` | Picker display name ([language-vN scheme](model-catalog-names.md), unique) |
| `provider` | Architecture the row runs on: `calamari` for Calamari HTR, `kraken` for BLLA segmentation, `ppocr` for PP-OCR. A label for humans reading the table; nothing dispatches on it, and `list_models` only orders by it |
| `task` | `segment` or `transcribe` |
| `artifact_ref` | `registry://<registry_model_id>?tag=stable` |
| `default_params` | JSON, e.g. `{"device": "cpu"}` |

### Development

Add a `(registry_id, display_name)` pair to `SEGMENT_MODELS` or `TRANSCRIBE_MODELS` in
[`scripts/platform/seed_dev_inference.py`](../../scripts/platform/seed_dev_inference.py) and run:

```bash
uv run --group platform python scripts/platform/seed_dev_inference.py
```

### Production

Insert the row (migration, admin script, or one-off SQL). `name` is the picker display name per the [language-vN scheme](model-catalog-names.md); `artifact_ref` carries the registry model id so resolution stays consistent with cloud jobs.

Optional: create a **ModelBinding** at project, document, or part scope so the new model is the default for a workspace.

### Registering the catalog row in production

Production Postgres is a Coolify-managed container, so catalog rows go
through `scripts/platform/apply_catalog_sql.sh` with a dated SQL file in
`scripts/platform/catalog/` (see the catalog README for the file rules).
Look first, then apply:

```bash
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<date>-<what>.sql
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<date>-<what>.sql --apply
```

The script finds the production database itself (it must already hold the
known models), shows the current `inference_models` rows, and sends the
file over stdin, so nothing is written on the server.

---

## 5. Tests and CI

Update or add coverage as needed:

| Area | Tests |
|------|-------|
| Registry parsing | `tests/inference/unit/test_registry.py` |
| Hosted registry endpoint | `tests/nomikos/integration/test_inference_registry.py` |
| Hub resolve / prefetch | `tests/hf/` |
| Platform catalog | `tests/nomikos/integration/test_inference_catalog.py` |
| ML integration (optional) | `tests/nomikos/integration/ml/` |

```bash
uv run --group inference --group export pytest tests/inference tests/hf -q
uv run --group platform --group inference pytest tests/nomikos/integration/test_inference_registry.py -q
```

---

## 6. Deploy

1. **Deploy platform API** - ships the updated `nomikos_inference/registry.yaml` in the container (`/app/nomikos_inference/registry.yaml`). The new endpoint is live:

   ```bash
   curl -s https://api.example.com/inference/v1/registry
   ```

2. **Publish the package.** A registry entry only reaches an **inference agent**
   inside the wheel it ships in, so a new model is a PyPI release - one wheel,
   one runner, no per-OS installer build. See
   [`nomikos_inference/README.md`](../../nomikos_inference/README.md#releasing-and-what-that-changed-about-security-patching).

3. **Roll out the hosted agent**, if cloud inference should serve the model, by
   upgrading the package on its host. It runs the same wheel a laptop does.

4. **Local inference** - researchers pick the model up with
   `uv tool upgrade nomikos-inference`, or automatically at the next **launch
   check** if you raise the platform's **version floor**
   (`INFERENCE_AGENT_MIN_VERSION`) past the last release without it. Weights
   download on the first run that needs them.

   Verify from a source checkout before releasing, with the registry read from
   the tree rather than from an installed wheel:

   ```bash
   NOMIKOS_API_URL=http://localhost:8000 \
     uv run --group inference python -m nomikos_inference.cli run --exit-when-empty
   ```

---

## 7. Verify end-to-end

- [ ] `curl …/inference/v1/registry` returns the new model id
- [ ] Authenticated `GET /inference/models` lists the new **InferenceModel**
- [ ] `host_eligibility` in the shipped `registry.yaml` matches what the model can actually run on
- [ ] Cloud job: hosted agent claims a segment/transcribe page → job completes with expected output
- [ ] Local path: with **host preference** on and a paired machine running `nomikos run`, the job announces the local host, completes, and leaves weights under `~/.nomikos/hf/cache/<registry_model_id>/stable/`

---

## Quick reference: new Calamari transcribe model

1. Stage `best.pt` → `nomikos_inference/publish/artifacts/staging/models/{script}/calamari/v1/stable/`, then export `best.onnx` beside it
2. `publish_model.py … --upload` (both files go up at one revision)
3. Add block to `nomikos_inference/registry.yaml` with `hf://…` **weights_source**, `hub_revision`, and the `artifact_sha256` **of `best.onnx`**
4. Add a `(registry_id, display_name)` pair to `TRANSCRIBE_MODELS` in `scripts/platform/seed_dev_inference.py`, which upserts the `InferenceModel` with `provider: calamari` and `artifact_ref: registry://{id}?tag=stable`
5. Run tests → deploy API → publish the package
6. Agents carry the entry once upgraded; weights download on first use

Re-pointing an existing id at a retrained model is the same six steps: the
registry entry is edited in place rather than added, and the catalog row already
exists, so the seed script rewrites it. What the old revision holds is
irrelevant to the new one, and it keeps resolving for anything still pinned to
it, which is why a stale pin fails by transcribing with last generation's
weights rather than by erroring.

## Related docs

- [`nomikos_inference/README.md`](../../nomikos_inference/README.md) - published package, CLI, and registry
- [`scripts/hf/README.md`](../../scripts/hf/README.md) - Hub publish and prefetch
- [`README.md`](../../README.md#self-hosting-and-local-inference) - local vs cloud inference architecture
- Issue **036** - registry id naming migration (historical; see git history on branches with `issues/done/`)
