# Release: syriac-ppocr-v1 (2026-09-20)

First publication of the PP-OCRv6 Syriac recognition model (`syriac-ppocr-v1`),
pinned in `nomikos_inference/registry.yaml` and verified end to end through the
registry path. Evidence for the export and the adapter lives in
`docs/inference/ppocr-rec-onnx-parity-2026-09-20.md` and
`docs/inference/ppocr-rec-adapter-2026-09-20.md`.

> Note (2026-10-11): the catalog row this release created was renamed to
> `syriac-v2` under the [language-vN scheme](model-catalog-names.md). The
> registry id `syriac-ppocr-v1` and the SQL below are unchanged history.


## What was published

| Field | Value |
|-------|-------|
| Hub repo | `nomikos-project/syriac-htr-ppocr_rec` |
| Tag | `stable` |
| Revision (`stable` points to) | `19e8d1f23fd7a2bc272d303fb02616bbdcfcd08e` |
| Files at that revision | `model.onnx` (63,524,245 bytes), `ppocr-syriac.safetensors` (63,779,644 bytes), `README.md` (generated model card) |
| `model.onnx` sha256 | `a839d951ea443f3fb04acbbde88ad263b984f39a4e50b49adb78e9ad742c8a22` (matches the export artifact; re-downloaded from the Hub at the pinned revision and re-hashed, match confirmed) |
| Export input sha256 | `2545eca289525c534c55f96f9941f6eddccae059cfa6fd4bf0042a2a7063e17e` (the `ppocr-syriac.safetensors` beside it, also recorded as `source_sha256` in the ONNX metadata) |

Published with `scripts/hf/publish_model.py --upload` from the Hub staging
tree (`nomikos_inference/publish/artifacts/staging/models/syriac/ppocr_rec/v1/stable/`).
The staging copies were left untracked and deleted after the upload, so no
binary is committed. Note: the first upload attempt failed on the XET upload
backend (`Operation not permitted` writing the XET staging cache); the retry
with `HF_HUB_DISABLE_XET=1` uploaded both files as plain LFS and succeeded.

## Registry block

Added after `syriac-calamari-v2` in `nomikos_inference/registry.yaml`:

```yaml
  syriac-ppocr-v1:
    task: transcribe
    architecture: ppocr_rec
    device: cpu
    host_eligibility: local
    line_crop: polygon-white
    line_crop_padding: 0
    versions:
      stable:
        weights_source: hf://nomikos-project/syriac-htr-ppocr_rec@stable
        hub_revision: 19e8d1f23fd7a2bc272d303fb02616bbdcfcd08e
        artifact_sha256: a839d951ea443f3fb04acbbde88ad263b984f39a4e50b49adb78e9ad742c8a22
```

The `line_crop_padding: 0` plus the comment in the file state the full story:
kraken extracts the line polygon without a margin, and the 16 px white
horizontal padding is applied inside the adapter from the ONNX metadata
(`pad 16`, `pad_fill 255`, line height 96, read back from the staged graph).
That padding path is unmeasured end to end on platform crops. Sanity figures
only, not a benchmark: the Calamari entry measures CER 0.29 on EastSyriac
Chapter4, while this model's Torch greedy decode measured 0.160 on the same
pages (display order on both sides).

The shipped-registry snapshot test
(`test_the_shipped_registry_states_a_crop_for_every_transcribe_model`) was
extended with the fifth entry; it still asserts every transcribe model states
its crop.

## Local verification

- `PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/inference/unit -q -p no:cacheprovider`: 252 passed, 4 skipped.
- Real transcription through the registry path
  (`nomikos_inference.jobs.runner.run_model`, task `transcribe`,
  `syriac-ppocr-v1@stable`) on
  `tests/fixtures/manuscripts/syriac/transcribe_line.jpg`: weights resolved
  from the Hub into the Hub cache with the sha check passing, and returned
  non-empty Syriac text (42 characters, line confidence 0.7174, one confidence
  per character):

  `ܥܐܝ ܐM̄ܚ ܫܝ̈ܝܝܘܢ ܗ̄ܚܥ ܘܿܗܘܥ ܥ̄ܟ ܚܝܗܬܘܟ ܐ ܬ`

  (This sandbox denies writes to `~/.nomikos`, so the run used the supported
  `HF_CACHE_ROOT` override pointing at a temp dir; the resolution, download,
  and sha-check code path is identical.)

## Remaining production steps

Checklist order follows `docs/inference/adding-inference-models.md` steps 4
to 6. The dev seed script already lists `syriac-ppocr-v1` in
`TRANSCRIBE_MODELS`, and the provider mapping already covers the architecture
(`RegistryArchitecture.ppocr_rec` to `"ppocr"`).

1. Dev catalog row. Run the seed script (upsert, safe to re-run):

   ```bash
   uv run --group platform python scripts/platform/seed_dev_inference.py
   ```

2. Production catalog row. The user runs this with psql against the
   production database (columns per
   `nomikos/backend/ml/infrastructure/orm_models.py`, table
   `inference_models`):

   The `id` column has only an app-side default (`uuid.uuid4` in the ORM, no
   server default), so raw SQL must supply it explicitly via `gen_random_uuid()`.

   ```sql
   INSERT INTO inference_models (id, name, provider, task, artifact_ref, default_params)
   VALUES (gen_random_uuid(), 'syriac-ppocr-v1', 'ppocr', 'transcribe', 'registry://syriac-ppocr-v1?tag=stable', '{"device": "cpu"}')
   ON CONFLICT (name) DO UPDATE SET
     provider = EXCLUDED.provider,
     task = EXCLUDED.task,
     artifact_ref = EXCLUDED.artifact_ref,
     default_params = EXCLUDED.default_params;
   ```

3. Platform API redeploy. The container carries `registry.yaml`, so deploy the
   API and confirm the entry shipped:

   ```bash
   curl -s https://api.example.com/inference/v1/registry
   ```

4. `nomikos-inference` PyPI release. The registry entry only reaches inference
   agents inside the wheel it ships in. See `nomikos_inference/README.md`
   (releasing section).

5. Hosted worker upgrade and restart, per
   `docs/deployment/cloud-inference-worker.md` (a running worker holds the old
   registry; `git pull` on the box does not change it):

   ```bash
   uv tool install --reinstall "nomikos-inference==<version containing the change>"
   sudo systemctl restart 'nomikos-worker@*'
   ```

   Then run that doc's namespace check to confirm the worker resolves the new
   weights source.

6. End-to-end checks from the adding-models checklist: authenticated
   `GET /inference/models` lists `syriac-ppocr-v1`; a cloud transcribe job
   completes; a local run with host preference leaves weights under
   `~/.nomikos/hf/cache/syriac-ppocr-v1/stable/`.
