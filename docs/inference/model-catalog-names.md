# Model catalog display names

`inference_models.name` is the picker text; dispatch resolves
`artifact_ref` (`registry://<registry_id>?tag=stable`), so renames never
touch serving. Scheme: `seg-vN` for segmenters, `<language>-vN` for
transcribers.

| Catalog name | Registry id | Task |
|---|---|---|
| `seg-v1` | `blla-segment` | segment |
| `seg-v2` | `ppocr-segment` | segment |
| `greek-v1` | `greek-calamari-v1` | transcribe |
| `armenian-v1` | `armenian-calamari-v1` | transcribe |
| `coptic-v1` | `coptic-calamari-v1` | transcribe |
| `syriac-v2` | `syriac-ppocr-v1` | transcribe |

Production: apply `scripts/platform/catalog/2026-10-11_model_names_v2.sql`
once with psql; the guard raises if the 7 pre-rename rows are absent, so a
second run fails instead of renaming twice. Bindings on the retired
`syriac-calamari-v2` row move to `syriac-v2` before it is deleted.
`jobs.model_id` is ON DELETE SET NULL, so every job that ran on
`syriac-calamari-v2` ends up with `model_id = NULL` (the script prints the
count first); jobs on the six renamed rows keep their UUID references and
are unaffected.
