# Catalog SQL changes

One SQL file per production catalog change, named `<date>-<what>.sql`.

Rules:

- Every file is idempotent (safe to run twice) and wrapped in `begin;`/`commit;`.
- Raw inserts must pass `gen_random_uuid()` for `id` (the PK has an app-side default only).
- Display name lives in `name`; dispatch reads `artifact_ref` (`registry://<registry id>?tag=stable`), so names are free to change while registry ids stay fixed.

Apply with `scripts/platform/apply_catalog_sql.sh`:

```bash
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<file>.sql
bash scripts/platform/apply_catalog_sql.sh scripts/platform/catalog/<file>.sql --apply
```

The first run only looks (shows the database, the current rows, the SQL).
The second run sends the file over stdin and shows the rows again.

Container, database, and user names from the server or env overrides must
match `^[A-Za-z0-9_.-]+$`. Anything else is refused with exit code 5
before it can reach a remote command.

## Writing a catalog file

A catalog file is a record of what ran: once a file has run on production
it is never edited. Fix forward with a new dated file.

A new file should converge: update the intended row by `artifact_ref`,
remove a legacy duplicate row when one can exist, and use
`on conflict (name) do update` (not `do nothing`) when the declared params
must win over what is already stored. Every file must be safe to run twice
and stay wrapped in `begin;`/`commit;`.

The 2026-09-20 file uses guarded renames
(`update ... where ... and not exists ...`) and `do nothing` because
production was known to hold no duplicates (checked with the look-only run
first).

## Order on production

Apply the files in date order; each one expects the state the previous one left:

1. `2026-09-20-ppocr-rows-and-segmenter-display-names.sql`: segmenters named
   `kraken` and `ppocr`, the `ppocr` and `syriac-ppocr-v1` rows inserted.
2. `2026-10-11_model_names_v2.sql`: renames to `seg-v1`, `seg-v2`, `greek-v1`,
   `armenian-v1`, `coptic-v1`, `syriac-v2` and retires `syriac-calamari-v2`. It
   refuses (nothing applied) unless all seven 2026-09-20 rows are present, so it
   refuses a second run and refuses a database that never got the first file.
