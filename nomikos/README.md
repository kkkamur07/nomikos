# nomikos

AI-assisted transcription platform for manuscript pages. `nomikos/` is the production app root: the Postgres-backed backend, Next.js frontend, and migration infrastructure live here, while the repository-level `model/` workspace remains outside nomikos.

See `CONTEXT.md` for domain glossary and [`docs/README.md`](../docs/README.md) for guides, deployment, and audits.

Public-facing documentation:
[overview](../README.md) ·
[use and hosting](../docs/guides/using-and-hosting.md) ·
[architecture](../docs/architecture.md)

## Data layout

Structured data lives in Postgres. Uploaded Document part images are stored
under `MEDIA_ROOT` (default `nomikos/backend/media`) or in Supabase Storage
when `STORAGE_BACKEND=supabase`, and referenced by `document_parts.image_key`.
Exports, annotation snapshots, and transcriptions are Postgres rows or
on-demand artifacts, not files under `data/`.

The `data/` directory and `NOMIKOS_DATA_ROOT` were part of an earlier on-disk
layout that the current backend does not read. Ignore both.

## Setup

### Backend environment

```bash
cd nomikos
uv sync --project .. --group platform
cp backend/core/.env.example backend/core/.env
```

### Frontend

```bash
cd nomikos/frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL defaults to http://localhost:8000
npm install
```

## Development (two terminals)

### Terminal 1 - API

```bash
docker compose -f infrastructure/docker-compose.yml up db -d
cd nomikos
uv run --project .. --group platform alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform uvicorn backend.core.app:create_app --factory --reload
```

Default API URL: `http://127.0.0.1:8000`

### Terminal 2 - Frontend

```bash
cd nomikos/frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Docker (one command)

Ensure ports **5173**, **8000**, and **5433** are free. Run Compose from the repository root:

```bash
cd ..
docker compose -f infrastructure/docker-compose.yml up --build      # foreground; Ctrl+C stops the stack
docker compose -f infrastructure/docker-compose.yml up --build -d   # detached background
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:5173](http://localhost:5173) |
| API | [http://localhost:8000](http://localhost:8000) |
| Postgres | `localhost:5433` |

Platform media is mounted at `nomikos/backend/media/`. The existing `data/` folder is not mounted or migrated by the production platform relocation.

Compose runs no inference service. The repository-level [`nomikos_inference/`](../nomikos_inference/) package supplies the contracts the platform imports and the model runtime an agent executes; the platform owns the only job queue (ADR 0003). See [`nomikos_inference/README.md`](../nomikos_inference/README.md).

Useful after `-d`: `docker compose -f infrastructure/docker-compose.yml ps`, `docker compose -f infrastructure/docker-compose.yml logs -f`, `docker compose -f infrastructure/docker-compose.yml down`.

- `NEXT_PUBLIC_API_BASE_URL` (build/runtime env) - URL the browser uses (`http://localhost:8000`)

Rebuild the frontend image if you change `NEXT_PUBLIC_API_BASE_URL`.

### Bumping the Docker version

**Release source of truth:** [`VERSION`](VERSION) at the repo root of `nomikos/`.

1. Edit `VERSION` (semver, one line - e.g. `0.2.1`).
2. From the repository root, export it and rebuild so Compose tags images correctly:

```bash
cd ..
export NOMIKOS_VERSION=$(cat nomikos/VERSION)
docker compose -f infrastructure/docker-compose.yml up --build -d
```

`infrastructure/docker-compose.yml` builds `nomikos-api:${NOMIKOS_VERSION}` and `nomikos-frontend:${NOMIKOS_VERSION}` and passes `APP_VERSION` into both Dockerfiles (API `/health` and frontend build).

3. Verify:

```bash
curl -s http://localhost:8000/health | python -m json.tool
docker images 'nomikos-*'
```

**When to bump:** any release you want to distinguish in image tags or `/health` - not required for every code change during dev (rebuild with the same `NOMIKOS_VERSION` is fine).

**Important:** set `NOMIKOS_VERSION` in a shell profile or in `infrastructure/.env`
next to the Compose file so the image tag always matches `VERSION`. The current
Compose fallback is `0.3.3`; do not rely on it for a release unless it has been
updated to match `VERSION`.

## Environment variables

| Variable | File | Default | Purpose |
|----------|------|---------|---------|
| `DATABASE_URL` | `backend/core/.env` | local Compose DB URL | Async platform database URL |
| `SYNC_DATABASE_URL` | `backend/core/.env` | local Compose DB URL | Alembic database URL |
| `JWT_SECRET` | `backend/core/.env` | development secret | Auth token signing key |
| `CORS_ORIGINS` | `backend/core/.env` | `http://localhost:3000,http://localhost:5173` | Allowed browser origins |
| `MEDIA_ROOT` | `backend/core/.env` | `nomikos/backend/media` | Uploaded document part media |
| `NEXT_PUBLIC_API_BASE_URL` | `frontend/.env.local` | `http://localhost:8000` | Frontend → API URL |

## Inference Catalog

`nomikos_inference/registry.yaml` is the runtime model catalog. The development seed
creates `InferenceModel` rows with `registry://<model-id>?tag=stable` artifact
references and project-level bindings. It seeds two segment rows (`blla-segment`,
`ppocr-segment`) and four transcribe rows (`greek-calamari-v1`, `armenian-calamari-v1`,
`coptic-calamari-v1`, `syriac-ppocr-v1`); the pickers show short display names
(`seg-v1`, `seg-v2`, `greek-v1`, …) per [`docs/inference/model-catalog-names.md`](../docs/inference/model-catalog-names.md).

```bash
uv run --group platform python scripts/platform/seed_dev_inference.py
```

Model bytes are resolved by the inference service from the registry's
`package://`, `hf://`, or optional `file://` source. See
[`docs/inference/adding-inference-models.md`](../docs/inference/adding-inference-models.md).

## Tests (TDD)

```bash
# From repository root
uv run --group platform --group inference pytest tests/nomikos/unit
uv run --group platform --group inference pytest tests/nomikos/integration -m "not ml"
```

See [`docs/guides/testing.md`](../docs/guides/testing.md) for the ML lane and full-suite commands.

## OpenAPI

```bash
python scripts/platform/export_openapi.py

cd nomikos/frontend
npm run codegen:api
```

Generated schema types: `frontend/src/api/schema.d.ts`. App-facing aliases live
in `frontend/src/api/client.ts`.

