# Use and host Nomikos

Nomikos lets researchers upload manuscript pages, segment written lines,
generate model transcription drafts, correct them, collaborate, publish
selected documents, and export paired line images and text.

This guide covers the supported local stack and the current hosted topology.

## Docker Compose quick start

The repository's Compose file is a development stack, not a hardened
internet-facing deployment. It uses local Postgres, local filesystem media,
bind mounts, reload mode, and development seed data.

Prerequisites:

- Git
- Docker Desktop with Compose
- about 10 GB of free disk space

From the repository root:

```bash
git clone https://github.com/kkkamur07/nomikos.git
cd nomikos
cp infrastructure/.env.compose.example infrastructure/.env
```

Replace the placeholders in `infrastructure/.env`:

```text
POSTGRES_PASSWORD
JWT_SECRET
INFERENCE_WEBHOOK_SECRET
```

Start the stack:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

Open <http://localhost:5173>. Development seed credentials are
`dev@example.com` / `dev-pass-123`.

| Service           | Address                      |
| ----------------- | ---------------------------- |
| Editor            | <http://localhost:5173>      |
| Platform API      | <http://localhost:8000>      |
| API documentation | <http://localhost:8000/docs> |
| Postgres          | `127.0.0.1:5433`             |

Compose runs no inference service. Models run in an **inference agent** you
start yourself (below), which reaches the platform outbound and needs no
published port. The first inference run downloads public weights into
`~/.nomikos/hf/cache`.

```bash
docker compose -f infrastructure/docker-compose.yml ps
curl -s http://localhost:8000/health | python -m json.tool
docker compose -f infrastructure/docker-compose.yml logs -f
docker compose -f infrastructure/docker-compose.yml down
```

## Run services individually

For service development, install Python 3.11 to 3.12, `uv`, Node.js 20+, and
Docker for Postgres:

```bash
uv sync --group platform --group inference
cp nomikos/backend/core/.env.example nomikos/backend/core/.env
docker compose -f infrastructure/docker-compose.yml up db -d
```

Start the platform API:

```bash
cd nomikos
uv run --project .. --group platform \
  alembic -c infrastructure/alembic.ini upgrade head
uv run --project .. --group platform \
  uvicorn backend.core.app:create_app --factory --reload --port 8000
```

Start the frontend:

```bash
cd nomikos/frontend
npm install
cp .env.local.example .env.local
npm run dev
```

This path does not start the asynchronous job worker. Start the platform worker
too, and an **inference agent**, when testing queued jobs.

## Local inference agent

Models run in the **inference agent**: one command-line program, distributed as
the **published package** `nomikos-inference`, that a researcher starts in a
terminal. A hosted worker runs the same package (ADR 0002), so there is no
separate local build and no per-OS installer.

```bash
uv tool install nomikos-inference
nomikos pair          # links this machine to your account
nomikos run           # takes pages from the queue until you stop it
```

No flags and no uv version floor: ADR 0006 replaced PyTorch with ONNX Runtime,
which publishes one CPU wheel per platform, so there is no accelerator variant to
pin against. See [`nomikos_inference/README.md`](../../nomikos_inference/README.md#install) for
what the instruction used to be and why it mattered.

The agent opens no port and accepts no connection. It asks the platform for a
page, downloads that one page image through a short-lived signed link, runs the
model, and reports the result through the platform's job callback - all
outbound, so nothing on the researcher's machine has to be reachable from a
browser, a proxy, or a VPN. Weights are cached under `~/.nomikos/hf/cache`;
the device credential lives at `~/.nomikos/device.json`.

Point it at a platform other than the hosted one with `NOMIKOS_API_URL` or
`--api-url`:

```bash
NOMIKOS_API_URL=http://localhost:8000 nomikos pair
NOMIKOS_API_URL=http://localhost:8000 nomikos run
```

From a source checkout, `uv run --group inference python -m nomikos_inference.cli run`
is the same entry point without installing the package first.

Where a job runs is decided once, at submission, from the account-level **host
preference** ("use my computer when it is available") and whether that host has
**capacity** - whether an agent for it was seen recently. There is no per-job
toggle, and an agent that is not running is an announced state rather than a
failure: the job goes to the cloud and says so.

## Supabase-backed development

1. Create a Supabase project.
2. Create a private Storage bucket named `document-media`.
3. Fill `nomikos/backend/core/.env.supabase.example`.
4. Set service secrets in `infrastructure/.env`.
5. Provision the least-privilege roles in
   [`../deployment/database-roles.md`](../deployment/database-roles.md).
6. Run migrations with the direct migrator connection:

```bash
./scripts/platform/migrate_supabase.sh
```

7. Start the overlay:

```bash
docker compose -f infrastructure/docker-compose.yml -f infrastructure/docker-compose.supabase.yml up --build
```

Runtime traffic uses configured pooler connections; migrations use the direct
connection. Keep migration URLs and Storage service keys server-side.

## Hosted production

The current production topology is manual:

- Vercel: landing page, Next.js editor, and request/response FastAPI API;
- Supabase: Postgres and private `document-media` Storage;
- persistent Docker host: platform worker, and a hosted inference agent if
  cloud inference is enabled.

There is no complete one-click hosting template. Operators must configure DNS,
secrets, roles, migrations, TLS, backups, monitoring, and worker supervision.

The documented default is local inference through the researcher-installed
agent. Vercel cannot run a long-lived model process. If cloud inference is
enabled, these persistent processes are required:

```text
platform-worker   -> runs the job types the platform executes itself
inference agent   -> claims pages from the platform, runs models, calls back
```

The hosted agent is the same package a laptop runs, with a **service
credential** in `NOMIKOS_SERVICE_TOKEN` instead of a device token. It needs no
inbound address and no database access.

Cloud inference is disabled in the documented Vercel defaults. Enable it only
after deploying and verifying the complete claim and callback path.

See the detailed operator runbooks:

- [`../deployment/production.md`](../deployment/production.md)
- [`../deployment/supabase.md`](../deployment/supabase.md)
- [`../../nomikos_inference/README.md`](../../nomikos_inference/README.md)
