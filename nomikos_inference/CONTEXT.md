# nomikos_inference

Standalone inference service for manuscript **segment** and **transcribe** jobs. Loads models from a file-based **Registry** and resolves weight files at runtime.

## Language

**Registry**:
The YAML catalog (`registry.yaml`) that lists runnable models by **registry model id**, task, architecture, and weight location.
_Avoid_: Model catalog (ambiguous with platform Postgres catalog)

**Registry model id**:
The stable runtime key for one inference model (e.g. `greek-calamari-v1`).
Language-specific format: `{script}-{architecture}-{model_version}`;
script-agnostic models may use a task-specific id such as `blla-segment`.
_Avoid_: model_id (ambiguous with platform UUID), Hub repo slug

**Hub repo slug**:
The single-segment name of a **Hub model repo**, derived from task and architecture: HTR repos use `{script}-htr-{architecture}` (e.g. `greek-htr-calamari`), while the script-agnostic BLLA segmenter uses `segmentation-blla`. The model card title may be human-friendly; the slug stays mechanical.
_Avoid_: ancient-greek-htr (fine as display title, not slug), registry model id (includes model_version)

**Model version**:
The family generation of a model (`v1`, `v2`) - distinct from **registry tag** (`stable`). Encoded in **registry model id** and local staging path, not in **Hub repo slug** when architecture already disambiguates repos.
_Avoid_: version (too generic), release

**Script**:
The writing system / language family the model targets (e.g. `greek`, `syriac`). Script-agnostic models use a reserved task namespace such as `segmentation`. It is the first segment of the local staging path and **registry model id**.
_Avoid_: language (fine in metadata), locale

**Registry tag**:
A named version slot for one registry model id (e.g. `stable`). Selects which weight snapshot to load.
_Avoid_: version (too generic), release

**Weights source**:
A URI telling the inference service where to find checkpoint files for one registry tag. Schemes: local `file://`, packaged `package://`, or remote `hf://<namespace>/<hub-repo-slug>@<registry-tag>` (full Hub URI including namespace).
_Avoid_: artifact path, model path

**Hub model repo**:
A Hugging Face **model** repository holding inference weights and a model card for one **registry model id** (e.g. `nomikos/greek-htr-calamari` for `greek-calamari-v1` or `nomikos/segmentation-blla` for `blla-segment`).
_Avoid_: model folder, checkpoint repo

**Hub dataset slug**:
The single-segment name of a **Hub dataset repo**, optimized for search: `{script}-manuscript-lines` or `{script}-{corpus}-htr-lines` (e.g. `greek-byzantine-manuscript-lines`). May carry a `nomos-` prefix for brand cohesion. Distinct from **Hub repo slug** (models use task-specific slugs).
_Avoid_: mirroring registry model id, generic `dataset-v1`

**Hub collection**:
A Hugging Face collection grouping **Hub model repos** and **Hub dataset repos** for discovery. Source of truth: `nomikos_inference/publish/collection.yaml`; synced via `scripts/hf/sync_collection.py`. Collection slug: `nomos`.
_Avoid_: monorepo, model bundle, src/hf/publish/collection.yaml (pre-move location)

**Hub revision**:
The immutable 40-character git commit on a **Hub model repo** selected by a **registry tag**. The tag remains a human-facing selector in the **weights source**, while the registry records its resolved commit separately.
_Avoid_: mutable tag as a runtime revision, version (too generic), release branch

**Hub artifact**:
The files published inside a **Hub model repo** at one **Hub revision**. Calamari and BLLA inference both load the ONNX graph (`best.onnx`, `blla.onnx`), which carries its own codec and input geometry. The native checkpoints (`.pt`, `.safetensors`) are published at the same revision and are the *export* input, never the run input.
_Avoid_: weights (too generic), model file, checkpoint (means the native export input here)

**Training recipe**:
The loader and the crop settings one model was trained under: how a line image is cut out of a page, how it was written to disk, and how that crop becomes the tensor handed to the graph. It belongs to the model, not to the architecture, and serving must reproduce it exactly (ADR 0007). Every Calamari entry in the **Registry** shares the loader, `src/models/calamari/data.py::_load_line_image`: grayscale, aspect-preserving bilinear resize to the graph's line height, raw `uint8`, no inversion, no dewarp, no tensor padding. They also share one crop function, `src/preprocessing_data/syriac/xml_to_data.py::crop_polygon`, a polygon mask on a white canvas. What they do *not* share is how far that crop was widened before masking, so each `task: transcribe` entry states its own in required `line_crop` (currently the single value `polygon-white`) and `line_crop_padding` fields: 0 for `greek-calamari-v1`, whose finetuning crops were exported with no padding, and 12 for `armenian-calamari-v1` and `syriac-calamari-v2`. `coptic-calamari-v1` is measured
at the same 12 px: CER 0.0177 at 12 against 0.1890 at 0 on 2823 platform Coptic project
lines through `best.pt` at batch size 1, minimum flat across 10 to 12. The exporter also wrote each crop as a grayscale JPEG at quality 82 (`save_crop`) and the loader read that file back, so serving re-encodes the same way rather than handing over a cleaner lossless image.
_Avoid_: "Calamari preprocessing" (it meant the legacy TensorFlow processors, which no model here was trained on), normalization, input spec, one recipe per architecture (the crop is per model)

**Artifact SHA-256**:
The required 64-character SHA-256 digest for the architecture-native **Hub artifact**. The inference service verifies it after download, before Hub-cache reuse, and before passing the artifact to an architecture loader.
_Avoid_: directory hash alone, unverified download

**Local bundled weights**:
Checkpoint files under `nomikos_inference/publish/artifacts/local/` used for offline dev and images built without Hub access. Referenced by `file://local/...` **weights source** URIs relative to `nomikos_inference/publish/artifacts/`. A source-checkout affordance only: the directory is excluded from both build targets, so an installed package has no such root and `local_bundled_root()` refuses rather than resolving inside site-packages. Override the root with `NOMIKOS_LOCAL_WEIGHTS_ROOT`.
_Avoid_: dev weights, nomikos_inference/weights, src/hf/local (pre-move location)

**Hub staging tree**:
Publish-ready **Hub artifact**s under `nomikos_inference/publish/artifacts/staging/` (models and datasets) before upload scripts push to the Hub.
_Avoid_: hf repo (ambiguous with Hub remote repo), src/hf/staging (pre-move location)

**Hub cache**:
Downloaded **Hub artifact**s at runtime under `~/.nomikos/hf/cache/<registry_model_id>/<registry_tag>/` (override: `HF_CACHE_ROOT`). It lives in the researcher's home directory, not beside the code, because **Hub integration** ships inside the installed package. Reused only when required files exist and a manifest matches the immutable **Hub revision** and **artifact SHA-256**.
_Avoid_: runtime weight cache, nomikos_inference/weights/cache, src/hf/cache (pre-package layout), nomikos_inference/publish/artifacts/cache (a CI and test fetch target under `HF_CACHE_ROOT`, not the default root)

**Hub cache manifest**:
An integrity record (e.g. `.hub-manifest.json`) stored alongside cached **Hub artifact**s. It records the Hub repo, immutable **Hub revision**, artifact path, and **artifact SHA-256**; all must match before cache reuse.
_Avoid_: revision file alone (insufficient when artifact bytes change)

**Hub integration**:
Python code at `nomikos_inference/hub/` that resolves `hf://` URIs, checks **Hub cache**, verifies **artifact SHA-256**, and downloads missing artifacts. It lives inside the published package because it is on the runtime path (ADR 0002). Publish-side code under `nomikos_inference/publish/` reuses it; the reverse dependency does not exist.
_Avoid_: huggingface module (too generic), src/hf/resolve (pre-package location), nomikos_inference/publish (that is the publish side, and it is excluded from the wheel)

**Published package**:
The one distribution, `nomikos-inference`, carrying the inference library, **Hub integration**, and the `nomikos` console entry point. A researcher's laptop and a hosted worker install the same package (ADR 0002), so there is no version-compatibility matrix between components that always ship together. Built from the repository root; the wheel is now the whole of `nomikos_inference/`, because the loopback HTTP surfaces it used to exclude were deleted (#60) rather than merely held out of the build.
_Avoid_: helper bundle (frozen-installer era), library package vs CLI package (there is one)

**Inference host**:
The machine where model weights are cached and inference executes - either the researcher's machine (**local inference**) or a hosted server (**remote inference**).
_Avoid_: Runtime (too generic), worker node (infra jargon)

**Lite model tier**:
A registry model id sized for CPU on a typical researcher laptop; eligible for **local inference**.
_Avoid_: Small model (vague), edge model (mobile jargon)

**Server model tier**:
A registry model id for **remote inference** only - too large or too slow for typical laptops.
_Avoid_: Large model (vague), cloud model (ambiguous with Hub hosting)

**Host eligibility**:
Whether a registry model id may run on the researcher's machine (`local`), only on a hosted server (`remote`), or either (`any`). Distinct from **Compute device** (`cpu` / `cuda`), which says what hardware to use once a host is chosen. Constrains which **execution target**s a job may choose; it does not choose one.
_Avoid_: device (already means cpu/cuda), tier alone (ambiguous)

**Execution target**:
The **inference host** a single job runs on - `local` or `cloud` - fixed when the job is submitted and never changed afterwards. A target may only be chosen when it has **capacity**; the researcher is told which host will run the job, and a job that fails reports which host it failed on.
_Avoid_: routing/preference (implies the platform may re-decide mid-flight), fallback (nothing changes target after submission)

**Capacity**:
Whether an **inference host** currently has a machine able to take work, answered by whether any device for that host was seen recently. The researcher's laptop and a hosted worker are the same kind of thing here, so cloud availability is not a separate concept. "Recently" is the device layer's existing idle window; a device with no **capacity** is not a failure, it is an announced state.
_Avoid_: helper available (loopback-era, meant a port answering), online (ambiguous with the researcher's own connectivity)

**Inference agent**:
The program that takes work from the platform and runs it - one implementation, run either on a researcher's machine (**local inference**) or on a hosted server (**remote inference**). Local and cloud differ by credential and uptime, not by code path.
_Avoid_: helper (loopback-era, meant a process listening on a port), worker (ambiguous with the platform's own job worker)

**Claim**:
One **inference agent** taking exactly one page of work from the platform's queue, over HTTP, authenticated as itself. A batch is N claims. The claim fixes nothing new: it hands over a job whose **execution target** was already decided at submission, and it may only hand over work for the target the presented credential is allowed to run.
_Avoid_: dispatch (implies the platform pushes), assignment (implies the platform chooses an agent), reservation

**Signed page image link**:
The short-lived URL a **claim** carries to the one page image it hands over. It reaches exactly one stored object, expires in about a minute, and is fetched with no device credential attached - the signature *is* the authorization. Its lifetime is deliberately **not** the **lease**'s: the agent downloads once, immediately after claiming. Accepted risk (ADR 0002): a bearer token in a URL leaks through logs and crash dumps, bounded to one object and one minute.
_Avoid_: image endpoint (an authenticated `GET /device/v1/jobs/{id}/image` was rejected), download token, presigned upload (nothing uploads here)

**Lease**:
How long a claimed page stays with the **inference agent** that took it before the platform may give it to another - 600 seconds, deliberately shorter than the platform's 1800-second job timeout, which is sized for a server that does not sleep. There is no heartbeat: work is seconds-to-minutes, so the lease covers it with margin, and a stopped agent loses one page rather than a document. When it expires the page returns to the queue and the claim is cleared, so any agent may take it next; it is never failed, because a closed lid is not a failed job. A hosted worker inherits the same lease - it is now the one timeout rather than one of two.
_Avoid_: timeout (ambiguous with the platform-wide job timeout), lock (nothing is held in the database between requests)

**Device credential file**:
Where one paired machine keeps its **device token**: `~/.nomikos/device.json`, mode `0600` in a `0700` directory (override the root with `NOMIKOS_HOME`). Owner-only is the only control the client side of ADR 0001's accepted risk actually owns - per-account scope and revocation are the platform's. It also records which platform minted the token, because a credential is only meaningful against that one.
_Avoid_: config file (it holds a credential, not settings), keychain entry (nothing uses one), token in the environment

**Confirmation code**:
A short keyed derivation of one pairing request, shown by the CLI and on the consent screen so a researcher can compare them. Not a `user_code` under another name: no endpoint accepts it, so it adds no brute-forceable surface. It is the only thing on the consent screen not supplied by whoever started the pairing, which is what makes it the check - and it only works if the client prints it.
_Avoid_: pairing code (that is the `device_code`), verification token (that is the browser handoff secret), OTP

**Service credential**:
The credential a hosted **inference agent** presents instead of a device token. It claims `cloud` work for the whole platform, which is why it is not a device token: a device token's entire authorization scope is the one account on its device row, and `cloud` work has no such owner. Its **capacity** row is owned by a service account no person can log into.
_Avoid_: device token for cloud, webhook secret (that authenticates the platform's own callback receiver), API key

**Version floor**:
The oldest **inference agent** the platform will hand a **claim** to, served by the platform rather than read from PyPI so it can be turned without a release. An agent below it is refused outright and told to upgrade; an agent that states no version, or one that cannot be compared, is refused on the same terms rather than assumed current. Asked for on its own at the **launch check**, and repeated on every claim response. Distinct from **outdated**, which is a notice delivered *with* the work.
_Avoid_: minimum supported version (fine in prose, but this is a runtime dial not a support policy), pinned version, auto-update setting

**Launch check**:
The one moment an **inference agent** may replace its own code: before it has claimed anything, it asks the platform for the **version floor**, installs a newer build and re-execs into it if it is below the floor, prints a notice if it is merely **outdated**, and then begins claiming. Never during a batch - a process that swaps its own code while a page is in flight has already told the platform which version it was. A failed upgrade is fatal and claims nothing. Accepted risk (ADR 0002): it executes newly fetched code without asking, so a compromised package reaches every laptop at next launch; mitigable by pinning to published hashes, not eliminable.
_Avoid_: auto-update (implies a background updater), self-healing, restart (the process is replaced, not restarted by anything outside it)

**Outdated**:
An **inference agent** at or above the **version floor** but behind the newest published release. It is served normally and told, on every claim response, page or no page. Deliberately not the same state as being below the floor: most upgrades are not urgent, and refusing them would make every release an outage for anyone who had not restarted.
_Avoid_: stale (that is what being below the **version floor** means), deprecated, unsupported

**Host preference**:
The account-level setting "use my computer when it is available", the only researcher input to **execution target** selection. Combined with **host eligibility** and **capacity** it fixes one target at submission; there is no per-job toggle, because a researcher cannot know which host is faster for a given page.
_Avoid_: per-job execution mode, `local_only` (retired by ADR 0002), routing rule

**Hub namespace**:
The Hugging Face account or org that owns **Hub model repos** and **Hub dataset repos**. Starts under a personal username; may later move to the `nomikos` org without changing repo slugs.
_Avoid_: org (when meaning the namespace generically)

## Relationships

- The **Registry** maps each **registry model id** + **registry tag** to one **weights source**
- One **Hub model repo** corresponds to one task/architecture pair; HTR uses `{script}-htr-{architecture}` and BLLA segmentation uses `segmentation-blla`
- **Registry model id** = `{script}-{architecture}-{model_version}`; maps to one **Hub repo slug** + **model version**
- Local **Hub staging tree**: `nomikos_inference/publish/artifacts/staging/models/{script}/{architecture}/{model_version}/{registry_tag}/`
- **Hub cache**: `~/.nomikos/hf/cache/{registry_model_id}/{registry_tag}/`
- **Device credential file**: `~/.nomikos/device.json` - the same home-directory root as the **Hub cache**, one **device token** per machine
- **Local bundled weights**: `nomikos_inference/publish/artifacts/local/{script}/{architecture}/{model_version}/{registry_tag}/`
- The **Hub staging tree** and **Local bundled weights** live under `nomikos_inference/publish/artifacts/`, which is inside the package directory and outside both build targets; the **Hub cache** is in the researcher's home directory and **Hub integration** code is in the package and in the wheel
- **Hub cache** reuse requires matching **Hub cache manifest** hash, not just present files
- **Hub integration** lazy-fetches at inference; `scripts/hf/fetch_model.py` for explicit prefetch
- One **registry tag** records one immutable **Hub revision** on that repo
- Training output is copied into **Hub staging tree** when ready to publish
- One **Hub dataset repo** may train many **registry model ids** over time
- **Host eligibility** constrains which **execution target**s a job may choose; **host preference** and **capacity** choose one from what is left
- **Capacity** for one **inference host** = any device recorded as that host was seen recently; a hosted worker is such a device, not a separate concept
- One **claim** = one page; the presented credential fixes the **execution target** it may take (device token -> `local`, own account only; **service credential** -> `cloud`, any account)
- A **claim** starts a **lease**; the page returns to the queue as `pending` when the lease expires - recovered opportunistically by the platform's existing stale sweep on read paths, never by a background worker - and completion or failure ends it through the platform's existing job callback contract
- One **claim** carries one **signed page image link**, to one object; the two lifetimes are separate dials (`DEVICE_PAGE_IMAGE_URL_TTL_SECONDS` ~60s, `DEVICE_LEASE_SECONDS` 600s) because the fetch happens once at the start of the run
- Every **claim** states which agent version is calling; below the **version floor** it is refused before it is authenticated, so a refused agent also stops reporting **capacity** and submission announces no host rather than creating pages nobody may claim
- The **launch check** asks for the same verdict with no page attached (`GET /device/v1/agent/version`), so an agent learns it is below the **version floor** while nothing is in flight; it runs once per process and has no call site inside the **claim** loop
- Pairing writes one **device credential file** per machine; the **confirmation code** is printed before the wait, and the pairing URL before any browser is opened
- A **Hub collection** (`nomos`) links to many **Hub model repos** and **Hub dataset repos**; defined in `nomikos_inference/publish/collection.yaml`

## Example dialogue

> **Dev:** "Should training crops live in the same HF repo as the Greek checkpoint?"
> **Domain expert:** "No. Weights go in a **Hub model repo**; labelled crops go in a **Hub dataset repo**. Inference only reads the Registry and **Hub model repos**."

## Flagged ambiguities

- "data" was used to mean both weights and training material - resolved: use **Hub model repo** vs **Hub dataset repo**.
- "kalamos" vs "nomikos" as public product name - resolved for Hub: product is **nomikos**; **Hub namespace** may be personal until the org exists.
- Checkpoint filename at repo root - resolved: use runtime artifact names (**Hub artifact**), e.g. Calamari `best.onnx` and BLLA `blla.onnx`, published beside the `best.pt` / `blla.safetensors` they were exported from.
- Calamari and BLLA runtime **Hub artifact** format is ONNX (`best.onnx`, `blla.onnx`). PyTorch was the runtime from 2026-08-04 to 2026-08-05 and is now the *exporter*: the trained artifact and the run artifact are different files again, and the parity between them is asserted by `tests/export/` rather than assumed away. The 445 MB that buys off a researcher's install is why. See ADR 0006, which supersedes ADR 0004.
- Legacy registry ids (`greek-calamariv1`) - resolved: migrate to `{script}-{architecture}-{model_version}` (e.g. `greek-calamari-v1`); **Hub repo slug** is task-specific.
- **Hub cache** invalidation - resolved: manifest hash (not files-exist-only).

## Implementation notes (not domain)

### Current (interim)

- Entry point: `run_model` in `nomikos_inference/jobs/runner.py`, called in process by whichever **inference agent** claimed the page. There is no HTTP surface under `nomikos_inference/`: `nomikos_inference/api` (sync `POST /inference/v1/run`) and `nomikos_inference/helper` (the loopback sidecar) were deleted by #60, and with them `run_errors.py`, `contracts/run.py`, and the `INFERENCE_SERVICE_SECRET` that authenticated their callers.
- Queued jobs: the platform owns the only queue (ADR 0003). `inference` holds no database, ORM, or claim loop of its own.
- **Claim**: `POST /device/v1/jobs/claim` on the platform - the one endpoint this layer adds (ADR 0005). Completion and failure are the existing `POST /internal/inference/job-complete` with a `JobCallbackRequest`; abandonment is the existing stale sweep. There is no heartbeat and no release endpoint.
- **Signed page image link**: `page_image_url` / `page_image_expires_at` on the claimed page. Minted by the media store, so Supabase signs a Storage URL and the local filesystem backend signs a path the platform serves at `/media/signed/{image_key}` - a route with no credential dependency, which refuses to answer unless `STORAGE_BACKEND=local`.
- **Version floor**: every claim sends `X-Nomikos-Agent-Version`. Below the floor the platform answers `426` with `error.code = AGENT_VERSION_UNSUPPORTED`, `reason` (`below_floor` / `missing` / `malformed`), `minimum_version`, `latest_version`, `package`, and `retryable: false`. At or above it, the 200 response carries an `agent` notice with `outdated`. Configured by `INFERENCE_AGENT_MIN_VERSION` / `INFERENCE_AGENT_LATEST_VERSION` on the platform (`backend/ml/domain/agent_version.py`, `backend/ml/api/agent_version.py`).
- **Launch check**: `GET /device/v1/agent/version` answers the same 426 or the same notice with nothing taken from the queue - unauthenticated, because the version dependency resolves before any credential is looked at. The CLI half is `nomikos_inference/cli/upgrade.py`, wired into `main.py` for the commands that claim and nowhere else; it upgrades with whichever installer already owns the environment (`pip` if this interpreter has one, otherwise `uv pip`) and re-execs through `os.execve`.
- Architectures implemented: **Calamari** (`nomikos_inference/architectures/calamari/`) and native **BLLA segment** (`nomikos_inference/architectures/blla/`)
- **Calamari runtime**: local ONNX Runtime session + local preprocessing; no TensorFlow, no PyTorch, and no vendored `calamari_ocr` import at inference time.
- **Training recipe** on the serving side: `architectures/calamari/preprocessing/pipeline.py` is the trainer's loader and `architectures/calamari/preprocessing/crop.py` is the trainer's crop writer (`TRAINING_CROP_PADDING = 12` is the exporter's default, not a rule), parameterized per model by the registry's `line_crop` and `line_crop_padding` fields, with `jobs/runner.py::_crop_line_image` re-encoding the crop as the exporter's grayscale JPEG; `tests/inference/unit/test_calamari_training_parity.py` compares all of it against the real `src/` functions and fails when they drift (ADR 0007).
- Weight resolution: `file://`, `hf://`, and `package://` (see `nomikos_inference/weights/__init__.py` and `nomikos_inference/hub/`)
- Runtime cache: `~/.nomikos/hf/cache/<registry_model_id>/<registry_tag>/`
- **CLI** (`nomikos_inference/cli/`): the only entry point a researcher has. `nomikos pair` and `nomikos version` (#56), `nomikos run` (#57), `nomikos upgrade` (#58). `pair` runs the pairing protocol above and writes the **device credential file**; `version` reports what the **version floor** will read, and asks the platform nothing; `upgrade` is the **launch check** run on demand, and prints nothing when this agent is current. `run` is the **claim** loop: one page in flight, fetched through the **signed page image link**, executed by the same `run_model` the platform's own worker calls, and ended through the existing job callback - `--exit-when-empty` for a script, waiting otherwise. `run` is also the only command that claims, so it is the only one the launch check gates, and it runs that check before its first claim and never again. A hosted **inference agent** runs the same loop with a **service credential** in `NOMIKOS_SERVICE_TOKEN` and a short poll. The platform base URL comes from `NOMIKOS_API_URL` or `--api-url`.

### Hub layout

Every path below except the Hub runtime cache is now under `nomikos_inference/`,
which is a statement about the import root and not about the wheel. Only the
first row ships: `pyproject.toml` excludes `nomikos_inference/publish` from the
wheel and the sdist, so the published file listing is the same as it was when
these lived under `src/hf/`.

| Piece | Location | Ships? |
|-------|----------|--------|
| `hf://` resolution, download, cache manifest | `nomikos_inference/hub/` | yes |
| Publish staging validation, model cards, collection sync | `nomikos_inference/publish/` | no |
| Local bundled weights for offline dev | `nomikos_inference/publish/artifacts/local/` | no |
| Publish-ready staging tree | `nomikos_inference/publish/artifacts/staging/` | no |
| Training-stack predictors (the export *input*, not the runtime) | deleted; was `nomikos_inference/predictors/` | no |
| Hub runtime cache | `~/.nomikos/hf/cache/` | n/a |
| Collection metadata | `nomikos_inference/publish/collection.yaml` | no |
| CLI entrypoints | `scripts/hf/` | no |
