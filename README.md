<div align="center">
  <h1>Nomikos</h1>
  <p><strong>Nomikos is an open-source platform for transcribing historical manuscripts.</strong></p>
  <img src="landing/assets/screenshots/editor-1280.webp" alt="The Nomikos page editor open on Grec1360 p.3, a two-page Greek manuscript spread with 49 line segments outlined in green, an HTR model selector in the toolbar, and a paired/unpaired legend" width="720">
  <p><em>Grec1360 p.3, segmented into 49 lines. Pick an HTR model in the toolbar, let it draft the first pass, then correct, review, share, publish, and export.</em></p>
  <p>
    <a href="#why-nomikos"><strong>Why Nomikos</strong></a> ·
    <a href="#current-model-support"><strong>Models</strong></a> ·
    <a href="#accuracy"><strong>Accuracy</strong></a> ·
    <a href="docs/talks/how-did-we-do-it.pdf"><strong>How we did it</strong></a> ·
    <a href="#quick-start"><strong>Quick Start</strong></a> ·
    <a href="docs/README.md"><strong>Documentation</strong></a> ·
    <a href="https://huggingface.co/nomikos-project"><strong>Hugging Face</strong></a> ·
    <a href="https://nomikos.app"><strong>Website</strong></a>
  </p>

  <a href="https://nomikos.app"><img src="https://img.shields.io/badge/Website-nomikos.app-navy" alt="Website"></a>
  <a href="https://app.nomikos.app"><img src="https://img.shields.io/badge/App-app.nomikos.app-green" alt="Application"></a>
  <a href="https://huggingface.co/nomikos-project"><img src="https://img.shields.io/badge/Models-Hugging_Face-yellow" alt="Hugging Face models"></a>
  <a href="docs/talks/how-did-we-do-it.pdf"><img src="https://img.shields.io/badge/Slides-How_we_did_it-purple" alt="Methodology slides"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License: Apache 2.0"></a>
</div>

Upload a manuscript page and Nomikos segments it into written lines, drafts a transcription where a compatible HTR model is available, and hands you a browser editor to correct, review, share, publish, and export. Behind that sit the editor, the API, storage, job state, streaming, and inference that runs on a researcher's laptop or in the cloud, all in this repository.

Nomikos is the whole ecosystem, data, models, and the application to use them: we trained transcription models for Byzantine Greek, Armenian, Syriac, and Coptic, and published them with open line-level datasets for Greek, Armenian, and Syriac on [Hugging Face](https://huggingface.co/nomikos-project).

## Why Nomikos

- **Keep the expert in the loop.** Models draft segments and transcriptions; researchers correct and approve. Model output is never treated as automatic ground truth.
- **Operate inside your own data boundary.** Keep application data behind an API you control, with your own hosting, data policy, and review conventions.
- **Run inference on your own computer.** Point the `nomikos` agent at the platform and run supported BLLA, Calamari, and PP-OCR models on a researcher's CPU. The agent only makes outbound requests, so it needs no inbound port, VPN, or proxy.
- **Collaborate through projects and sharing.** Organize work in projects, share documents with colleagues, and publish read-only views behind a secret link that a reader opens without an account.
- **Correct rather than retype.** Fixing a model draft is faster than transcribing a blank page, and approved work exports in a predictable format for publication or retraining.

## Built for Research

Nomikos is being developed for the Nomos research ecosystem, with a focus on Syriac, Coptic, Armenian, Byzantine Greek, and related scripts. The platform itself is script agnostic: if you want to collaborate or develop models for your own script, get in touch via [GitHub](https://github.com/kkkamur07/nomikos) or the [Hugging Face organization](https://huggingface.co/nomikos-project).

The system is expert-in-the-loop by design. Models draft, and researchers decide what is correct. Approved work produces processed line images and transcription files for publication or future model training.

## Complete Workflow

- **Turn pages into editable data.** Upload or open a page, segment it into lines, generate a model transcription, and pair text with segments.
- **Watch the queue while you work.** Edit the draft in the browser editor, with job state that says which host ran each job and what is still queued.
- **Review, share, and publish.** Move documents through review, share them with collaborators, and publish read-only views for readers.
- **Export training-ready data.** Produce processed line images and transcription files from approved work for publication or future model training.
- **Extend to new scripts.** Add models through the registry, weights, and publishing workflow. Data preparation, training, and Hub publishing tools are in the repo.

## Current Model Support

Through the pinned runtime registry, Nomikos pages can use:

| Registry id | Task and script | Architecture | Weights |
| --- | --- | --- | --- |
| `blla-segment` | Page segmentation, any script | Kraken BLLA | [segmentation-blla](https://huggingface.co/nomikos-project/segmentation-blla) |
| `ppocr-segment` | Page segmentation, any script | PP-OCRv6 detection (ONNX Runtime) | [segmentation-ppocrv6-det](https://huggingface.co/nomikos-project/segmentation-ppocrv6-det) |
| `greek-calamari-v1` | Line HTR, Byzantine Greek (`grc`) | Calamari | [greek-htr-calamari](https://huggingface.co/nomikos-project/greek-htr-calamari) |
| `armenian-calamari-v1` | Line HTR, Armenian (`hy`) | Calamari | [armenian-htr-calamari](https://huggingface.co/nomikos-project/armenian-htr-calamari) |
| `syriac-calamari-v2` | Line HTR, Syriac (`syr`) | Calamari | [syriac-htr-calamari](https://huggingface.co/nomikos-project/syriac-htr-calamari) |
| `syriac-ppocr-v1` | Line HTR, Syriac (`syr`) | PP-OCRv6 recognition (ONNX Runtime) | [syriac-htr-ppocr_rec](https://huggingface.co/nomikos-project/syriac-htr-ppocr_rec) |
| `coptic-calamari-v1` | Line HTR, Coptic (`cop`) | Calamari | [coptic-htr-calamari](https://huggingface.co/nomikos-project/coptic-htr-calamari) |

In the platform pickers the segmenters appear as seg-v1 (blla-segment) and seg-v2 (ppocr-segment); each picker starts on the project's default model, else seg-v1, and there is no Default entry.

All four Calamari HTR checkpoints share one architecture: a 40/60-filter CNN followed by two 200-unit
bidirectional LSTM layers at line height 48. Read off the Coptic and Syriac checkpoints, whose
structural fields agree exactly (3x3 convolutions 1 to 40 to 60, BiLSTM weights at `layers.4`
and `layers.6` with 200-unit hidden states, blank index 0, temperature -1.0); each trains
its own codec.

A model is runtime-supported only after its weights are published, pinned, verified, and added to [nomikos_inference/registry.yaml](nomikos_inference/registry.yaml). Public weights live on [Hugging Face](https://huggingface.co/nomikos-project) and are cached under `~/.nomikos/hf/cache` on first inference. See [models and datasets](docs/inference/models-and-datasets.md) and the [publishing workflow](scripts/hf/README.md) for the pinning, verification, and release steps.

Run supported inference locally:

```bash
uv tool install nomikos-inference   # or: pip install nomikos-inference
nomikos pair          # links this machine to your account
nomikos run           # takes pages from the queue until you stop it
```

Point it at a different platform with `NOMIKOS_API_URL` or `--api-url`.

## Accuracy

Test character error rate (CER) of each transcription model. Lower is better. Values come from the Hugging Face model cards, except `syriac-ppocr-v1`, whose approximate figure is reported by the model's authors and is not directly comparable.

| Model | Script | Test CER |
| --- | --- | ---: |
| `greek-calamari-v1` | Byzantine Greek | 0.226 |
| `armenian-calamari-v1` | Armenian | 0.072 |
| `syriac-calamari-v2` | Syriac | 0.210 |
| `syriac-ppocr-v1` | Syriac | about 0.07 |
| `coptic-calamari-v1` | Coptic | 0.082 |

These figures come from specific manuscripts and hands; expect CER to move with the script, the hand and the image quality. The [methodology slides](docs/talks/how-did-we-do-it.pdf) explain how the models were trained and evaluated.

## Quick Start

The fastest way to evaluate the complete application is the development Compose stack. It is not a hardened internet-facing production deployment.

Prerequisites: Git, Docker Desktop with Compose, and about 10 GB of free disk space.

```bash
git clone https://github.com/kkkamur07/nomikos.git
cd nomikos
cp infrastructure/.env.compose.example infrastructure/.env
```

Replace the placeholder values in `infrastructure/.env` for `POSTGRES_PASSWORD`, `JWT_SECRET`, and `INFERENCE_WEBHOOK_SECRET`. Then start the stack:

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

Open the editor:

```bash
open http://localhost:5173   # or visit manually
# Development seed login: dev@example.com / dev-pass-123
```

| Service      | Address                                        |
| ------------ | ---------------------------------------------- |
| Editor       | [http://localhost:5173](http://localhost:5173) |
| Platform API | [http://localhost:8000](http://localhost:8000) |
| API docs     | [http://localhost:8000/docs](http://localhost:8000/docs) |
| Postgres     | `127.0.0.1:5433`                               |

The first inference request downloads public weights into `~/.nomikos/hf/cache`.

```bash
docker compose -f infrastructure/docker-compose.yml ps
curl -s http://localhost:8000/health | python -m json.tool
docker compose -f infrastructure/docker-compose.yml logs -f
docker compose -f infrastructure/docker-compose.yml down
```

## Ways to Run Nomikos

- **As a Docker stack.** Run the published Compose services (editor, API, Postgres, workers) for evaluation and development.
- **As a self-hosted platform.** Deploy the API, storage, and workers on infrastructure you operate. Production today is manual Supabase, Vercel, and Docker-host configuration rather than one-click hosting.
- **As a local inference agent.** Run `nomikos pair` once, then `nomikos run` to process your queue on your own CPU, against either a local stack or [app.nomikos.app](https://app.nomikos.app).

## Data Control and Review

Data controls define the boundary around Nomikos:

- Runs in infrastructure you control, including local Docker evaluation today and manual Supabase + Vercel + worker setup for production.
- Nothing outside the platform reaches Postgres or private storage. The API owns authentication, authorization, project sharing, and document and job state, and it is the only thing that reads a page image off disk.
- The public reader is the same API seen through a narrower door: no session, and every route it can reach demands the document's secret share token or answers exactly as if the document did not exist.
- The local agent opens no port. It claims one page over a short-lived signed link, runs the model, and reports back. An agent that is not running is an announced state rather than a failure: work goes to the cloud, and the page says so.
- Job updates travel over Postgres `NOTIFY` to API listeners, then SSE with polling fallback. There is no email or push provider in the current implementation.

See [`docs/security/`](docs/security/), [`docs/architecture.md`](docs/architecture.md), and [`docs/database-design.md`](docs/database-design.md), including auth boundaries, share-link behavior, and rate limiting.

## Explore Nomikos

- [How we did it: methodology slides (PDF)](docs/talks/how-did-we-do-it.pdf), the final project talk on data, models, training and evaluation for Armenian, Byzantine Greek, Coptic and Syriac
- [Use and host Nomikos](docs/guides/using-and-hosting.md)
- [Models and datasets](docs/inference/models-and-datasets.md)
- [Technical architecture](docs/architecture.md)
- [Architecture decision records](docs/adr/)
- [Inference service reference](nomikos_inference/README.md)
- [Model publishing workflow](scripts/hf/README.md)
- [Testing guide](docs/guides/testing.md)
- [Production deployment](docs/deployment/production.md)
- [Documentation index](docs/README.md)

## Community

Questions, script requests, and model contributions are welcome via [GitHub](https://github.com/kkkamur07/nomikos), the [Hugging Face organization](https://huggingface.co/nomikos-project), and the [website](https://nomikos.app). The hosted editor is at [app.nomikos.app](https://app.nomikos.app).

## License

Copyright 2026 Astha Gupta and Krrish Agarwalla. Nomikos is released under the [Apache License 2.0](LICENSE).
