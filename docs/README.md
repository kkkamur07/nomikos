# Nomikos documentation

**Picking the work back up?** Read
[`final-code-review-2026-08-06.md`](final-code-review-2026-08-06.md) first. It is the
current audit of `main`: four criticals, the confirmed defect list per bounded context,
the dead-file inventory, and the ranked architecture candidates. (The 2026-08-04/05
inference-redesign handoffs and the ADRs they replaced are retired; that work is merged.)

Start with the audience-appropriate guide:

| Doc                                                     | Use it when                                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [Root README](../README.md)                             | You want the product overview and a quick start                                          |
| [Use and host Nomikos](guides/using-and-hosting.md)    | You want Docker, local inference, Supabase, or deployment steps                          |
| [Models and datasets](inference/models-and-datasets.md) | You want the runtime catalog, HTR/segmentation models, or dataset workflow               |
| [How we did it (PDF)](talks/how-did-we-do-it.pdf)       | You want the methodology: data, models, training and evaluation, as presented in the final talk |
| [Technical architecture](architecture.md)               | You want the system design, stack rationale, jobs, notifications, or security boundaries |

## Developer guides

| Doc                                              | Scope                                              |
| ------------------------------------------------ | -------------------------------------------------- |
| [Local development](guides/local-development.md) | Existing service-by-service development reference  |
| [Testing](guides/testing.md)                     | Python, frontend, integration, ML, and CI commands |
| [Learnings](guides/learnings.md)                 | Operational lessons and known platform pitfalls    |

## Deployment and operations

| Doc                                                      | Scope                                                     |
| -------------------------------------------------------- | --------------------------------------------------------- |
| [Production deployment](deployment/production.md)        | Vercel surfaces, Supabase, workers, DNS, and rollback     |
| [Supabase](deployment/supabase.md)                       | Hosted Postgres, private Storage, poolers, and migrations |
| [Database roles](deployment/database-roles.md)           | Least-privilege service roles                             |
| [Vercel platform API](deployment/vercel-platform-api.md) | Python bundle and serverless constraints                  |
| [Release evidence](deployment/release-evidence.md)       | Per-release verification record                           |

## Inference and publishing

| Doc                                                             | Scope                                                             |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Adding inference models](inference/adding-inference-models.md) | Registry, weights, platform catalog, and deployment checklist     |
| [Inference service](../nomikos_inference/README.md)                     | Install from PyPI, releasing, contracts, limits, and runtime reference |
| [Hugging Face publishing](../scripts/hf/README.md)              | Model and dataset staging, validation, and upload workflow        |

## Platform references

- [`nomikos/CONTEXT.md`](../nomikos/CONTEXT.md) — domain terminology and
  annotation semantics
- [`database-design.md`](database-design.md) — schema, ownership, pooling,
  job state, `NOTIFY`, SSE, and polling
- [`nomikos/backend/README.md`](../nomikos/backend/README.md) — backend
  bounded contexts and routes
- [`nomikos/frontend/README.md`](../nomikos/frontend/README.md) — editor
  development and generated API client

## Status, handoff, and reviews

Point-in-time documents. They carry file paths and line numbers that drift — and, in the
review's case, paths that have since been deleted — so read them alongside the code rather
than as current reference.

| Doc                                                                             | Scope                                                                        |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [Final code review 2026-08-06](final-code-review-2026-08-06.md)                  | **Current.** Full-codebase five-axis audit of `main` @ `198738a`: 4 criticals, 49 confirmed required findings, dead-file inventory, architecture candidates |

## Security

Security-specific records live under [`security/`](security/). They include
dependency vulnerability exceptions and their remediation plans.

## Internal backlog

Deferred work lives in the repository-root [`todo.md`](../todo.md) — P0/P1/P2, one
backlog, no second copy. It is an internal backlog, not a product capability list.
