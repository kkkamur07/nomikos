"""Translate Product jobs into the run instruction an agent claims.

**No page image is read here.** The claim response carries a signed link to the
one page object (``page_image_url``) and the agent fetches it directly from
storage; putting the scan in the response as well meant every claim streamed a
manuscript page through the API, base64-encoded at about 1.33x its stored size,
for a field no client has ever read. ADR 0002 rejected an authenticated image
route on exactly that reasoning - the production API is serverless, so streaming
scans through it costs money for nothing - and the claim response was doing the
same thing by another name.

So what a claim carries is the instruction: which task, which registry model and
tag, and the parameters. Everything here is small and derived from Postgres.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from sqlalchemy.orm import Session

from backend.document.application.transcribe_merge_service import (
    TranscribeJobHandlerError,
    TranscribeMergeService,
)
from backend.document.infrastructure.orm_models import DocumentPart, Line
from backend.jobs.infrastructure.orm_models import Job, JobType
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from infrastructure.db import sync_system_session
from nomikos_inference.admission import validate_request_params
from nomikos_inference.contracts.common import InferenceTask as WireInferenceTask
from nomikos_inference.settings import get_inference_settings

_DEFAULT_SEGMENT_REGISTRY_MODEL = "blla-segment"
_DEFAULT_SEGMENT_REGISTRY_TAG = "stable"
_DEFAULT_TRANSCRIBE_REGISTRY_MODEL = "syriac-calamari-v2"
_DEFAULT_TRANSCRIBE_REGISTRY_TAG = "stable"


@dataclass(frozen=True)
class PageRunRequest:
    """What to run on one page - and deliberately not the page itself.

    This used to be ``nomikos_inference.contracts.jobs.JobSubmitRequest``, which carries
    ``image_bytes``, because the platform once POSTed that body into a second
    queue and the image had to travel with it. Since ADR 0003 there is no second
    queue: the agent claims from the platform's own table and fetches the scan
    from the signed link beside this object. Reusing the submit contract kept the
    image field alive with nothing to put in it but the whole page.

    The fields are the ones the shipped agent actually reads off ``request``.
    """

    task: WireInferenceTask
    registry_model_id: str
    registry_tag: str
    product_job_id: uuid.UUID
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """The two checks ``JobSubmitRequest`` made that still have a subject.

        Its third was an admission bound on ``image_bytes`` plus ``params``
        together; with the image gone, ``params`` is the entire body, so it is
        bounded on its own. A transcribe page carries one entry per line and is
        the only thing here that can grow without limit.
        """
        if self.task not in (WireInferenceTask.segment, WireInferenceTask.transcribe):
            raise ValueError(f"unsupported job task: {self.task.value}")
        if self.task == WireInferenceTask.transcribe and not self.params.get("lines"):
            raise ValueError("transcribe jobs require non-empty params.lines")
        validate_request_params(self.params, get_inference_settings())


@dataclass(frozen=True)
class PageRun:
    """A claimed page's run instruction and the single stored object it may reach.

    ``image_key`` is resolved from the job's own ``document_part_id``, not a
    prefix and not the document. Whoever signs it can therefore only ever sign a
    link to this page, which is the property that makes a link safe to hand out
    with no credential behind it.
    """

    request: PageRunRequest
    image_key: str


@dataclass(frozen=True)
class RegistrySelection:
    model_id: str
    tag: str


def _registry_selection_from_artifact_ref(artifact_ref: str) -> RegistrySelection:
    parsed = urlparse(artifact_ref)

    if parsed.scheme != "registry" or not parsed.netloc or parsed.path not in ("", "/"):
        raise ValueError(
            "Inference model artifact_ref must be registry://<registry_model_id>?tag=<tag>"
        )

    tag = parse_qs(parsed.query).get("tag", ["stable"])[0] or "stable"
    return RegistrySelection(model_id=parsed.netloc, tag=tag)


def _job_registry_selection(
    session: Session,
    job: Job,
    *,
    task: InferenceTask,
    fallback_model_id: str,
    fallback_tag: str,
) -> RegistrySelection:
    if job.model_id is None:
        return RegistrySelection(model_id=fallback_model_id, tag=fallback_tag)

    model = session.get(InferenceModel, job.model_id)
    if model is None:
        raise ValueError("Selected inference model not found")
    if model.task != task:
        raise ValueError(f"Selected inference model does not support {task.value}")

    return _registry_selection_from_artifact_ref(model.artifact_ref)


def _ml_params(job: Job) -> dict[str, object]:
    return dict((job.payload or {}).get("ml_params") or {})


def _segment_run(session: Session, job: Job) -> PageRun:
    if job.document_part_id is None:
        raise ValueError("Segment job is missing its target document part")
    part = session.get(DocumentPart, job.document_part_id)
    if part is None:
        raise ValueError("Document part not found")
    selection = _job_registry_selection(
        session,
        job,
        task=InferenceTask.segment,
        fallback_model_id=_DEFAULT_SEGMENT_REGISTRY_MODEL,
        fallback_tag=_DEFAULT_SEGMENT_REGISTRY_TAG,
    )
    request = PageRunRequest(
        task=WireInferenceTask.segment,
        registry_model_id=selection.model_id,
        registry_tag=selection.tag,
        product_job_id=job.id,
        params=_ml_params(job),
    )
    return PageRun(request=request, image_key=part.image_key)


def _selected_lines(session: Session, job: Job, part_id: uuid.UUID) -> list[Line]:
    """The part's lines in reading order, narrowed to ``payload.line_ids`` if set."""
    lines = TranscribeMergeService.load_lines(session, part_id)
    selected_line_ids = (job.payload or {}).get("line_ids")
    if not selected_line_ids:
        return lines
    allowed = {uuid.UUID(str(line_id)) for line_id in selected_line_ids}
    lines = [line for line in lines if line.id in allowed]
    if not lines:
        raise TranscribeJobHandlerError("No matching lines to transcribe")
    return lines


def _transcribe_run(session: Session, job: Job) -> PageRun:
    if job.document_id is None or job.document_part_id is None:
        raise TranscribeJobHandlerError("Transcribe job is missing its target document part")
    part = session.get(DocumentPart, job.document_part_id)
    if part is None or part.document_id != job.document_id:
        raise TranscribeJobHandlerError("Document part not found")
    selection = _job_registry_selection(
        session,
        job,
        task=InferenceTask.transcribe,
        fallback_model_id=_DEFAULT_TRANSCRIBE_REGISTRY_MODEL,
        fallback_tag=_DEFAULT_TRANSCRIBE_REGISTRY_TAG,
    )
    line_regions = [
        {"line_id": str(line.id), "line_index": index, "points": line.points}
        for index, line in enumerate(_selected_lines(session, job, part.id))
    ]
    request = PageRunRequest(
        task=WireInferenceTask.transcribe,
        registry_model_id=selection.model_id,
        registry_tag=selection.tag,
        product_job_id=job.id,
        params={**_ml_params(job), "lines": line_regions},
    )
    return PageRun(request=request, image_key=part.image_key)


_RUN_BUILDERS: dict[JobType, Callable[[Session, Job], PageRun]] = {
    JobType.segment: _segment_run,
    JobType.transcribe: _transcribe_run,
}


def build_page_run(job: Job) -> PageRun:
    """The run instruction for *job* and its page's image key, read in one session."""
    builder = _RUN_BUILDERS.get(job.type)
    if builder is None:
        raise ValueError(f"No inference run instruction for job type {job.type.value}")
    with sync_system_session() as session:
        return builder(session, job)
