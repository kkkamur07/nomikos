"""FastAPI application factory: wires routers from core and bounded contexts."""

import asyncio
import logging
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import infrastructure.models  # noqa: F401 - register all ORM mappers before first query
from backend.annotation.api.history import router as annotation_history_router
from backend.annotation.api.segment_health import router as segment_health_router
from backend.core.api.admin import router as admin_router
from backend.core.api.body_limit import BodySizeLimitMiddleware
from backend.core.api.client_failures import router as client_failures_router
from backend.core.api.feedback import router as feedback_router
from backend.core.api.health import router as health_router
from backend.core.api.request_metrics import RequestMetricsMiddleware
from backend.core.api.root import router as root_router
from backend.core.exceptions import (
    AccessDeniedError,
    ConflictError,
    GreekOCRException,
    InvalidCredentialsError,
    NotFoundError,
    ValidationError,
)
from backend.core.schemas.errors import ApiErrorResponse
from backend.core.settings import (
    get_app_settings,
    get_auth_settings,
    get_inference_settings,
    get_infrastructure_settings,
    get_storage_settings,
)
from backend.core.settings.device import get_device_settings
from backend.core.settings.job import JobSettings, get_job_settings
from backend.core.version import get_version
from backend.document.api.document_batch import router as document_batch_router
from backend.document.api.documents import router as documents_router
from backend.document.api.media import router as media_router
from backend.document.api.public import router as public_router
from backend.document.api.public_media import router as public_media_router
from backend.document.api.segmentation_runs import router as segmentation_runs_router
from backend.document.api.signed_media import router as signed_media_router
from backend.document.infrastructure.media_gc import media_gc_loop
from backend.jobs.api.device_claim import router as device_claim_router
from backend.jobs.api.internal_inference import router as internal_inference_router
from backend.jobs.api.jobs import router as jobs_router
from backend.jobs.infrastructure.notifications import platform_job_notification_loop
from backend.jobs.infrastructure.worker import worker_loop
from backend.ml.api.agent_version import (
    AGENT_VERSION_REFUSED_STATUS,
    AgentVersionRefusedError,
)
from backend.ml.api.agent_version import router as agent_version_router
from backend.ml.api.device_pairing import router as device_pairing_router
from backend.ml.api.device_self import router as device_self_router
from backend.ml.api.devices import router as devices_router
from backend.ml.api.execution_preference import router as execution_preference_router
from backend.ml.api.models import router as ml_models_router
from backend.ml.api.registry import router as ml_registry_router
from backend.project.api.projects import router as projects_router
from backend.users.api.auth import router as auth_router

logger = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

COMMON_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {"model": ApiErrorResponse, "description": "Not authenticated"},
    403: {"model": ApiErrorResponse, "description": "Not authorized"},
    404: {"model": ApiErrorResponse, "description": "Resource not found"},
    409: {"model": ApiErrorResponse, "description": "Conflict with current state"},
    422: {"model": ApiErrorResponse, "description": "Validation error"},
    429: {"model": ApiErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ApiErrorResponse, "description": "Internal server error"},
}
_HTTP_PUBLIC_MESSAGES = {
    400: "Bad request",
    401: "Authentication required",
    403: "Access denied",
    404: "Resource not found",
    409: "Request conflicts with current state",
    413: "Request entity too large",
    422: "Invalid request",
    429: "Too many requests",
    503: "Service unavailable",
}


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
    correlation_id: str | None = None,
) -> JSONResponse:
    error: dict[str, object] = {"code": code, "message": message}
    if correlation_id:
        error["ref"] = correlation_id
    content: dict[str, object] = {"error": error}
    response_headers = dict(headers or {})
    if correlation_id:
        response_headers["X-Error-ID"] = correlation_id
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(content),
        headers=response_headers,
    )


def _http_error_code(status_code: int) -> str:
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMITED",
    }.get(status_code, "HTTP_ERROR")


def _correlated_error(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    exc: BaseException,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    correlation_id = uuid.uuid4().hex
    logger.warning(
        "api_error correlation_id=%s method=%s path=%s status=%s code=%s exception=%s",
        correlation_id,
        request.method,
        request.url.path,
        status_code,
        code,
        type(exc).__name__,
    )
    return _error_response(
        status_code=status_code,
        code=code,
        message=message,
        headers=headers,
        correlation_id=correlation_id,
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=404,
            code="NOT_FOUND",
            message="Resource not found",
            exc=exc,
        )

    @app.exception_handler(InvalidCredentialsError)
    async def invalid_credentials_handler(
        request: Request, exc: InvalidCredentialsError
    ) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=401,
            code="UNAUTHORIZED",
            message="Authentication failed",
            exc=exc,
        )

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(request: Request, exc: AccessDeniedError) -> JSONResponse:
        return _correlated_error(
            request=request, status_code=403, code="FORBIDDEN", message="Access denied", exc=exc
        )

    @app.exception_handler(ValidationError)
    async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=422,
            code="VALIDATION_ERROR",
            message=str(exc).strip() or "Invalid request",
            exc=exc,
        )

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=409,
            code="CONFLICT",
            message=str(exc).strip() or "Request conflicts with current state",
            exc=exc,
        )

    @app.exception_handler(GreekOCRException)
    async def greekocr_handler(request: Request, exc: GreekOCRException) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            exc=exc,
        )

    @app.exception_handler(AgentVersionRefusedError)
    async def agent_version_refused_handler(
        request: Request, exc: AgentVersionRefusedError
    ) -> JSONResponse:
        """The one error handler that returns detail instead of a fixed public message.

        The caller here is an agent, not a browser: it needs the actual version
        and reason to self-upgrade (ADR 0002). Nothing here is secret, the
        version floor is public and the CLI needs to read it.
        """
        logger.info(
            "agent_version_refused reason=%s presented=%r minimum=%s path=%s",
            exc.refusal.reason,
            exc.refusal.agent_version,
            exc.refusal.minimum_version,
            request.url.path,
        )
        return JSONResponse(
            status_code=AGENT_VERSION_REFUSED_STATUS,
            content=jsonable_encoder({"error": exc.refusal.model_dump()}),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=exc.status_code,
            code=_http_error_code(exc.status_code),
            message=_HTTP_PUBLIC_MESSAGES.get(exc.status_code, "Request failed"),
            exc=exc,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _correlated_error(
            request=request,
            status_code=422,
            code="VALIDATION_ERROR",
            message="Invalid request",
            exc=exc,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = uuid.uuid4().hex
        logger.exception(
            "unhandled_api_error correlation_id=%s method=%s path=%s",
            correlation_id,
            request.method,
            request.url.path,
        )
        return _error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            correlation_id=correlation_id,
        )


@dataclass(frozen=True)
class LifespanPlan:
    """Which startup work the lifespan runs. Pure data, decided by settings."""

    ensure_dev_user: bool
    notifications: bool
    worker: bool
    media_gc: bool


def plan_lifespan_tasks(*, environment: str, job_settings: JobSettings) -> LifespanPlan:
    """Decide the lifespan's startup work from settings alone, so it is testable."""
    return LifespanPlan(
        ensure_dev_user=environment == "development",
        notifications=job_settings.job_sse_notifications_enabled,
        worker=job_settings.job_worker_enabled,
        media_gc=job_settings.media_gc_enabled,
    )


async def _ensure_dev_user_at_startup() -> None:
    from backend.dev.bootstrap import ensure_dev_user_exists
    from infrastructure.db import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            if await ensure_dev_user_exists(session):
                logger.info("Recreated missing dev login user after database reset")
    except Exception:
        logger.exception("Failed to ensure dev user exists at startup")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from infrastructure.db import engine, sync_engine

    job_settings = get_job_settings()
    plan = plan_lifespan_tasks(
        environment=get_infrastructure_settings().environment,
        job_settings=job_settings,
    )
    stop_event = asyncio.Event()
    if plan.ensure_dev_user:
        await _ensure_dev_user_at_startup()
    notification_task = (
        asyncio.create_task(platform_job_notification_loop(stop_event))
        if plan.notifications
        else None
    )
    worker_task = asyncio.create_task(worker_loop(stop_event)) if plan.worker else None
    media_gc_task = asyncio.create_task(media_gc_loop(stop_event)) if plan.media_gc else None
    yield
    stop_event.set()
    if worker_task is not None:
        worker_task.cancel()
    if notification_task is not None:
        notification_task.cancel()
    if media_gc_task is not None:
        media_gc_task.cancel()
    if notification_task is not None:
        with suppress(asyncio.CancelledError):
            await notification_task
    if media_gc_task is not None:
        with suppress(asyncio.CancelledError):
            await media_gc_task
    if worker_task is not None:
        with suppress(asyncio.CancelledError):
            await worker_task
    await engine.dispose()
    sync_engine.dispose()


def create_app() -> FastAPI:
    # Resolve every API runtime setting before registering routes so a production
    # deployment cannot serve traffic with missing or placeholder credentials.
    infrastructure_settings = get_infrastructure_settings()
    get_auth_settings()
    # Refuses to start when a production deployment would key device tokens off
    # JWT_SECRET, which would make a routine JWT rotation unpair every helper.
    get_device_settings()
    app_settings = get_app_settings()
    job_settings = get_job_settings()
    ml_settings = get_inference_settings()
    if ml_settings.cloud_inference_enabled or job_settings.job_worker_enabled:
        ml_settings.require_callback_receiver_configuration()
    get_storage_settings()
    # Docs/OpenAPI expose every route, body schema, and auth requirement, a dev
    # aid that shouldn't be reachable by unauthenticated callers in production.
    docs_enabled = not infrastructure_settings.is_production
    app = FastAPI(
        title="greekOCR Platform",
        version=get_version(),
        description="Greek manuscript OCR and annotation platform",
        lifespan=_lifespan,
        responses=COMMON_ERROR_RESPONSES,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # Innermost of the three, so CORS preflights answered by CORSMiddleware are not
    # counted and the router has filled in ``scope["route"]`` by the time it reads it.
    # In-process counters only (ADR "Admin dashboard and request metrics").
    app.add_middleware(RequestMetricsMiddleware)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
        expose_headers=["X-Error-ID"],
    )
    _register_exception_handlers(app)
    app.include_router(root_router)
    app.include_router(health_router)
    app.include_router(client_failures_router)
    app.include_router(feedback_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(projects_router)
    app.include_router(documents_router)
    # Document-level fan-out over the same prefix: the per-page routes above address one
    # part, these address every part of a chapter at once. Mounted after them so the
    # per-page surface stays the one an ambiguous path would resolve to.
    app.include_router(document_batch_router)
    app.include_router(annotation_history_router)
    app.include_router(segment_health_router)
    app.include_router(segmentation_runs_router)
    app.include_router(media_router)
    app.include_router(public_router)
    app.include_router(public_media_router)
    # Landing site for signed page image links on the local storage backend. No
    # auth dependency by design: the signature is the authorization (ADR 0002),
    # and it only responds when STORAGE_BACKEND=local.
    app.include_router(signed_media_router)
    app.include_router(ml_models_router)
    app.include_router(ml_registry_router)
    app.include_router(execution_preference_router)
    # Always mounted; each device router carries require_device_pairing_enabled,
    # so DEVICE_PAIRING_ENABLED turns the surface off per request, not at boot.
    # Off by default in production until the /pair consent page exists.
    app.include_router(device_pairing_router)
    app.include_router(devices_router)
    app.include_router(device_self_router)
    # Device route by credential, job route by subject. Same DEVICE_PAIRING_ENABLED
    # gate as the rest of the device surface: one switch for the whole agent layer.
    app.include_router(device_claim_router)
    # The version floor on its own, so an agent can learn it is too old at launch
    # instead of finding out by claiming a page it would then have to hold while
    # it upgraded (ADR 0002, issue 058).
    app.include_router(agent_version_router)
    app.include_router(jobs_router)
    app.include_router(internal_inference_router)
    return app
