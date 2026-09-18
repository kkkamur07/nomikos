"""Import all ORM models so Alembic metadata is complete."""

from backend.annotation.infrastructure.orm_models import AnnotationHistorySnapshot  # noqa: F401
from backend.document.infrastructure.orm_models import (  # noqa: F401
    Block,
    Document,
    DocumentPart,
    Line,
    LineTranscription,
    MediaDeletionIntent,
    PageTranscriptionLine,
    SegmentationRun,
    Transcription,
)
from backend.jobs.infrastructure.orm_models import Job  # noqa: F401
from backend.ml.infrastructure.device_orm_models import (  # noqa: F401
    HelperDevice,
    HelperPairing,
)
from backend.ml.infrastructure.orm_models import (  # noqa: F401
    InferenceModel,
    ModelBinding,
)
from backend.project.infrastructure.orm_models import Project, project_shared_users  # noqa: F401
from backend.users.infrastructure.orm_models import (  # noqa: F401
    AuthRateLimitAttempt,
    AuthSession,
    User,
)
