"""Platform exception types, mapped to HTTP responses in API layers (issue 001+)."""


class GreekOCRException(Exception):
    """Base for domain/application errors raised inside bounded contexts."""


class NotFoundError(GreekOCRException):
    """Entity does not exist or is not visible to the caller."""


class AccessDeniedError(GreekOCRException):
    """Caller is not allowed to perform the operation."""


class InvalidCredentialsError(GreekOCRException):
    """Authentication failed (wrong password, unknown account, invalid token)."""


class ConflictError(GreekOCRException):
    """Request conflicts with current state (duplicate slug, invalid transition, etc.)."""


class ValidationError(GreekOCRException):
    """Input or state failed business validation."""
