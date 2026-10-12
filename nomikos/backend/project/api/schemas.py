"""Project API request/response schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=512)
    slug: str = Field(min_length=1, max_length=512)
    guidelines: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    slug: str | SkipJsonSchema[None] = Field(default=None, min_length=1, max_length=512)
    guidelines: str | None = None

    @field_validator("name", "slug", mode="before")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("must not be null")
        return value


class ShareUserRequest(BaseModel):
    """Who to share with, named exactly one of three ways.

    ``email`` and ``username`` say which kind of identifier this is, for a
    caller that knows. ``identifier`` says "one or the other, you work it out",
    which is what a single UI box has to send: a username may legally contain
    an ``@`` (registration constrains only its length), so the client cannot
    tell the two apart by looking. Resolution order is documented on
    ``ProjectService._find_collaborator``.

    Matching on email is case-insensitive, like login.
    """

    username: str | None = Field(default=None, min_length=1, max_length=150)
    email: EmailStr | None = None
    identifier: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str | None) -> str | None:
        return value.strip().lower() if value is not None else None

    @field_validator("identifier")
    @classmethod
    def strip_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("identifier must not be blank")
        return stripped

    @model_validator(mode="after")
    def exactly_one_identifier(self) -> "ShareUserRequest":
        given = [f for f in (self.username, self.email, self.identifier) if f is not None]
        if len(given) != 1:
            raise ValueError("provide exactly one of username, email or identifier")
        return self


class ProjectCollaboratorResponse(BaseModel):
    id: UUID
    username: str
    email: str

    model_config = {"from_attributes": True}


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    guidelines: str | None
    owner_id: UUID | None
    owner_username: str | None = None
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectPageResponse(BaseModel):
    items: list[ProjectResponse]
    next_cursor: str | None = None
