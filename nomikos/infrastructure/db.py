"""Shared SQLAlchemy engine and session factories."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import cast

from sqlalchemy import CursorResult, MetaData, Result, create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.settings import get_infrastructure_settings

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=convention)


_infra = get_infrastructure_settings()

_pool_kwargs = {
    "pool_size": _infra.db_pool_size,
    "max_overflow": _infra.db_max_overflow,
    "pool_recycle": _infra.db_pool_recycle,
}

_STATEMENT_TIMEOUT_MS = 30_000


def _asyncpg_database_url(url: str) -> str:
    """asyncpg uses ``ssl=``, not libpq ``sslmode=``."""
    if "sslmode=" not in url:
        return url
    return url.replace("sslmode=", "ssl=")


def _asyncpg_connect_args(url: str) -> dict:
    """PgBouncer transaction mode (Supabase pooler :6543) breaks prepared statements."""
    if "pooler.supabase.com" in url or ":6543/" in url:
        return {"statement_cache_size": 0}
    return {}


def _install_statement_timeout(sync_engine) -> None:
    @event.listens_for(sync_engine, "connect")
    def _set_statement_timeout(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"SET statement_timeout = {_STATEMENT_TIMEOUT_MS}")
        finally:
            cursor.close()


engine = create_async_engine(
    _asyncpg_database_url(_infra.database_url),
    connect_args=_asyncpg_connect_args(_infra.database_url),
    echo=_infra.environment == "development",
    pool_pre_ping=True,
    **_pool_kwargs,
)
_install_statement_timeout(engine.sync_engine)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

sync_engine = create_engine(
    _infra.sync_database_url,
    pool_pre_ping=True,
    **_pool_kwargs,
)
_install_statement_timeout(sync_engine)

SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def system_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


@contextmanager
def sync_system_session() -> Generator[Session, None, None]:
    with SyncSessionLocal() as session:
        yield session


def rows_affected(result: Result[tuple[object, ...]]) -> int:
    """Rows an UPDATE or DELETE touched.

    ``Session.execute`` is typed as returning ``Result`` for every statement,
    but a DML statement always yields a ``CursorResult``, which is where
    ``rowcount`` lives.
    """
    return cast(CursorResult[tuple[object, ...]], result).rowcount or 0
