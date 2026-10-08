from __future__ import annotations

import os

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from environment import load_environment


# --------- ENVIRONMENT ------------------------

load_environment()

DB_URL = str(
    os.getenv("DB_URL")
    or ""
).strip()

if not DB_URL:
    raise RuntimeError(
        "DB_URL is required"
    )


# --------- ENGINE ------------------------

engine = create_async_engine(
    DB_URL,
    pool_pre_ping=True,

    # Keep each Cloud Run instance modest.
    pool_size=4,
    max_overflow=2,
)


# --------- SESSION FACTORY ------------------------

SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# --------- SESSION ------------------------

@asynccontextmanager
async def session_scope() -> AsyncIterator[
    AsyncSession
]:
    async with SessionFactory() as session:
        async with session.begin():
            yield session


# --------- SHUTDOWN ------------------------

async def close_database() -> None:
    await engine.dispose()
