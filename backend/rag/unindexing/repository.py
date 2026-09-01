from __future__ import annotations

import os

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Sequence

import psycopg

from dotenv import load_dotenv
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag.unindexing.models import KnowledgeSource


load_dotenv()


#------------------------REPOSITORY-------------------------

class KnowledgeSourceRepository:
    def __init__(
        self,
        connection: AsyncConnection[Any],
    ) -> None:
        self.connection = connection

    @classmethod
    async def connect(
        cls,
    ) -> KnowledgeSourceRepository:
        connection = (
            await psycopg.AsyncConnection.connect(
                **_postgres_connect_kwargs(),
                autocommit=True,
                row_factory=dict_row,
            )
        )

        return cls(
            connection
        )

    async def close(
        self,
    ) -> None:
        await self.connection.close()

    #------------------------READ-------------------------

    async def fetch_selected_sources(
        self,
        *,
        tenant_id: str,
        knowledge_source_ids: Sequence[str],
    ) -> dict[str, KnowledgeSource]:
        if not knowledge_source_ids:
            return {}

        cursor = await self.connection.execute(
            """
            SELECT
                knowledge_source_id,
                tenant_id,
                job_id,
                source_id,
                filename,
                status,
                qdrant_collection,
                COALESCE(
                    indexed_chunk_count,
                    0
                ) AS indexed_chunk_count
            FROM knowledge.sources
            WHERE tenant_id = %s
              AND knowledge_source_id = ANY(%s)
            """,
            (
                tenant_id,
                list(
                    knowledge_source_ids
                ),
            ),
        )

        rows = await cursor.fetchall()

        return {
            str(
                row[
                    "knowledge_source_id"
                ]
            ): KnowledgeSource(
                knowledge_source_id=str(
                    row[
                        "knowledge_source_id"
                    ]
                ),
                tenant_id=str(
                    row[
                        "tenant_id"
                    ]
                ),
                job_id=str(
                    row[
                        "job_id"
                    ]
                ),
                source_id=str(
                    row[
                        "source_id"
                    ]
                ),
                filename=str(
                    row[
                        "filename"
                    ]
                ),
                status=str(
                    row[
                        "status"
                    ]
                ),
                qdrant_collection=str(
                    row[
                        "qdrant_collection"
                    ]
                    or ""
                ),
                indexed_chunk_count=int(
                    row[
                        "indexed_chunk_count"
                    ]
                    or 0
                ),
            )
            for row in rows
        }

    #------------------------LIST-------------------------

    async def list_indexed_sources(
        self,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT
                knowledge_source_id,
                filename,
                media_type,
                size_bytes,
                status,
                indexed_chunk_count,
                indexed_at,
                created_at
            FROM knowledge.sources
            WHERE tenant_id = %s
              AND status = 'indexed'
            ORDER BY
                indexed_at DESC NULLS LAST,
                created_at DESC
            """,
            (tenant_id,),
        )

        rows = await cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    #------------------------WRITE-------------------------

    async def mark_unindexed(
        self,
        *,
        source: KnowledgeSource,
        qdrant_points_deleted: int,
    ) -> None:
        metadata = {
            "last_unindex": {
                "qdrant_points_deleted": (
                    qdrant_points_deleted
                ),
                "completed_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }
        }

        async with self.connection.transaction():
            cursor = await self.connection.execute(
                """
                UPDATE knowledge.sources
                SET
                    status = 'unindexed',
                    indexed_chunk_count = 0,
                    unindexed_at = now(),
                    last_error_code = NULL,
                    last_error_message = NULL,
                    metadata = (
                        metadata
                        || %s::jsonb
                    )
                WHERE knowledge_source_id = %s
                  AND tenant_id = %s
                  AND job_id = %s
                  AND source_id = %s
                """,
                (
                    Jsonb(
                        metadata
                    ),
                    source.knowledge_source_id,
                    source.tenant_id,
                    source.job_id,
                    source.source_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    "Postgres knowledge source update "
                    "did not affect exactly one row: "
                    f"{source.knowledge_source_id}"
                )


#------------------------POSTGRES CONFIG-------------------------

def _postgres_connect_kwargs(
) -> dict[str, Any]:
    database = (
        os.getenv(
            "CONVERSATION_DB_NAME",
            "docs-gpt",
        ).strip()
        or "docs-gpt"
    )

    host = os.getenv(
        "CONVERSATION_DB_HOST",
        "",
    ).strip()

    port = os.getenv(
        "CONVERSATION_DB_PORT",
        "5432",
    ).strip()

    user = os.getenv(
        "CONVERSATION_DB_USER",
        "",
    ).strip()

    password = os.getenv(
        "CONVERSATION_DB_PASSWORD",
        "",
    ).strip()

    kwargs: dict[str, Any] = {
        "dbname": database,
        "connect_timeout": 5,
    }

    if host:
        kwargs[
            "host"
        ] = host

    if port:
        kwargs[
            "port"
        ] = int(
            port
        )

    if user:
        kwargs[
            "user"
        ] = user

    if password:
        kwargs[
            "password"
        ] = password

    return kwargs


__all__ = [
    "KnowledgeSourceRepository",
]
