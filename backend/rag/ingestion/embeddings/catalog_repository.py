from __future__ import annotations

import hashlib
import json
import os

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import psycopg

from dotenv import load_dotenv
from psycopg import sql
from psycopg.types.json import Jsonb

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.artifact_reader import ChunkArtifactFile
from rag.ingestion.embeddings.config import EmbeddingPipelineConfig


load_dotenv()


#------------------------MODELS-------------------------

@dataclass(frozen=True)
class IndexedSourceCatalogRecord:
    tenant_id: str
    job_id: str
    source_id: str
    filename: str
    relative_path: str
    media_type: str
    size_bytes: int
    content_sha256: str
    qdrant_collection: str
    indexed_chunk_count: int
    source_artifact_root: str
    chunked_artifact_root: str
    indexed_artifact_root: str
    upload_count: int
    source_count: int
    duplicate_count: int
    metadata: dict[str, Any]


#------------------------REPOSITORY-------------------------

class KnowledgeCatalogRepository:
    def __init__(
        self,
    ) -> None:
        self._connect_kwargs = _postgres_connect_kwargs()

    def record_indexed_source(
        self,
        record: IndexedSourceCatalogRecord,
    ) -> None:
        with psycopg.connect(
            **self._connect_kwargs
        ) as connection:
            with connection.transaction():
                self._ensure_ingestion_job(
                    connection,
                    record,
                )

                self._upsert_knowledge_source(
                    connection,
                    record,
                )

    def mark_job_succeeded(
        self,
        *,
        tenant_id: str,
        job_id: str,
    ) -> None:
        self._mark_job_status(
            tenant_id=tenant_id,
            job_id=job_id,
            status="succeeded",
        )

    def mark_job_failed(
        self,
        *,
        tenant_id: str,
        job_id: str,
    ) -> None:
        self._mark_job_status(
            tenant_id=tenant_id,
            job_id=job_id,
            status="failed",
        )

    #------------------------JOB-------------------------

    def _ensure_ingestion_job(
        self,
        connection: psycopg.Connection[Any],
        record: IndexedSourceCatalogRecord,
    ) -> None:
        columns = _table_columns(
            connection,
            schema="ingestion",
            table="jobs",
        )

        if not columns:
            raise RuntimeError(
                "Postgres table ingestion.jobs was not found."
            )

        values: dict[str, Any] = {
            "tenant_id": record.tenant_id,
            "job_id": record.job_id,
            "status": "processing",
            "upload_count": record.upload_count,
            "source_count": record.source_count,
            "duplicate_count": record.duplicate_count,
            "source_artifact_root": record.source_artifact_root,
            "output_artifact_root": record.chunked_artifact_root,
            "chunked_artifact_root": record.chunked_artifact_root,
            "indexed_artifact_root": record.indexed_artifact_root,
            "processing_started_at": datetime.now(
                timezone.utc
            ),
            "metadata": Jsonb(
                {
                    "writer": "embedding_indexer",
                    "qdrant_collection": (
                        record.qdrant_collection
                    ),
                }
            ),
        }

        _insert_known_columns(
            connection,
            schema="ingestion",
            table="jobs",
            available_columns=columns,
            values=values,
        )

        assignments: dict[str, Any] = {}

        if "status" in columns:
            assignments["status"] = "processing"

        if (
            "processing_started_at" in columns
        ):
            assignments[
                "processing_started_at"
            ] = datetime.now(
                timezone.utc
            )

        if assignments:
            _update_row(
                connection,
                schema="ingestion",
                table="jobs",
                assignments=assignments,
                where={
                    "tenant_id": record.tenant_id,
                    "job_id": record.job_id,
                },
            )

    #------------------------SOURCE-------------------------

    def _upsert_knowledge_source(
        self,
        connection: psycopg.Connection[Any],
        record: IndexedSourceCatalogRecord,
    ) -> None:
        columns = _table_columns(
            connection,
            schema="knowledge",
            table="sources",
        )

        if not columns:
            raise RuntimeError(
                "Postgres table knowledge.sources was not found."
            )

        now = datetime.now(
            timezone.utc
        )

        values: dict[str, Any] = {
            "tenant_id": record.tenant_id,
            "job_id": record.job_id,
            "source_id": record.source_id,
            "filename": record.filename,
            "relative_path": record.relative_path,
            "media_type": record.media_type,
            "size_bytes": record.size_bytes,
            "content_sha256": record.content_sha256,
            "status": "indexed",
            "qdrant_collection": record.qdrant_collection,
            "indexed_chunk_count": (
                record.indexed_chunk_count
            ),
            "indexed_at": now,
            "unindexed_at": None,
            "last_error_code": None,
            "last_error_message": None,
            "metadata": Jsonb(
                record.metadata
            ),
        }

        _insert_known_columns(
            connection,
            schema="knowledge",
            table="sources",
            available_columns=columns,
            values=values,
        )

        update_values = {
            key: value
            for key, value in values.items()
            if key in columns
            and key
            not in {
                "tenant_id",
                "job_id",
                "source_id",
            }
        }

        _update_row(
            connection,
            schema="knowledge",
            table="sources",
            assignments=update_values,
            where={
                "tenant_id": record.tenant_id,
                "job_id": record.job_id,
                "source_id": record.source_id,
            },
        )

    #------------------------JOB STATUS-------------------------

    def _mark_job_status(
        self,
        *,
        tenant_id: str,
        job_id: str,
        status: str,
    ) -> None:
        with psycopg.connect(
            **self._connect_kwargs
        ) as connection:
            with connection.transaction():
                columns = _table_columns(
                    connection,
                    schema="ingestion",
                    table="jobs",
                )

                assignments: dict[str, Any] = {}

                if "status" in columns:
                    assignments[
                        "status"
                    ] = status

                if "completed_at" in columns:
                    assignments[
                        "completed_at"
                    ] = datetime.now(
                        timezone.utc
                    )

                if "updated_at" in columns:
                    assignments[
                        "updated_at"
                    ] = datetime.now(
                        timezone.utc
                    )

                if not assignments:
                    return

                _update_row(
                    connection,
                    schema="ingestion",
                    table="jobs",
                    assignments=assignments,
                    where={
                        "tenant_id": tenant_id,
                        "job_id": job_id,
                    },
                )


#------------------------RECORD BUILDER-------------------------

def build_indexed_source_catalog_record(
    *,
    artifact_file: ChunkArtifactFile,
    chunks: Sequence[Chunk],
    config: EmbeddingPipelineConfig,
    indexed_chunk_count: int,
) -> IndexedSourceCatalogRecord:
    if not chunks:
        raise ValueError(
            "Cannot build catalog record without chunks."
        )

    source_path = Path(
        str(
            chunks[0].source or ""
        )
    ).resolve()

    if (
        not source_path.exists()
        or not source_path.is_file()
    ):
        raise FileNotFoundError(
            "Original source artifact was not found: "
            f"{source_path}"
        )

    source_job_root = (
        source_path.parent.parent
    )

    manifest = _read_json(
        source_job_root
        / "manifest.json"
    )

    source_entry = _manifest_source_entry(
        manifest,
        source_id=artifact_file.source_id,
    )

    filename = str(
        source_entry.get(
            "filename"
        )
        or source_path.name
    ).strip()

    relative_path = str(
        source_entry.get(
            "path"
        )
        or f"files/{filename}"
    ).strip()

    media_type = str(
        source_entry.get(
            "media_type"
        )
        or _media_type_for_path(
            source_path
        )
    ).strip()

    size_bytes = _as_int(
        source_entry.get(
            "size_bytes"
        ),
        default=source_path.stat().st_size,
    )

    content_sha256 = str(
        source_entry.get(
            "content_sha256"
        )
        or ""
    ).strip()

    if not content_sha256:
        content_sha256 = (
            _sha256_file(
                source_path
            )
        )

    files = manifest.get(
        "files"
    )

    duplicates = manifest.get(
        "duplicates"
    )

    manifest_files = (
        files
        if isinstance(
            files,
            list,
        )
        else []
    )

    manifest_duplicates = (
        duplicates
        if isinstance(
            duplicates,
            list,
        )
        else []
    )

    upload_count = _as_int(
        manifest.get(
            "upload_count"
        ),
        default=(
            len(manifest_files)
            + len(manifest_duplicates)
        ),
    )

    source_count = _as_int(
        manifest.get(
            "source_count"
        ),
        default=len(
            manifest_files
        ),
    )

    duplicate_count = _as_int(
        manifest.get(
            "duplicate_count"
        ),
        default=len(
            manifest_duplicates
        ),
    )

    indexed_root = (
        config.artifacts.indexed_root
        / artifact_file.tenant_id
        / artifact_file.job_id
    )

    return IndexedSourceCatalogRecord(
        tenant_id=artifact_file.tenant_id,
        job_id=artifact_file.job_id,
        source_id=artifact_file.source_id,
        filename=filename,
        relative_path=relative_path,
        media_type=media_type,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        qdrant_collection=(
            config.qdrant.collection_name
        ),
        indexed_chunk_count=(
            indexed_chunk_count
        ),
        source_artifact_root=(
            source_job_root.as_posix()
        ),
        chunked_artifact_root=(
            artifact_file.path
            .parent
            .parent
            .as_posix()
        ),
        indexed_artifact_root=(
            indexed_root.as_posix()
        ),
        upload_count=upload_count,
        source_count=source_count,
        duplicate_count=duplicate_count,
        metadata={
            "source_artifact_path": (
                source_path.as_posix()
            ),
            "chunk_artifact_path": (
                artifact_file.path.as_posix()
            ),
            "embedding_schema_version": (
                config.schema_version
            ),
        },
    )


#------------------------SQL HELPERS-------------------------

def _postgres_connect_kwargs(
) -> dict[str, Any]:
    database = os.getenv(
        "CONVERSATION_DB_NAME",
        "docs-gpt",
    ).strip() or "docs-gpt"

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
        kwargs["host"] = host

    if port:
        kwargs["port"] = int(
            port
        )

    if user:
        kwargs["user"] = user

    if password:
        kwargs[
            "password"
        ] = password

    return kwargs


def _table_columns(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
    table: str,
) -> set[str]:
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        """,
        (
            schema,
            table,
        ),
    ).fetchall()

    return {
        str(
            row[0]
        )
        for row in rows
    }


def _insert_known_columns(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
    table: str,
    available_columns: set[str],
    values: Mapping[str, Any],
) -> None:
    active = {
        key: value
        for key, value
        in values.items()
        if key
        in available_columns
    }

    if not active:
        raise RuntimeError(
            f"No compatible columns found for {schema}.{table}."
        )

    query = sql.SQL(
        "INSERT INTO {}.{} ({}) "
        "VALUES ({}) "
        "ON CONFLICT DO NOTHING"
    ).format(
        sql.Identifier(
            schema
        ),
        sql.Identifier(
            table
        ),
        sql.SQL(", ").join(
            sql.Identifier(
                column
            )
            for column in active
        ),
        sql.SQL(", ").join(
            sql.Placeholder()
            for _ in active
        ),
    )

    connection.execute(
        query,
        list(
            active.values()
        ),
    )


def _update_row(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
    table: str,
    assignments: Mapping[str, Any],
    where: Mapping[str, Any],
) -> None:
    if not assignments:
        return

    set_sql = sql.SQL(
        ", "
    ).join(
        sql.SQL(
            "{} = %s"
        ).format(
            sql.Identifier(
                column
            )
        )
        for column in assignments
    )

    where_sql = sql.SQL(
        " AND "
    ).join(
        sql.SQL(
            "{} = %s"
        ).format(
            sql.Identifier(
                column
            )
        )
        for column in where
    )

    query = sql.SQL(
        "UPDATE {}.{} "
        "SET {} "
        "WHERE {}"
    ).format(
        sql.Identifier(
            schema
        ),
        sql.Identifier(
            table
        ),
        set_sql,
        where_sql,
    )

    connection.execute(
        query,
        [
            *assignments.values(),
            *where.values(),
        ],
    )


#------------------------MANIFEST HELPERS-------------------------

def _manifest_source_entry(
    manifest: Mapping[str, Any],
    *,
    source_id: str,
) -> dict[str, Any]:
    files = manifest.get(
        "files"
    )

    if not isinstance(
        files,
        list,
    ):
        return {}

    for item in files:
        if not isinstance(
            item,
            Mapping,
        ):
            continue

        active_source_id = str(
            item.get(
                "source_id"
            )
            or ""
        ).strip()

        if (
            active_source_id
            == source_id
        ):
            return dict(
                item
            )

    return {}


def _read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        Mapping,
    ):
        return {}

    return dict(
        value
    )


def _media_type_for_path(
    path: Path,
) -> str:
    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".mdx":
        return "text/mdx"

    return "text/markdown"


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _as_int(
    value: Any,
    *,
    default: int,
) -> int:
    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return int(
            default
        )


__all__ = [
    "IndexedSourceCatalogRecord",
    "KnowledgeCatalogRepository",
    "build_indexed_source_catalog_record",
]
