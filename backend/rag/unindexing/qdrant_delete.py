from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from rag.ingestion.embeddings.config import QdrantConfig
from rag.ingestion.embeddings.config import QdrantMode
from rag.ingestion.embeddings.config import load_embedding_pipeline_config


#------------------------RESULT-------------------------

@dataclass(frozen=True)
class QdrantDeleteResult:
    collection_name: str
    points_before: int
    points_deleted: int
    points_remaining: int


#------------------------STORE-------------------------

class QdrantKnowledgeDeleteStore:
    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        default_collection_name: str,
    ) -> None:
        self.client = client
        self.default_collection_name = (
            default_collection_name
        )

    @classmethod
    def from_embedding_config(
        cls,
        config_path: str | Path | None = None,
    ) -> QdrantKnowledgeDeleteStore:
        pipeline_config = (
            load_embedding_pipeline_config(
                config_path
            )
        )

        qdrant = pipeline_config.qdrant

        client = _build_client(
            qdrant
        )

        return cls(
            client=client,
            default_collection_name=(
                qdrant.collection_name
            ),
        )

    async def close(
        self,
    ) -> None:
        await self.client.close()

    #------------------------FILTER-------------------------

    def source_filter(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
    ) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id
                    ),
                ),
                models.FieldCondition(
                    key="job_id",
                    match=models.MatchValue(
                        value=job_id
                    ),
                ),
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(
                        value=source_id
                    ),
                ),
            ]
        )

    #------------------------COUNT-------------------------

    async def count_source(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
        collection_name: str | None = None,
    ) -> int:
        collection = (
            collection_name
            or self.default_collection_name
        )

        query_filter = self.source_filter(
            tenant_id=tenant_id,
            job_id=job_id,
            source_id=source_id,
        )

        result = await self.client.count(
            collection_name=collection,
            count_filter=query_filter,
            exact=True,
        )

        return int(
            result.count
        )

    #------------------------DELETE SOURCE-------------------------

    async def delete_source(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
        collection_name: str | None = None,
    ) -> QdrantDeleteResult:
        collection = (
            collection_name
            or self.default_collection_name
        )

        query_filter = self.source_filter(
            tenant_id=tenant_id,
            job_id=job_id,
            source_id=source_id,
        )

        before = await self.client.count(
            collection_name=collection,
            count_filter=query_filter,
            exact=True,
        )

        points_before = int(
            before.count
        )

        if points_before > 0:
            await self.client.delete(
                collection_name=collection,
                points_selector=(
                    models.FilterSelector(
                        filter=query_filter
                    )
                ),
                wait=True,
            )

        after = await self.client.count(
            collection_name=collection,
            count_filter=query_filter,
            exact=True,
        )

        points_remaining = int(
            after.count
        )

        if points_remaining != 0:
            raise RuntimeError(
                "Qdrant source deletion verification "
                "failed: "
                f"tenant_id={tenant_id!r} "
                f"job_id={job_id!r} "
                f"source_id={source_id!r} "
                f"collection={collection!r} "
                f"remaining={points_remaining}"
            )

        return QdrantDeleteResult(
            collection_name=collection,
            points_before=points_before,
            points_deleted=points_before,
            points_remaining=0,
        )


#------------------------CLIENT-------------------------

def _build_client(
    config: QdrantConfig,
) -> AsyncQdrantClient:
    if config.mode == QdrantMode.MEMORY:
        return AsyncQdrantClient(
            ":memory:"
        )

    if config.mode == QdrantMode.REMOTE:
        if not config.url:
            raise ValueError(
                "Qdrant URL is required for "
                "remote unindexing."
            )

        return AsyncQdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            prefer_grpc=config.prefer_grpc,
        )

    raise ValueError(
        "Unsupported Qdrant mode for "
        f"unindexing: {config.mode!r}"
    )


__all__ = [
    "QdrantDeleteResult",
    "QdrantKnowledgeDeleteStore",
]
