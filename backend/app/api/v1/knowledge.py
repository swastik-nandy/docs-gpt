from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException

from app.core.settings import Settings, get_settings
from app.core.tenancy import TenantResolutionError, resolve_tenant_context
from app.schemas.knowledge import (
    KnowledgeSourceResponse,
    KnowledgeSourcesResponse,
    UnindexSourcesRequest,
    UnindexSourcesResponse,
)
from rag.unindexing.qdrant_delete import QdrantKnowledgeDeleteStore
from rag.unindexing.repository import KnowledgeSourceRepository
from rag.unindexing.service import UnindexingService


router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
)


#------------------------DEPENDENCIES-------------------------

async def get_knowledge_repository(
) -> AsyncIterator[KnowledgeSourceRepository]:
    repository = await KnowledgeSourceRepository.connect()

    try:
        yield repository

    finally:
        await repository.close()


async def get_qdrant_delete_store(
) -> AsyncIterator[QdrantKnowledgeDeleteStore]:
    store = (
        QdrantKnowledgeDeleteStore
        .from_embedding_config()
    )

    try:
        yield store

    finally:
        await store.close()


#------------------------AVAILABLE KNOWLEDGE-------------------------

@router.get(
    "/sources",
    response_model=KnowledgeSourcesResponse,
)
async def list_knowledge_sources(
    settings: Settings = Depends(get_settings),
    repository: KnowledgeSourceRepository = Depends(
        get_knowledge_repository
    ),
) -> KnowledgeSourcesResponse:
    try:
        tenant = resolve_tenant_context(
            requested_tenant_id=None,
            settings=settings,
        )

    except TenantResolutionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    rows = await repository.list_indexed_sources(
        tenant_id=tenant.tenant_id,
    )

    sources = [
        KnowledgeSourceResponse(
            knowledge_source_id=str(
                row["knowledge_source_id"]
            ),
            filename=str(
                row["filename"]
            ),
            media_type=str(
                row["media_type"]
            ),
            size_bytes=int(
                row["size_bytes"]
            ),
            status="indexed",
            indexed_chunk_count=int(
                row["indexed_chunk_count"]
                or 0
            ),
            indexed_at=row["indexed_at"],
            created_at=row["created_at"],
        )
        for row in rows
    ]

    return KnowledgeSourcesResponse(
        total=len(sources),
        sources=sources,
    )


#------------------------UNINDEX-------------------------

@router.post(
    "/sources/unindex",
    response_model=UnindexSourcesResponse,
)
async def unindex_knowledge_sources(
    request: UnindexSourcesRequest,
    settings: Settings = Depends(get_settings),
    repository: KnowledgeSourceRepository = Depends(
        get_knowledge_repository
    ),
    qdrant: QdrantKnowledgeDeleteStore = Depends(
        get_qdrant_delete_store
    ),
) -> UnindexSourcesResponse:
    try:
        tenant = resolve_tenant_context(
            requested_tenant_id=None,
            settings=settings,
        )

        service = UnindexingService(
            repository=repository,
            qdrant=qdrant,
        )

        result = await service.unindex_sources(
            tenant_id=tenant.tenant_id,
            knowledge_source_ids=(
                request.knowledge_source_ids
            ),
        )

    except TenantResolutionError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return UnindexSourcesResponse.model_validate(
        result.to_dict()
    )
