from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


#------------------------SOURCES-------------------------

class KnowledgeSourceResponse(BaseModel):
    knowledge_source_id: str
    filename: str
    media_type: str
    size_bytes: int
    status: Literal["indexed"]
    indexed_chunk_count: int
    indexed_at: datetime | None
    created_at: datetime


class KnowledgeSourcesResponse(BaseModel):
    total: int
    sources: list[KnowledgeSourceResponse]


#------------------------UNINDEX REQUEST-------------------------

class UnindexSourcesRequest(BaseModel):
    knowledge_source_ids: list[str] = Field(
        min_length=1,
        max_length=100,
    )


#------------------------UNINDEX RESPONSE-------------------------

class UnindexSourceResponse(BaseModel):
    knowledge_source_id: str
    filename: str
    status: Literal[
        "unindexed",
        "not_found",
        "failed",
    ]
    qdrant_points_deleted: int
    qdrant_points_remaining: int
    error: str = ""


class UnindexSourcesResponse(BaseModel):
    tenant_id: str
    requested_count: int
    succeeded_count: int
    failed_count: int
    results: list[UnindexSourceResponse]
