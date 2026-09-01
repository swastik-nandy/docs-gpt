from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


#------------------------SOURCE-------------------------

@dataclass(frozen=True)
class KnowledgeSource:
    knowledge_source_id: str
    tenant_id: str
    job_id: str
    source_id: str
    filename: str
    status: str
    qdrant_collection: str
    indexed_chunk_count: int


#------------------------RESULTS-------------------------

UnindexStatus = Literal[
    "unindexed",
    "not_found",
    "failed",
]


@dataclass(frozen=True)
class UnindexSourceResult:
    knowledge_source_id: str
    filename: str
    status: UnindexStatus
    qdrant_points_deleted: int
    qdrant_points_remaining: int
    error: str = ""

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "knowledge_source_id": self.knowledge_source_id,
            "filename": self.filename,
            "status": self.status,
            "qdrant_points_deleted": (
                self.qdrant_points_deleted
            ),
            "qdrant_points_remaining": (
                self.qdrant_points_remaining
            ),
            "error": self.error,
        }


@dataclass(frozen=True)
class UnindexBatchResult:
    tenant_id: str
    requested_count: int
    succeeded_count: int
    failed_count: int
    results: tuple[
        UnindexSourceResult,
        ...
    ]

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "requested_count": self.requested_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


__all__ = [
    "KnowledgeSource",
    "UnindexBatchResult",
    "UnindexSourceResult",
]
