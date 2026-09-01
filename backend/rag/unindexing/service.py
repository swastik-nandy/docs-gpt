from __future__ import annotations

import logging

from collections.abc import Sequence

from rag.unindexing.models import UnindexBatchResult
from rag.unindexing.models import UnindexSourceResult
from rag.unindexing.qdrant_delete import QdrantKnowledgeDeleteStore
from rag.unindexing.repository import KnowledgeSourceRepository


logger = logging.getLogger(
    __name__
)


#------------------------SERVICE-------------------------

class UnindexingService:
    def __init__(
        self,
        *,
        repository: KnowledgeSourceRepository,
        qdrant: QdrantKnowledgeDeleteStore,
    ) -> None:
        self.repository = repository
        self.qdrant = qdrant

    async def unindex_sources(
        self,
        *,
        tenant_id: str,
        knowledge_source_ids: Sequence[str],
    ) -> UnindexBatchResult:
        normalized_ids = (
            _normalize_source_ids(
                knowledge_source_ids
            )
        )

        if not normalized_ids:
            raise ValueError(
                "At least one knowledge source "
                "must be selected."
            )

        sources = (
            await self.repository.fetch_selected_sources(
                tenant_id=tenant_id,
                knowledge_source_ids=normalized_ids,
            )
        )

        results: list[
            UnindexSourceResult
        ] = []

        for knowledge_source_id in normalized_ids:
            source = sources.get(
                knowledge_source_id
            )

            if source is None:
                results.append(
                    UnindexSourceResult(
                        knowledge_source_id=(
                            knowledge_source_id
                        ),
                        filename="",
                        status="not_found",
                        qdrant_points_deleted=0,
                        qdrant_points_remaining=0,
                        error=(
                            "Knowledge source was not "
                            "found for this tenant."
                        ),
                    )
                )

                continue

            try:
                delete_result = (
                    await self.qdrant.delete_source(
                        tenant_id=source.tenant_id,
                        job_id=source.job_id,
                        source_id=source.source_id,
                        collection_name=(
                            source.qdrant_collection
                            or None
                        ),
                    )
                )

            except Exception as exc:
                logger.exception(
                    "Qdrant source unindex failed | "
                    "knowledge_source_id=%s "
                    "tenant_id=%s "
                    "job_id=%s "
                    "source_id=%s",
                    source.knowledge_source_id,
                    source.tenant_id,
                    source.job_id,
                    source.source_id,
                )

                results.append(
                    UnindexSourceResult(
                        knowledge_source_id=(
                            source.knowledge_source_id
                        ),
                        filename=source.filename,
                        status="failed",
                        qdrant_points_deleted=0,
                        qdrant_points_remaining=-1,
                        error=str(
                            exc
                        ),
                    )
                )

                continue

            # Qdrant has been verified empty here.
            # Only now may Postgres claim unindexed.
            try:
                await self.repository.mark_unindexed(
                    source=source,
                    qdrant_points_deleted=(
                        delete_result.points_deleted
                    ),
                )

            except Exception as exc:
                logger.exception(
                    "Postgres source reconciliation "
                    "failed after Qdrant deletion | "
                    "knowledge_source_id=%s "
                    "tenant_id=%s "
                    "job_id=%s "
                    "source_id=%s",
                    source.knowledge_source_id,
                    source.tenant_id,
                    source.job_id,
                    source.source_id,
                )

                results.append(
                    UnindexSourceResult(
                        knowledge_source_id=(
                            source.knowledge_source_id
                        ),
                        filename=source.filename,
                        status="failed",
                        qdrant_points_deleted=(
                            delete_result.points_deleted
                        ),
                        qdrant_points_remaining=0,
                        error=(
                            "Qdrant is empty for this "
                            "source but Postgres update "
                            f"failed: {exc}"
                        ),
                    )
                )

                continue

            results.append(
                UnindexSourceResult(
                    knowledge_source_id=(
                        source.knowledge_source_id
                    ),
                    filename=source.filename,
                    status="unindexed",
                    qdrant_points_deleted=(
                        delete_result.points_deleted
                    ),
                    qdrant_points_remaining=0,
                )
            )

        succeeded_count = sum(
            result.status == "unindexed"
            for result in results
        )

        return UnindexBatchResult(
            tenant_id=tenant_id,
            requested_count=len(
                normalized_ids
            ),
            succeeded_count=(
                succeeded_count
            ),
            failed_count=(
                len(results)
                - succeeded_count
            ),
            results=tuple(
                results
            ),
        )


#------------------------NORMALIZATION-------------------------

def _normalize_source_ids(
    values: Sequence[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        cleaned = str(
            value or ""
        ).strip()

        if not cleaned:
            continue

        if cleaned in seen:
            continue

        seen.add(
            cleaned
        )

        output.append(
            cleaned
        )

    return tuple(
        output
    )


__all__ = [
    "UnindexingService",
]
