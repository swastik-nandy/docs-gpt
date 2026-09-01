from __future__ import annotations

import argparse
import asyncio
import json

from rag.unindexing.qdrant_delete import QdrantKnowledgeDeleteStore
from rag.unindexing.repository import KnowledgeSourceRepository
from rag.unindexing.service import UnindexingService


#------------------------CLI-------------------------

async def _run(
    *,
    tenant_id: str,
    knowledge_source_ids: list[str],
) -> int:
    repository = (
        await KnowledgeSourceRepository.connect()
    )

    qdrant = (
        QdrantKnowledgeDeleteStore.from_embedding_config()
    )

    try:
        service = UnindexingService(
            repository=repository,
            qdrant=qdrant,
        )

        result = await service.unindex_sources(
            tenant_id=tenant_id,
            knowledge_source_ids=(
                knowledge_source_ids
            ),
        )

        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )

        return (
            0
            if result.failed_count == 0
            else 1
        )

    finally:
        await qdrant.close()
        await repository.close()


def main(
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove selected knowledge sources "
            "from Qdrant and reconcile Postgres."
        )
    )

    parser.add_argument(
        "--tenant-id",
        required=True,
    )

    parser.add_argument(
        "knowledge_source_ids",
        nargs="+",
    )

    args = parser.parse_args()

    return asyncio.run(
        _run(
            tenant_id=args.tenant_id,
            knowledge_source_ids=(
                args.knowledge_source_ids
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
