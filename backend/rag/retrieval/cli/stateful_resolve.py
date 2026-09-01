from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from rag.retrieval.contracts.request import RetrievalRequest, SourceConstraints, TenantContext
from rag.retrieval.stateful import prepare_stateful_retrieval_request
from tenancy.dev import load_dev_identity_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a conversational retrieval query using Postgres state.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--parser-doc-id", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--ir-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    request = RetrievalRequest.create(
        query=args.query,
        tenant=TenantContext(
            tenant_id=args.tenant_id or dev.tenant_id,
            user_id=args.user_id or dev.user_id,
            session_id=args.session_id or dev.session_id,
        ),
        conversation_id=args.conversation_id or dev.conversation_id,
        source_constraints=SourceConstraints(
            job_id=args.job_id or dev.job_id,
            source_id=args.source_id or dev.source_id,
            parser_doc_id=args.parser_doc_id or dev.parser_doc_id,
            section_id=args.section_id or dev.section_id,
            ir_id=args.ir_id or dev.ir_id,
        ),
        top_k=args.top_k,
        metadata={
            "cli": "stateful_resolve",
        },
    )

    prepared = await prepare_stateful_retrieval_request(request)

    context = prepared.context
    payload: dict[str, Any] = {
        "raw_query": prepared.resolution.raw_query,
        "resolved_query": prepared.resolution.resolved_query,
        "query_changed": prepared.query_changed,
        "needed_resolution": prepared.resolution.needed_resolution,
        "resolved_entity": prepared.resolution.resolved_entity,
        "strategy": prepared.resolution.strategy,
        "confidence": prepared.resolution.confidence,
        "conversation_id": prepared.request.conversation_id,
        "tenant_id": prepared.request.tenant.tenant_id,
        "user_id": prepared.request.tenant.user_id,
        "session_id": prepared.request.tenant.session_id,
        "context": {
            "recent_turn_count": len(context.recent_turns) if context else 0,
            "recent_entity_count": len(context.recent_entities) if context else 0,
            "recent_entities": [
                {
                    "entity_text": item.entity_text,
                    "entity_type": item.entity_type,
                    "turn_index": item.turn_index,
                    "salience_rank": item.salience_rank,
                }
                for item in (context.recent_entities if context else ())
            ],
            "has_summary": context.has_summary if context else False,
            "working_memory_count": len(context.working_memory) if context else 0,
        },
        "request_metadata": prepared.request.metadata.get("stateful_retrieval"),
    }

    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
