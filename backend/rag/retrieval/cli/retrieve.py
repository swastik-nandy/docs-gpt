from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from rag.retrieval.contracts.request import (
    RetrievalRequest,
    SourceConstraints,
    TenantContext,
)
from rag.retrieval.service import RetrievalService, RetrievalServiceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive retrieval through the LangGraph-only service.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--parser-doc-id", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--ir-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--enable-stateful-resolution", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


async def run() -> None:
    args = parse_args()

    request = RetrievalRequest.create(
        query=args.query,
        tenant=TenantContext(
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            session_id=args.session_id,
        ),
        conversation_id=args.conversation_id,
        source_constraints=SourceConstraints(
            job_id=args.job_id,
            source_id=args.source_id,
            parser_doc_id=args.parser_doc_id,
            section_id=args.section_id,
            ir_id=args.ir_id,
        ),
        top_k=args.top_k,
        metadata={
            "cli": "retrieve",
        },
    )

    service = RetrievalService(
        config=RetrievalServiceConfig(
            include_debug_metadata=not args.no_debug,
            enable_stateful_resolution=args.enable_stateful_resolution,
        )
    )

    output = await service.retrieve(
        request,
        metadata={
            "source": "cli.retrieve",
            "orchestration": "langgraph",
        },
    )

    payload: dict[str, Any] = {
        "normalized_query": output.get("normalized_query"),
        "decision": output.get("decision"),
        "context_blocks": output.get("context_blocks", ()),
        "reranked_candidates": output.get("reranked_candidates", ()),
        "trace": output.get("trace", ()),
        "warnings": output.get("warnings", ()),
        "errors": output.get("errors", ()),
        "metadata": output.get("metadata", {}),
    }

    print(
        json.dumps(
            payload,
            indent=2 if args.pretty else None,
            sort_keys=True,
            default=str,
        )
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
