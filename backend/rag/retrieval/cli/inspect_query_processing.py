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
from rag.retrieval.query_processing.processor import AdaptiveQueryProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect adaptive query processing without running retrieval.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
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
        ),
        metadata={
            "cli": "inspect_query_processing",
        },
    )

    plan = await AdaptiveQueryProcessor().process(request)
    plan.validate()

    payload: dict[str, Any] = {
        "planner": plan.planner,
        "query": plan.query,
        "intent": plan.intent.value,
        "defects": [item.value for item in plan.defects],
        "retrieval_mode": plan.retrieval_mode.value,
        "tool_hints": [item.value for item in plan.tool_hints],
        "semantic_queries": list(plan.semantic_queries),
        "exact_terms": list(plan.exact_terms),
        "source_hints": list(plan.source_hints),
        "requires_memory": plan.requires_memory,
        "needs_exact_search": plan.needs_exact_search,
        "needs_source_navigation": plan.needs_source_navigation,
        "needs_agentic_loop": plan.needs_agentic_loop,
        "confidence": plan.confidence,
        "reasoning": plan.reasoning,
        "metadata": plan.metadata,
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
