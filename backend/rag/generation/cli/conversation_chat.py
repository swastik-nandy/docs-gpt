from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from rag.conversation import PostgresConversationRepository, extract_entity_mentions
from rag.conversation.models import EntityMention
from rag.generation.conversation_io import (
    build_generation_state_from_turn,
    persist_assistant_turn_from_generation_state,
)
from rag.generation.embedder import RetrievalDenseGenerationEmbedder
from rag.generation.retrieval_node import build_generation_retrieval_node
from rag.generation.service import build_generation_runtime
from tenancy.dev import load_dev_identity_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one full stateful RAG generation turn.",
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
    parser.add_argument("--runtime-config-path", default=None)
    parser.add_argument("--print-state", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    tenant_id = args.tenant_id or dev.tenant_id
    user_id = args.user_id or dev.user_id
    session_id = args.session_id or dev.session_id
    conversation_id = args.conversation_id or dev.conversation_id

    source_constraints = {
        "job_id": args.job_id or dev.job_id,
        "source_id": args.source_id or dev.source_id,
        "parser_doc_id": args.parser_doc_id or dev.parser_doc_id,
        "section_id": args.section_id or dev.section_id,
        "ir_id": args.ir_id or dev.ir_id,
        "payload_filters": {},
    }

    repo = await PostgresConversationRepository.connect()

    await repo.ensure_conversation(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata={"source": "generation_conversation_chat"},
    )

    raw_entities = extract_entity_mentions(args.query)

    user_turn = await repo.insert_turn(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        raw_text=args.query,
        entities_extracted=[
            {
                "entity_text": item.entity_text,
                "entity_type": item.entity_type,
                "salience_rank": item.salience_rank,
            }
            for item in raw_entities
        ],
        metadata={
            "source": "generation_conversation_chat",
            "phase": "pre_graph_user_turn",
        },
    )

    await repo.insert_entities(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_index=user_turn.turn_index,
        entities=raw_entities,
    )

    state = await build_generation_state_from_turn(
        repository=repo,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_turn_id=user_turn.turn_id,
        current_user_message=args.query,
    )

    state["session_id"] = session_id
    state["top_k"] = args.top_k
    state["retrieval_source_constraints"] = source_constraints

    runtime = await build_generation_runtime(
        retrieval_node=build_generation_retrieval_node(
            top_k=args.top_k,
        ),
        embedder=RetrievalDenseGenerationEmbedder(),
        runtime_config_path=args.runtime_config_path,
    )

    try:
        final_state = await runtime.graph.ainvoke(state)
    finally:
        await runtime.close()

    retrieval_state = dict(final_state.get("retrieval_state") or {})
    stateful = dict(retrieval_state.get("stateful_retrieval") or {})

    resolved_query = str(
        stateful.get("resolved_query")
        or retrieval_state.get("rewritten_query")
        or args.query
    )
    query_changed = bool(stateful.get("query_changed"))
    retrieval_trace_id = retrieval_state.get("retrieval_trace_id")

    updated_user_turn = await repo.update_turn_after_retrieval(
        turn_id=user_turn.turn_id,
        resolved_text=resolved_query if query_changed else None,
        retrieval_trace_id=str(retrieval_trace_id) if retrieval_trace_id else None,
        metadata={
            "source": "generation_conversation_chat",
            "phase": "post_graph_user_turn",
            "stateful_retrieval": stateful,
        },
    )

    if retrieval_trace_id:
        await repo.attach_retrieval_trace_to_turn(
            retrieval_trace_id=str(retrieval_trace_id),
            turn_id=user_turn.turn_id,
        )

    resolved_entities = extract_entity_mentions(resolved_query)
    new_entities = _entities_not_in(raw_entities, resolved_entities)

    await repo.insert_entities(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_index=updated_user_turn.turn_index,
        entities=new_entities,
    )

    assistant_turn = await persist_assistant_turn_from_generation_state(
        repository=repo,
        state=final_state,
        metadata={
            "source": "generation_conversation_chat",
            "retrieval_trace_id": retrieval_trace_id,
        },
    )

    answer = str(final_state.get("final_answer") or "").strip()

    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "user_turn_id": user_turn.turn_id,
        "user_turn_index": user_turn.turn_index,
        "assistant_turn_id": assistant_turn.turn_id,
        "assistant_turn_index": assistant_turn.turn_index,
        "raw_query": args.query,
        "resolved_query": resolved_query,
        "query_changed": query_changed,
        "retrieval_trace_id": retrieval_trace_id,
        "resolved_entities_added": [item.entity_text for item in new_entities],
        "answer": answer,
    }

    if args.print_state:
        payload["final_state_keys"] = sorted(final_state.keys())
        payload["retrieval_state"] = retrieval_state
        payload["generation_metadata"] = final_state.get("generation_metadata")
        payload["grounding_result"] = final_state.get("grounding_result")

    print(json.dumps(payload, indent=2, default=str, sort_keys=True))

    await repo.close()


def _entities_not_in(
    existing: tuple[EntityMention, ...],
    candidates: tuple[EntityMention, ...],
) -> tuple[EntityMention, ...]:
    existing_keys = {item.entity_text.lower() for item in existing}
    result: list[EntityMention] = []

    for item in candidates:
        if item.entity_text.lower() in existing_keys:
            continue
        result.append(item)

    return tuple(result)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
