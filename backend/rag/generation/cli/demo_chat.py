from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any
from uuid import uuid4

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


Payload = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean demo chatbot: fresh stateful RAG conversation per launch.",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runtime-config-path", default=None)

    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)

    # Demo default: do not accidentally restrict retrieval with stale dev IDs.
    parser.add_argument(
        "--use-source-constraints",
        action="store_true",
        help="Use dev/source constraints instead of broad tenant retrieval.",
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--parser-doc-id", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--ir-id", default=None)

    # Demo default: fresh conversation state only.
    parser.add_argument("--enable-long-term-memory", action="store_true")
    parser.add_argument("--enable-semantic-memory-write", action="store_true")

    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    tenant_id = args.tenant_id or dev.tenant_id
    user_id = args.user_id or dev.user_id
    session_id = args.session_id or f"ses_demo_{int(time.time())}_{uuid4().hex[:8]}"

    # Critical demo behavior:
    # every launch creates a new conversation, so no old Postgres chat history leaks in.
    conversation_id = f"cnv_demo_{int(time.time())}_{uuid4().hex[:8]}"

    repository = await PostgresConversationRepository.connect()

    runtime = await build_generation_runtime(
        retrieval_node=build_generation_retrieval_node(
            top_k=args.top_k,
        ),
        embedder=RetrievalDenseGenerationEmbedder(),
        runtime_config_path=args.runtime_config_path,
    )

    await repository.ensure_conversation(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata={
            "source": "generation_demo_chat",
            "fresh_session": True,
        },
    )

    print("Demo chat ready. Type :quit to exit.")

    try:
        while True:
            try:
                query = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query:
                continue

            if query in {":quit", ":q", "quit", "exit"}:
                break

            if query in {":clear", "clear"}:
                print("\033c", end="")
                continue

            answer, model_used = await run_turn(
                repository=repository,
                runtime=runtime,
                args=args,
                dev=dev,
                query=query,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )

            print()
            print(f"assistant [{model_used}]>")
            print(answer)
            print()

    finally:
        await runtime.close()
        await repository.close()


async def run_turn(
    *,
    repository: PostgresConversationRepository,
    runtime: Any,
    args: argparse.Namespace,
    dev: Any,
    query: str,
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
) -> tuple[str, str]:
    raw_entities = extract_entity_mentions(query)

    user_turn = await repository.insert_turn(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        raw_text=query,
        entities_extracted=[
            {
                "entity_text": item.entity_text,
                "entity_type": item.entity_type,
                "salience_rank": item.salience_rank,
            }
            for item in raw_entities
        ],
        metadata={
            "source": "generation_demo_chat",
            "phase": "pre_graph_user_turn",
        },
    )

    await repository.insert_entities(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_index=user_turn.turn_index,
        entities=raw_entities,
    )

    state = await build_generation_state_from_turn(
        repository=repository,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_turn_id=user_turn.turn_id,
        current_user_message=query,
    )

    state["session_id"] = session_id
    state["top_k"] = args.top_k
    # Full-collection conversation test: search across all indexed Qdrant docs.
    state["retrieval_source_constraints"] = {}

    # Fresh demo mode:
    # Postgres conversation memory is active within this session.
    # Cross-conversation semantic memory is disabled unless explicitly enabled.
    state["disable_long_term_memory"] = not args.enable_long_term_memory
    state["disable_memory_write"] = not args.enable_semantic_memory_write

    final_state = await runtime.graph.ainvoke(state)

    retrieval_state = dict(final_state.get("retrieval_state") or {})
    stateful = dict(retrieval_state.get("stateful_retrieval") or {})

    resolved_query = str(
        stateful.get("resolved_query")
        or retrieval_state.get("rewritten_query")
        or query
    )
    query_changed = bool(stateful.get("query_changed"))
    retrieval_trace_id = retrieval_state.get("retrieval_trace_id")

    updated_user_turn = await repository.update_turn_after_retrieval(
        turn_id=user_turn.turn_id,
        resolved_text=resolved_query if query_changed else None,
        retrieval_trace_id=str(retrieval_trace_id) if retrieval_trace_id else None,
        metadata={
            "source": "generation_demo_chat",
            "phase": "post_graph_user_turn",
            "stateful_retrieval": stateful,
            "turn_plan": final_state.get("turn_plan") or {},
        },
    )

    if retrieval_trace_id:
        await repository.attach_retrieval_trace_to_turn(
            retrieval_trace_id=str(retrieval_trace_id),
            turn_id=user_turn.turn_id,
        )

    resolved_entities = extract_entity_mentions(resolved_query)
    new_entities = entities_not_in(raw_entities, resolved_entities)

    await repository.insert_entities(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_index=updated_user_turn.turn_index,
        entities=new_entities,
    )

    assistant_turn = await persist_assistant_turn_from_generation_state(
        repository=repository,
        state=final_state,
        metadata={
            "source": "generation_demo_chat",
            "retrieval_trace_id": retrieval_trace_id,
            "turn_plan": final_state.get("turn_plan") or {},
        },
    )

    answer = str(final_state.get("final_answer") or "").strip()
    if not answer:
        answer = "I do not have enough grounded information to answer that confidently."

    return answer, model_label(final_state)


def source_constraints(*, args: argparse.Namespace, dev: Any) -> Payload:
    if not args.use_source_constraints:
        return {"payload_filters": {}}

    return {
        "job_id": args.job_id or dev.job_id,
        "source_id": args.source_id or dev.source_id,
        "parser_doc_id": args.parser_doc_id or dev.parser_doc_id,
        "section_id": args.section_id or dev.section_id,
        "ir_id": args.ir_id or dev.ir_id,
        "payload_filters": {},
    }


def model_label(state: Payload) -> str:
    metadata = dict(state.get("generation_metadata") or {})

    provider = str(metadata.get("provider") or "unknown").strip()
    model = str(metadata.get("model") or "unknown").strip()

    if provider and model:
        return f"{provider}:{model}"

    return model or provider or "unknown"


def entities_not_in(
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