from __future__ import annotations

import argparse
import asyncio
from typing import Any

from rag.conversation import PostgresConversationRepository, extract_entity_mentions
from rag.conversation.models import EntityMention
from rag.retrieval.contracts.request import RetrievalRequest, SourceConstraints, TenantContext
from rag.retrieval.service import RetrievalService, RetrievalServiceConfig
from tenancy.dev import load_dev_identity_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one stateful conversational retrieval turn with Postgres write-back.",
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
    parser.add_argument("--no-store-turn", action="store_true")
    parser.add_argument("--max-text-chars", type=int, default=500)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    tenant_id = args.tenant_id or dev.tenant_id
    user_id = args.user_id or dev.user_id
    session_id = args.session_id or dev.session_id
    conversation_id = args.conversation_id or dev.conversation_id

    repo = await PostgresConversationRepository.connect()

    stored_turn = None
    raw_entities: tuple[EntityMention, ...] = ()

    if not args.no_store_turn:
        await repo.ensure_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={"source": "conversation_retrieve"},
        )

        raw_entities = extract_entity_mentions(args.query)

        stored_turn = await repo.insert_turn(
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
                "source": "conversation_retrieve",
                "phase": "pre_retrieval_user_turn",
            },
        )

        await repo.insert_entities(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            turn_index=stored_turn.turn_index,
            entities=raw_entities,
        )

        print("stored_user_turn_id:", stored_turn.turn_id)
        print("stored_user_turn_index:", stored_turn.turn_index)
        print("raw_entities:", [item.entity_text for item in raw_entities])

    request = RetrievalRequest.create(
        query=args.query,
        tenant=TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
        ),
        conversation_id=conversation_id,
        source_constraints=SourceConstraints(
            job_id=args.job_id or dev.job_id,
            source_id=args.source_id or dev.source_id,
            parser_doc_id=args.parser_doc_id or dev.parser_doc_id,
            section_id=args.section_id or dev.section_id,
            ir_id=args.ir_id or dev.ir_id,
        ),
        top_k=args.top_k,
        metadata={
            "cli": "conversation_retrieve",
        },
    )

    service = RetrievalService(
        config=RetrievalServiceConfig(
            include_debug_metadata=True,
            enable_stateful_resolution=True,
        )
    )

    output = await service.retrieve(
        request,
        metadata={
            "source": "conversation_retrieve",
        },
    )

    metadata = output.get("metadata") or {}
    stateful = metadata.get("stateful_retrieval") or {}
    retrieval_trace_id = metadata.get("retrieval_trace_id")

    raw_query = str(stateful.get("raw_query") or args.query)
    resolved_query = str(stateful.get("resolved_query") or args.query)
    query_changed = bool(stateful.get("query_changed"))

    if stored_turn is not None:
        resolved_text_for_db = resolved_query if query_changed else None

        updated_turn = await repo.update_turn_after_retrieval(
            turn_id=stored_turn.turn_id,
            resolved_text=resolved_text_for_db,
            retrieval_trace_id=str(retrieval_trace_id) if retrieval_trace_id else None,
            metadata={
                "phase": "post_retrieval_user_turn",
                "stateful_retrieval": stateful,
            },
        )

        if retrieval_trace_id:
            await repo.attach_retrieval_trace_to_turn(
                retrieval_trace_id=str(retrieval_trace_id),
                turn_id=stored_turn.turn_id,
            )

        resolved_entities = extract_entity_mentions(resolved_query)
        new_entities = _entities_not_in(raw_entities, resolved_entities)

        await repo.insert_entities(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            turn_index=updated_turn.turn_index,
            entities=new_entities,
        )

        print("updated_user_turn_id:", updated_turn.turn_id)
        print("updated_resolved_text:", updated_turn.resolved_text)
        print("retrieval_trace_id:", updated_turn.retrieval_trace_id)
        print("resolved_entities_added:", [item.entity_text for item in new_entities])

    await repo.close()

    _print_output(
        output=output,
        max_text_chars=args.max_text_chars,
        raw_query=raw_query,
        resolved_query=resolved_query,
    )


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


def _print_output(
    *,
    output: dict[str, Any],
    max_text_chars: int,
    raw_query: str,
    resolved_query: str,
) -> None:
    metadata = output.get("metadata") or {}
    stateful = metadata.get("stateful_retrieval") or {}
    decision = output.get("decision") or {}
    context_blocks = output.get("context_blocks") or []
    errors = output.get("errors") or []
    warnings = output.get("warnings") or []

    print("stateful_raw_query:", stateful.get("raw_query") or raw_query)
    print("stateful_resolved_query:", stateful.get("resolved_query") or resolved_query)
    print("stateful_query_changed:", stateful.get("query_changed"))
    print("stateful_strategy:", stateful.get("strategy"))
    print("stateful_resolved_entity:", stateful.get("resolved_entity"))
    print("output_retrieval_trace_id:", metadata.get("retrieval_trace_id"))
    print("decision:", decision)
    print("errors:", len(errors))
    print("warnings:", len(warnings))
    print("context_blocks:", len(context_blocks))

    if errors:
        print("\nErrors")
        print("------")
        for error in errors:
            print(error)

    if warnings:
        print("\nWarnings")
        print("--------")
        for warning in warnings:
            print(warning)

    for index, block in enumerate(context_blocks, start=1):
        print(f"\n--- context {index} ---")
        print("source_id:", getattr(block, "source_id", None))
        print("chunk_id:", getattr(block, "chunk_id", None))
        print("text:", getattr(block, "text", "")[:max_text_chars])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
