from __future__ import annotations

import argparse
import asyncio
import json
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
        description=(
            "Run a fresh full RAG chatbot session: "
            "Postgres conversation state + retrieval + LLM generation."
        ),
    )

    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)

    parser.add_argument("--conversation-prefix", default="cnv_chat")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runtime-config-path", default=None)

    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--parser-doc-id", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--ir-id", default=None)
    parser.add_argument("--no-source-constraints", action="store_true")

    parser.add_argument(
        "--enable-long-term-memory",
        action="store_true",
        help=(
            "Allow cross-conversation semantic memory retrieval. "
            "Off by default for clean demos."
        ),
    )
    parser.add_argument(
        "--enable-semantic-memory-write",
        action="store_true",
        help=(
            "Allow writing durable semantic memory to Qdrant. "
            "Off by default for clean demos."
        ),
    )

    parser.add_argument("--print-context", action="store_true")
    parser.add_argument("--print-state", action="store_true")
    parser.add_argument("--max-context-chars", type=int, default=420)

    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    tenant_id = args.tenant_id or dev.tenant_id
    user_id = args.user_id or dev.user_id
    session_id = args.session_id or f"ses_chat_{int(time.time())}_{uuid4().hex[:8]}"

    # Critical demo behavior:
    # Always start a fresh conversation on every launch.
    # Never silently reuse dev.conversation_id / cnv_local_dev.
    conversation_id = f"{args.conversation_prefix}_{int(time.time())}_{uuid4().hex[:8]}"

    repo = await PostgresConversationRepository.connect()

    runtime = await build_generation_runtime(
        retrieval_node=build_generation_retrieval_node(
            top_k=args.top_k,
        ),
        embedder=RetrievalDenseGenerationEmbedder(),
        runtime_config_path=args.runtime_config_path,
    )

    await repo.ensure_conversation(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        metadata={
            "source": "generation_cli_chat",
            "fresh_session": True,
            "long_term_memory_enabled": args.enable_long_term_memory,
            "semantic_memory_write_enabled": args.enable_semantic_memory_write,
        },
    )

    print("")
    print("Full RAG chatbot")
    print("----------------")
    print(f"conversation_id: {conversation_id}")
    print(f"session_id:      {session_id}")
    print(f"tenant_id:       {tenant_id}")
    print(f"user_id:         {user_id}")
    print("")
    print("Fresh conversation created. No old Postgres turns/entities are used.")
    print("Commands: :quit, :id, :context, :state")
    print("")

    last_state: Payload | None = None

    try:
        while True:
            try:
                query = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                break

            if not query:
                continue

            if query in {":quit", ":q", "quit", "exit"}:
                break

            if query in {":clear", "clear"}:
                print("\033c", end="")
                continue

            if query == ":id":
                print(f"conversation_id: {conversation_id}")
                continue

            if query == ":context":
                await _print_conversation_context(
                    repo=repo,
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                continue

            if query == ":state":
                if last_state is None:
                    print("No turn has run yet.")
                else:
                    print(json.dumps(last_state, indent=2, default=str, sort_keys=True))
                continue

            last_state = await _run_turn(
                repo=repo,
                runtime=runtime,
                args=args,
                dev=dev,
                query=query,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )

    finally:
        await runtime.close()
        await repo.close()


async def _run_turn(
    *,
    repo: PostgresConversationRepository,
    runtime: Any,
    args: argparse.Namespace,
    dev: Any,
    query: str,
    conversation_id: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
) -> Payload:
    raw_entities = extract_entity_mentions(query)

    user_turn = await repo.insert_turn(
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
            "source": "generation_cli_chat",
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
        current_user_message=query,
    )

    state["session_id"] = session_id
    state["top_k"] = args.top_k
    state["retrieval_source_constraints"] = _source_constraints(args=args, dev=dev)

    # Fresh demo safety:
    # The session is stateful through Postgres conversation rows,
    # but old cross-conversation semantic memory stays off unless explicitly enabled.
    state["disable_long_term_memory"] = not args.enable_long_term_memory
    state["disable_memory_write"] = not args.enable_semantic_memory_write

    final_state = await runtime.graph.ainvoke(state)

    retrieval_state = dict(final_state.get("retrieval_state") or {})
    stateful = dict(retrieval_state.get("stateful_retrieval") or {})
    turn_plan = dict(final_state.get("turn_plan") or {})

    resolved_query = str(
        stateful.get("resolved_query")
        or retrieval_state.get("rewritten_query")
        or query
    )
    query_changed = bool(stateful.get("query_changed"))
    retrieval_trace_id = retrieval_state.get("retrieval_trace_id")

    updated_user_turn = await repo.update_turn_after_retrieval(
        turn_id=user_turn.turn_id,
        resolved_text=resolved_query if query_changed else None,
        retrieval_trace_id=str(retrieval_trace_id) if retrieval_trace_id else None,
        metadata={
            "source": "generation_cli_chat",
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
            "source": "generation_cli_chat",
            "retrieval_trace_id": retrieval_trace_id,
        },
    )

    answer = str(final_state.get("final_answer") or "").strip()

    print("")
    print("assistant>")
    print(answer or "[empty answer]")
    print("")

    _print_debug(
        query=query,
        resolved_query=resolved_query,
        query_changed=query_changed,
        stateful=stateful,
        retrieval_state=retrieval_state,
        final_state=final_state,
        user_turn_id=user_turn.turn_id,
        assistant_turn_id=assistant_turn.turn_id,
        print_context=args.print_context,
        print_state=args.print_state,
        max_context_chars=args.max_context_chars,
    )

    return {
        "conversation_id": conversation_id,
        "user_turn_id": user_turn.turn_id,
        "assistant_turn_id": assistant_turn.turn_id,
        "raw_query": query,
        "resolved_query": resolved_query,
        "query_changed": query_changed,
        "stateful_retrieval": stateful,
        "turn_plan": turn_plan,
        "retrieval_query": final_state.get("retrieval_query"),
        "retrieval_trace_id": retrieval_trace_id,
        "generation_metadata": final_state.get("generation_metadata"),
        "grounding_result": final_state.get("grounding_result"),
        "written_memory_ids": final_state.get("written_memory_ids"),
        "skipped_memory_ids": final_state.get("skipped_memory_ids"),
        "memory_write_warnings": final_state.get("memory_write_warnings"),
    }


def _source_constraints(*, args: argparse.Namespace, dev: Any) -> Payload:
    if args.no_source_constraints:
        return {"payload_filters": {}}

    return {
        "job_id": args.job_id or dev.job_id,
        "source_id": args.source_id or dev.source_id,
        "parser_doc_id": args.parser_doc_id or dev.parser_doc_id,
        "section_id": args.section_id or dev.section_id,
        "ir_id": args.ir_id or dev.ir_id,
        "payload_filters": {},
    }


def _print_debug(
    *,
    query: str,
    resolved_query: str,
    query_changed: bool,
    stateful: Payload,
    retrieval_state: Payload,
    final_state: Payload,
    user_turn_id: str,
    assistant_turn_id: str,
    print_context: bool,
    print_state: bool,
    max_context_chars: int,
) -> None:
    print("debug>")
    print(f"  raw_query:       {query}")
    print(f"  resolved_query:  {resolved_query}")
    print(f"  query_changed:   {query_changed}")
    print(f"  strategy:        {stateful.get('strategy')}")
    print(f"  resolved_entity: {stateful.get('resolved_entity')}")
    turn_plan = dict(final_state.get("turn_plan") or {})
    print(f"  turn_action:     {turn_plan.get('action')}")
    print(f"  retrieval_query: {final_state.get('retrieval_query')}")
    print(f"  retrieval_trace: {retrieval_state.get('retrieval_trace_id')}")
    print(f"  user_turn_id:    {user_turn_id}")
    print(f"  assistant_turn:  {assistant_turn_id}")

    if print_context:
        selected_context = list(retrieval_state.get("selected_context") or [])
        print("")
        print("context>")
        if not selected_context:
            print("  [none]")
        for index, block in enumerate(selected_context, start=1):
            text = str(block.get("text") or "")
            print(
                f"  [{index}] "
                f"source={block.get('source_id')} "
                f"chunk={block.get('chunk_id')} "
                f"score={block.get('score')}"
            )
            print(f"      {text[:max_context_chars]}")

    if print_state:
        payload = {
            "final_state_keys": sorted(final_state.keys()),
            "retrieval_state": retrieval_state,
            "generation_metadata": final_state.get("generation_metadata"),
            "grounding_result": final_state.get("grounding_result"),
            "written_memory_ids": final_state.get("written_memory_ids"),
            "skipped_memory_ids": final_state.get("skipped_memory_ids"),
            "memory_write_warnings": final_state.get("memory_write_warnings"),
        }
        print("")
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))

    print("")


async def _print_conversation_context(
    *,
    repo: PostgresConversationRepository,
    conversation_id: str,
    tenant_id: str,
    user_id: str,
) -> None:
    context = await repo.fetch_context(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_limit=8,
        entity_limit=8,
    )

    print("")
    print("conversation context")
    print("--------------------")
    print("conversation_id:", conversation_id)
    print("recent_turn_count:", len(context.recent_turns))
    print("recent_entity_count:", len(context.recent_entities))
    print("has_summary:", context.has_summary)
    print("working_memory_count:", len(context.working_memory))

    if context.recent_entities:
        print("")
        print("entities:")
        for item in context.recent_entities:
            print(
                f"- {item.entity_text} "
                f"type={item.entity_type} "
                f"turn={item.turn_index} "
                f"rank={item.salience_rank}"
            )

    if context.recent_turns:
        print("")
        print("turns:")
        for item in context.recent_turns:
            text = item.text_for_context.replace("\\n", " ")
            print(f"- {item.turn_index}. {item.role}: {text[:220]}")

    print("")


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
