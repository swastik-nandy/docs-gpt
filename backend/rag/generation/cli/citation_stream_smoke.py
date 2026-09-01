from __future__ import annotations

import asyncio
import re
import time

from typing import Any
from uuid import uuid4

from rag.conversation import PostgresConversationRepository
from rag.conversation import extract_entity_mentions
from rag.conversation.models import EntityMention
from rag.generation.conversation_io import build_generation_state_from_turn
from rag.generation.conversation_io import persist_assistant_turn_from_generation_state
from rag.generation.embedder import RetrievalDenseGenerationEmbedder
from rag.generation.retrieval_node import build_generation_retrieval_node
from rag.generation.service import build_generation_runtime
from rag.generation.streaming import GenerationStreamEventType
from tenancy.dev import load_dev_identity_config


QUERY = "What is vector similarity search?"

_INTERNAL_DOC_ID_RE = re.compile(
    r"doc_[a-f0-9]+:[A-Za-z0-9_:-]+",
    re.IGNORECASE,
)


#--------------------HELPERS--------------------

def entities_not_in(
    existing: tuple[EntityMention, ...],
    candidates: tuple[EntityMention, ...],
) -> tuple[EntityMention, ...]:
    existing_keys = {
        item.entity_text.lower()
        for item in existing
    }

    return tuple(
        item
        for item in candidates
        if item.entity_text.lower()
        not in existing_keys
    )


def human_citation_names(
    final_state: dict[str, Any],
) -> list[str]:
    retrieval_state = dict(
        final_state.get(
            "retrieval_state"
        )
        or {}
    )

    citations = list(
        retrieval_state.get(
            "citations"
        )
        or []
    )

    names: list[str] = []

    for citation in citations:
        if not isinstance(
            citation,
            dict,
        ):
            continue

        name = str(
            citation.get(
                "source_name"
            )
            or citation.get(
                "citation_label"
            )
            or ""
        ).strip()

        if (
            name
            and name not in names
        ):
            names.append(
                name
            )

    return names


#--------------------MAIN--------------------

async def main() -> None:
    dev = load_dev_identity_config()

    tenant_id = dev.tenant_id
    user_id = dev.user_id

    conversation_id = (
        "cnv_citation_test_"
        f"{int(time.time())}_"
        f"{uuid4().hex[:8]}"
    )

    session_id = (
        "ses_citation_test_"
        f"{uuid4().hex[:8]}"
    )

    repository = (
        await
        PostgresConversationRepository
        .connect()
    )

    runtime = await build_generation_runtime(
        retrieval_node=(
            build_generation_retrieval_node(
                top_k=5,
            )
        ),
        embedder=(
            RetrievalDenseGenerationEmbedder()
        ),
    )

    try:
        #--------------------CONVERSATION--------------------

        await repository.ensure_conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "source": "citation_stream_smoke",
                "fresh_session": True,
            },
        )

        print()
        print("CITATION STREAM SMOKE")
        print("---------------------")
        print(
            "conversation_id:",
            conversation_id,
        )
        print(
            "query:",
            QUERY,
        )

        #--------------------USER TURN--------------------

        raw_entities = extract_entity_mentions(
            QUERY
        )

        user_turn = await repository.insert_turn(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role="user",
            raw_text=QUERY,
            entities_extracted=[
                {
                    "entity_text": (
                        item.entity_text
                    ),
                    "entity_type": (
                        item.entity_type
                    ),
                    "salience_rank": (
                        item.salience_rank
                    ),
                }
                for item in raw_entities
            ],
            metadata={
                "source": "citation_stream_smoke",
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

        #--------------------STATE--------------------

        state = await build_generation_state_from_turn(
            repository=repository,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            user_turn_id=user_turn.turn_id,
            current_user_message=QUERY,
        )

        state["session_id"] = session_id
        state["top_k"] = 5
        state["retrieval_source_constraints"] = {}
        state["disable_long_term_memory"] = True
        state["disable_memory_write"] = True

        #--------------------STREAM--------------------

        final_state: (
            dict[str, Any]
            | None
        ) = None

        streamed_parts: list[str] = []

        print()
        print("ANSWER")
        print("------")

        async for event in runtime.stream(
            state
        ):
            if (
                event.event_type
                == GenerationStreamEventType
                .ANSWER_DELTA
            ):
                text = event.text or ""

                streamed_parts.append(
                    text
                )

                print(
                    text,
                    end="",
                    flush=True,
                )

                continue

            if (
                event.event_type
                == GenerationStreamEventType
                .ANSWER_REPLACE
            ):
                streamed_parts = [
                    event.text or ""
                ]

                print()
                print()
                print(
                    "[answer.replace]"
                )
                print(
                    event.text or ""
                )

                continue

            if (
                event.event_type
                == GenerationStreamEventType
                .STREAM_ERROR
            ):
                print()
                print(
                    "STREAM ERROR:",
                    event.payload,
                )

            if (
                event.event_type
                == GenerationStreamEventType
                .STREAM_COMPLETED
            ):
                if (
                    event.final_state
                    is not None
                ):
                    final_state = dict(
                        event.final_state
                    )

        print()
        print()

        if final_state is None:
            raise RuntimeError(
                "Stream completed without final_state."
            )

        final_answer = str(
            final_state.get(
                "final_answer"
            )
            or "".join(
                streamed_parts
            )
        ).strip()

        #--------------------CITATION CHECK--------------------

        citation_names = (
            human_citation_names(
                final_state
            )
        )

        print("CITATIONS")
        print("---------")

        if not citation_names:
            raise AssertionError(
                "No human-facing citation names "
                "were present in retrieval_state."
            )

        for name in citation_names:
            print(
                "-",
                name,
            )

        internal_id_match = (
            _INTERNAL_DOC_ID_RE.search(
                final_answer
            )
        )

        if internal_id_match:
            raise AssertionError(
                "Internal document ID leaked into "
                "the generated answer: "
                f"{internal_id_match.group(0)}"
            )

        matching_names = [
            name
            for name in citation_names
            if f"[{name}]" in final_answer
        ]

        if not matching_names:
            raise AssertionError(
                "The generated answer did not cite "
                "any retrieved document using its "
                "human-facing filename.\n"
                f"Known citation names: {citation_names}"
            )

        print()
        print(
            "visible citations:",
            matching_names,
        )

        #--------------------POST RETRIEVAL--------------------

        retrieval_state = dict(
            final_state.get(
                "retrieval_state"
            )
            or {}
        )

        stateful = dict(
            retrieval_state.get(
                "stateful_retrieval"
            )
            or {}
        )

        resolved_query = str(
            stateful.get(
                "resolved_query"
            )
            or retrieval_state.get(
                "rewritten_query"
            )
            or QUERY
        )

        query_changed = bool(
            stateful.get(
                "query_changed"
            )
        )

        retrieval_trace_id = (
            retrieval_state.get(
                "retrieval_trace_id"
            )
        )

        updated_user_turn = (
            await repository
            .update_turn_after_retrieval(
                turn_id=user_turn.turn_id,
                resolved_text=(
                    resolved_query
                    if query_changed
                    else None
                ),
                retrieval_trace_id=(
                    str(
                        retrieval_trace_id
                    )
                    if retrieval_trace_id
                    else None
                ),
                metadata={
                    "source": (
                        "citation_stream_smoke"
                    ),
                    "phase": (
                        "post_graph_user_turn"
                    ),
                    "stateful_retrieval": (
                        stateful
                    ),
                    "turn_plan": (
                        final_state.get(
                            "turn_plan"
                        )
                        or {}
                    ),
                },
            )
        )

        if retrieval_trace_id:
            await (
                repository
                .attach_retrieval_trace_to_turn(
                    retrieval_trace_id=str(
                        retrieval_trace_id
                    ),
                    turn_id=(
                        user_turn.turn_id
                    ),
                )
            )

        #--------------------ENTITIES--------------------

        resolved_entities = (
            extract_entity_mentions(
                resolved_query
            )
        )

        new_entities = entities_not_in(
            raw_entities,
            resolved_entities,
        )

        await repository.insert_entities(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            turn_index=(
                updated_user_turn.turn_index
            ),
            entities=new_entities,
        )

        #--------------------ASSISTANT TURN--------------------

        assistant_turn = (
            await
            persist_assistant_turn_from_generation_state(
                repository=repository,
                state=final_state,
                metadata={
                    "source": (
                        "citation_stream_smoke"
                    ),
                    "retrieval_trace_id": (
                        retrieval_trace_id
                    ),
                    "turn_plan": (
                        final_state.get(
                            "turn_plan"
                        )
                        or {}
                    ),
                },
            )
        )

        #--------------------STRUCTURED PROVENANCE--------------------

        citations = list(
            retrieval_state.get(
                "citations"
            )
            or []
        )

        print()
        print("STRUCTURED PROVENANCE")
        print("---------------------")

        for citation in citations:
            if not isinstance(
                citation,
                dict,
            ):
                continue

            print()
            print(
                "source_name:",
                citation.get(
                    "source_name"
                ),
            )

            print(
                "heading:",
                citation.get(
                    "heading"
                ),
            )

            print(
                "chunk_id:",
                citation.get(
                    "chunk_id"
                ),
            )

        #--------------------RESULT--------------------

        print()
        print("RESULT")
        print("------")

        print(
            "user_turn_id:",
            user_turn.turn_id,
        )

        print(
            "assistant_turn_id:",
            assistant_turn.turn_id,
        )

        print(
            "retrieval_trace_id:",
            retrieval_trace_id,
        )

        print(
            "visible_source_names:",
            matching_names,
        )

        print()
        print(
            "PASS: human document citations "
            "are frontend-safe"
        )

    finally:
        await runtime.close()
        await repository.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
