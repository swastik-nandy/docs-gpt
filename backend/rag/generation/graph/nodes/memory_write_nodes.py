from __future__ import annotations

from rag.generation.graph.nodes.memory_nodes import QueryEmbedder
from rag.generation.graph.state import ConversationState
from rag.generation.llm.router import ModelRouter
from rag.generation.llm.schema import ChatMessage, ModelRequest
from rag.generation.memory.extraction import (
    MemoryExtractionParseError,
    parse_memory_extraction_response,
)
from rag.generation.memory.memory_gate import should_consider_memory_write
from rag.generation.memory.qdrant_memory_store import QdrantMemoryStore
from rag.generation.memory.schema import MemoryCandidate
from rag.generation.prompt.prompt_config import PromptRegistry


def build_memory_write_node(
    *,
    memory_store: QdrantMemoryStore,
    embedder: QueryEmbedder,
    model_router: ModelRouter,
    prompt_registry: PromptRegistry | None = None,
    min_confidence: float = 0.65,
    duplicate_min_score: float = 0.92,
):
    registry = prompt_registry or PromptRegistry.from_toml()

    async def memory_write_node(state: ConversationState) -> dict:
        if state.get("disable_memory_write"):
            return {
                "written_memory_ids": [],
                "skipped_memory_ids": [],
                "memory_write_warnings": ["semantic_memory_write_disabled"],
            }

        user_message = str(
            state.get("raw_user_message")
            or state["current_user_message"]
        )

        if not should_consider_memory_write(user_message):
            return {
                "written_memory_ids": [],
                "skipped_memory_ids": [],
                "memory_write_warnings": [],
            }

        extraction_prompt = registry.memory_extraction.user_template.format(
            current_user_message=user_message,
            conversation_summary=state.get("conversation_summary") or registry.rendering.empty_value,
            draft_answer=state.get("draft_answer", ""),
        )

        try:
            response = await model_router.generate(
                ModelRequest(
                    role="memory_extraction",
                    system=registry.memory_extraction.system,
                    messages=[ChatMessage(role="user", content=extraction_prompt)],
                    response_mime_type="application/json",
                )
            )
        except Exception as exc:
            return {
                "written_memory_ids": [],
                "skipped_memory_ids": [],
                "memory_write_warnings": [
                    f"memory_extraction_failed:{type(exc).__name__}:{exc}"
                ],
            }

        try:
            facts = parse_memory_extraction_response(response.text)
        except MemoryExtractionParseError as exc:
            return {
                "written_memory_ids": [],
                "skipped_memory_ids": [],
                "memory_write_warnings": [f"memory_extraction_parse_failed:{exc}"],
            }

        written_ids: list[str] = []
        skipped_ids: list[str] = []
        warnings: list[str] = []

        for fact in facts:
            if fact.confidence < min_confidence:
                warnings.append(f"low_confidence_memory_skipped:{fact.confidence}:{fact.text}")
                continue

            try:
                vector = await embedder.embed_query(fact.text)
            except Exception as exc:
                warnings.append(f"memory_embedding_failed:{type(exc).__name__}:{exc}")
                continue

            try:
                duplicate = await memory_store.find_duplicate(
                    tenant_id=state["tenant_id"],
                    user_id=state["user_id"],
                    query_vector=vector,
                    fact_type=fact.fact_type,
                    min_score=duplicate_min_score,
                )
            except Exception as exc:
                warnings.append(f"memory_duplicate_check_failed:{type(exc).__name__}:{exc}")
                continue

            if duplicate is not None:
                skipped_ids.append(duplicate.memory_id)
                continue

            try:
                memory_id = await memory_store.put(
                    MemoryCandidate(
                        tenant_id=state["tenant_id"],
                        user_id=state["user_id"],
                        text=fact.text,
                        vector=vector,
                        fact_type=fact.fact_type,
                        conversation_id=state["conversation_id"],
                        source_turn_id=state["turn_id"],
                        confidence=fact.confidence,
                        metadata={
                            **fact.metadata,
                            "source": "generation_memory_write_node",
                            "source_message": user_message,
                        },
                    )
                )
            except Exception as exc:
                warnings.append(f"memory_write_failed:{type(exc).__name__}:{exc}")
                continue

            written_ids.append(memory_id)

        return {
            "written_memory_ids": written_ids,
            "skipped_memory_ids": skipped_ids,
            "memory_write_warnings": warnings,
        }

    return memory_write_node
