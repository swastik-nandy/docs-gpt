from __future__ import annotations

from rag.generation.graph.state import ConversationState


_ESCAPE_HATCHES = (
    "based on general knowledge",
    "outside the provided context",
    "i don't have access",
    "i cannot verify",
)


async def basic_grounding_node(state: ConversationState) -> dict:
    draft_answer = str(state.get("draft_answer") or "").strip()
    context_bundle = state.get("context_bundle") or {}
    retrieved_documents = context_bundle.get("retrieved_documents") or []
    retrieved_memory = context_bundle.get("retrieved_memory") or []
    turn_plan = state.get("turn_plan") or {}
    generation_metadata = state.get("generation_metadata") or {}

    if generation_metadata.get("model") == "generation_error_fallback":
        return {
            "grounding_result": {
                "passed": True,
                "score": 0.0,
                "reasons": ["generation_model_failed_after_retrieval"],
            }
        }

    if not draft_answer:
        return {
            "grounding_result": {
                "passed": False,
                "score": 0.0,
                "reasons": ["empty_answer"],
            }
        }

    if (
        turn_plan.get("allow_conversation_answer")
        and not turn_plan.get("needs_document_retrieval", True)
    ):
        return {
            "grounding_result": {
                "passed": True,
                "score": 0.85,
                "reasons": ["conversation_only_answer_allowed_by_turn_plan"],
            }
        }

    if not retrieved_documents and not retrieved_memory:
        return {
            "grounding_result": {
                "passed": False,
                "score": 0.0,
                "reasons": ["no_grounding_evidence"],
            }
        }

    lowered = draft_answer.lower()
    escape_hatches = [phrase for phrase in _ESCAPE_HATCHES if phrase in lowered]
    if escape_hatches:
        return {
            "grounding_result": {
                "passed": False,
                "score": 0.25,
                "reasons": ["unsupported_escape_hatch"],
                "details": {
                    "phrases": escape_hatches,
                },
            }
        }

    return {
        "grounding_result": {
            "passed": True,
            "score": 0.75,
            "reasons": ["basic_evidence_present"],
        }
    }
