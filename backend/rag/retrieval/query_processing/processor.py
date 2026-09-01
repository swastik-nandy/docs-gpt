from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.contracts.enums import RetrievalMode
from rag.retrieval.contracts.query import ExtractedFilter
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.query_processing.analysis import QueryAnalysis, analyze_query
from rag.retrieval.query_processing.contracts import (
    ProcessedQueryPlan,
    QueryProcessingConfig,
    RetrievalToolHint,
    build_normalized_query_from_plan,
    clean_unique_strings,
    merge_tool_hints,
)


@dataclass(frozen=True)
class AdaptiveQueryProcessor:
    """
    Canonical query-processing brain.

    Generic by design:
    - no document/vendor/domain terms
    - no action-verb dictionary
    - no AWS/ECS/Qdrant-specific handling
    - no LLM calls
    - no I/O

    It emits structural/statistical query analysis for downstream retrieval.
    """

    config: QueryProcessingConfig = QueryProcessingConfig(processor="adaptive")

    async def process(self, request: RetrievalRequest) -> ProcessedQueryPlan:
        request.validate()
        self.config.validate()

        analysis = analyze_query(
            request.query,
            wants_memory=request.wants_memory,
            has_chat_history=bool(request.chat_history),
            has_conversation_id=bool(request.conversation_id),
        )

        extracted_filters = _request_filters(request)
        metadata = {
            "planner": "adaptive",
            "query_analysis": analysis.to_payload(),
            "generic": True,
        }

        normalized_query = build_normalized_query_from_plan(
            request=request,
            normalized_query=analysis.normalized_query,
            intent=analysis.intent,
            defects=analysis.defects,
            semantic_queries=analysis.semantic_queries,
            extracted_terms=analysis.exact_terms,
            extracted_filters=extracted_filters,
            requires_memory=analysis.requires_memory,
            metadata=metadata,
        )

        result = ProcessedQueryPlan(
            normalized_query=normalized_query,
            retrieval_mode=RetrievalMode.SPARSE,
            tool_hints=_tool_hints(analysis),
            semantic_queries=clean_unique_strings(
                analysis.semantic_queries,
                limit=self.config.max_semantic_queries,
            ),
            exact_terms=clean_unique_strings(
                analysis.exact_terms,
                limit=self.config.max_exact_terms,
            ),
            source_hints=clean_unique_strings(
                analysis.source_hints,
                limit=self.config.max_source_hints,
            ),
            extracted_filters=extracted_filters,
            needs_agentic_loop=False,
            needs_exact_search=bool(analysis.exact_terms),
            needs_source_navigation=bool(analysis.source_hints),
            requires_memory=analysis.requires_memory,
            confidence=analysis.confidence,
            planner="adaptive",
            reasoning="; ".join(analysis.reasoning),
            metadata={
                "source": "adaptive_query_processor",
                "query_analysis": analysis.to_payload(),
                "request_mode": request.mode.value,
            },
        )
        result.validate(config=self.config)
        return result


def _tool_hints(analysis: QueryAnalysis) -> tuple[RetrievalToolHint, ...]:
    hints: list[RetrievalToolHint] = []

    if analysis.exact_terms:
        hints.append(RetrievalToolHint.EXACT_PAYLOAD_SEARCH)

    if analysis.source_hints:
        hints.extend(
            (
                RetrievalToolHint.SOURCE_NAVIGATION,
                RetrievalToolHint.FILE_READ,
                RetrievalToolHint.CHUNK_READ,
            )
        )

    if analysis.requires_memory:
        hints.append(RetrievalToolHint.MEMORY_LOOKUP)

    return merge_tool_hints(hints)


def _request_filters(request: RetrievalRequest) -> tuple[ExtractedFilter, ...]:
    constraints = request.source_constraints
    filters: list[ExtractedFilter] = []

    for key, value in (
        ("source_id", constraints.source_id),
        ("job_id", constraints.job_id),
        ("parser_doc_id", constraints.parser_doc_id),
        ("section_id", constraints.section_id),
        ("ir_id", constraints.ir_id),
    ):
        if value:
            filters.append(ExtractedFilter(key=key, value=value, operator="eq"))

    for key, value in constraints.payload_filters.items():
        if isinstance(value, str | int | float | bool):
            filters.append(ExtractedFilter(key=key, value=value, operator="eq"))

    for item in filters:
        item.validate()

    return tuple(filters)
