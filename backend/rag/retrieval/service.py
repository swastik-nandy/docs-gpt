from __future__ import annotations

from rag.conversation.repository import PostgresConversationRepository

from dataclasses import dataclass
from typing import Any, Sequence

from rag.retrieval.adaptive import (
    AdaptiveRetrievalGraphRuntime,
    AdaptiveRetrievalGraphRuntimeConfig,
)
from rag.retrieval.contracts.context import ContextBlock
from rag.retrieval.contracts.request import RetrievalRequest, TenantContext
from rag.retrieval.graph.state import RetrievalGraphOutput
from rag.retrieval.stateful import prepare_stateful_retrieval_request


Payload = dict[str, Any]


@dataclass(frozen=True)
class RetrievalServiceConfig:
    recursion_limit: int = 16
    include_debug_metadata: bool = True
    enable_stateful_resolution: bool = False
    persist_retrieval_trace: bool = True

    def validate(self) -> None:
        if self.recursion_limit <= 0:
            raise ValueError("recursion_limit must be greater than zero.")

        if not isinstance(self.include_debug_metadata, bool):
            raise ValueError("include_debug_metadata must be boolean.")

        if not isinstance(self.enable_stateful_resolution, bool):
            raise ValueError("enable_stateful_resolution must be boolean.")

        if not isinstance(self.persist_retrieval_trace, bool):
            raise ValueError("persist_retrieval_trace must be boolean.")


class RetrievalService:
    """
    Application-facing retrieval entrypoint.

    This wraps the retrieval orchestration runtime so API/generation layers do
    not need to know about LangGraph, adaptive actions, Qdrant retrievers,
    fusion, rerankers, or packers.
    """

    def __init__(
        self,
        *,
        graph_runtime: AdaptiveRetrievalGraphRuntime | None = None,
        config: RetrievalServiceConfig | None = None,
    ) -> None:
        self.config = config or RetrievalServiceConfig()
        self.config.validate()

        self.graph_runtime = graph_runtime or AdaptiveRetrievalGraphRuntime(
            config=AdaptiveRetrievalGraphRuntimeConfig(
                recursion_limit=self.config.recursion_limit,
                include_debug_metadata=self.config.include_debug_metadata,
            ),
        )

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        metadata: Payload | None = None,
    ) -> RetrievalGraphOutput:
        request.validate()

        original_request = request
        retrieval_request = request
        retrieval_metadata: Payload = dict(metadata or {})

        stateful_enabled = self.config.enable_stateful_resolution

        if stateful_enabled:
            prepared = await prepare_stateful_retrieval_request(request)
            retrieval_request = prepared.request

            stateful_metadata = retrieval_request.metadata.get("stateful_retrieval")
            if stateful_metadata is not None:
                retrieval_metadata["stateful_retrieval"] = stateful_metadata
                retrieval_metadata["raw_query"] = prepared.original_request.query
                retrieval_metadata["resolved_query"] = prepared.request.query
                retrieval_metadata["query_changed"] = prepared.query_changed
                retrieval_metadata["resolution_strategy"] = prepared.resolution.strategy

        output = await self.graph_runtime.retrieve(
            retrieval_request,
            metadata=retrieval_metadata,
        )

        if self.config.persist_retrieval_trace:
            await _persist_retrieval_trace(
                original_request=original_request,
                retrieval_request=retrieval_request,
                output=output,
                metadata=retrieval_metadata,
            )

        return output

    async def retrieve_context(
        self,
        *,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        metadata: Payload | None = None,
    ) -> Sequence[ContextBlock]:
        request = RetrievalRequest.create(
            query=query,
            tenant=TenantContext(tenant_id=tenant_id),
            top_k=top_k,
        )

        output = await self.retrieve(
            request,
            metadata=metadata,
        )

        return tuple(output["context_blocks"])

    async def retrieve_raw(
        self,
        *,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        metadata: Payload | None = None,
    ) -> RetrievalGraphOutput:
        request = RetrievalRequest.create(
            query=query,
            tenant=TenantContext(tenant_id=tenant_id),
            top_k=top_k,
        )

        return await self.retrieve(
            request,
            metadata=metadata,
        )


async def _persist_retrieval_trace(
    *,
    original_request: RetrievalRequest,
    retrieval_request: RetrievalRequest,
    output: RetrievalGraphOutput,
    metadata: Payload,
) -> None:
    conversation_id = retrieval_request.conversation_id
    if not conversation_id:
        return

    try:
        repository = await PostgresConversationRepository.connect()
        try:
            retrieval_trace_id = await repository.insert_retrieval_trace(
                conversation_id=conversation_id,
                tenant_id=retrieval_request.tenant.tenant_id,
                user_id=retrieval_request.tenant.user_id,
                raw_query=original_request.query,
                resolved_query=retrieval_request.query,
                query_signals=_safe_query_signals(output),
                retrieval_decision=_safe_decision(output),
                selected_context_refs=_safe_context_refs(output),
                trace=_safe_trace(output),
                metadata={
                    **metadata,
                    "request_id": retrieval_request.request_id,
                    "source": "retrieval_service",
                },
            )
        finally:
            await repository.close()

        output_metadata = output.setdefault("metadata", {})
        output_metadata["retrieval_trace_id"] = retrieval_trace_id

    except Exception as exc:
        warnings = output.setdefault("warnings", [])
        warnings.append(f"failed to persist retrieval trace: {exc}")


def _safe_decision(output: RetrievalGraphOutput) -> Payload:
    decision = output.get("decision")
    if decision is None:
        return {}

    if hasattr(decision, "to_dict"):
        return decision.to_dict()

    if hasattr(decision, "__dict__"):
        return dict(decision.__dict__)

    if isinstance(decision, dict):
        return dict(decision)

    return {"repr": repr(decision)}


def _safe_query_signals(output: RetrievalGraphOutput) -> Payload:
    metadata = output.get("metadata") or {}
    stateful = metadata.get("stateful_retrieval") or {}

    return {
        "stateful_retrieval": stateful,
        "normalized_query": str(output.get("normalized_query") or ""),
    }


def _safe_context_refs(output: RetrievalGraphOutput) -> list[Payload]:
    refs: list[Payload] = []

    for block in output.get("context_blocks") or []:
        refs.append(
            {
                "source_id": getattr(block, "source_id", None),
                "chunk_id": getattr(block, "chunk_id", None),
                "score": getattr(block, "score", None),
                "metadata": getattr(block, "metadata", None),
            }
        )

    return refs


def _safe_trace(output: RetrievalGraphOutput) -> list[Payload]:
    trace_items: list[Payload] = []

    for item in output.get("trace") or []:
        if isinstance(item, dict):
            trace_items.append(dict(item))
        elif hasattr(item, "__dict__"):
            trace_items.append(dict(item.__dict__))
        else:
            trace_items.append({"repr": repr(item)})

    return trace_items
