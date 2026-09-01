from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.contracts.candidates import CandidateOrigin, RetrievalCandidate
from rag.retrieval.contracts.embeddings import QueryEmbedding
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.vector_store.qdrant_candidate import QdrantCandidateMapper
from rag.retrieval.vector_store.qdrant_client import (
    QdrantClientFactory,
    QdrantConnectionConfig,
)
from rag.retrieval.vector_store.qdrant_filter import (
    QdrantFilterBuilder,
    QdrantFilterBuilderConfig,
)


Payload = dict[str, Any]


@dataclass(frozen=True)
class DenseQdrantRetrieverConfig:
    collection_name: str
    vector_name: str = "dense"
    limit_multiplier: int = 1
    score_threshold: float | None = None
    with_payload: bool = True
    with_vectors: bool = False

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> DenseQdrantRetrieverConfig:
        config = cls(
            collection_name=runtime_config.qdrant.collection_name,
            vector_name=runtime_config.qdrant.dense_vector_name,
            limit_multiplier=runtime_config.dense.limit_multiplier,
            score_threshold=runtime_config.dense.score_threshold,
            with_payload=runtime_config.dense.with_payload,
            with_vectors=runtime_config.dense.with_vectors,
        )
        config.validate()

        return config

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        if not self.vector_name.strip():
            raise ValueError("vector_name cannot be empty.")

        if self.limit_multiplier <= 0:
            raise ValueError("limit_multiplier must be greater than zero.")

        if self.score_threshold is not None and self.score_threshold < 0:
            raise ValueError("score_threshold cannot be negative.")

        if not isinstance(self.with_payload, bool):
            raise ValueError("with_payload must be boolean.")

        if not isinstance(self.with_vectors, bool):
            raise ValueError("with_vectors must be boolean.")


class DenseQdrantRetriever:
    """
    Dense vector retriever backed by Qdrant named dense vectors.

    Retrieval behavior is TOML-controlled through RetrievalRuntimeConfig.
    Qdrant connection details can still come from environment/runtime secrets.
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient | None = None,
        connection_config: QdrantConnectionConfig | None = None,
        runtime_config: RetrievalRuntimeConfig | None = None,
        config: DenseQdrantRetrieverConfig | None = None,
        filter_builder: QdrantFilterBuilder | None = None,
        candidate_mapper: QdrantCandidateMapper | None = None,
    ) -> None:
        self.runtime_config = runtime_config or load_retrieval_runtime_config()
        self.runtime_config.validate()

        self.connection_config = connection_config or QdrantConnectionConfig.from_env()
        self.config = config or DenseQdrantRetrieverConfig.from_runtime_config(
            self.runtime_config,
        )
        self.config.validate()

        self.client = client or QdrantClientFactory(
            self.connection_config,
        ).async_client()
        self.filter_builder = filter_builder or QdrantFilterBuilder(
            QdrantFilterBuilderConfig(),
        )
        self.candidate_mapper = candidate_mapper or QdrantCandidateMapper()

    async def retrieve(
        self,
        *,
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
        query_embedding: QueryEmbedding,
    ) -> Sequence[RetrievalCandidate]:
        request.validate()
        normalized_query.validate()
        query_embedding.validate()

        if not query_embedding.has_dense:
            raise ValueError("Dense retrieval requires query_embedding.dense_vector.")

        filter_result = self.filter_builder.build(
            request=request,
            normalized_query=normalized_query,
        )

        limit = max(1, request.top_k * self.config.limit_multiplier)

        points = await self._query_points(
            dense_vector=query_embedding.dense_vector,
            query_filter=filter_result.filter,
            limit=limit,
        )

        return tuple(
            self.candidate_mapper.from_scored_point(
                point,
                origin=CandidateOrigin.DENSE,
                rank=index + 1,
            )
            for index, point in enumerate(points)
        )

    async def _query_points(
        self,
        *,
        dense_vector: list[float],
        query_filter: Any,
        limit: int,
    ) -> tuple[Any, ...]:
        if hasattr(self.client, "query_points"):
            response = await self.client.query_points(
                collection_name=self.config.collection_name,
                query=dense_vector,
                using=self.config.vector_name,
                query_filter=query_filter,
                limit=limit,
                score_threshold=self.config.score_threshold,
                with_payload=self.config.with_payload,
                with_vectors=self.config.with_vectors,
            )

            return _extract_points(response)

        response = await self.client.search(
            collection_name=self.config.collection_name,
            query_vector=(self.config.vector_name, dense_vector),
            query_filter=query_filter,
            limit=limit,
            score_threshold=self.config.score_threshold,
            with_payload=self.config.with_payload,
            with_vectors=self.config.with_vectors,
        )

        return _extract_points(response)


def _extract_points(response: Any) -> tuple[Any, ...]:
    if response is None:
        return ()

    points = getattr(response, "points", None)
    if points is not None:
        return tuple(points)

    if isinstance(response, (list, tuple)):
        return tuple(response)

    try:
        return tuple(response)
    except TypeError:
        return (response,)
