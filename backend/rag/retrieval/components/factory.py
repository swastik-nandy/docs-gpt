from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from qdrant_client import AsyncQdrantClient

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.context_packing import DefaultContextPacker
from rag.retrieval.embeddings import RetrievalQueryEmbedder
from rag.retrieval.expansion import LinkedStructureExpander
from rag.retrieval.fusion import RrfCandidateFusion
from rag.retrieval.query_processing.contracts import QueryProcessor
from rag.retrieval.query_processing.processor import AdaptiveQueryProcessor
from rag.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.rerankers.score_reranker import ScoreReranker
from rag.retrieval.retrievers import DenseQdrantRetriever, SparseQdrantRetriever
from rag.retrieval.vector_store import (
    QdrantClientFactory,
    QdrantConnectionConfig,
)


@dataclass
class RetrievalComponents:
    """
    Lazy component registry for the retrieval engine.

    Heavy/model-backed objects are only created when accessed. This keeps imports
    and graph construction light, while still giving the graph one stable place
    to fetch retrieval tools from.
    """

    runtime_config: RetrievalRuntimeConfig
    connection_config: QdrantConnectionConfig
    client: AsyncQdrantClient | None = None

    @cached_property
    def qdrant_client(self) -> AsyncQdrantClient:
        if self.client is not None:
            return self.client

        return QdrantClientFactory(self.connection_config).async_client()

    @cached_property
    def query_processor(self) -> QueryProcessor:
        return AdaptiveQueryProcessor()

    @cached_property
    def query_embedder(self) -> RetrievalQueryEmbedder:
        return RetrievalQueryEmbedder()

    @cached_property
    def dense_retriever(self) -> DenseQdrantRetriever:
        return DenseQdrantRetriever(
            client=self.qdrant_client,
            connection_config=self.connection_config,
            runtime_config=self.runtime_config,
        )

    @cached_property
    def sparse_retriever(self) -> SparseQdrantRetriever:
        return SparseQdrantRetriever(
            client=self.qdrant_client,
            connection_config=self.connection_config,
            runtime_config=self.runtime_config,
        )

    @cached_property
    def fusion(self) -> RrfCandidateFusion:
        return RrfCandidateFusion(
            runtime_config=self.runtime_config,
        )

    @cached_property
    def structure_expander(self) -> LinkedStructureExpander:
        return LinkedStructureExpander(
            client=self.qdrant_client,
            connection_config=self.connection_config,
            runtime_config=self.runtime_config,
        )

    @cached_property
    def score_reranker(self) -> ScoreReranker:
        return ScoreReranker(
            runtime_config=self.runtime_config,
        )

    @cached_property
    def cross_encoder_reranker(self) -> CrossEncoderReranker:
        return CrossEncoderReranker(
            runtime_config=self.runtime_config,
        )

    @cached_property
    def context_packer(self) -> DefaultContextPacker:
        return DefaultContextPacker(
            runtime_config=self.runtime_config,
        )

    async def close(self) -> None:
        if "qdrant_client" not in self.__dict__:
            return

        close_method = getattr(self.qdrant_client, "close", None)
        if close_method is None:
            return

        result = close_method()

        if hasattr(result, "__await__"):
            await result


@dataclass(frozen=True)
class RetrievalComponentFactory:
    runtime_config: RetrievalRuntimeConfig
    connection_config: QdrantConnectionConfig

    @classmethod
    def from_runtime(
        cls,
        *,
        runtime_config: RetrievalRuntimeConfig | None = None,
        connection_config: QdrantConnectionConfig | None = None,
    ) -> RetrievalComponentFactory:
        resolved_runtime = runtime_config or load_retrieval_runtime_config()
        resolved_runtime.validate()

        resolved_connection = connection_config or QdrantConnectionConfig.from_env()

        return cls(
            runtime_config=resolved_runtime,
            connection_config=resolved_connection,
        )

    def create(
        self,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> RetrievalComponents:
        return RetrievalComponents(
            runtime_config=self.runtime_config,
            connection_config=self.connection_config,
            client=client,
        )


def create_retrieval_components(
    *,
    runtime_config: RetrievalRuntimeConfig | None = None,
    connection_config: QdrantConnectionConfig | None = None,
    client: AsyncQdrantClient | None = None,
) -> RetrievalComponents:
    return RetrievalComponentFactory.from_runtime(
        runtime_config=runtime_config,
        connection_config=connection_config,
    ).create(
        client=client,
    )
