from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.config import QdrantConfig, QdrantMode, VectorDistance
from rag.ingestion.embeddings.schema import build_vector_points
from rag.ingestion.embeddings.types import (
    SearchResult,
    SearchResults,
    StoreUpsertResult,
    Vector,
)


logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant storage boundary for RAG chunks.

    Supports:
      - local disk Qdrant for development
      - in-memory Qdrant for tests
      - remote Qdrant server/cloud for later scaling
      - deterministic upserts through schema.py
      - source/file/doc filters for multi-file retrieval
    """

    def __init__(
        self,
        *,
        config: Optional[QdrantConfig] = None,
        vector_size: int = 768,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        self.config = config or self._legacy_config(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self.config.validate()

        self.collection_name = self.config.collection_name
        self.vector_size = int(vector_size)

        if self.vector_size <= 0:
            raise ValueError("vector_size must be greater than zero.")

        self.client = self._build_client()
        self._ensure_collection()

    # ---------------- CLIENT ---------------- #

    def _build_client(self) -> QdrantClient:
        if self.config.mode == QdrantMode.MEMORY:
            logger.info("Starting in-memory Qdrant client")
            return QdrantClient(":memory:")

        if self.config.mode == QdrantMode.LOCAL:
            local_path = Path(self.config.local_path)
            local_path.mkdir(parents=True, exist_ok=True)

            logger.info("Starting local Qdrant client at %s", local_path.as_posix())
            return QdrantClient(path=local_path.as_posix())

        if self.config.mode == QdrantMode.REMOTE:
            logger.info("Connecting to remote Qdrant at %s", self.config.url)
            return QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
                prefer_grpc=self.config.prefer_grpc,
            )

        raise ValueError(f"Unsupported Qdrant mode: {self.config.mode!r}")

    @staticmethod
    def _legacy_config(
        *,
        collection_name: Optional[str],
        persist_directory: Optional[str],
    ) -> QdrantConfig:
        base = QdrantConfig.from_env()

        if collection_name:
            base = QdrantConfig(
                collection_name=collection_name,
                mode=base.mode,
                local_path=base.local_path,
                url=base.url,
                api_key=base.api_key,
                timeout_seconds=base.timeout_seconds,
                prefer_grpc=base.prefer_grpc,
                distance=base.distance,
                recreate_collection=base.recreate_collection,
                create_payload_indexes=base.create_payload_indexes,
            )

        if persist_directory:
            base = QdrantConfig(
                collection_name=base.collection_name,
                mode=base.mode,
                local_path=Path(persist_directory),
                url=base.url,
                api_key=base.api_key,
                timeout_seconds=base.timeout_seconds,
                prefer_grpc=base.prefer_grpc,
                distance=base.distance,
                recreate_collection=base.recreate_collection,
                create_payload_indexes=base.create_payload_indexes,
            )

        base.validate()
        return base

    # ---------------- COLLECTION ---------------- #

    def _ensure_collection(self) -> None:
        exists = self._collection_exists()

        if exists and self.config.recreate_collection:
            logger.warning("Recreating Qdrant collection: %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self._distance(),
                ),
            )

            logger.info(
                "Created Qdrant collection: name=%s vector_size=%s distance=%s",
                self.collection_name,
                self.vector_size,
                self.config.distance.value,
            )

            if self.config.create_payload_indexes:
                self._create_payload_indexes()

            return

        current_size = self._get_existing_vector_size()

        if current_size != self.vector_size:
            raise ValueError(
                f"Collection {self.collection_name!r} already exists with vector size "
                f"{current_size}, but embedder produced {self.vector_size}. "
                "Use a new collection name or reset/recreate the collection."
            )

        if self.config.create_payload_indexes:
            self._create_payload_indexes()

    def _collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(item.name == self.collection_name for item in collections)

    def _get_existing_vector_size(self) -> int:
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors_config = info.config.params.vectors

        if hasattr(vectors_config, "size"):
            return int(vectors_config.size)

        if isinstance(vectors_config, dict):
            if not vectors_config:
                raise ValueError(
                    f"Collection {self.collection_name!r} has no vector config."
                )

            first_config = next(iter(vectors_config.values()))
            return int(first_config.size)

        raise ValueError(
            f"Unable to read vector size for collection {self.collection_name!r}."
        )

    def _distance(self) -> Distance:
        mapping = {
            VectorDistance.COSINE: Distance.COSINE,
            VectorDistance.DOT: Distance.DOT,
            VectorDistance.EUCLID: Distance.EUCLID,
        }
        return mapping[self.config.distance]

    def _create_payload_indexes(self) -> None:
        """
        Create indexes for fields commonly used in filters.

        Safe to call repeatedly. Some Qdrant versions may complain if an index
        already exists, so we log and continue instead of breaking ingestion.
        """

        keyword_fields = [
            "source",
            "file_name",
            "doc_id",
            "chunk_id",
            "content_hash",
            "schema_version",
            "embedding_model",
            "chunk_type",
            "subtype",
        ]

        integer_fields = [
            "tokens",
            "chunk_index",
            "split_part",
        ]

        for field_name in keyword_fields:
            self._create_payload_index(
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        for field_name in integer_fields:
            self._create_payload_index(
                field_name=field_name,
                field_schema=PayloadSchemaType.INTEGER,
            )

    def _create_payload_index(
        self,
        *,
        field_name: str,
        field_schema: PayloadSchemaType,
    ) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            logger.debug(
                "Skipping payload index creation for %s.%s: %s",
                self.collection_name,
                field_name,
                exc,
            )

    # ---------------- ADD / UPSERT ---------------- #

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Vector],
        *,
        embedding_model: Optional[str] = None,
        schema_version: str = "embedding_payload_v1",
        batch_size: int = 64,
    ) -> StoreUpsertResult:
        """
        Upsert chunks + embeddings into Qdrant.

        Upsert means deterministic IDs from schema.py will replace existing
        matching points instead of creating duplicates.
        """

        if not chunks or not embeddings:
            return StoreUpsertResult(
                collection_name=self.collection_name,
                attempted_count=0,
                upserted_count=0,
                skipped_count=0,
                status="empty",
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        points = build_vector_points(
            chunks,
            embeddings,
            embedding_model=embedding_model,
            schema_version=schema_version,
        )

        attempted_count = len(points)
        upserted_count = 0
        skipped_count = 0

        for batch in self._batch(points, batch_size):
            qdrant_points: List[PointStruct] = []

            for point in batch:
                if not self._valid_vector(point.vector):
                    skipped_count += 1
                    continue

                qdrant_points.append(
                    PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=point.payload,
                    )
                )

            if not qdrant_points:
                continue

            operation = self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
                wait=True,
            )

            upserted_count += len(qdrant_points)

            logger.debug(
                "Qdrant upsert operation: collection=%s status=%s count=%s",
                self.collection_name,
                getattr(operation, "status", None),
                len(qdrant_points),
            )

        return StoreUpsertResult(
            collection_name=self.collection_name,
            attempted_count=attempted_count,
            upserted_count=upserted_count,
            skipped_count=skipped_count,
            status="completed",
        )

    def _valid_vector(self, vector: Sequence[float]) -> bool:
        if not vector:
            return False

        if len(vector) != self.vector_size:
            return False

        return all(isinstance(value, (float, int)) for value in vector)

    # ---------------- QUERY ---------------- #

    def query(
        self,
        query_embedding: Vector,
        n_results: int = 5,
        query_filter: Optional[Filter] = None,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible query method.

        Returns a list of dictionaries for older retriever/test code.
        """

        active_filter = query_filter if query_filter is not None else filter
        results = self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            query_filter=active_filter,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.text,
                "file_name": result.payload.get("file_name"),
                "heading": result.payload.get("heading"),
                "source": result.payload.get("source"),
                "doc_id": result.payload.get("doc_id"),
                "chunk_id": result.payload.get("chunk_id"),
                "payload": result.payload,
            }
            for result in results.results
        ]

    def search(
        self,
        *,
        query_embedding: Vector,
        n_results: int = 5,
        query_filter: Optional[Filter] = None,
        query_text: str = "",
    ) -> SearchResults:
        """
        Semantic search over stored chunks.

        Uses query_points when available and falls back to search for older
        qdrant-client versions.
        """

        if not query_embedding:
            return SearchResults(query=query_text, results=[])

        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding mismatch: got {len(query_embedding)}, "
                f"expected {self.vector_size}."
            )

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        points = self._query_points(
            query_embedding=query_embedding,
            n_results=n_results,
            query_filter=query_filter,
        )

        results: List[SearchResult] = []

        for point in points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    id=str(point.id),
                    score=float(point.score),
                    text=str(payload.get("text", "")),
                    payload=payload,
                )
            )

        return SearchResults(query=query_text, results=results)

    def _query_points(
        self,
        *,
        query_embedding: Vector,
        n_results: int,
        query_filter: Optional[Filter],
    ) -> List[Any]:
        query_points = getattr(self.client, "query_points", None)

        if callable(query_points):
            response = query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
                with_vectors=False,
            )

            return list(getattr(response, "points", response))

        search = getattr(self.client, "search", None)

        if not callable(search):
            raise RuntimeError(
                "Installed qdrant-client supports neither query_points nor search."
            )

        return list(
            search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
                with_vectors=False,
            )
        )

    # ---------------- FILTER HELPERS ---------------- #

    def build_file_filter(self, file_name: str) -> Filter:
        return self._exact_filter("file_name", file_name)

    def build_source_filter(self, source: str) -> Filter:
        return self._exact_filter("source", source)

    def build_doc_filter(self, doc_id: str) -> Filter:
        return self._exact_filter("doc_id", doc_id)

    def build_schema_filter(self, schema_version: str) -> Filter:
        return self._exact_filter("schema_version", schema_version)

    @staticmethod
    def _exact_filter(field_name: str, value: Any) -> Filter:
        if value is None or str(value).strip() == "":
            raise ValueError(f"Cannot build filter for empty field {field_name!r}.")

        return Filter(
            must=[
                FieldCondition(
                    key=field_name,
                    match=MatchValue(value=value),
                )
            ]
        )

    # ---------------- DELETE / REFRESH ---------------- #

    def delete_by_source(self, source: str) -> int:
        """
        Delete all points for one source path.

        Used by pipeline refresh_source=True.
        """

        return self._delete_by_filter(self.build_source_filter(source))

    def delete_by_file_name(self, file_name: str) -> int:
        return self._delete_by_filter(self.build_file_filter(file_name))

    def delete_by_doc_id(self, doc_id: str) -> int:
        return self._delete_by_filter(self.build_doc_filter(doc_id))

    def _delete_by_filter(self, query_filter: Filter) -> int:
        before = self.count(query_filter=query_filter)

        if before == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=query_filter),
            wait=True,
        )

        return before

    # ---------------- RESET ---------------- #

    def reset(self) -> None:
        """
        Delete and recreate the collection.
        """

        if self._collection_exists():
            logger.warning("Deleting Qdrant collection: %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)

        self._ensure_collection()

    # ---------------- COUNT / SCROLL ---------------- #

    def count(self, query_filter: Optional[Filter] = None) -> int:
        response = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(response.count)

    def get_all(
        self,
        *,
        limit: int = 100,
        query_filter: Optional[Filter] = None,
    ) -> Dict[str, List[Any]]:
        """
        Fetch all payloads for debugging, BM25 side-indexing, or inspection.
        """

        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        all_ids: List[str] = []
        all_payloads: List[Dict[str, Any]] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                all_ids.append(str(point.id))
                all_payloads.append(point.payload or {})

            if offset is None:
                break

        return {
            "ids": all_ids,
            "payloads": all_payloads,
        }

    # ---------------- HELPERS ---------------- #

    @staticmethod
    def _batch(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
        if size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        return [items[index : index + size] for index in range(0, len(items), size)]