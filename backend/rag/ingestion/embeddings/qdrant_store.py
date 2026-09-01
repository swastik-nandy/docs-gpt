from __future__ import annotations

import logging
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from rag.ingestion.embeddings.config import QdrantConfig, QdrantMode, VectorDistance
from rag.ingestion.embeddings.types import (
    SearchResult,
    SearchResults,
    SparseVectorData,
    StoreUpsertResult,
    Vector,
    VectorPoint,
)


logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant storage boundary for named dense+sparse vectors.
    """

    def __init__(
        self,
        *,
        config: QdrantConfig,
        vector_size: int,
    ) -> None:
        self.config = config
        self.config.validate()

        self.collection_name = self.config.collection_name
        self.vector_size = int(vector_size)

        if self.vector_size <= 0:
            raise ValueError("vector_size must be greater than zero.")

        self.client = self._build_client()
        self.ensure_collection()

    def _build_client(self) -> QdrantClient:
        if self.config.mode == QdrantMode.MEMORY:
            logger.info("Starting in-memory Qdrant client")
            return QdrantClient(":memory:")

        if self.config.mode == QdrantMode.REMOTE:
            logger.info("Connecting to Qdrant at %s", self.config.url)
            return QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key or None,
                timeout=self.config.timeout_seconds,
                prefer_grpc=self.config.prefer_grpc,
            )

        raise ValueError(f"Unsupported Qdrant mode: {self.config.mode!r}")

    def ensure_collection(self) -> None:
        exists = self.collection_exists()

        if exists and self.config.recreate_collection:
            logger.warning("Recreating Qdrant collection: %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.config.dense_vector_name: models.VectorParams(
                        size=self.vector_size,
                        distance=self._distance(),
                        hnsw_config=models.HnswConfigDiff(
                            m=self.config.hnsw_m,
                            ef_construct=self.config.hnsw_ef_construct,
                            full_scan_threshold=self.config.hnsw_full_scan_threshold,
                        ),
                        on_disk=self.config.dense_on_disk,
                    )
                },
                sparse_vectors_config=self._sparse_vectors_config(),
            )

            logger.info(
                "Created Qdrant collection: name=%s dense=%s sparse=%s dim=%s distance=%s",
                self.collection_name,
                self.config.dense_vector_name,
                self.config.sparse_vector_name if self.config.enable_sparse_vectors else "",
                self.vector_size,
                self.config.distance.value,
            )
        else:
            existing_size = self.existing_dense_vector_size()
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Collection {self.collection_name!r} dense vector size is "
                    f"{existing_size}, but embedder produced {self.vector_size}. "
                    "Use a new collection name or recreate the collection."
                )

        if self.config.create_payload_indexes and self.config.mode != QdrantMode.MEMORY:
            self.create_payload_indexes()

    def _sparse_vectors_config(self) -> dict[str, models.SparseVectorParams] | None:
        if not self.config.enable_sparse_vectors:
            return None

        return {
            self.config.sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=self.config.sparse_on_disk,
                )
            )
        }

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(item.name == self.collection_name for item in collections)

    def existing_dense_vector_size(self) -> int:
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors_config = info.config.params.vectors

        if hasattr(vectors_config, "size"):
            return int(vectors_config.size)

        if isinstance(vectors_config, dict):
            dense_config = vectors_config.get(self.config.dense_vector_name)
            if dense_config is None:
                raise ValueError(
                    f"Collection {self.collection_name!r} has no dense vector "
                    f"{self.config.dense_vector_name!r}."
                )

            return int(dense_config.size)

        raise ValueError(
            f"Unable to read dense vector size for collection {self.collection_name!r}."
        )

    def _distance(self) -> models.Distance:
        mapping = {
            VectorDistance.COSINE: models.Distance.COSINE,
            VectorDistance.DOT: models.Distance.DOT,
            VectorDistance.EUCLID: models.Distance.EUCLID,
        }
        return mapping[self.config.distance]

    def create_payload_indexes(self) -> None:
        keyword_fields = [
            "tenant_id",
            "job_id",
            "source_id",
            "chunk_id",
            "chunk_type",
            "subtype",
            "source",
            "chunk_artifact_file",
            "text_sha256",
            "schema_version",
            "point_identity_version",
            "embedding_model",
            "dense_embedding_model",
            "sparse_embedding_model",
            "vector_backend",
        ]

        integer_fields = [
            "tokens",
            "order",
        ]

        bool_fields = [
            "has_dense_vector",
            "has_sparse_vector",
        ]

        for field_name in keyword_fields:
            self._create_payload_index(
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        for field_name in integer_fields:
            self._create_payload_index(
                field_name=field_name,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        for field_name in bool_fields:
            self._create_payload_index(
                field_name=field_name,
                field_schema=models.PayloadSchemaType.BOOL,
            )

    def _create_payload_index(
        self,
        *,
        field_name: str,
        field_schema: models.PayloadSchemaType,
    ) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:
            logger.debug(
                "Skipping payload index creation for %s.%s: %s",
                self.collection_name,
                field_name,
                exc,
            )

    def upsert_points(
        self,
        points: Sequence[VectorPoint],
        *,
        batch_size: int | None = None,
    ) -> StoreUpsertResult:
        if not points:
            return StoreUpsertResult(
                collection_name=self.collection_name,
                attempted_count=0,
                upserted_count=0,
                skipped_count=0,
                status="empty",
            )

        active_batch_size = int(batch_size or self.config.upsert_batch_size)
        if active_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        attempted_count = len(points)
        upserted_count = 0
        skipped_count = 0

        for batch in _batch(points, active_batch_size):
            qdrant_points: list[models.PointStruct] = []

            for point in batch:
                if not self._valid_point(point):
                    skipped_count += 1
                    continue

                qdrant_points.append(
                    models.PointStruct(
                        id=point.id,
                        vector=self._qdrant_vector(point),
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
                "Qdrant upsert: collection=%s status=%s count=%s",
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

    def _qdrant_vector(self, point: VectorPoint) -> dict[str, Any]:
        vector: dict[str, Any] = {
            self.config.dense_vector_name: point.dense_vector,
        }

        if (
            self.config.enable_sparse_vectors
            and point.sparse_vector is not None
            and not point.sparse_vector.is_empty
        ):
            vector[self.config.sparse_vector_name] = models.SparseVector(
                indices=point.sparse_vector.indices,
                values=point.sparse_vector.values,
            )

        return vector

    def search_dense(
        self,
        *,
        query_embedding: Vector,
        tenant_id: str,
        n_results: int = 5,
        source_id: str | None = None,
        query_text: str = "",
    ) -> SearchResults:
        if not query_embedding:
            return SearchResults(query=query_text, results=[])

        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding mismatch: got {len(query_embedding)}, expected {self.vector_size}."
            )

        query_filter = self.build_tenant_filter(
            tenant_id=tenant_id,
            source_id=source_id,
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.config.dense_vector_name,
            query_filter=query_filter,
            limit=n_results,
            with_payload=True,
            with_vectors=False,
        )

        return SearchResults(
            query=query_text,
            results=self._points_to_results(getattr(response, "points", response)),
        )

    def search_hybrid(
        self,
        *,
        query_dense: Vector,
        query_sparse: SparseVectorData,
        tenant_id: str,
        n_results: int = 5,
        source_id: str | None = None,
        query_text: str = "",
    ) -> SearchResults:
        self._require_hybrid_models()

        query_filter = self.build_tenant_filter(
            tenant_id=tenant_id,
            source_id=source_id,
        )

        fusion = (
            models.Fusion.DBSF
            if self.config.hybrid_fusion == "DBSF"
            else models.Fusion.RRF
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using=self.config.dense_vector_name,
                    filter=query_filter,
                    limit=self.config.hybrid_prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using=self.config.sparse_vector_name,
                    filter=query_filter,
                    limit=self.config.hybrid_prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=n_results,
            with_payload=True,
            with_vectors=False,
        )

        return SearchResults(
            query=query_text,
            results=self._points_to_results(getattr(response, "points", response)),
        )

    def build_tenant_filter(
        self,
        *,
        tenant_id: str,
        source_id: str | None = None,
    ) -> models.Filter:
        must: list[models.FieldCondition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            )
        ]

        if source_id:
            must.append(
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(value=source_id),
                )
            )

        return models.Filter(must=must)

    def count(
        self,
        *,
        query_filter: models.Filter | None = None,
    ) -> int:
        response = self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(response.count)

    def build_source_filter(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
    ) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id
                    ),
                ),
                models.FieldCondition(
                    key="job_id",
                    match=models.MatchValue(
                        value=job_id
                    ),
                ),
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(
                        value=source_id
                    ),
                ),
            ]
        )

    def count_source(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
    ) -> int:
        return self.count(
            query_filter=self.build_source_filter(
                tenant_id=tenant_id,
                job_id=job_id,
                source_id=source_id,
            )
        )

    def delete_source_points(
        self,
        *,
        tenant_id: str,
        job_id: str,
        source_id: str,
    ) -> int:
        query_filter = self.build_source_filter(
            tenant_id=tenant_id,
            job_id=job_id,
            source_id=source_id,
        )

        before = self.count(
            query_filter=query_filter
        )

        if before <= 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=query_filter
            ),
            wait=True,
        )

        after = self.count(
            query_filter=query_filter
        )

        if after != 0:
            raise RuntimeError(
                "Qdrant source deletion verification failed: "
                f"tenant_id={tenant_id!r} "
                f"job_id={job_id!r} "
                f"source_id={source_id!r} "
                f"remaining={after}"
            )

        return before

    def count_tenant(self, tenant_id: str) -> int:
        return self.count(
            query_filter=self.build_tenant_filter(
                tenant_id=tenant_id
            ),
        )

    def _points_to_results(self, points: Sequence[Any]) -> list[SearchResult]:
        results: list[SearchResult] = []

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

        return results

    def _valid_point(self, point: VectorPoint) -> bool:
        if not point.id:
            return False

        if not point.dense_vector:
            return False

        if len(point.dense_vector) != self.vector_size:
            return False

        if not all(isinstance(value, (float, int)) for value in point.dense_vector):
            return False

        if not point.payload:
            return False

        if not point.payload.get("tenant_id"):
            return False

        if self.config.enable_sparse_vectors:
            if point.sparse_vector is None or point.sparse_vector.is_empty:
                return False

            if len(point.sparse_vector.indices) != len(point.sparse_vector.values):
                return False

        return True

    @staticmethod
    def _require_hybrid_models() -> None:
        missing = [
            name
            for name in ("Prefetch", "Fusion", "FusionQuery", "SparseVector")
            if not hasattr(models, name)
        ]

        if missing:
            raise RuntimeError(
                "Hybrid search requires newer qdrant-client models. "
                f"Missing: {missing}. Run: uv add 'qdrant-client>=1.18.0,<2'"
            )


def _batch(
    items: Sequence[Any],
    size: int,
) -> list[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be greater than zero.")

    return [
        items[index : index + size]
        for index in range(0, len(items), size)
    ]