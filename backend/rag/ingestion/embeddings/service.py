from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.artifact_reader import (
    ChunkArtifactFile,
    ChunkArtifactJob,
    discover_chunk_artifact_jobs,
    load_chunk_artifact_file,
)
from rag.ingestion.embeddings.catalog_repository import KnowledgeCatalogRepository
from rag.ingestion.embeddings.catalog_repository import build_indexed_source_catalog_record
from rag.ingestion.embeddings.config import (
    EmbeddingPipelineConfig,
    get_embedding_pipeline_config,
)
from rag.ingestion.embeddings.embedder import Embedder
from rag.ingestion.embeddings.payload import build_vector_points_from_chunks
from rag.ingestion.embeddings.qdrant_store import QdrantVectorStore
from rag.ingestion.embeddings.types import (
    EmbeddingValidationReport,
    ValidEmbeddingPairs,
)
from rag.ingestion.embeddings.validator import filter_valid_pairs


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedSourceReport:
    tenant_id: str
    job_id: str
    source_id: str
    chunk_artifact_path: str
    status: str
    chunk_count: int = 0
    embedded_count: int = 0
    upserted_count: int = 0
    skipped_count: int = 0
    vector_size: int = 0
    embedding_model: str = ""
    sparse_embedding_model: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "source_id": self.source_id,
            "chunk_artifact_path": self.chunk_artifact_path,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "embedded_count": self.embedded_count,
            "upserted_count": self.upserted_count,
            "skipped_count": self.skipped_count,
            "vector_size": self.vector_size,
            "embedding_model": self.embedding_model,
            "sparse_embedding_model": self.sparse_embedding_model,
            "validation": self.validation,
            "error": self.error,
        }


@dataclass(frozen=True)
class IndexedJobReport:
    tenant_id: str
    job_id: str
    status: str
    collection_name: str
    source_count: int
    succeeded: int
    failed: int
    chunk_count: int
    embedded_count: int
    upserted_count: int
    skipped_count: int
    seconds: float
    sources: tuple[IndexedSourceReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "status": self.status,
            "collection_name": self.collection_name,
            "source_count": self.source_count,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "chunk_count": self.chunk_count,
            "embedded_count": self.embedded_count,
            "upserted_count": self.upserted_count,
            "skipped_count": self.skipped_count,
            "seconds": round(self.seconds, 6),
            "sources": [source.to_dict() for source in self.sources],
        }


class ChunkArtifactIndexingService:
    """
    Consumes chunk artifacts and writes dense+sparse points to Qdrant.

    Important batching model:
      - embedding.batch_size controls GPU inference memory
      - qdrant.upsert_batch_size controls Qdrant HTTP/write batching
    """

    def __init__(
        self,
        config: EmbeddingPipelineConfig | None = None,
    ) -> None:
        self.config = config or get_embedding_pipeline_config()
        self.config.validate()

        self.embedder = Embedder(self.config.embedding)
        self.store = QdrantVectorStore(
            config=self.config.qdrant,
            vector_size=self.embedder.vector_size,
        )

        self.catalog = KnowledgeCatalogRepository()

    def index_available_jobs(self) -> list[IndexedJobReport]:
        jobs = discover_chunk_artifact_jobs(self.config)
        jobs = jobs[: self.config.indexing.max_jobs_per_iteration]

        reports: list[IndexedJobReport] = []

        for job in jobs:
            try:
                reports.append(self.index_job(job))
            except Exception:
                LOGGER.exception(
                    "Indexing job crashed | tenant_id=%s job_id=%s",
                    job.tenant_id,
                    job.job_id,
                )

                if not self.config.indexing.continue_on_error:
                    raise

        return reports

    def index_job(self, job: ChunkArtifactJob) -> IndexedJobReport:
        started = time.perf_counter()

        source_reports: list[IndexedSourceReport] = []
        indexed_root = self._indexed_job_root(job)

        indexed_root.mkdir(parents=True, exist_ok=True)
        (indexed_root / "reports").mkdir(parents=True, exist_ok=True)

        files = job.files[: self.config.indexing.max_files_per_job]

        LOGGER.info(
            "Indexing job started | tenant_id=%s job_id=%s files=%s",
            job.tenant_id,
            job.job_id,
            len(files),
        )

        for artifact_file in files:
            try:
                report = self.index_source_file(artifact_file)
            except Exception as exc:
                LOGGER.exception(
                    "Source indexing failed | tenant_id=%s job_id=%s source_id=%s",
                    artifact_file.tenant_id,
                    artifact_file.job_id,
                    artifact_file.source_id,
                )

                report = IndexedSourceReport(
                    tenant_id=artifact_file.tenant_id,
                    job_id=artifact_file.job_id,
                    source_id=artifact_file.source_id,
                    chunk_artifact_path=artifact_file.path.as_posix(),
                    status="failed",
                    vector_size=self.embedder.vector_size,
                    embedding_model=self.embedder.model_name,
                    sparse_embedding_model=self.embedder.sparse_model_name,
                    error=f"{type(exc).__name__}: {exc}",
                )

            source_reports.append(report)

            _write_json_atomic(
                indexed_root / "reports" / f"{artifact_file.source_id}.index_report.json",
                report.to_dict(),
            )

            if report.status == "failed" and not self.config.indexing.continue_on_error:
                break

        failed = sum(1 for report in source_reports if report.status != "succeeded")
        status = "succeeded" if failed == 0 and source_reports else "failed"

        job_report = IndexedJobReport(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            status=status,
            collection_name=self.config.qdrant.collection_name,
            source_count=len(source_reports),
            succeeded=len(source_reports) - failed,
            failed=failed,
            chunk_count=sum(report.chunk_count for report in source_reports),
            embedded_count=sum(report.embedded_count for report in source_reports),
            upserted_count=sum(report.upserted_count for report in source_reports),
            skipped_count=sum(report.skipped_count for report in source_reports),
            seconds=time.perf_counter() - started,
            sources=tuple(source_reports),
        )

        _write_json_atomic(
            indexed_root / "reports" / "job_report.json",
            job_report.to_dict(),
        )

        _write_json_atomic(
            indexed_root / "manifest.json",
            {
                "tenant_id": job.tenant_id,
                "job_id": job.job_id,
                "status": job_report.status,
                "collection_name": job_report.collection_name,
                "chunked_job_root": job.root.as_posix(),
                "source_count": job_report.source_count,
                "embedding_model": self.embedder.model_name,
                "sparse_embedding_model": self.embedder.sparse_model_name,
                "dense_vector_name": self.config.qdrant.dense_vector_name,
                "sparse_vector_name": self.config.qdrant.sparse_vector_name,
            },
        )

        if status == "succeeded":
            self.catalog.mark_job_succeeded(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
            )

            self._touch_marker(
                indexed_root
                / self.config.markers.indexed
            )

        else:
            self.catalog.mark_job_failed(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
            )

            self._touch_marker(
                indexed_root
                / self.config.markers.index_failed
            )

        LOGGER.info(
            "Indexing job finished | tenant_id=%s job_id=%s status=%s sources=%s chunks=%s embedded=%s upserted=%s skipped=%s seconds=%.2f",
            job.tenant_id,
            job.job_id,
            status,
            job_report.source_count,
            job_report.chunk_count,
            job_report.embedded_count,
            job_report.upserted_count,
            job_report.skipped_count,
            job_report.seconds,
        )

        return job_report

    def index_source_file(
        self,
        artifact_file: ChunkArtifactFile,
    ) -> IndexedSourceReport:
        started = time.perf_counter()

        loaded = load_chunk_artifact_file(artifact_file)
        chunks = list(loaded.chunks)

        if not chunks:
            return IndexedSourceReport(
                tenant_id=artifact_file.tenant_id,
                job_id=artifact_file.job_id,
                source_id=artifact_file.source_id,
                chunk_artifact_path=artifact_file.path.as_posix(),
                status="failed",
                vector_size=self.embedder.vector_size,
                embedding_model=self.embedder.model_name,
                sparse_embedding_model=self.embedder.sparse_model_name,
                error="No chunks found in artifact file.",
            )

        embedded_count = 0
        upserted_count = 0
        skipped_count = 0
        qdrant_write_count = 0

        validation_reports: list[EmbeddingValidationReport] = []
        pending_points: list[Any] = []

        def flush_pending_points() -> None:
            nonlocal pending_points
            nonlocal upserted_count
            nonlocal skipped_count
            nonlocal qdrant_write_count

            if not pending_points:
                return

            upsert_result = self.store.upsert_points(
                pending_points,
                batch_size=self.config.qdrant.upsert_batch_size,
            )

            upserted_count += upsert_result.upserted_count
            skipped_count += upsert_result.skipped_count
            qdrant_write_count += 1
            pending_points = []

        for chunk_batch in _batch(chunks, self.config.embedding.batch_size):
            embedding_batch = self.embedder.embed_documents(chunk_batch)

            pairs = self._validate_pairs(
                chunks=chunk_batch,
                dense_vectors=embedding_batch.dense_vectors,
                sparse_vectors=embedding_batch.sparse_vectors,
                expected_dim=self.embedder.vector_size,
            )

            validation_reports.append(pairs.report)

            skipped_count += len(chunk_batch) - pairs.count

            if not pairs.chunks:
                continue

            points = build_vector_points_from_chunks(
                artifact_file=artifact_file,
                chunks=pairs.chunks,
                dense_vectors=pairs.dense_vectors,
                sparse_vectors=pairs.sparse_vectors,
                dense_embedding_model=embedding_batch.dense_model_name,
                sparse_embedding_model=embedding_batch.sparse_model_name,
                schema_version=self.config.schema_version,
            )

            pending_points.extend(points)
            embedded_count += pairs.count

            if len(pending_points) >= self.config.qdrant.upsert_batch_size:
                flush_pending_points()

        flush_pending_points()

        merged_validation = _merge_validation_reports(
            reports=validation_reports,
            total_chunks=len(chunks),
            expected_dim=self.embedder.vector_size,
        )

        verified_count = self.store.count_source(
            tenant_id=artifact_file.tenant_id,
            job_id=artifact_file.job_id,
            source_id=artifact_file.source_id,
        )

        expected_count = len(chunks)

        qdrant_verified = (
            upserted_count == expected_count
            and embedded_count == expected_count
            and skipped_count == 0
            and verified_count == expected_count
        )

        if not qdrant_verified:
            self.store.delete_source_points(
                tenant_id=artifact_file.tenant_id,
                job_id=artifact_file.job_id,
                source_id=artifact_file.source_id,
            )

            seconds = time.perf_counter() - started

            error = (
                "Qdrant source verification failed: "
                f"expected={expected_count} "
                f"embedded={embedded_count} "
                f"upserted={upserted_count} "
                f"verified={verified_count} "
                f"skipped={skipped_count}"
            )

            LOGGER.error(
                "Source verification failed | tenant_id=%s job_id=%s source_id=%s expected=%s embedded=%s upserted=%s verified=%s skipped=%s",
                artifact_file.tenant_id,
                artifact_file.job_id,
                artifact_file.source_id,
                expected_count,
                embedded_count,
                upserted_count,
                verified_count,
                skipped_count,
            )

            return IndexedSourceReport(
                tenant_id=artifact_file.tenant_id,
                job_id=artifact_file.job_id,
                source_id=artifact_file.source_id,
                chunk_artifact_path=artifact_file.path.as_posix(),
                status="failed",
                chunk_count=len(chunks),
                embedded_count=embedded_count,
                upserted_count=upserted_count,
                skipped_count=skipped_count,
                vector_size=self.embedder.vector_size,
                embedding_model=self.embedder.model_name,
                sparse_embedding_model=self.embedder.sparse_model_name,
                validation=_validation_report_to_dict(
                    merged_validation
                ),
                error=error,
            )

        catalog_record = (
            build_indexed_source_catalog_record(
                artifact_file=artifact_file,
                chunks=chunks,
                config=self.config,
                indexed_chunk_count=verified_count,
            )
        )

        try:
            self.catalog.record_indexed_source(
                catalog_record
            )

        except Exception:
            LOGGER.exception(
                "Postgres catalog write failed after Qdrant indexing. "
                "Compensating Qdrant delete will run | "
                "tenant_id=%s job_id=%s source_id=%s",
                artifact_file.tenant_id,
                artifact_file.job_id,
                artifact_file.source_id,
            )

            self.store.delete_source_points(
                tenant_id=artifact_file.tenant_id,
                job_id=artifact_file.job_id,
                source_id=artifact_file.source_id,
            )

            raise

        seconds = time.perf_counter() - started

        LOGGER.info(
            "Source indexed and catalogued | tenant_id=%s job_id=%s source_id=%s chunks=%s verified=%s qdrant_writes=%s seconds=%.2f",
            artifact_file.tenant_id,
            artifact_file.job_id,
            artifact_file.source_id,
            len(chunks),
            verified_count,
            qdrant_write_count,
            seconds,
        )

        return IndexedSourceReport(
            tenant_id=artifact_file.tenant_id,
            job_id=artifact_file.job_id,
            source_id=artifact_file.source_id,
            chunk_artifact_path=artifact_file.path.as_posix(),
            status="succeeded",
            chunk_count=len(chunks),
            embedded_count=embedded_count,
            upserted_count=upserted_count,
            skipped_count=skipped_count,
            vector_size=self.embedder.vector_size,
            embedding_model=self.embedder.model_name,
            sparse_embedding_model=self.embedder.sparse_model_name,
            validation=_validation_report_to_dict(
                merged_validation
            ),
            error="",
        )

    def _validate_pairs(
        self,
        *,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[list[float]],
        sparse_vectors: Sequence[Any],
        expected_dim: int,
    ) -> ValidEmbeddingPairs:
        return filter_valid_pairs(
            chunks,
            dense_vectors,
            sparse_vectors=sparse_vectors,
            expected_dim=expected_dim,
        )

    def _indexed_job_root(self, job: ChunkArtifactJob) -> Path:
        return self.config.artifacts.indexed_root / job.tenant_id / job.job_id

    @staticmethod
    def _touch_marker(path: Path) -> None:
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text("", encoding="utf-8")
        temp_path.replace(path)


def _batch(
    items: Sequence[Any],
    size: int,
) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be greater than zero.")

    return [
        list(items[index : index + size])
        for index in range(0, len(items), size)
    ]


def _merge_validation_reports(
    *,
    reports: Sequence[EmbeddingValidationReport],
    total_chunks: int,
    expected_dim: int,
) -> EmbeddingValidationReport:
    if not reports:
        return EmbeddingValidationReport(
            total_chunks=total_chunks,
            total_embeddings=0,
            valid_count=0,
            invalid_count=total_chunks,
            expected_dim=expected_dim,
            issues=tuple(),
        )

    issues = []
    for report in reports:
        issues.extend(report.issues)

    return EmbeddingValidationReport(
        total_chunks=total_chunks,
        total_embeddings=sum(report.total_embeddings for report in reports),
        valid_count=sum(report.valid_count for report in reports),
        invalid_count=sum(report.invalid_count for report in reports),
        expected_dim=expected_dim,
        issues=tuple(issues),
    )


def _validation_report_to_dict(report: EmbeddingValidationReport) -> dict[str, Any]:
    return {
        "total_chunks": report.total_chunks,
        "total_embeddings": report.total_embeddings,
        "valid_count": report.valid_count,
        "invalid_count": report.invalid_count,
        "expected_dim": report.expected_dim,
        "issues": [
            {
                "index": issue.index,
                "chunk_id": issue.chunk_id,
                "reason": issue.reason,
            }
            for issue in report.issues[:50]
        ],
        "truncated_issues": max(0, len(report.issues) - 50),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)