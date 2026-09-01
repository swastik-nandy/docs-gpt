from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from rag.ingestion.data_transformation.chunking.chunker import chunk_document
from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.chunking.validator import validate_chunks
from rag.ingestion.embeddings.config import (
    EmbeddingPipelineConfig,
    QdrantMode,
)
from rag.ingestion.embeddings.embedder import Embedder
from rag.ingestion.embeddings.store import QdrantStore
from rag.ingestion.embeddings.types import (
    EmbeddingValidationReport,
    IngestionResult,
    StoreUpsertResult,
    ValidEmbeddingPairs,
    Vector,
)
from rag.ingestion.embeddings.validator import filter_valid_pairs
from rag.ingestion.data_transformation.parser import parse


logger = logging.getLogger(__name__)


SUPPORTED_DOCUMENT_EXTENSIONS = {".md", ".mdx"}


# ---------------- FILE DISCOVERY ---------------- #


def discover_markdown_files(
    path: str | Path,
    *,
    recursive: bool = True,
) -> List[Path]:
    """
    Discover markdown-like files from a file or directory.

    Supported:
      - .md
      - .mdx

    The output is sorted for deterministic ingestion.
    """

    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {root.suffix!r}. "
                f"Supported: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
            )

        return [root]

    if not root.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {root}")

    pattern = "**/*" if recursive else "*"

    files: List[Path] = []
    for candidate in root.glob(pattern):
        if not candidate.is_file():
            continue

        if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        if _is_ignored_path(candidate):
            continue

        files.append(candidate)

    return sorted(files, key=lambda item: item.as_posix())


def _is_ignored_path(path: Path) -> bool:
    ignored_parts = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".venv",
        "venv",
        "env",
    }

    return any(part in ignored_parts for part in path.parts)


# ---------------- PUBLIC INGESTION API ---------------- #


def ingest_path(
    path: str | Path,
    *,
    config: Optional[EmbeddingPipelineConfig] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    recursive: bool = True,
    refresh_source: bool = False,
) -> List[IngestionResult]:
    """
    Ingest a single markdown file or a directory of markdown files.

    For your current use case, this supports:

      docs/mellisearch/claude/commands/

    It will discover all .md and .mdx files under that folder and ingest them
    into the same Qdrant collection.
    """

    files = discover_markdown_files(path, recursive=recursive)

    if not files:
        raise ValueError(f"No markdown files found under: {path}")

    runtime_config = _build_runtime_config(
        config=config,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    embedder = Embedder(runtime_config.embedding)
    store = _build_store(runtime_config, vector_size=embedder.vector_size)

    results: List[IngestionResult] = []

    logger.info(
        "Starting ingestion: files=%s collection=%s vector_size=%s model=%s",
        len(files),
        runtime_config.qdrant.collection_name,
        embedder.vector_size,
        embedder.model_name,
    )

    for file_path in files:
        result = _ingest_one_file(
            file_path=file_path,
            config=runtime_config,
            embedder=embedder,
            store=store,
            refresh_source=refresh_source,
        )
        results.append(result)

    stored_total = sum(item.stored_count for item in results)
    chunk_total = sum(item.chunk_count for item in results)

    logger.info(
        "Ingestion finished: files=%s chunks=%s stored=%s collection=%s",
        len(results),
        chunk_total,
        stored_total,
        runtime_config.qdrant.collection_name,
    )

    return results


def ingest_file(
    file_path: str | Path,
    *,
    config: Optional[EmbeddingPipelineConfig] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    refresh_source: bool = False,
) -> IngestionResult:
    """
    Backward-friendly single-file ingestion entrypoint.
    """

    results = ingest_path(
        file_path,
        config=config,
        collection_name=collection_name,
        persist_directory=persist_directory,
        recursive=False,
        refresh_source=refresh_source,
    )

    return results[0]


def ingest_directory(
    directory_path: str | Path,
    *,
    config: Optional[EmbeddingPipelineConfig] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    recursive: bool = True,
    refresh_source: bool = False,
) -> List[IngestionResult]:
    """
    Directory ingestion entrypoint.

    Use this for:
      docs/mellisearch/claude/commands/
    """

    return ingest_path(
        directory_path,
        config=config,
        collection_name=collection_name,
        persist_directory=persist_directory,
        recursive=recursive,
        refresh_source=refresh_source,
    )


# ---------------- SINGLE FILE FLOW ---------------- #


def _ingest_one_file(
    *,
    file_path: Path,
    config: EmbeddingPipelineConfig,
    embedder: Embedder,
    store: QdrantStore,
    refresh_source: bool,
) -> IngestionResult:
    source = file_path.as_posix()
    warnings: List[str] = []

    logger.info("Ingesting file: %s", source)

    try:
        sections = _parse_sections(file_path)
    except Exception as exc:
        logger.exception("Failed to parse file: %s", source)
        return _failed_result(
            source_path=source,
            config=config,
            embedder=embedder,
            warning=f"Parse failed: {exc}",
        )

    if not sections:
        return _failed_result(
            source_path=source,
            config=config,
            embedder=embedder,
            parsed=True,
            warning="No parser sections found.",
        )

    try:
        raw_chunks = chunk_document(sections, source=source)
    except Exception as exc:
        logger.exception("Failed to chunk file: %s", source)
        return _failed_result(
            source_path=source,
            config=config,
            embedder=embedder,
            parsed=True,
            warning=f"Chunking failed: {exc}",
        )

    if not raw_chunks:
        return _failed_result(
            source_path=source,
            config=config,
            embedder=embedder,
            parsed=True,
            warning="Chunker returned no chunks.",
        )

    valid_chunks = validate_chunks(raw_chunks)

    if not valid_chunks:
        return _failed_result(
            source_path=source,
            config=config,
            embedder=embedder,
            parsed=True,
            warning="Chunk validation removed all chunks.",
        )

    if refresh_source:
        refresh_warning = _refresh_existing_source(store, source)
        if refresh_warning:
            warnings.append(refresh_warning)

    embedded_count = 0
    stored_count = 0
    validation_reports: List[EmbeddingValidationReport] = []

    for batch_index, batch in enumerate(
        _batch(valid_chunks, config.embedding.batch_size),
        start=1,
    ):
        logger.info(
            "Embedding batch: source=%s batch=%s size=%s",
            source,
            batch_index,
            len(batch),
        )

        try:
            embedding_batch = embedder.embed_documents(batch)
        except Exception as exc:
            logger.exception("Embedding failed: source=%s batch=%s", source, batch_index)
            warnings.append(f"Embedding failed for batch {batch_index}: {exc}")
            continue

        pairs = _validate_pairs(
            chunks=batch,
            vectors=embedding_batch.vectors,
            expected_dim=embedder.vector_size,
        )

        validation_reports.append(pairs.report)

        if not pairs.chunks:
            warnings.append(f"Batch {batch_index} had no valid embeddings.")
            continue

        embedded_count += pairs.count

        try:
            upsert_result = _upsert_pairs(
                store=store,
                pairs=pairs,
                embedding_model=embedding_batch.model_name,
                schema_version=config.schema_version,
            )
        except Exception as exc:
            logger.exception("Qdrant upsert failed: source=%s batch=%s", source, batch_index)
            warnings.append(f"Store upsert failed for batch {batch_index}: {exc}")
            continue

        stored_count += upsert_result.upserted_count

    merged_report = _merge_validation_reports(
        reports=validation_reports,
        total_chunks=len(valid_chunks),
        expected_dim=embedder.vector_size,
    )

    return IngestionResult(
        source_path=source,
        collection_name=config.qdrant.collection_name,
        parsed=True,
        chunk_count=len(valid_chunks),
        embedded_count=embedded_count,
        stored_count=stored_count,
        embedding_model=embedder.model_name,
        vector_size=embedder.vector_size,
        validation_report=merged_report,
        warnings=tuple(warnings),
    )


# ---------------- PARSING ---------------- #


def _parse_sections(file_path: Path) -> List[Any]:
    documents = parse(file_path.as_posix())

    if not documents:
        return []

    sections: List[Any] = []

    for document in documents:
        metadata = getattr(document, "metadata", None) or {}
        document_sections = metadata.get("ast") or []

        if isinstance(document_sections, list):
            sections.extend(document_sections)

    return sections


# ---------------- VALIDATION BRIDGE ---------------- #


def _validate_pairs(
    *,
    chunks: Sequence[Chunk],
    vectors: Sequence[Vector],
    expected_dim: int,
) -> ValidEmbeddingPairs:
    """
    Supports both the upcoming production validator and your older validator.

    New expected validator signature:
      filter_valid_pairs(chunks, vectors, expected_dim=...)

    Old validator signature:
      filter_valid_pairs(chunks, vectors)
    """

    try:
        result = filter_valid_pairs(
            chunks,
            vectors,
            expected_dim=expected_dim,
        )
    except TypeError:
        result = filter_valid_pairs(chunks, vectors)

    if isinstance(result, ValidEmbeddingPairs):
        return result

    if isinstance(result, tuple) and len(result) == 2:
        valid_chunks, valid_vectors = result

        report = EmbeddingValidationReport(
            total_chunks=len(chunks),
            total_embeddings=len(vectors),
            valid_count=len(valid_chunks),
            invalid_count=max(len(chunks), len(vectors)) - len(valid_chunks),
            expected_dim=expected_dim,
            issues=tuple(),
        )

        return ValidEmbeddingPairs(
            chunks=list(valid_chunks),
            vectors=list(valid_vectors),
            report=report,
        )

    raise TypeError(
        "filter_valid_pairs returned an unsupported value. "
        "Expected ValidEmbeddingPairs or tuple(valid_chunks, valid_vectors)."
    )


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

    total_embeddings = sum(report.total_embeddings for report in reports)
    valid_count = sum(report.valid_count for report in reports)
    invalid_count = sum(report.invalid_count for report in reports)

    issues = []
    for report in reports:
        issues.extend(report.issues)

    return EmbeddingValidationReport(
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        valid_count=valid_count,
        invalid_count=invalid_count,
        expected_dim=expected_dim,
        issues=tuple(issues),
    )


# ---------------- STORE BRIDGE ---------------- #


def _build_store(
    config: EmbeddingPipelineConfig,
    *,
    vector_size: int,
) -> QdrantStore:
    """
    Prefer the new production store signature.

    Expected upcoming store signature:
      QdrantStore(config=config.qdrant, vector_size=vector_size)

    Fallback supports your older local-only store while you are migrating.
    """

    try:
        return QdrantStore(
            config=config.qdrant,
            vector_size=vector_size,
        )
    except TypeError:
        if config.qdrant.mode != QdrantMode.LOCAL:
            raise RuntimeError(
                "Current store.py does not support remote/memory Qdrant config yet. "
                "Rewrite store.py next before using QDRANT_MODE other than local."
            )

        return QdrantStore(
            collection_name=config.qdrant.collection_name,
            persist_directory=config.qdrant.local_path.as_posix(),
            vector_size=vector_size,
        )


def _upsert_pairs(
    *,
    store: QdrantStore,
    pairs: ValidEmbeddingPairs,
    embedding_model: str,
    schema_version: str,
) -> StoreUpsertResult:
    """
    Prefer production store.add signature with schema/model traceability.

    Falls back to old:
      store.add(chunks, vectors)
    """

    try:
        result = store.add(
            pairs.chunks,
            pairs.vectors,
            embedding_model=embedding_model,
            schema_version=schema_version,
        )
    except TypeError:
        result = store.add(pairs.chunks, pairs.vectors)

    if isinstance(result, StoreUpsertResult):
        return result

    return StoreUpsertResult(
        collection_name=getattr(store, "collection_name", "unknown"),
        attempted_count=pairs.count,
        upserted_count=pairs.count,
        skipped_count=0,
        status="completed",
    )


def _refresh_existing_source(store: QdrantStore, source: str) -> Optional[str]:
    """
    Delete existing chunks for the same source before re-ingestion.

    This needs store.py to implement:
      delete_by_source(source: str) -> int

    Without that method, deterministic point IDs still reduce duplicates, but
    stale chunks from heavily edited files may remain.
    """

    delete_method = getattr(store, "delete_by_source", None)

    if not callable(delete_method):
        return (
            "refresh_source=True was requested, but store.delete_by_source(...) "
            "is not implemented yet."
        )

    deleted_count = delete_method(source)
    logger.info("Deleted existing source chunks: source=%s deleted=%s", source, deleted_count)
    return None


# ---------------- CONFIG ---------------- #


def _build_runtime_config(
    *,
    config: Optional[EmbeddingPipelineConfig],
    collection_name: Optional[str],
    persist_directory: Optional[str],
) -> EmbeddingPipelineConfig:
    runtime_config = config or EmbeddingPipelineConfig.from_env()

    qdrant_config = runtime_config.qdrant

    if collection_name:
        qdrant_config = replace(
            qdrant_config,
            collection_name=collection_name,
        )

    if persist_directory:
        qdrant_config = replace(
            qdrant_config,
            local_path=Path(persist_directory),
        )

    runtime_config = replace(
        runtime_config,
        qdrant=qdrant_config,
    )

    runtime_config.validate()
    return runtime_config


# ---------------- RESULT HELPERS ---------------- #


def _failed_result(
    *,
    source_path: str,
    config: EmbeddingPipelineConfig,
    embedder: Embedder,
    warning: str,
    parsed: bool = False,
) -> IngestionResult:
    return IngestionResult(
        source_path=source_path,
        collection_name=config.qdrant.collection_name,
        parsed=parsed,
        chunk_count=0,
        embedded_count=0,
        stored_count=0,
        embedding_model=embedder.model_name,
        vector_size=embedder.vector_size,
        validation_report=None,
        warnings=(warning,),
    )


def _batch(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    for index in range(0, len(items), size):
        yield list(items[index : index + size])


# ---------------- CLI ---------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest markdown files into Qdrant embeddings."
    )

    parser.add_argument(
        "path",
        help="Markdown file or directory containing .md/.mdx files.",
    )

    parser.add_argument(
        "--collection",
        default=None,
        help="Override QDRANT_COLLECTION from backend/.env.",
    )

    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Override QDRANT_LOCAL_PATH from backend/.env for local mode.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only ingest files directly inside the directory.",
    )

    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Delete existing chunks for each source before re-ingesting. Requires store.delete_by_source.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    results = ingest_path(
        args.path,
        collection_name=args.collection,
        persist_directory=args.persist_directory,
        recursive=not args.no_recursive,
        refresh_source=args.refresh_source,
    )

    total_chunks = sum(result.chunk_count for result in results)
    total_embedded = sum(result.embedded_count for result in results)
    total_stored = sum(result.stored_count for result in results)

    print("\nIngestion summary")
    print("-----------------")
    print(f"Files:    {len(results)}")
    print(f"Chunks:   {total_chunks}")
    print(f"Embedded: {total_embedded}")
    print(f"Stored:   {total_stored}")

    failed = [result for result in results if not result.success]
    if failed:
        print(f"Warnings: {len(failed)} file(s) had issues")
        for result in failed:
            print(f"- {result.source_path}")
            for warning in result.warnings:
                print(f"  {warning}")


if __name__ == "__main__":
    main()