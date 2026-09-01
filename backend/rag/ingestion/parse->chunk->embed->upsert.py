from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag.ingestion.data_transformation.parser import parse
from rag.ingestion.data_transformation.chunking.chunker import chunk_document
from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.chunking.validator import validate_chunks

from rag.ingestion.embeddings.config import EmbeddingPipelineConfig
from rag.ingestion.embeddings.embedder import Embedder
from rag.ingestion.embeddings.store import QdrantStore
from rag.ingestion.embeddings.validator import filter_valid_pairs
from rag.ingestion.embeddings.types import EmbeddingValidationReport


# -------------------------------------- CONFIG -------------------------------------------

DEFAULT_TARGET_PATH = "docs/mellisearch/claude/commands"
SUPPORTED_EXTENSIONS = {".md", ".mdx"}

DEFAULT_PARSE_CONCURRENCY = 4
DEFAULT_BATCH_SIZE = 32

logger = logging.getLogger("rag.embed")


# -------------------------------------- RESULT TYPES -------------------------------------------

@dataclass(frozen=True)
class ParsedChunkedFile:
    path: Path
    sections_count: int
    raw_chunk_count: int
    chunks: List[Chunk]
    ok: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class EmbeddedFileResult:
    path: Path
    ok: bool
    sections_count: int
    raw_chunk_count: int
    valid_chunk_count: int
    embedded_count: int
    stored_count: int
    errors: List[str]
    validation_reports: List[EmbeddingValidationReport]


# -------------------------------------- PATH HELPERS -------------------------------------------

def backend_root() -> Path:
    """
    Expected file location:
        backend/rag/embed.py

    Returns:
        backend/
    """

    return Path(__file__).resolve().parents[1]


def resolve_target(path_text: str) -> Path:
    """
    Resolve paths in this order:
      1. absolute path or cwd-relative path
      2. backend-root-relative path
    """

    raw_path = Path(path_text).expanduser()

    if raw_path.exists():
        return raw_path.resolve()

    backend_candidate = backend_root() / raw_path

    if backend_candidate.exists():
        return backend_candidate.resolve()

    raise FileNotFoundError(
        f"Path not found: {path_text}. Tried current directory and backend root."
    )


def discover_markdown_files(
    target: Path,
    *,
    recursive: bool = True,
) -> List[Path]:
    """
    Accept one markdown file or a directory containing markdown/mdx files.
    """

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {target.suffix!r}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        return [target]

    if not target.is_dir():
        raise ValueError(f"Target is neither a file nor a directory: {target}")

    pattern = "**/*" if recursive else "*"

    files = [
        path
        for path in target.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not is_ignored_path(path)
    ]

    return sorted(files, key=lambda item: item.as_posix())


def is_ignored_path(path: Path) -> bool:
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


# -------------------------------------- PARSER EXTRACTION -------------------------------------------

def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_sections(documents: Sequence[Any]) -> List[Dict[str, Any]]:
    """
    Parser contract used by the current parser layer:

        documents = parse(str(path))
        sections = documents[0].metadata["ast"]
    """

    sections: List[Dict[str, Any]] = []

    for document in documents:
        metadata = safe_dict(getattr(document, "metadata", {}))
        ast_sections = metadata.get("ast", [])

        if not isinstance(ast_sections, list):
            continue

        sections.extend(
            section
            for section in ast_sections
            if isinstance(section, dict)
        )

    return sections


# -------------------------------------- PARSE -> CHUNK -------------------------------------------

def parse_and_chunk_sync(file_path: Path) -> ParsedChunkedFile:
    """
    Blocking parse + chunk flow.

    It is intentionally sync because your parser and chunker are sync.
    The async runner executes this in worker threads.
    """

    try:
        documents = parse(str(file_path))

        if not documents:
            return ParsedChunkedFile(
                path=file_path,
                sections_count=0,
                raw_chunk_count=0,
                chunks=[],
                ok=False,
                error="parse returned no documents",
            )

        sections = extract_sections(documents)

        if not sections:
            return ParsedChunkedFile(
                path=file_path,
                sections_count=0,
                raw_chunk_count=0,
                chunks=[],
                ok=False,
                error="no sections found in document metadata ast",
            )

        raw_chunks = chunk_document(
            sections,
            source=str(file_path),
        )

        if not raw_chunks:
            return ParsedChunkedFile(
                path=file_path,
                sections_count=len(sections),
                raw_chunk_count=0,
                chunks=[],
                ok=False,
                error="chunk_document returned no chunks",
            )

        valid_chunks = validate_chunks(raw_chunks)

        if not valid_chunks:
            return ParsedChunkedFile(
                path=file_path,
                sections_count=len(sections),
                raw_chunk_count=len(raw_chunks),
                chunks=[],
                ok=False,
                error="chunk validation removed all chunks",
            )

        return ParsedChunkedFile(
            path=file_path,
            sections_count=len(sections),
            raw_chunk_count=len(raw_chunks),
            chunks=valid_chunks,
            ok=True,
        )

    except Exception as exc:
        return ParsedChunkedFile(
            path=file_path,
            sections_count=0,
            raw_chunk_count=0,
            chunks=[],
            ok=False,
            error=str(exc),
        )


async def parse_and_chunk_file(
    file_path: Path,
    semaphore: asyncio.Semaphore,
) -> ParsedChunkedFile:
    async with semaphore:
        logger.info("Parsing and chunking: %s", file_path)
        return await asyncio.to_thread(parse_and_chunk_sync, file_path)


# -------------------------------------- EMBED -> STORE -------------------------------------------

def batch_items(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


async def embed_and_store_file(
    parsed: ParsedChunkedFile,
    *,
    embedder: Embedder,
    store: Optional[QdrantStore],
    config: EmbeddingPipelineConfig,
    batch_size: int,
    refresh_source: bool,
    dry_run: bool,
) -> EmbeddedFileResult:
    if not parsed.ok:
        return EmbeddedFileResult(
            path=parsed.path,
            ok=False,
            sections_count=parsed.sections_count,
            raw_chunk_count=parsed.raw_chunk_count,
            valid_chunk_count=0,
            embedded_count=0,
            stored_count=0,
            errors=[parsed.error or "parse/chunk failed"],
            validation_reports=[],
        )

    errors: List[str] = []
    validation_reports: List[EmbeddingValidationReport] = []

    embedded_count = 0
    stored_count = 0

    if refresh_source and not dry_run:
        if store is None:
            raise RuntimeError("store is required when refresh_source=True")

        try:
            deleted_count = await asyncio.to_thread(
                store.delete_by_source,
                str(parsed.path),
            )
            logger.info(
                "Deleted existing chunks for source=%s count=%s",
                parsed.path,
                deleted_count,
            )
        except Exception as exc:
            errors.append(f"source refresh failed: {exc}")

    for batch_index, chunk_batch in enumerate(
        batch_items(parsed.chunks, batch_size),
        start=1,
    ):
        try:
            logger.info(
                "Embedding batch: file=%s batch=%s size=%s",
                parsed.path.name,
                batch_index,
                len(chunk_batch),
            )

            embedding_batch = await asyncio.to_thread(
                embedder.embed_documents,
                chunk_batch,
            )

            pairs = filter_valid_pairs(
                chunk_batch,
                embedding_batch.vectors,
                expected_dim=embedder.vector_size,
            )

            validation_reports.append(pairs.report)

            if not pairs.chunks:
                errors.append(f"batch {batch_index}: no valid embedding pairs")
                continue

            embedded_count += pairs.count

            if dry_run:
                continue

            if store is None:
                raise RuntimeError("Qdrant store is not initialized")

            upsert_result = await asyncio.to_thread(
                store.add,
                pairs.chunks,
                pairs.vectors,
                embedding_model=embedding_batch.model_name,
                schema_version=config.schema_version,
                batch_size=batch_size,
            )

            stored_count += upsert_result.upserted_count

        except Exception as exc:
            errors.append(f"batch {batch_index}: {exc}")

    ok = not errors and embedded_count > 0 and (dry_run or stored_count > 0)

    return EmbeddedFileResult(
        path=parsed.path,
        ok=ok,
        sections_count=parsed.sections_count,
        raw_chunk_count=parsed.raw_chunk_count,
        valid_chunk_count=len(parsed.chunks),
        embedded_count=embedded_count,
        stored_count=stored_count,
        errors=errors,
        validation_reports=validation_reports,
    )


# -------------------------------------- CONFIG BUILDING -------------------------------------------

def build_config(
    *,
    collection: Optional[str],
    persist_directory: Optional[str],
    batch_size: Optional[int],
) -> EmbeddingPipelineConfig:
    config = EmbeddingPipelineConfig.from_env()

    qdrant_config = config.qdrant
    embedding_config = config.embedding

    if collection:
        qdrant_config = replace(
            qdrant_config,
            collection_name=collection,
        )

    if persist_directory:
        qdrant_config = replace(
            qdrant_config,
            local_path=Path(persist_directory),
        )

    if batch_size:
        embedding_config = replace(
            embedding_config,
            batch_size=batch_size,
        )

    config = replace(
        config,
        qdrant=qdrant_config,
        embedding=embedding_config,
    )

    config.validate()
    return config


# -------------------------------------- MAIN PIPELINE -------------------------------------------

async def run_embedding_pipeline(
    *,
    target_path: str,
    collection: Optional[str],
    persist_directory: Optional[str],
    recursive: bool,
    parse_concurrency: int,
    batch_size: int,
    refresh_source: bool,
    reset_collection: bool,
    dry_run: bool,
) -> List[EmbeddedFileResult]:
    started = time.perf_counter()

    target = resolve_target(target_path)
    files = discover_markdown_files(
        target,
        recursive=recursive,
    )

    if not files:
        raise ValueError(f"No markdown/mdx files found under: {target}")

    config = build_config(
        collection=collection,
        persist_directory=persist_directory,
        batch_size=batch_size,
    )

    logger.info("Backend root       : %s", backend_root())
    logger.info("Target             : %s", target)
    logger.info("Files discovered   : %s", len(files))
    logger.info("Collection         : %s", config.qdrant.collection_name)
    logger.info("Embedding model    : %s", config.embedding.model_name)
    logger.info("Embedding batch    : %s", config.embedding.batch_size)
    logger.info("Parse concurrency  : %s", parse_concurrency)
    logger.info("Dry run            : %s", dry_run)

    embedder = Embedder(config.embedding)

    store: Optional[QdrantStore] = None

    if not dry_run:
        store = QdrantStore(
            config=config.qdrant,
            vector_size=embedder.vector_size,
        )

        if reset_collection:
            store.reset()

    parse_semaphore = asyncio.Semaphore(parse_concurrency)

    parsed_results = await asyncio.gather(
        *[
            parse_and_chunk_file(
                file_path=file_path,
                semaphore=parse_semaphore,
            )
            for file_path in files
        ]
    )

    results: List[EmbeddedFileResult] = []

    for parsed in parsed_results:
        logger.info(
            "Embedding file: %s chunks=%s ok=%s",
            parsed.path.name,
            len(parsed.chunks),
            parsed.ok,
        )

        result = await embed_and_store_file(
            parsed,
            embedder=embedder,
            store=store,
            config=config,
            batch_size=batch_size,
            refresh_source=refresh_source,
            dry_run=dry_run,
        )

        results.append(result)

    elapsed_seconds = time.perf_counter() - started

    report_summary(
        results=results,
        collection_name=config.qdrant.collection_name,
        vector_size=embedder.vector_size,
        elapsed_seconds=elapsed_seconds,
        dry_run=dry_run,
    )

    return results


# -------------------------------------- REPORTING -------------------------------------------

def report_summary(
    *,
    results: Sequence[EmbeddedFileResult],
    collection_name: str,
    vector_size: int,
    elapsed_seconds: float,
    dry_run: bool,
) -> None:
    total_files = len(results)
    ok_files = sum(1 for result in results if result.ok)

    total_sections = sum(result.sections_count for result in results)
    total_raw_chunks = sum(result.raw_chunk_count for result in results)
    total_valid_chunks = sum(result.valid_chunk_count for result in results)
    total_embedded = sum(result.embedded_count for result in results)
    total_stored = sum(result.stored_count for result in results)

    print("\n" + "=" * 100)
    print("RAG EMBEDDING INGESTION SUMMARY")
    print("=" * 100)
    print(f"Collection      : {collection_name}")
    print(f"Vector size     : {vector_size}")
    print(f"Dry run         : {dry_run}")
    print(f"Files           : {ok_files}/{total_files} ok")
    print(f"Sections        : {total_sections}")
    print(f"Raw chunks      : {total_raw_chunks}")
    print(f"Valid chunks    : {total_valid_chunks}")
    print(f"Embedded        : {total_embedded}")
    print(f"Stored          : {total_stored}")
    print(f"Elapsed         : {elapsed_seconds:.2f}s")

    print("\nFiles")
    print("-" * 100)

    for result in results:
        status = "OK" if result.ok else "FAIL"

        print(
            f"{status} | {result.path} | "
            f"sections={result.sections_count} "
            f"raw_chunks={result.raw_chunk_count} "
            f"valid_chunks={result.valid_chunk_count} "
            f"embedded={result.embedded_count} "
            f"stored={result.stored_count}"
        )

        for error in result.errors:
            print(f"  - {error}")

        validation_issue_count = sum(
            report.invalid_count
            for report in result.validation_reports
        )

        if validation_issue_count:
            print(f"  - embedding validation issues: {validation_issue_count}")

    print("=" * 100 + "\n")


# -------------------------------------- CLI -------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse -> chunk -> embed -> store markdown docs in Qdrant."
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_TARGET_PATH,
        help=f"Markdown file or directory. Default: {DEFAULT_TARGET_PATH}",
    )

    parser.add_argument(
        "--collection",
        default=None,
        help="Override QDRANT_COLLECTION from backend/.env.",
    )

    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Override QDRANT_LOCAL_PATH from backend/.env in local mode.",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan direct children when path is a directory.",
    )

    parser.add_argument(
        "--parse-concurrency",
        type=int,
        default=DEFAULT_PARSE_CONCURRENCY,
        help="Number of files to parse/chunk concurrently.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding and Qdrant upsert batch size.",
    )

    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Delete existing chunks for each source before re-ingesting.",
    )

    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate the whole collection before ingestion.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, chunk, and embed, but do not write to Qdrant.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.parse_concurrency <= 0:
        raise ValueError("--parse-concurrency must be greater than zero.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    if args.refresh_source and args.reset_collection:
        raise ValueError(
            "Use either --refresh-source or --reset-collection, not both."
        )


async def async_main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    await run_embedding_pipeline(
        target_path=args.path,
        collection=args.collection,
        persist_directory=args.persist_directory,
        recursive=not args.no_recursive,
        parse_concurrency=args.parse_concurrency,
        batch_size=args.batch_size,
        refresh_source=args.refresh_source,
        reset_collection=args.reset_collection,
        dry_run=args.dry_run,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()