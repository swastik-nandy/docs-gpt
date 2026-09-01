from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from rag.ingestion.data_transformation.parser.inspection.markdown import inspect_markdown_file
from rag.ingestion.data_transformation.parser.inspection.schema import MarkdownInspection
from rag.ingestion.data_transformation.parser.adapters.markdown import parse_markdown


@dataclass
class MarkdownParseExecutionResult:
    """
    In-memory execution result.

    Later this same shape can map to:
    - S3 artifacts
    - queue jobs
    - partial IR files
    - merge jobs

    For now, everything stays in memory.
    """

    inspection: MarkdownInspection
    strategy: str
    documents: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fragments: List[dict[str, Any]] = field(default_factory=list)


def _mark_documents(
    documents: List[Any],
    inspection: MarkdownInspection,
    strategy: str,
) -> List[Any]:
    """
    Attach inspection/strategy metadata to current Document objects.
    """

    for document in documents:
        metadata = getattr(document, "metadata", None)

        if not isinstance(metadata, dict):
            continue

        metadata["parse_strategy"] = strategy
        metadata["inspection"] = inspection.to_dict()
        metadata["spine_node_count"] = len(inspection.spine_nodes)
        metadata["size_class"] = inspection.size_class
        metadata["risk_level"] = inspection.risk_level
        metadata["target_queue"] = inspection.target_queue

    return documents


def parse_markdown_with_strategy(
    file_path: str,
) -> MarkdownParseExecutionResult:
    """
    Strategy-aware Markdown parser entrypoint.

    Current version:
    - keeps everything in memory
    - uses inspection decision
    - executes existing parser for safe paths
    - does not yet implement real spine-fragmented parsing

    Later:
    - spine_fragmented will create fragment parse results
    - fragments will become partial IR artifacts
    - merge will become a separate stage
    """

    inspection = inspect_markdown_file(file_path)
    strategy = inspection.parse_strategy
    warnings = list(inspection.warnings or [])

    if strategy == "single_document":
        documents = parse_markdown(file_path)

        return MarkdownParseExecutionResult(
            inspection=inspection,
            strategy=strategy,
            documents=_mark_documents(
                documents=documents,
                inspection=inspection,
                strategy=strategy,
            ),
            warnings=warnings,
        )

    if strategy == "full_parse_then_chunk_fanout":
        documents = parse_markdown(file_path)

        warnings.append(
            "Parsed as full document. Chunking layer should fan out later."
        )

        return MarkdownParseExecutionResult(
            inspection=inspection,
            strategy=strategy,
            documents=_mark_documents(
                documents=documents,
                inspection=inspection,
                strategy=strategy,
            ),
            warnings=warnings,
        )

    if strategy == "spine_fragmented":
        warnings.append(
            "Spine-fragmented parsing is planned but not implemented yet."
        )

        return MarkdownParseExecutionResult(
            inspection=inspection,
            strategy=strategy,
            documents=[],
            warnings=warnings,
            fragments=[
                {
                    "node_id": node.node_id,
                    "path": node.path,
                    "byte_start": node.byte_start,
                    "byte_end": node.byte_end,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "estimated_tokens": node.estimated_tokens,
                }
                for node in inspection.spine_nodes
            ],
        )

    if strategy == "quarantine":
        warnings.append(
            "Document was not parsed because inspection marked it for quarantine."
        )

        return MarkdownParseExecutionResult(
            inspection=inspection,
            strategy=strategy,
            documents=[],
            warnings=warnings,
        )

    raise ValueError(f"Unsupported markdown parse strategy: {strategy!r}")