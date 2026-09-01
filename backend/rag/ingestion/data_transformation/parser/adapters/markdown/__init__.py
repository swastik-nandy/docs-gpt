from __future__ import annotations

from rag.ingestion.data_transformation.parser.adapters.markdown.document import Document
from rag.ingestion.data_transformation.parser.adapters.markdown.parser import parse_markdown


# Backward-compatible alias.
parse = parse_markdown


# Backward-compatible strategy-aware legacy parser.
# Keep this import after parse_markdown is already available because
# strategy.py imports parse_markdown from this package.
from rag.ingestion.data_transformation.parser.adapters.markdown.compat import (
    MarkdownParseExecutionResult,
    parse_markdown_with_strategy,
)


__all__ = [
    "Document",
    "MarkdownParseExecutionResult",
    "parse",
    "parse_markdown",
    "parse_markdown_with_strategy",
]