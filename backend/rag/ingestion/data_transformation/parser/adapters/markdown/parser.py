from __future__ import annotations

import os
from pathlib import Path

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config

from rag.ingestion.data_transformation.parser.adapters.markdown.adapters import normalize_mdx
from rag.ingestion.data_transformation.parser.adapters.markdown.ast import parse_md_ast
from rag.ingestion.data_transformation.parser.adapters.markdown.preprocessing import clean_markdown
from rag.ingestion.data_transformation.parser.adapters.markdown.document import Document
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.enricher import enrich
from rag.ingestion.data_transformation.parser.adapters.markdown.io import load_markdown


def _log(stage: str, msg: str) -> None:
    print(f"[MarkdownParser::{stage}] {msg}")


def _read_extension(file_path: str) -> str:
    if not file_path:
        return ""

    return Path(file_path).suffix.lower()


def _detect_markdown_variant(file_path: str) -> str:
    """
    Detect markdown-family variant from TOML config.

    Important:
    - inspection/router decides this file belongs to the markdown parser family.
    - this function only decides whether markdown-specific handling is needed.
    """

    extension = _read_extension(file_path)
    config = load_parser_heuristics_config()

    return config.file_types.markdown_variants.get(extension, "unknown")


def parse_markdown(file_path: str):
    """
    End-to-end Markdown-family parsing pipeline.

    Supports variants configured in parser/config/heuristics.toml.

    Flow:
        file
        -> detect markdown variant
        -> load
        -> optional MDX normalization
        -> clean
        -> AST
        -> enrich
        -> legacy Document wrapper
    """

    _log("START", file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    doc_type = _detect_markdown_variant(file_path)
    _log("VARIANT", doc_type)

    if doc_type == "unknown":
        raise ValueError(f"Unsupported markdown file type: {file_path}")

    raw_text = load_markdown(file_path)
    _log("LOAD", f"chars={len(raw_text)}")

    if not raw_text.strip():
        raise ValueError("Empty file content")

    if doc_type == "mdx":
        raw_text = normalize_mdx(raw_text)
        _log("MDX_NORMALIZE", f"chars={len(raw_text)}")

    clean_text = clean_markdown(raw_text)
    _log("CLEAN", f"chars={len(clean_text)}")

    if not clean_text.strip():
        raise ValueError("Empty content after cleaning")

    elements = parse_md_ast(clean_text)
    _log("AST", f"elements={len(elements)}")

    if not elements:
        raise ValueError("AST parsing failed")

    sections = enrich(elements)
    _log("ENRICH", f"sections={len(sections)}")

    if not sections:
        raise ValueError("Enrichment failed")

    final_text = "\n\n".join(
        section.get("text", "")
        for section in sections
        if section.get("text")
    ).strip()

    document = Document(
        text=final_text,
        metadata={
            "source": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_type": "markdown",
            "doc_type": doc_type,
            "ast": sections,
        },
    )

    _log("DONE", file_path)

    return [document]