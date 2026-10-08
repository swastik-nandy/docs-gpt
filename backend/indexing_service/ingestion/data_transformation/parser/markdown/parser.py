from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ingestion.data_transformation.parser.markdown.ast import (
    parse_md_ast,
)
from ingestion.data_transformation.parser.markdown.enrichment.enricher import (
    enrich,
)
from ingestion.data_transformation.parser.markdown.io import (
    load_markdown,
)
from ingestion.data_transformation.parser.markdown.ir.block_builder import (
    refresh_retrieval_text,
)
from ingestion.data_transformation.parser.markdown.preprocessing import (
    clean_markdown,
    extract_document_prolog,
)
from ingestion.data_transformation.parser.markdown.schema import (
    MarkdownParsedUnit,
    MarkdownParseResult,
)


def _doc_id(
    text: str,
) -> str:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        f"doc_{digest[:16]}"
    )


def _collect_blocks(
    sections: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    return [
        block

        for section
        in sections

        for block
        in section.get(
            "blocks",
            [],
        )

        if isinstance(
            block,
            dict,
        )
    ]


def _attach_document_meta(
    sections: list[
        dict[str, Any]
    ],
    metadata: dict[
        str,
        Any,
    ],
) -> None:
    if not metadata:
        return

    for section in sections:
        section[
            "document_meta"
        ] = metadata

        for block in section.get(
            "blocks",
            [],
        ):
            if not isinstance(
                block,
                dict,
            ):
                continue

            meta = block.setdefault(
                "meta",
                {},
            )

            meta[
                "document_meta"
            ] = metadata

            for key, value in (
                metadata.items()
            ):
                meta.setdefault(
                    key,
                    value,
                )

            refresh_retrieval_text(
                block
            )


def parse_markdown(
    file_path: str | Path,
) -> MarkdownParseResult:
    path = (
        Path(file_path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Markdown source does "
            f"not exist: {path}"
        )

    raw_text = load_markdown(
        str(path)
    )

    if not raw_text.strip():
        raise ValueError(
            f"Markdown source is empty: {path}"
        )

    doc_id = _doc_id(
        raw_text
    )

    prolog = extract_document_prolog(
        raw_text
    )

    clean_text = clean_markdown(
        prolog.text
    )

    if not clean_text.strip():
        raise ValueError(
            "Markdown source is empty "
            f"after cleaning: {path}"
        )

    elements = parse_md_ast(
        clean_text
    )

    if not elements:
        raise ValueError(
            "Markdown AST produced "
            f"no elements: {path}"
        )

    sections = enrich(
        elements,
        doc_id=doc_id,
    )

    if not sections:
        raise ValueError(
            "Markdown enrichment produced "
            f"no sections: {path}"
        )

    _attach_document_meta(
        sections,
        prolog.metadata,
    )

    blocks = _collect_blocks(
        sections
    )

    if not blocks:
        raise ValueError(
            "Markdown enrichment produced "
            f"no IR blocks: {path}"
        )

    unit = MarkdownParsedUnit(
        unit_id="full",

        doc_id=doc_id,

        source_doc_id=doc_id,

        parse_strategy=
            "single_document",

        sections=sections,

        blocks=blocks,

        meta={
            "unit_kind":
                "full_document",

            "source_path":
                str(path),

            "parser_variant":
                "markdown",

            "parse_strategy":
                "single_document",

            "doc_id":
                doc_id,

            "source_doc_id":
                doc_id,

            "section_count":
                len(sections),

            "block_count":
                len(blocks),

            "document_meta":
                prolog.metadata,

            "raw_prolog_comments":
                prolog.raw_comments,
        },
    )

    return MarkdownParseResult(
        doc_id=doc_id,

        source_path=
            str(path),

        units=[
            unit
        ],

        metrics={
            "section_count":
                len(sections),

            "block_count":
                len(blocks),
        },
    )
