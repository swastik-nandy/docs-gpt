# rag/ingestion/data_transformation/parser/adapters/markdown/ast/paragraphs.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rag.ingestion.data_transformation.parser.adapters.markdown.ast.inline import render_inline
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.mdx_markers import parse_mdx_marker
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import (
    clean_text,
    compact_dict,
    find_matching_close,
    merge_inline_meta,
)


def parse_paragraph(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse paragraph_open ... paragraph_close into a paragraph AST element.

    Handles:
    - plain paragraph text
    - links
    - images
    - inline code
    - inline HTML
    - normalized MDX marker lines

    Output example:
        {
            "type": "paragraph",
            "text": "Read the FastAPI docs and run `uvicorn`.",
            "links": [...],
            "inline_code": ["uvicorn"]
        }
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    close_index = find_matching_close(tokens, start_index, "paragraph_close")

    paragraph: Dict[str, Any] = {
        "type": "paragraph",
        "text": "",
    }

    text_parts: List[str] = []

    for index in range(start_index + 1, close_index):
        token = tokens[index]

        if getattr(token, "type", "") != "inline":
            continue

        inline_result = render_inline(token)
        inline_text = inline_result.get("text", "")

        if inline_text:
            text_parts.append(inline_text)

        merge_inline_meta(paragraph, inline_result)

    text = clean_text(" ".join(text_parts))

    if not text:
        return None, close_index + 1

    # Normalized MDX markers are emitted as plain paragraph text by markdown-it.
    
    marker = parse_mdx_marker(text)

    if marker:
        return marker, close_index + 1

    paragraph["text"] = text

    return compact_dict(paragraph), close_index + 1