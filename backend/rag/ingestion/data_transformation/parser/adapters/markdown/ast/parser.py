# rag/ingestion/data_transformation/parser/adapters/markdown/ast/parser.py

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Sequence

from markdown_it import MarkdownIt

from rag.ingestion.data_transformation.parser.adapters.markdown.ast.blockquotes import (
    parse_blockquote,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.code import (
    parse_code_block,
    parse_fence,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.headings import (
    parse_heading,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.inline import (
    render_inline,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.lists import (
    parse_list,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.mdx_markers import (
    parse_mdx_marker,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.paragraphs import (
    parse_paragraph,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.provenance import (
    attach_source_provenance,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.tables import (
    parse_table,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import (
    clean_text,
    compact_dict,
    merge_inline_meta,
)


md = MarkdownIt("gfm-like", {"linkify": False})


def _parse_html_block(token: Any) -> Dict[str, Any] | None:
    """
    Preserve raw HTML blocks as paragraph-like AST elements.

    The Markdown AST does not attempt to become a general HTML parser.
    Raw HTML is retained so later structural passes can classify or transform
    it without losing source information.
    """

    raw_html = getattr(token, "content", "") or ""
    text = clean_text(raw_html)

    if not text:
        return None

    marker = parse_mdx_marker(text)

    if marker:
        return marker

    return compact_dict(
        {
            "type": "paragraph",
            "text": text,
            "html_block": [raw_html.rstrip()],
        }
    )


def _parse_loose_inline(token: Any) -> Dict[str, Any] | None:
    """
    Handle rare inline tokens emitted outside a normal paragraph wrapper.

    Most inline tokens are consumed by paragraph, heading and table parsers.
    This fallback prevents meaningful text from being silently discarded when
    Markdown-it plugins emit a loose inline token.
    """

    inline_result = render_inline(token)
    text = clean_text(inline_result.get("text", ""))

    if not text:
        return None

    marker = parse_mdx_marker(text)

    if marker:
        return marker

    paragraph: Dict[str, Any] = {
        "type": "paragraph",
        "text": text,
    }

    merge_inline_meta(paragraph, inline_result)

    return compact_dict(paragraph)


def _append_element(
    elements: List[Dict[str, Any]],
    element: MutableMapping[str, Any] | None,
    *,
    source_token: Any,
    source_lines: Sequence[str],
) -> None:
    """
    Append one AST element and attach deterministic source provenance.

    `block_index` represents the emitted AST order, not the Markdown-it token
    index. This remains stable even when one structural parser consumes several
    tokens for a single logical element.
    """

    if not element:
        return

    attach_source_provenance(
        element,
        token=source_token,
        source_lines=source_lines,
        block_index=len(elements),
    )

    elements.append(dict(element))


def parse_md_ast(text: str) -> List[Dict[str, Any]]:
    """
    Parse Markdown or normalized MDX into a structural AST.

    Supported output atoms:

    - heading
    - paragraph
    - blockquote
    - list
    - table
    - code
    - component_open
    - component_close
    - component_self

    Design boundaries:

    - This layer preserves source structure.
    - It does not decide final retrieval chunk boundaries.
    - It does not contain product-specific or document-specific rules.
    - Every emitted top-level atom retains its source provenance.
    """

    if not text:
        return []

    tokens = md.parse(text)

    # Keep line endings so source fragments can be reconstructed exactly from
    # Markdown-it's zero-based, end-exclusive token maps.
    source_lines = text.splitlines(keepends=True)

    elements: List[Dict[str, Any]] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        token_type = getattr(token, "type", "")

        # ---------------- HEADINGS ---------------- #

        if token_type == "heading_open":
            element, next_index = parse_heading(tokens, index)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index = next_index
            continue

        # ---------------- PARAGRAPHS ---------------- #

        if token_type == "paragraph_open":
            element, next_index = parse_paragraph(tokens, index)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index = next_index
            continue

        # ---------------- BLOCKQUOTES / CALLOUTS ---------------- #

        if token_type == "blockquote_open":
            element, next_index = parse_blockquote(tokens, index)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index = next_index
            continue

        # ---------------- LISTS ---------------- #

        if token_type in {
            "bullet_list_open",
            "ordered_list_open",
        }:
            element, next_index = parse_list(tokens, index)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index = next_index
            continue

        # ---------------- TABLES ---------------- #

        if token_type == "table_open":
            element, next_index = parse_table(tokens, index)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index = next_index
            continue

        # ---------------- FENCED CODE ---------------- #

        if token_type == "fence":
            element = parse_fence(token)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index += 1
            continue

        # ---------------- INDENTED CODE ---------------- #

        if token_type == "code_block":
            element = parse_code_block(token)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index += 1
            continue

        # ---------------- HTML BLOCK ---------------- #

        if token_type == "html_block":
            element = _parse_html_block(token)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index += 1
            continue

        # ---------------- LOOSE INLINE FALLBACK ---------------- #

        if token_type == "inline":
            element = _parse_loose_inline(token)

            _append_element(
                elements,
                element,
                source_token=token,
                source_lines=source_lines,
            )

            index += 1
            continue

        index += 1

    return elements