# rag/ingestion/data_transformation/parser/adapters/markdown/ast/tables.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ingestion.data_transformation.parser.markdown.ast.inline import render_inline
from ingestion.data_transformation.parser.markdown.ast.utils import (
    clean_single_line,
    compact_dict,
    find_matching_close,
    merge_inline_meta,
)


def _empty_cell() -> Dict[str, Any]:
    """
    Standard table cell structure.
    """

    return {
        "text": "",
        "links": [],
        "images": [],
        "inline_code": [],
        "html_inline": [],
    }


def _compact_cell(cell: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove empty metadata from a cell but keep text.
    """

    return compact_dict({
        "text": clean_single_line(cell.get("text", "")),
        "links": cell.get("links") or [],
        "images": cell.get("images") or [],
        "inline_code": cell.get("inline_code") or [],
        "html_inline": cell.get("html_inline") or [],
    })


def parse_table(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse markdown-it table tokens into a raw table AST element.

    Output keeps backward-compatible rows:

        {
            "type": "table",
            "rows": [
                ["Option", "Description"],
                ["--port", "Specify port"]
            ],
            "cell_meta": [
                [{"text": "Option"}, {"text": "Description"}],
                [{"text": "--port", "inline_code": ["--port"]}, ...]
            ]
        }

    The enrichment layer can still use rows exactly as before,
    while richer metadata remains available for better retrieval later.
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    close_index = find_matching_close(tokens, start_index, "table_close")

    rows: List[List[str]] = []
    cell_meta_rows: List[List[Dict[str, Any]]] = []

    current_row: Optional[List[str]] = None
    current_meta_row: Optional[List[Dict[str, Any]]] = None
    current_cell: Optional[Dict[str, Any]] = None

    index = start_index + 1

    while index < close_index:
        token = tokens[index]
        token_type = getattr(token, "type", "")

        # ---------------- ROW START ---------------- #

        if token_type == "tr_open":
            current_row = []
            current_meta_row = []
            index += 1
            continue

        # ---------------- CELL START ---------------- #

        if token_type in {"td_open", "th_open"}:
            current_cell = _empty_cell()
            index += 1
            continue

        # ---------------- CELL INLINE CONTENT ---------------- #

        if token_type == "inline" and current_cell is not None:
            inline_result = render_inline(token)
            text = clean_single_line(inline_result.get("text", ""))

            if text:
                current_cell["text"] = (
                    f"{current_cell.get('text', '')} {text}"
                ).strip()

            merge_inline_meta(current_cell, inline_result)

            index += 1
            continue

        # ---------------- CELL END ---------------- #

        if token_type in {"td_close", "th_close"}:
            if current_row is not None and current_meta_row is not None and current_cell is not None:
                compact_cell = _compact_cell(current_cell)
                current_row.append(compact_cell.get("text", ""))
                current_meta_row.append(compact_cell)

            current_cell = None
            index += 1
            continue

        # ---------------- ROW END ---------------- #

        if token_type == "tr_close":
            if current_row is not None:
                rows.append(current_row)

            if current_meta_row is not None:
                cell_meta_rows.append(current_meta_row)

            current_row = None
            current_meta_row = None

            index += 1
            continue

        index += 1

    if not rows:
        return None, close_index + 1

    return compact_dict({
        "type": "table",
        "rows": rows,
        "cell_meta": cell_meta_rows,
    }), close_index + 1