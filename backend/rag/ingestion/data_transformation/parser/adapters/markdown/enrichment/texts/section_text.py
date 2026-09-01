# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/texts/section_text.py

from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.texts.common import clean_list_items, clean_text


def _blockquote_to_text(item: Dict[str, Any]) -> str:
    """
    Convert blockquote/callout content into section-level text.
    """

    text = clean_text(item.get("text", ""))

    if not text:
        return ""

    label = clean_text(item.get("label", ""))
    kind = clean_text(item.get("kind", ""))

    prefix = label or kind

    if prefix and prefix.lower() not in {"quote", "blockquote"}:
        return f"{prefix}: {text}"

    return text


def _list_to_text(item: Dict[str, Any]) -> List[str]:
    """
    Convert list content into readable section text.
    """

    items = clean_list_items(item.get("items", []))

    if not items:
        return []

    if item.get("ordered"):
        return [
            f"{index + 1}. {list_item}"
            for index, list_item in enumerate(items)
        ]

    return [f"- {list_item}" for list_item in items]


def _table_rows_to_text(item: Dict[str, Any]) -> List[str]:
    """
    Convert table content into retrieval-friendly row text.
    """

    row_texts = item.get("row_text", []) or []

    if row_texts:
        return [
            text
            for text in (clean_text(row) for row in row_texts)
            if text
        ]

    rows = item.get("rows", []) or []
    lines: List[str] = []

    for row in rows:
        if isinstance(row, dict):
            line = " | ".join(str(value) for value in row.values())
        else:
            line = " | ".join(str(value) for value in row)

        line = clean_text(line)

        if line:
            lines.append(line)

    return lines


def build_section_text(content: List[Dict[str, Any]]) -> str:
    """
    Build section-level text from enriched content blocks.

    Used by:
    - metadata enrichment
    - keyword extraction
    - final document text

    This should preserve structure without inventing meaning.
    """

    parts: List[str] = []

    for item in content or []:
        if not item:
            continue

        item_type = item.get("type")

        if item_type == "paragraph":
            text = clean_text(item.get("text", ""))
            if text:
                parts.append(text)
            continue

        if item_type == "blockquote":
            text = _blockquote_to_text(item)
            if text:
                parts.append(text)
            continue

        if item_type == "list":
            parts.extend(_list_to_text(item))
            continue

        if item_type in {"table", "generic_table"}:
            parts.extend(_table_rows_to_text(item))
            continue

        if item_type == "code":
            code_text = str(item.get("text", "") or "").strip()
            if code_text:
                parts.append(code_text)
            continue

        if item_type == "component":
            text = clean_text(item.get("text", ""))
            if text:
                parts.append(text)
            continue

        fallback = clean_text(item.get("text", ""))
        if fallback:
            parts.append(fallback)

    return "\n\n".join(parts).strip()