# rag/ingestion/data_transformation/parser/adapters/markdown/ast/lists.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ingestion.data_transformation.parser.markdown.ast.inline import render_inline
from ingestion.data_transformation.parser.markdown.ast.utils import (
    clean_single_line,
    clean_text,
    compact_dict,
    find_matching_close,
    merge_inline_meta,
    normalize_language,
)


def _list_close_type(open_type: str) -> str:
    if open_type == "ordered_list_open":
        return "ordered_list_close"

    return "bullet_list_close"


def _format_nested_item(text: str, ordered: bool, index: int) -> str:
    """
    Format nested list item text when flattening it into parent item text.
    """

    value = clean_text(text)

    if not value:
        return ""

    marker = f"{index + 1}." if ordered else "-"
    return f"{marker} {value}"


def _format_nested_list(list_element: Dict[str, Any]) -> str:
    """
    Convert a nested list into readable text.

    Keeps nested list information visible for retrieval while still preserving
    the structured version in raw_items.children.
    """

    if not list_element:
        return ""

    ordered = bool(list_element.get("ordered", False))
    items = list_element.get("items") or []

    lines: List[str] = []

    for index, item in enumerate(items):
        line = _format_nested_item(str(item), ordered=ordered, index=index)

        if line:
            lines.append(line)

    return clean_text("\n".join(lines))


def _parse_code_like_token(token: Any) -> Optional[Dict[str, Any]]:
    """
    Parse code blocks inside list items.
    """

    token_type = getattr(token, "type", "")

    if token_type not in {"fence", "code_block"}:
        return None

    code = getattr(token, "content", "") or ""
    code = code.rstrip("\n")

    if not code.strip():
        return None

    language = ""

    if token_type == "fence":
        language = normalize_language(getattr(token, "info", "") or "")

    return compact_dict({
        "language": language,
        "text": code,
    })


def _append_code_to_parts(parts: List[str], code_block: Dict[str, Any]) -> None:
    """
    Add a readable fenced representation of a code block into item text.
    """

    code = code_block.get("text", "")

    if not code:
        return

    language = clean_single_line(code_block.get("language", ""))

    if language:
        parts.append(f"```{language}\n{code}\n```")
    else:
        parts.append(f"```\n{code}\n```")


def parse_list_item(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse list_item_open ... list_item_close.

    Keeps:
    - visible item text
    - inline links/images/code/html metadata
    - nested lists
    - code blocks inside list items

    Output example:
        {
            "text": "Install dependencies\n```bash\nnpm install\n```",
            "inline_code": ["npm install"],
            "children": [...],
            "code_blocks": [...]
        }
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    close_index = find_matching_close(tokens, start_index, "list_item_close")

    item: Dict[str, Any] = {
        "text": "",
        "links": [],
        "images": [],
        "inline_code": [],
        "html_inline": [],
    }

    parts: List[str] = []
    children: List[Dict[str, Any]] = []
    code_blocks: List[Dict[str, Any]] = []

    index = start_index + 1

    while index < close_index:
        token = tokens[index]
        token_type = getattr(token, "type", "")

        # ---------------- INLINE CONTENT ---------------- #

        if token_type == "inline":
            inline_result = render_inline(token)
            inline_text = clean_text(inline_result.get("text", ""))

            if inline_text:
                parts.append(inline_text)

            merge_inline_meta(item, inline_result)

            index += 1
            continue

        # ---------------- NESTED LISTS ---------------- #

        if token_type in {"bullet_list_open", "ordered_list_open"}:
            nested_list, next_index = parse_list(tokens, index)

            if nested_list:
                children.append(nested_list)

                nested_text = _format_nested_list(nested_list)

                if nested_text:
                    parts.append(nested_text)

            index = next_index
            continue

        # ---------------- CODE INSIDE LIST ITEM ---------------- #

        if token_type in {"fence", "code_block"}:
            code_block = _parse_code_like_token(token)

            if code_block:
                code_blocks.append(code_block)
                _append_code_to_parts(parts, code_block)

            index += 1
            continue

        # ---------------- HTML BLOCK INSIDE LIST ITEM ---------------- #
        # Keep raw HTML as visible text/metadata. For MD/MDX docs this is
        # usually documentation markup, not arbitrary page layout.

        if token_type == "html_block":
            html = clean_text(getattr(token, "content", "") or "")

            if html:
                item.setdefault("html_block", []).append(html)
                parts.append(html)

            index += 1
            continue

        index += 1

    text = clean_text("\n".join(part for part in parts if clean_text(part)))

    if not text:
        return None, close_index + 1

    item["text"] = text

    if children:
        item["children"] = children

    if code_blocks:
        item["code_blocks"] = code_blocks

    return compact_dict(item), close_index + 1


def parse_list(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse bullet_list_open / ordered_list_open into a list AST element.

    Backward-compatible output:
        {
            "type": "list",
            "items": ["item one", "item two"],
            "ordered": false,
            "raw_items": [...]
        }

    raw_items preserves richer structure for future enrichment.
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    open_token = tokens[start_index]
    open_type = getattr(open_token, "type", "")

    if open_type not in {"bullet_list_open", "ordered_list_open"}:
        return None, start_index + 1

    ordered = open_type == "ordered_list_open"
    close_type = _list_close_type(open_type)
    close_index = find_matching_close(tokens, start_index, close_type)

    items: List[str] = []
    raw_items: List[Dict[str, Any]] = []

    index = start_index + 1

    while index < close_index:
        token = tokens[index]
        token_type = getattr(token, "type", "")

        if token_type == "list_item_open":
            parsed_item, next_index = parse_list_item(tokens, index)

            if parsed_item:
                text = clean_text(parsed_item.get("text", ""))

                if text:
                    items.append(text)
                    raw_items.append(parsed_item)

            index = next_index
            continue

        index += 1

    if not items:
        return None, close_index + 1

    return compact_dict({
        "type": "list",
        "items": items,
        "ordered": ordered,
        "raw_items": raw_items,
    }), close_index + 1