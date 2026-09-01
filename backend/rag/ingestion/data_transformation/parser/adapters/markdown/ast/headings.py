# rag/ingestion/data_transformation/parser/adapters/markdown/ast/headings.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.inline import render_inline
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.mdx_markers import parse_mdx_marker
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import (
    clean_single_line,
    compact_dict,
    find_matching_close,
    merge_inline_meta,
)


# Common heading permalink markers emitted by scraped documentation sites.
# Loaded from parser/config/heuristics.toml.
PERMALINK_TEXTS = load_parser_heuristics_config().markdown.heading_permalink_texts


def _heading_level_from_tag(tag: Any) -> int:
    """
    Convert markdown-it heading tag into numeric level.

    Example:
        h1 -> 1
        h2 -> 2
    """

    value = str(tag or "").lower().strip()

    if len(value) == 2 and value.startswith("h") and value[1].isdigit():
        level = int(value[1])

        if 1 <= level <= 6:
            return level

    return 1


def _is_permalink_link(link: Dict[str, Any]) -> bool:
    """
    Detect heading permalink anchors.

    Common scraped docs patterns:
        ### Check it[¶](https://example.com#check-it)
        ## Usage[#](https://example.com#usage)
        ## API[](https://example.com#api)

    These links are navigation anchors, not meaningful heading text.
    """

    text = clean_single_line(link.get("text", ""))
    href = clean_single_line(link.get("href", ""))

    if not href:
        return False

    if text.lower() in PERMALINK_TEXTS:
        return True

    if href.startswith("#") and text.lower() in PERMALINK_TEXTS:
        return True

    return False


def _extract_heading_anchor(links: List[Dict[str, Any]]) -> str:
    """
    Extract permalink anchor URL from heading links.
    """

    for link in links or []:
        if _is_permalink_link(link):
            return clean_single_line(link.get("href", ""))

    return ""


def _remove_permalink_links(
    links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove heading permalink links from normal link metadata.

    The anchor is stored separately as heading_anchor.
    """

    cleaned: List[Dict[str, Any]] = []

    for link in links or []:
        if _is_permalink_link(link):
            continue

        cleaned.append(link)

    return cleaned


def _strip_permalink_suffix(text: str) -> str:
    """
    Remove permalink marker text that became visible heading text.

    Examples:
        "Check it¶" -> "Check it"
        "Usage #" -> "Usage"
        "OpenAPI 🔗" -> "OpenAPI"
    """

    value = clean_single_line(text)

    if not value:
        return ""

    # Remove repeated trailing permalink symbols.
    value = re.sub(r"\s*[¶#🔗§∞]+\s*$", "", value).strip()

    # Remove trailing textual marker in case a scraper emitted "permalink".
    value = re.sub(
        r"\s+(?:link|permalink)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    return value


def parse_heading(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse heading_open ... heading_close into a heading AST element.

    Preserves meaningful inline metadata while removing common heading
    permalink artifacts like ¶ / # / 🔗.
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    open_token = tokens[start_index]
    close_index = find_matching_close(tokens, start_index, "heading_close")

    level = _heading_level_from_tag(getattr(open_token, "tag", ""))

    inline_result: Dict[str, Any] = {
        "text": "",
        "links": [],
        "images": [],
        "inline_code": [],
        "html_inline": [],
    }

    for index in range(start_index + 1, close_index):
        token = tokens[index]

        if getattr(token, "type", "") == "inline":
            inline_result = render_inline(token)
            break

    raw_text = clean_single_line(inline_result.get("text", ""))
    text = _strip_permalink_suffix(raw_text)

    if not text:
        return None, close_index + 1

    # Rare case: normalized MDX marker appears in heading position.
    
    marker = parse_mdx_marker(text)

    if marker:
        return marker, close_index + 1

    links = inline_result.get("links") or []
    heading_anchor = _extract_heading_anchor(links)
    cleaned_links = _remove_permalink_links(links)

    heading: Dict[str, Any] = {
        "type": "heading",
        "level": level,
        "text": text,
        "heading_anchor": heading_anchor,
    }

    inline_result = dict(inline_result)
    inline_result["text"] = text
    inline_result["links"] = cleaned_links

    merge_inline_meta(heading, inline_result)

    return compact_dict(heading), close_index + 1