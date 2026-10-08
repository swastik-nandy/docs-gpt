# rag/ingestion/data_transformation/parser/adapters/markdown/ast/blockquotes.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ingestion.data_transformation.parser.markdown.config import get_markdown_config
from ingestion.data_transformation.parser.markdown.ast.inline import render_inline
from ingestion.data_transformation.parser.markdown.ast.utils import (
    clean_single_line,
    clean_text,
    compact_dict,
    find_matching_close,
    merge_inline_meta,
)


# Generic documentation callout labels loaded from parser/config/heuristics.toml.
CALLOUT_KIND_MAP = get_markdown_config().callouts.kind_map


def _detect_callout(text: str) -> Dict[str, str]:
    """
    Detect common documentation callout labels inside blockquotes.

    Handles:
        > **Good to know**: ...
        > **Warning**: ...
        > Note: ...

    The parser stays generic. It does not know anything about Next.js,
    FastAPI, product docs, changelogs, or company-specific wording.

    Label vocabulary comes from parser/config/heuristics.toml.
    """

    value = clean_text(text)

    if not value:
        return {}

    # Remove common Markdown emphasis wrappers that can survive in edge cases.
    normalized = value
    normalized = re.sub(r"^\*\*(.+?)\*\*\s*:", r"\1:", normalized)
    normalized = re.sub(r"^__(.+?)__\s*:", r"\1:", normalized)

    if ":" not in normalized:
        return {}

    label, body = normalized.split(":", 1)

    label = clean_single_line(label)
    body = clean_text(body)

    if not label or not body:
        return {}

    kind = CALLOUT_KIND_MAP.get(label.lower())

    if not kind:
        return {}

    return compact_dict({
        "label": label,
        "kind": kind,
        "body": body,
    })


def _parse_nested_inline_content(
    tokens: List[Any],
    start_index: int,
    close_index: int,
) -> Dict[str, Any]:
    """
    Extract visible text and inline metadata from tokens inside a blockquote.
    """

    result: Dict[str, Any] = {
        "text": "",
        "links": [],
        "images": [],
        "inline_code": [],
        "html_inline": [],
    }

    parts: List[str] = []

    index = start_index + 1

    while index < close_index:
        token = tokens[index]
        token_type = getattr(token, "type", "")

        if token_type == "inline":
            inline_result = render_inline(token)
            inline_text = inline_result.get("text", "")

            if inline_text:
                parts.append(inline_text)

            merge_inline_meta(result, inline_result)

        elif token_type == "fence":
            code = getattr(token, "content", "") or ""
            code = code.rstrip("\n")

            if code.strip():
                parts.append(code)
                result.setdefault("code_blocks", []).append({
                    "language": clean_single_line(getattr(token, "info", "")),
                    "text": code,
                })

        elif token_type == "code_block":
            code = getattr(token, "content", "") or ""
            code = code.rstrip("\n")

            if code.strip():
                parts.append(code)
                result.setdefault("code_blocks", []).append({
                    "language": "",
                    "text": code,
                })

        index += 1

    result["text"] = clean_text("\n".join(parts))

    return result


def parse_blockquote(
    tokens: List[Any],
    start_index: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Parse blockquote_open ... blockquote_close into a blockquote AST element.

    Output example:
        {
            "type": "blockquote",
            "text": "With `npm run`, use `--` before CLI flags...",
            "label": "Good to know",
            "kind": "note",
            "inline_code": ["npm run", "--", "next"]
        }

    For non-callout quotes:
        {
            "type": "blockquote",
            "text": "Quoted text..."
        }
    """

    if start_index < 0 or start_index >= len(tokens):
        return None, start_index + 1

    close_index = find_matching_close(tokens, start_index, "blockquote_close")

    parsed = _parse_nested_inline_content(tokens, start_index, close_index)
    text = clean_text(parsed.get("text", ""))

    if not text:
        return None, close_index + 1

    quote: Dict[str, Any] = {
        "type": "blockquote",
        "text": text,
    }

    merge_inline_meta(quote, parsed)

    callout = _detect_callout(text)

    if callout:
        label = callout.get("label", "")
        body = callout.get("body", "")

        quote["label"] = label
        quote["kind"] = callout.get("kind", "note")

        if body:
            quote["text"] = body

    if parsed.get("code_blocks"):
        quote["code_blocks"] = parsed["code_blocks"]

    return compact_dict(quote), close_index + 1