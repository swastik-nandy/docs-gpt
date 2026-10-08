from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------- PATTERNS --------------------------------

LEADING_KEY_VALUE_COMMENT_RE = re.compile(
    r"""
    \A
    \s*
    <!--
    \s*
    (?P<key>[A-Za-z0-9_.:-]+)
    \s*
    :
    \s*
    (?P<value>.*?)
    \s*
    -->
    \s*
    """,
    re.DOTALL | re.VERBOSE,
)

SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9_]+")


# --------------- SCHEMA --------------------------------

@dataclass(frozen=True)
class DocumentProlog:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_comments: list[str] = field(default_factory=list)


# --------------- NORMALIZATION --------------------------------

def normalize_metadata_key(key: Any) -> str:
    value = str(key or "").strip().lower()

    if not value:
        return ""

    value = value.replace("-", "_")
    value = value.replace(".", "_")
    value = SAFE_KEY_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    return value


def normalize_metadata_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


# --------------- EXTRACTION --------------------------------

def extract_document_prolog(text: str) -> DocumentProlog:
    """
    Extract generic leading key-value HTML comments.

    This is domain-agnostic.

    Accepted:
        <!-- source: docs -->
        <!-- topic: Install guide -->
        <!-- canonical_url: https://example.com/page -->

    Rejected as content:
        comments that are not at the top
        comments without key:value shape
        normal HTML blocks
    """

    if not text:
        return DocumentProlog(text="")

    remaining = text
    metadata: dict[str, Any] = {}
    raw_comments: list[str] = []

    while True:
        match = LEADING_KEY_VALUE_COMMENT_RE.match(remaining)

        if not match:
            break

        raw_comment = match.group(0).strip()
        key = normalize_metadata_key(match.group("key"))
        value = normalize_metadata_value(match.group("value"))

        if key and value:
            metadata[key] = value
            raw_comments.append(raw_comment)

        remaining = remaining[match.end():]

    return DocumentProlog(
        text=remaining.lstrip("\n"),
        metadata=metadata,
        raw_comments=raw_comments,
    )


def extract_document_prolog_from_file(
    file_path: str,
    *,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    """
    Extract leading metadata without reading huge files fully.
    """

    path = Path(file_path)

    if not path.exists():
        return {}

    with path.open("rb") as file:
        data = file.read(max_bytes)

    text = data.decode("utf-8", errors="replace")
    return extract_document_prolog(text).metadata