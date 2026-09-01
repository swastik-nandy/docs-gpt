from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------- PATTERNS --------------------------------

LEADING_HTML_COMMENT_RE = re.compile(
    r"\A\s*<!--\s*(?P<key>[A-Za-z0-9_.:-]+)\s*:\s*(?P<value>.*?)\s*-->\s*",
    re.DOTALL,
)

SAFE_META_KEY_RE = re.compile(r"[^a-zA-Z0-9_]+")


# --------------- SCHEMA --------------------------------

@dataclass(frozen=True)
class DocumentMetadataExtraction:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_comments: list[str] = field(default_factory=list)


# --------------- HELPERS --------------------------------

def normalize_metadata_key(key: str) -> str:
    value = str(key or "").strip().lower()
    value = value.replace("-", "_")
    value = value.replace(".", "_")
    value = SAFE_META_KEY_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    return value


def normalize_metadata_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


# --------------- EXTRACTION --------------------------------

def extract_leading_document_metadata(text: str) -> DocumentMetadataExtraction:
    """
    Extract leading HTML comment metadata from Markdown.

    Example:
        <!-- source_family: aws-ecs -->
        <!-- topic: Amazon ECS Anywhere -->
        <!-- source_url: https://docs.aws.amazon.com/... -->

    These comments are document metadata, not retrieval content.
    """

    if not text:
        return DocumentMetadataExtraction(text="")

    remaining = text
    metadata: dict[str, Any] = {}
    raw_comments: list[str] = []

    while True:
        match = LEADING_HTML_COMMENT_RE.match(remaining)

        if not match:
            break

        raw_comments.append(match.group(0).strip())

        key = normalize_metadata_key(match.group("key"))
        value = normalize_metadata_value(match.group("value"))

        if key and value:
            metadata[key] = value

        remaining = remaining[match.end():]

    return DocumentMetadataExtraction(
        text=remaining.lstrip("\n"),
        metadata=metadata,
        raw_comments=raw_comments,
    )


def extract_leading_document_metadata_from_file(
    file_path: str,
    *,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    """
    Read only the beginning of a file and extract leading document metadata.

    This is safe for huge files because source metadata comments should appear
    at the top of the document.
    """

    path = Path(file_path)

    if not path.exists():
        return {}

    with path.open("rb") as file:
        data = file.read(max_bytes)

    text = data.decode("utf-8", errors="replace")
    extraction = extract_leading_document_metadata(text)

    return extraction.metadata