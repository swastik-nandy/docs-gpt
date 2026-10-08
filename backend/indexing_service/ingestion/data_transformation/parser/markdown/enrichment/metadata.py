# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/metadata.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from ingestion.data_transformation.parser.markdown.config import get_markdown_config


_METADATA_CONFIG = get_markdown_config().metadata

STOPWORDS = _METADATA_CONFIG.keyword_stopwords
PLACEHOLDER_HEADINGS = _METADATA_CONFIG.placeholder_headings


# -------------------------------------- TEXT UTILS -------------------------------------------

def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _normalize_key(text: Any) -> str:
    return _normalize(text).lower()


def _tokenize_words(text: Any) -> List[str]:
    return re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9._-]{3,}\b",
        str(text or "").lower(),
    )


def extract_keywords(text: Any, limit: int = 12) -> List[str]:
    """
    Lightweight keyword extraction.

    Purely lexical:
    - no domain assumptions
    - no target-word dependency
    """

    words = _tokenize_words(text)
    words = [word for word in words if word not in STOPWORDS]

    seen: Set[str] = set()
    result: List[str] = []

    for word in words:
        if word in seen:
            continue

        seen.add(word)
        result.append(word)

        if len(result) >= limit:
            break

    return result


# -------------------------------------- PATH / CONTEXT -------------------------------------------

def _is_placeholder_heading(text: Any) -> bool:
    return _normalize_key(text) in PLACEHOLDER_HEADINGS


def _safe_heading_path(path: List[str]) -> List[str]:
    """
    Remove empty/root placeholders from heading path.
    """

    clean: List[str] = []

    for part in path or []:
        value = _normalize(part)

        if not value:
            continue

        if _is_placeholder_heading(value):
            continue

        clean.append(value)

    return clean


def _build_heading_path(path: List[str], heading: str) -> List[str]:
    """
    Build full heading path.

    section["path"] should represent parents.
    heading_path should represent parents + current heading.
    """

    clean_path = _safe_heading_path(path)
    clean_heading = _normalize(heading)

    if clean_heading and not _is_placeholder_heading(clean_heading):
        return clean_path + [clean_heading]

    return clean_path


def build_context(path: List[str], heading: str) -> str:
    """
    Context represents parent hierarchy.

    If parent path is empty, use current heading as a useful fallback.
    """

    clean_path = _safe_heading_path(path)

    if clean_path:
        return " > ".join(clean_path)

    return _normalize(heading) or "root"


def extract_local_context(path: List[str], heading: str) -> Optional[str]:
    """
    Return the most local meaningful label.

    Usually the current heading.
    Falls back to nearest parent heading.
    """

    clean_heading = _normalize(heading)

    if clean_heading and not _is_placeholder_heading(clean_heading):
        return clean_heading

    for part in reversed(_safe_heading_path(path)):
        if part:
            return part

    return None


# -------------------------------------- STRUCTURAL SIGNALS -------------------------------------------

def _content_types(content: List[Dict[str, Any]]) -> Set[str]:
    return {
        item.get("type", "")
        for item in content or []
        if item
    }


def _count_type(content: List[Dict[str, Any]], *types: str) -> int:
    wanted = set(types)
    return sum(
        1
        for item in content or []
        if item and item.get("type") in wanted
    )


def _word_count(text: Any) -> int:
    return len(str(text or "").split())


def build_structure_profile(
    content: List[Dict[str, Any]],
    text: str,
) -> Dict[str, Any]:
    """
    Generic structural profile of a section.
    """

    block_types = _content_types(content)

    return {
        "block_count": len(content or []),
        "block_types": sorted(block_types),
        "paragraph_count": _count_type(content, "paragraph"),
        "blockquote_count": _count_type(content, "blockquote"),
        "table_count": _count_type(content, "table", "generic_table"),
        "list_count": _count_type(content, "list", "bullets", "procedure"),
        "code_count": _count_type(content, "code"),
        "word_count": _word_count(text),
        "is_mixed": len(block_types) > 1,
    }


# -------------------------------------- COMMAND-LIKE CONTEXT -------------------------------------------

_CODEISH_RE = re.compile(r"`([^`]+)`")
_OPTION_TOKEN_RE = re.compile(r"(--[a-zA-Z0-9][a-zA-Z0-9-]*|-[a-zA-Z])")


def _is_command_like(text: Any) -> bool:
    """
    Heuristic for short command-like anchors from headings.

    This remains structural and format-based.
    """

    value = _normalize(text)

    if not value:
        return False

    if len(value.split()) > 6:
        return False

    # Do not treat pure option flags as command context.
    if _OPTION_TOKEN_RE.search(value):
        return False

    if any(symbol in value for symbol in ["|", ">", "<", "=", "{", "}"]):
        return False

    return True


def _extract_command_context(path: List[str], heading: str) -> Optional[str]:
    """
    Extract command-like anchor from heading/path without domain assumptions.
    """

    candidates = [heading] + list(reversed(path or []))

    for candidate in candidates:
        for code in _CODEISH_RE.findall(candidate or ""):
            if _is_command_like(code):
                return _normalize(code)

        if _is_command_like(candidate):
            return _normalize(candidate)

    return None


# -------------------------------------- SECTION TYPE -------------------------------------------

def detect_section_type(
    heading: str,
    path: List[str],
    content: List[Dict[str, Any]],
) -> str:
    """
    Structure-based section classification.
    """

    types = _content_types(content)

    if not types:
        return "empty"

    if "procedure" in types:
        return "procedure"

    if "table" in types or "generic_table" in types:
        return "structured"

    if "code" in types and len(types) == 1:
        return "code"

    if "blockquote" in types and len(types) == 1:
        return "callout"

    if "component" in types and len(types) == 1:
        return "component"

    if "list" in types or "bullets" in types:
        if len(types) == 1:
            return "list"

    if len(types) > 1:
        return "mixed"

    return "text"


def detect_content_role(content: List[Dict[str, Any]]) -> str:
    """
    Broad dominant content role.
    """

    types = _content_types(content)

    if "generic_table" in types or "table" in types:
        return "table"

    if "procedure" in types:
        return "procedure"

    if "code" in types:
        return "code"

    if "blockquote" in types:
        return "callout"

    if "component" in types:
        return "component"

    if "bullets" in types or "list" in types:
        return "list"

    return "text"


# -------------------------------------- MAIN -------------------------------------------

def enrich_metadata(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach structural metadata to a section.

    No domain-specific inference.
    """

    text = section.get("text", "")
    path = section.get("path", []) or []
    heading = section.get("heading", "")
    content = section.get("content", []) or []

    clean_path = _safe_heading_path(path)

    section_type = detect_section_type(
        heading=heading,
        path=clean_path,
        content=content,
    )

    section["context"] = build_context(clean_path, heading)
    section["section_type"] = section_type
    section["type"] = section_type
    section["content_role"] = detect_content_role(content)
    section["local_context"] = extract_local_context(clean_path, heading)
    section["command_context"] = _extract_command_context(clean_path, heading)
    section["parent_path"] = clean_path
    section["heading_path"] = _build_heading_path(clean_path, heading)
    section["keywords"] = extract_keywords(text)
    section["structure_profile"] = build_structure_profile(content, text)

    anchors = list(section.get("anchors", []) or [])

    if anchors:
        section["anchors"] = anchors

        if not section.get("heading_anchor"):
            section["heading_anchor"] = anchors[0]

    return section