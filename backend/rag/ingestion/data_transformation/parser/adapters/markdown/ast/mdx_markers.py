# rag/ingestion/data_transformation/parser/adapters/markdown/ast/mdx_markers.py

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config
from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import (
    clean_single_line,
    clean_text,
    compact_dict,
)


MARKER_PREFIX = load_parser_heuristics_config().markdown.mdx_markers.prefix

MARKER_RE = re.compile(
    rf"^{re.escape(MARKER_PREFIX)}(?P<kind>open|close|self)\s*(?P<attrs>.*)$",
    re.IGNORECASE,
)

# Supports attrs like:
# name="Tabs"
# label='Good to know'
# count="3"
# variant="warning"
ATTR_RE = re.compile(
    r"""
    (?P<key>[A-Za-z_:][A-Za-z0-9_:.-]*)
    \s*=\s*
    (?:
        " (?P<double> [^"]* ) "
        |
        ' (?P<single> [^']* ) '
    )
    """,
    re.VERBOSE,
)


def is_mdx_marker(text: Any) -> bool:
    """
    True if text looks like a normalized MDX component marker.

    Example:
        ::mdx-component-open name="Tabs"
        ::mdx-component-self name="Callout" type="warning"
        ::mdx-component-close name="Tabs"
    """

    value = clean_single_line(text)
    return value.startswith(MARKER_PREFIX)


def parse_marker_attrs(raw_attrs: Any) -> Dict[str, str]:
    """
    Parse normalized marker attributes.

    This intentionally supports only quoted string attrs because mdx_adapter.py
    should already have normalized unsafe/complex MDX props into strings.
    """

    raw = str(raw_attrs or "")
    attrs: Dict[str, str] = {}

    for match in ATTR_RE.finditer(raw):
        key = clean_single_line(match.group("key"))
        value = match.group("double")

        if value is None:
            value = match.group("single")

        key = clean_single_line(key)
        value = clean_text(value)

        if key:
            attrs[key] = value

    return attrs


def parse_mdx_marker(text: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a normalized MDX marker into an AST element.

    Supported marker kinds:
        open  -> component_open
        close -> component_close
        self  -> component_self
    """

    value = clean_single_line(text)

    if not value.startswith(MARKER_PREFIX):
        return None

    match = MARKER_RE.match(value)

    if not match:
        return None

    kind = clean_single_line(match.group("kind")).lower()
    attrs = parse_marker_attrs(match.group("attrs"))

    name = clean_single_line(attrs.pop("name", ""))

    if not name:
        name = "UnknownComponent"

    marker_type = {
        "open": "component_open",
        "close": "component_close",
        "self": "component_self",
    }.get(kind)

    if not marker_type:
        return None

    return compact_dict({
        "type": marker_type,
        "name": name,
        "props": attrs,
    })


def marker_to_text(marker: Dict[str, Any]) -> str:
    """
    Convert a parsed marker into readable fallback text.

    Useful for debugging or for enrichment layers that want component context.
    """

    if not marker:
        return ""

    marker_type = marker.get("type", "")
    name = marker.get("name", "")
    props = marker.get("props") or {}

    if not name:
        return ""

    if marker_type == "component_open":
        prefix = "MDX component open"
    elif marker_type == "component_close":
        prefix = "MDX component close"
    elif marker_type == "component_self":
        prefix = "MDX component"
    else:
        prefix = "MDX component"

    if not props:
        return f"{prefix}: {name}"

    prop_text = ", ".join(
        f"{key}={value}"
        for key, value in props.items()
        if clean_single_line(key) and clean_single_line(value)
    )

    if not prop_text:
        return f"{prefix}: {name}"

    return f"{prefix}: {name} ({prop_text})"


def is_component_open(element: Dict[str, Any]) -> bool:
    return bool(element and element.get("type") == "component_open")


def is_component_close(element: Dict[str, Any]) -> bool:
    return bool(element and element.get("type") == "component_close")


def is_component_self(element: Dict[str, Any]) -> bool:
    return bool(element and element.get("type") == "component_self")


def is_component_marker_element(element: Dict[str, Any]) -> bool:
    return bool(
        element
        and element.get("type")
        in {"component_open", "component_close", "component_self"}
    )