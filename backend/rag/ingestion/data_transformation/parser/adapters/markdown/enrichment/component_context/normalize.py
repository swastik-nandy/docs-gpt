# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/normalize.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def clean_text(value: Any) -> str:
    """
    Normalize small metadata strings.
    """

    if value is None:
        return ""

    return str(value).strip()


def clean_component_name(name: Any) -> str:
    """
    Normalize component name.
    """

    value = clean_text(name)
    return value or "Component"


def clean_props(props: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Normalize component props without deleting structured values.

    Keeps:
    - strings
    - numbers
    - booleans
    - simple structured values if upstream provides them
    """

    cleaned: Dict[str, Any] = {}

    for key, value in dict(props or {}).items():
        key_text = clean_text(key)

        if not key_text:
            continue

        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

            if not value:
                continue

        cleaned[key_text] = value

    return cleaned