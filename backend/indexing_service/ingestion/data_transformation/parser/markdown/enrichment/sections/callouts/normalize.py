from __future__ import annotations

from typing import Any

from ingestion.data_transformation.parser.markdown.enrichment.sections.callouts.config import (
    CALLOUT_KIND_MAP,
)


# --------------- NORMALIZATION --------------------------------

def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def label_key(text: Any) -> str:
    value = clean_text(text)
    value = value.strip(":").strip()

    return value.lower()


def display_label(label: Any) -> str:
    value = clean_text(label)

    return value.strip(":").strip()


def known_callout_kind(label: Any) -> str:
    return CALLOUT_KIND_MAP.get(label_key(label), "")


def is_known_callout_label(label: Any) -> bool:
    return bool(known_callout_kind(label))


def remove_label_prefix(text: str, label: str) -> str:
    value = clean_text(text)
    label_value = clean_text(label)

    if not value or not label_value:
        return value

    if not value.lower().startswith(label_value.lower()):
        return value

    return value[len(label_value):].strip(" :-–—\n\t")
