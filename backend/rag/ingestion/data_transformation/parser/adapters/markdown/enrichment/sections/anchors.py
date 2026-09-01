from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


# --------------- PATTERNS --------------------------------

ANCHOR_ONLY_RE = re.compile(
    r"""
    ^\s*
    <a
      \s+
      (?:
        name|id
      )
      \s*=\s*
      ["'](?P<anchor>[^"']+)["']
      \s*
    >
    \s*
    </a>
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


# --------------- HELPERS --------------------------------

def clean_anchor(value: Any) -> str:
    return str(value or "").strip()


def extract_anchor_from_text(text: Any) -> str:
    value = str(text or "").strip()
    match = ANCHOR_ONLY_RE.match(value)

    if not match:
        return ""

    return clean_anchor(match.group("anchor"))


def is_anchor_only_block(block: dict[str, Any]) -> bool:
    if not block:
        return False

    if block.get("type") != "paragraph":
        return False

    return bool(extract_anchor_from_text(block.get("text", "")))


# --------------- SECTION PATCH --------------------------------

def attach_anchors_and_strip_anchor_blocks(
    section: dict[str, Any],
) -> dict[str, Any]:
    """
    Move standalone HTML anchors into section metadata.

    Input content block:
        {"type": "paragraph", "text": "<a name=\\"ecs-anywhere\\"></a>"}

    Output:
        section["anchors"] = ["ecs-anywhere"]
        section["heading_anchor"] = "ecs-anywhere"

    The anchor block is removed from retrieval content.
    """

    if not section:
        return section

    content = section.get("content", []) or []

    if not content:
        return section

    clean_content: list[dict[str, Any]] = []
    anchors: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue

        anchor = extract_anchor_from_text(block.get("text", ""))

        if anchor and is_anchor_only_block(block):
            anchors.append(anchor)
            continue

        clean_content.append(block)

    if not anchors:
        return section

    patched = deepcopy(section)
    existing_anchors = list(patched.get("anchors", []) or [])

    for anchor in anchors:
        if anchor not in existing_anchors:
            existing_anchors.append(anchor)

    patched["anchors"] = existing_anchors

    if not patched.get("heading_anchor") and existing_anchors:
        patched["heading_anchor"] = existing_anchors[0]

    patched["content"] = clean_content

    return patched