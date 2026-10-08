from __future__ import annotations

from typing import Any


OVERVIEW_HEADING = "Overview"


def new_section(
    heading: str,
    path: list[str],
    *,
    synthetic: bool = False,
    synthetic_reason: str = "",
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "heading": heading or "",
        "path": path or [],
        "content": [],
        "blocks": [],
    }

    if synthetic:
        section["synthetic"] = True
        section[
            "synthetic_reason"
        ] = synthetic_reason

    return section


def new_overview_section() -> dict[str, Any]:
    return new_section(
        heading=OVERVIEW_HEADING,
        path=[],
        synthetic=True,
        synthetic_reason=(
            "content_before_first_heading"
        ),
    )


def add_content_block(
    section: dict[str, Any],
    block: dict[str, Any] | None,
) -> None:
    if not block:
        return

    section.setdefault(
        "content",
        [],
    ).append(
        block
    )
