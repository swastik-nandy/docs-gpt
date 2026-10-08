from __future__ import annotations

import re
from copy import deepcopy
from typing import Any
from typing import Mapping


#--------------------IDENTITY--------------------

def slugify(
    text: str,
    fallback: str = "section",
) -> str:
    value = (
        text
        or ""
    ).strip().lower()

    value = re.sub(
        r"`+",
        "",
        value,
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    ).strip(
        "_"
    )

    return (
        value
        or fallback
    )


def build_section_id(
    section: dict[str, Any],
    section_index: int,
) -> str:
    heading = section.get(
        "heading",
        "",
    )

    return (
        f"s{section_index}_"
        f"{slugify(heading)}"
    )


#--------------------METADATA--------------------

def base_meta(
    item_index: int,
    source_type: str,
    section: dict[str, Any],
    *,
    item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[
        str,
        Any
    ] = {
        "source_type": source_type,
        "source_index": item_index,
        "synthetic_section": section.get(
            "synthetic",
            False,
        ),
        "heading_anchor": section.get(
            "heading_anchor",
            "",
        ),
        "anchors": deepcopy(
            section.get(
                "anchors",
                [],
            )
        ),
        "heading_path": deepcopy(
            section.get(
                "heading_path",
                [],
            )
        ),
        "parent_path": deepcopy(
            section.get(
                "parent_path",
                [],
            )
        ),
    }

    if not isinstance(
        item,
        Mapping,
    ):
        return metadata

    for key in (
        "source",
        "local_scope_id",
        "local_scope_title",
        "structural_role",
        "selector_metadata",
        "semantic_packed",
        "semantic_packed_block_count",
        "semantic_source_indexes",
    ):
        value = item.get(
            key
        )

        if value is None:
            continue

        if isinstance(
            value,
            str,
        ) and not value.strip():
            continue

        if isinstance(
            value,
            (
                list,
                dict,
                tuple,
                set,
            ),
        ) and not value:
            continue

        metadata[
            key
        ] = deepcopy(
            value
        )

    return metadata


#--------------------LOCAL CONTEXT--------------------

def resolve_base_local_context(
    section: dict[str, Any],
) -> str:
    return str(
        section.get(
            "local_context"
        )
        or section.get(
            "command_context"
        )
        or section.get(
            "heading"
        )
        or ""
    ).strip()


#--------------------COLLECTION HELPERS--------------------

def append_block(
    blocks: list[dict[str, Any]],
    block: dict[str, Any] | None,
) -> None:
    if block:
        blocks.append(
            block
        )


def extend_blocks(
    blocks: list[dict[str, Any]],
    new_blocks: list[dict[str, Any]] | None,
) -> None:
    if new_blocks:
        blocks.extend(
            new_blocks
        )


#--------------------COMPACTION--------------------

def clean_meta_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        str,
    ):
        return value.strip()

    return value


def compact_meta(
    meta: dict[str, Any],
) -> dict[str, Any]:
    compacted: dict[
        str,
        Any
    ] = {}

    for key, value in (
        meta
        or {}
    ).items():
        value = clean_meta_value(
            value
        )

        if value is None:
            continue

        if (
            isinstance(
                value,
                str,
            )
            and not value
        ):
            continue

        if (
            isinstance(
                value,
                (
                    list,
                    dict,
                    tuple,
                    set,
                ),
            )
            and not value
        ):
            continue

        compacted[
            key
        ] = value

    return compacted
