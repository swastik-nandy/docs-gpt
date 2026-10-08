from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import Chunk
from .utils import clean_text
from .utils import estimate_tokens


#--------------------TYPE MAPPING--------------------

IR_TYPE_TO_CHUNK_TYPE = {
    "text": "text",
    "code": "code",
    "table": "table",
    "list": "list",
    "component": "component",
}


#--------------------PUBLIC API--------------------

def chunk_from_ir_block(
    section: dict[str, Any],
    block: dict[str, Any],
    section_index: int,
    block_index: int,
    source: str = "",
) -> Chunk | None:
    if not block:
        return None

    text = clean_text(
        block.get(
            "text",
            "",
        )
    )

    if not text:
        return None

    chunk_type = _resolve_chunk_type(
        block
    )

    subtype = _resolve_subtype(
        block
    )

    metadata = _build_metadata(
        section=section,
        block=block,
        section_index=section_index,
        block_index=block_index,
        subtype=subtype,
    )

    return Chunk(
        text=text,
        chunk_type=chunk_type,
        heading=_resolve_heading(
            section,
            block,
        ),
        context=_resolve_context(
            section,
            block,
        ),
        subtype=subtype,
        order=_resolve_order(
            block,
            block_index,
        ),
        metadata=metadata,
        tokens=estimate_tokens(
            text
        ),
        source=source,
        chunk_id=_build_chunk_id(
            section=section,
            block=block,
            fallback_index=block_index,
        ),
    )


def chunks_from_ir_blocks(
    section: dict[str, Any],
    section_index: int,
    source: str = "",
) -> list[Chunk]:
    blocks = _safe_list(
        section.get(
            "blocks"
        )
    )

    if not blocks:
        return []

    chunks: list[Chunk] = []

    for block_index, block in enumerate(
        blocks
    ):
        if not isinstance(
            block,
            dict,
        ):
            continue

        chunk = chunk_from_ir_block(
            section=section,
            block=block,
            section_index=section_index,
            block_index=block_index,
            source=source,
        )

        if chunk is not None:
            chunks.append(
                chunk
            )

    return _attach_dependent_lead_ins(
        chunks
    )


#--------------------DEPENDENT LEAD-INS--------------------

_DEPENDENT_ENDINGS = (
    "as follows",
    "as follows.",
    "shown below",
    "shown below.",
    "looks like this",
    "looks like this.",
    "in the following example",
    "in the following example.",
    "in the following examples",
    "in the following examples.",
)

_STRUCTURAL_TARGET_SUBTYPES = {
    "code",
    "list",
    "table",
    "component",
}


def _attach_dependent_lead_ins(
    chunks: list[Chunk],
) -> list[Chunk]:
    if not chunks:
        return []

    output: list[Chunk] = []

    index = 0

    while index < len(
        chunks
    ):
        current = chunks[
            index
        ]

        if (
            index + 1
            < len(
                chunks
            )
        ):
            following = chunks[
                index + 1
            ]

            if _can_attach_lead_in(
                current,
                following,
            ):
                output.append(
                    _compose_lead_in_with_target(
                        lead=current,
                        target=following,
                    )
                )

                index += 2

                continue

        output.append(
            current
        )

        index += 1

    return output


def _can_attach_lead_in(
    lead: Chunk,
    target: Chunk,
) -> bool:
    if lead.subtype not in {
        "paragraph",
        "blockquote",
    }:
        return False

    if (
        target.subtype
        not in _STRUCTURAL_TARGET_SUBTYPES
    ):
        return False

    if not _is_dependent_lead_in(
        lead.text
    ):
        return False

    if (
        lead.heading
        and target.heading
        and lead.heading
        != target.heading
    ):
        return False

    if (
        lead.context
        and target.context
        and lead.context
        != target.context
    ):
        return False

    lead_scope = _scope_identity(
        lead
    )

    target_scope = _scope_identity(
        target
    )

    if (
        lead_scope
        and target_scope
        and lead_scope
        != target_scope
    ):
        return False

    return True


def _is_dependent_lead_in(
    text: str,
) -> bool:
    value = clean_text(
        text
    )

    if not value:
        return False

    if value.endswith(
        ":"
    ):
        return True

    normalized = " ".join(
        value.lower().split()
    )

    return normalized.endswith(
        _DEPENDENT_ENDINGS
    )


def _scope_identity(
    chunk: Chunk,
) -> tuple[str, str] | None:
    metadata = chunk.metadata

    scope_id = _as_text(
        metadata.get(
            "local_scope_id"
        )
    )

    scope_title = _as_text(
        metadata.get(
            "local_scope_title"
        )
    )

    if (
        not scope_id
        and not scope_title
    ):
        return None

    return (
        scope_id,
        scope_title,
    )


def _compose_lead_in_with_target(
    *,
    lead: Chunk,
    target: Chunk,
) -> Chunk:
    text = (
        f"{lead.text.rstrip()}\n\n"
        f"{target.text.lstrip()}"
    ).strip()

    metadata = deepcopy(
        target.metadata
    )

    source_ir_ids = _ordered_unique(
        _chunk_source_ir_ids(
            lead
        )
        + _chunk_source_ir_ids(
            target
        )
    )

    if source_ir_ids:
        metadata[
            "source_ir_ids"
        ] = source_ir_ids

    absorbed_chunk_ids = _ordered_unique(
        [
            *(
                lead.metadata.get(
                    "absorbed_chunk_ids"
                )
                if isinstance(
                    lead.metadata.get(
                        "absorbed_chunk_ids"
                    ),
                    list,
                )
                else []
            ),
            lead.chunk_id,
        ]
    )

    metadata[
        "absorbed_chunk_ids"
    ] = [
        value
        for value in absorbed_chunk_ids
        if isinstance(
            value,
            str,
        )
        and value.strip()
    ]

    metadata[
        "lead_in_chunk_id"
    ] = lead.chunk_id

    metadata[
        "lead_in_text"
    ] = lead.text

    metadata[
        "retrieval_composite"
    ] = True

    metadata[
        "retrieval_composite_kind"
    ] = (
        "dependent_lead_in+"
        f"{target.subtype}"
    )

    metadata[
        "composite_start_order"
    ] = lead.order

    metadata[
        "composite_end_order"
    ] = target.order

    for key in (
        "content",
        "relationships",
        "links",
        "images",
        "inline_code",
    ):
        values = _merge_metadata_sequence(
            lead.metadata.get(
                key
            ),
            target.metadata.get(
                key
            ),
        )

        if values:
            metadata[
                key
            ] = values

    for key in (
        "local_scope_id",
        "local_scope_title",
        "local_context",
        "parent_context",
        "parent_context_id",
    ):
        if metadata.get(
            key
        ) not in (
            None,
            "",
        ):
            continue

        value = lead.metadata.get(
            key
        )

        if value not in (
            None,
            "",
        ):
            metadata[
                key
            ] = deepcopy(
                value
            )

    return target.copy_with(
        text=text,
        heading=(
            target.heading
            or lead.heading
        ),
        context=(
            target.context
            or lead.context
        ),
        order=lead.order,
        metadata=metadata,
        tokens=estimate_tokens(
            text
        ),
        source=(
            target.source
            or lead.source
        ),
    )


def _chunk_source_ir_ids(
    chunk: Chunk,
) -> list[str]:
    metadata = chunk.metadata

    output: list[str] = []

    values = metadata.get(
        "source_ir_ids"
    )

    if isinstance(
        values,
        list,
    ):
        for value in values:
            normalized = _as_text(
                value
            )

            if normalized:
                output.append(
                    normalized
                )

    for key in (
        "source_ir_id",
        "ir_id",
    ):
        value = _as_text(
            metadata.get(
                key
            )
        )

        if value:
            output.append(
                value
            )

    return _ordered_unique(
        output
    )


def _merge_metadata_sequence(
    left: Any,
    right: Any,
) -> list[Any]:
    output: list[Any] = []

    for raw in (
        left,
        right,
    ):
        if not isinstance(
            raw,
            list,
        ):
            continue

        for value in raw:
            if value not in output:
                output.append(
                    deepcopy(
                        value
                    )
                )

    return output


def _ordered_unique(
    values: list[Any],
) -> list[Any]:
    output: list[Any] = []

    for value in values:
        if value in (
            None,
            "",
        ):
            continue

        if value not in output:
            output.append(
                value
            )

    return output


#--------------------SUBTYPE RESOLUTION--------------------

def _resolve_subtype(
    block: dict[str, Any],
) -> str:
    block_type = _as_text(
        block.get(
            "type"
        )
    ).lower()

    form = _as_text(
        block.get(
            "form"
        )
    ).lower()

    if block_type == "code":
        return "code"

    if block_type == "table":
        return "table"

    if block_type == "list":
        return "list"

    if block_type == "component":
        return "component"

    if form in {
        "callout",
        "blockquote",
        "quote",
    }:
        return "blockquote"

    return "paragraph"


def _resolve_chunk_type(
    block: dict[str, Any],
) -> str:
    block_type = _as_text(
        block.get(
            "type"
        )
    ).lower()

    return IR_TYPE_TO_CHUNK_TYPE.get(
        block_type,
        block_type or "text",
    )


#--------------------IDENTITY--------------------

def _build_chunk_id(
    *,
    section: dict[str, Any],
    block: dict[str, Any],
    fallback_index: int,
) -> str:
    block_id = _as_text(
        block.get(
            "id"
        )
    )

    if block_id:
        return block_id

    section_id = _as_text(
        section.get(
            "section_id"
        )
    )

    if not section_id:
        section_id = "section"

    return (
        f"{section_id}:"
        f"b{fallback_index}"
    )


def _resolve_order(
    block: dict[str, Any],
    fallback_index: int,
) -> int:
    value = block.get(
        "block_index",
        fallback_index,
    )

    if isinstance(
        value,
        bool,
    ):
        return fallback_index

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return fallback_index


#--------------------CONTEXT--------------------

def _resolve_heading(
    section: dict[str, Any],
    block: dict[str, Any],
) -> str:
    return (
        _as_text(
            block.get(
                "heading"
            )
        )
        or _as_text(
            section.get(
                "heading"
            )
        )
        or _as_text(
            section.get(
                "title"
            )
        )
    )


def _resolve_context(
    section: dict[str, Any],
    block: dict[str, Any],
) -> str:
    return (
        _as_text(
            block.get(
                "context"
            )
        )
        or _as_text(
            section.get(
                "context"
            )
        )
        or _as_text(
            section.get(
                "parent_context"
            )
        )
    )


def _resolve_local_context(
    section: dict[str, Any],
    block: dict[str, Any],
) -> str:
    meta = _safe_dict(
        block.get(
            "meta"
        )
    )

    return (
        _as_text(
            block.get(
                "local_context"
            )
        )
        or _as_text(
            meta.get(
                "local_context"
            )
        )
        or _as_text(
            meta.get(
                "parent_context"
            )
        )
        or _as_text(
            meta.get(
                "parent_instruction"
            )
        )
        or _as_text(
            section.get(
                "local_context"
            )
        )
        or _as_text(
            section.get(
                "command_context"
            )
        )
        or _as_text(
            section.get(
                "heading"
            )
        )
    )


#--------------------METADATA--------------------

def _build_metadata(
    *,
    section: dict[str, Any],
    block: dict[str, Any],
    section_index: int,
    block_index: int,
    subtype: str,
) -> dict[str, Any]:
    block_meta = deepcopy(
        _safe_dict(
            block.get(
                "meta"
            )
        )
    )

    heading_path = (
        section.get(
            "heading_path"
        )
        or section.get(
            "path"
        )
        or block_meta.get(
            "heading_path"
        )
        or []
    )

    ir_id = _as_text(
        block.get(
            "id"
        )
    )

    metadata: dict[str, Any] = {
        **block_meta,
        "ir_id": ir_id,
        "source_ir_ids": (
            [
                ir_id
            ]
            if ir_id
            else []
        ),
        "section_id": _as_text(
            section.get(
                "section_id"
            )
        ),
        "block_index": block.get(
            "block_index",
            block_index,
        ),
        "section_index": section_index,
        "ir_type": block.get(
            "type"
        ),
        "ir_form": block.get(
            "form"
        ),
        "function": block.get(
            "function"
        ),
        "source_type": block_meta.get(
            "source_type"
        ),
        "heading_path": deepcopy(
            heading_path
        ),
        "local_context": _resolve_local_context(
            section,
            block,
        ),
        "parent_context": block_meta.get(
            "parent_context"
        ),
        "parent_context_id": block_meta.get(
            "parent_context_id"
        ),
        "parent_instruction": block_meta.get(
            "parent_instruction"
        ),
        "parent_instruction_id": block_meta.get(
            "parent_instruction_id"
        ),
        "relationships": deepcopy(
            _safe_list(
                block.get(
                    "relationships"
                )
            )
        ),
        "content": [
            deepcopy(
                block
            )
        ],
        "subtype": subtype,
    }

    return _drop_empty(
        metadata
    )


#--------------------HELPERS--------------------

def _safe_dict(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        return {}

    return value


def _safe_list(
    value: Any,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        return []

    return value


def _as_text(
    value: Any,
) -> str:
    return str(
        value
        or ""
    ).strip()


def _drop_empty(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if value is not None
        and value != ""
        and value != []
    }
