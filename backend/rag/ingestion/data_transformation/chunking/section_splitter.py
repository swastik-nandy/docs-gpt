from __future__ import annotations

from typing import Any

from .ir_adapter import chunks_from_ir_blocks
from .models import Chunk
from .utils import clean_text
from .utils import estimate_tokens


#--------------------PUBLIC API--------------------

def split_sections(
    sections: list[dict[str, Any]],
    source: str = "",
) -> list[Chunk]:
    if not sections:
        return []

    chunks: list[Chunk] = []

    for section_index, section in enumerate(
        sections
    ):
        if not isinstance(
            section,
            dict,
        ):
            continue

        chunks.extend(
            _split_section(
                section=section,
                section_index=section_index,
                source=source,
            )
        )

    return _reindex(
        chunks
    )


#--------------------SECTION ROUTING--------------------

def _split_section(
    *,
    section: dict[str, Any],
    section_index: int,
    source: str,
) -> list[Chunk]:
    blocks = section.get(
        "blocks"
    )

    if isinstance(
        blocks,
        list,
    ) and blocks:
        return chunks_from_ir_blocks(
            section=section,
            section_index=section_index,
            source=source,
        )

    return _fallback_chunks(
        section=section,
        section_index=section_index,
        source=source,
    )


#--------------------FALLBACK--------------------

def _fallback_chunks(
    *,
    section: dict[str, Any],
    section_index: int,
    source: str,
) -> list[Chunk]:
    content = section.get(
        "content"
    )

    if not isinstance(
        content,
        list,
    ):
        return []

    heading = _resolve_heading(
        section
    )

    context = _resolve_context(
        section
    )

    output: list[Chunk] = []

    for block_index, block in enumerate(
        content
    ):
        if not isinstance(
            block,
            dict,
        ):
            continue

        block_type = _text(
            block.get(
                "type"
            )
        ).lower()

        if block_type == "list":
            output.extend(
                _list_chunks(
                    section=section,
                    block=block,
                    section_index=section_index,
                    block_index=block_index,
                    heading=heading,
                    context=context,
                    source=source,
                )
            )
            continue

        if block_type in {
            "table",
            "generic_table",
        }:
            output.extend(
                _table_chunks(
                    section=section,
                    block=block,
                    section_index=section_index,
                    block_index=block_index,
                    heading=heading,
                    context=context,
                    source=source,
                )
            )
            continue

        subtype = _fallback_subtype(
            block_type
        )

        chunk_type = _fallback_chunk_type(
            block_type
        )

        text = _fallback_text(
            block,
            block_type,
        )

        chunk = _make_chunk(
            text=text,
            heading=heading,
            context=context,
            subtype=subtype,
            chunk_type=chunk_type,
            section=section,
            block=block,
            section_index=section_index,
            block_index=block_index,
            source=source,
        )

        if chunk is not None:
            output.append(
                chunk
            )

    return output


#--------------------LISTS--------------------

def _list_chunks(
    *,
    section: dict[str, Any],
    block: dict[str, Any],
    section_index: int,
    block_index: int,
    heading: str,
    context: str,
    source: str,
) -> list[Chunk]:
    items = block.get(
        "items"
    )

    if not isinstance(
        items,
        list,
    ):
        return []

    output: list[Chunk] = []

    for item_index, item in enumerate(
        items
    ):
        text = _list_item_text(
            item
        )

        child_block = {
            **block,
            "item_index": item_index,
        }

        chunk = _make_chunk(
            text=text,
            heading=heading,
            context=context,
            subtype="list",
            chunk_type="list",
            section=section,
            block=child_block,
            section_index=section_index,
            block_index=block_index,
            source=source,
        )

        if chunk is not None:
            output.append(
                chunk
            )

    return output


def _list_item_text(
    item: Any,
) -> str:
    if isinstance(
        item,
        dict,
    ):
        for key in (
            "text",
            "content",
            "value",
            "label",
        ):
            value = item.get(
                key
            )

            if isinstance(
                value,
                str,
            ) and value.strip():
                return clean_text(
                    value
                )

    return clean_text(
        str(
            item
            or ""
        )
    )


#--------------------TABLES--------------------

def _table_chunks(
    *,
    section: dict[str, Any],
    block: dict[str, Any],
    section_index: int,
    block_index: int,
    heading: str,
    context: str,
    source: str,
) -> list[Chunk]:
    row_text = block.get(
        "row_text"
    )

    if isinstance(
        row_text,
        list,
    ) and row_text:
        rows = row_text
    else:
        raw_rows = block.get(
            "rows"
        )

        rows = (
            raw_rows
            if isinstance(
                raw_rows,
                list,
            )
            else []
        )

    output: list[Chunk] = []

    for row_index, row in enumerate(
        rows
    ):
        text = _table_row_text(
            row
        )

        child_block = {
            **block,
            "row_index": row_index,
        }

        chunk = _make_chunk(
            text=text,
            heading=heading,
            context=context,
            subtype="table",
            chunk_type="table",
            section=section,
            block=child_block,
            section_index=section_index,
            block_index=block_index,
            source=source,
        )

        if chunk is not None:
            output.append(
                chunk
            )

    return output


def _table_row_text(
    row: Any,
) -> str:
    if isinstance(
        row,
        str,
    ):
        return clean_text(
            row
        )

    if isinstance(
        row,
        dict,
    ):
        parts: list[str] = []

        for key, value in row.items():
            key_text = clean_text(
                str(
                    key
                    or ""
                )
            )

            value_text = clean_text(
                str(
                    value
                    or ""
                )
            )

            if not value_text:
                continue

            if key_text:
                parts.append(
                    f"{key_text}: {value_text}"
                )
            else:
                parts.append(
                    value_text
                )

        return " | ".join(
            parts
        )

    if isinstance(
        row,
        list,
    ):
        parts = [
            clean_text(
                str(
                    value
                    or ""
                )
            )
            for value in row
        ]

        return " | ".join(
            part
            for part in parts
            if part
        )

    return ""


#--------------------CHUNK CREATION--------------------

def _make_chunk(
    *,
    text: Any,
    heading: str,
    context: str,
    subtype: str,
    chunk_type: str,
    section: dict[str, Any],
    block: dict[str, Any],
    section_index: int,
    block_index: int,
    source: str,
) -> Chunk | None:
    normalized = clean_text(
        text
    )

    if not normalized:
        return None

    metadata = {
        "content": [
            dict(
                block
            )
        ],
        "heading_path": _heading_path(
            section
        ),
        "section_id": section.get(
            "section_id"
        ),
        "section_index": section_index,
        "block_index": block_index,
        "subtype": subtype,
        "fallback_chunking": True,
        "source_content_type": block.get(
            "type"
        ),
    }

    return Chunk(
        text=normalized,
        heading=heading,
        context=context,
        chunk_type=chunk_type,
        subtype=subtype,
        metadata=_drop_empty(
            metadata
        ),
        tokens=estimate_tokens(
            normalized
        ),
        source=source,
        order=block_index,
    )


#--------------------TYPE RESOLUTION--------------------

def _fallback_subtype(
    block_type: str,
) -> str:
    if block_type == "code":
        return "code"

    if block_type in {
        "blockquote",
        "callout",
        "quote",
    }:
        return "blockquote"

    if block_type == "component":
        return "component"

    return "paragraph"


def _fallback_chunk_type(
    block_type: str,
) -> str:
    if block_type == "code":
        return "code"

    if block_type == "component":
        return "component"

    return "text"


def _fallback_text(
    block: dict[str, Any],
    block_type: str,
) -> str:
    text = clean_text(
        block.get(
            "text",
            ""
        )
    )

    if (
        block_type
        in {
            "blockquote",
            "callout",
            "quote",
        }
    ):
        label = clean_text(
            block.get(
                "label",
                ""
            )
        )

        if label and text:
            return f"{label}: {text}"

    return text


#--------------------CONTEXT--------------------

def _resolve_heading(
    section: dict[str, Any],
) -> str:
    heading = (
        _text(
            section.get(
                "heading"
            )
        )
        or _text(
            section.get(
                "title"
            )
        )
    )

    if heading:
        return heading

    path = _heading_path(
        section
    )

    return (
        path[-1]
        if path
        else ""
    )


def _resolve_context(
    section: dict[str, Any],
) -> str:
    context = _text(
        section.get(
            "context"
        )
    )

    if context:
        return context

    path = _heading_path(
        section
    )

    if len(
        path
    ) >= 2:
        return path[-2]

    return ""


def _heading_path(
    section: dict[str, Any],
) -> list[str]:
    path = (
        section.get(
            "heading_path"
        )
        or section.get(
            "path"
        )
        or []
    )

    if not isinstance(
        path,
        list,
    ):
        return []

    return [
        _text(
            item
        )
        for item in path
        if _text(
            item
        )
    ]


#--------------------ORDERING--------------------

def _reindex(
    chunks: list[Chunk],
) -> list[Chunk]:
    return [
        chunk.copy_with(
            order=index
        )
        for index, chunk in enumerate(
            chunks
        )
    ]


#--------------------HELPERS--------------------

def _text(
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
