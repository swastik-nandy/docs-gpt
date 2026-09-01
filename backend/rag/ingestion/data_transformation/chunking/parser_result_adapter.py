# rag/ingestion/data_transformation/chunking/parser_result_adapter.py

from __future__ import annotations

from typing import Any

from rag.ingestion.data_transformation.chunking.chunker import chunk_document
from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.chunking.config import ChunkingConfig


#--------------------PUBLIC API--------------------

def chunk_parse_result(
    parse_result: Any,
    *,
    source: str = "",
    enable_overlap: bool = False,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    if parse_result is None:
        return []

    units = _result_units(
        parse_result
    )

    if units:
        return _chunk_units(
            parse_result=parse_result,
            units=units,
            source=source,
            enable_overlap=enable_overlap,
            config=config,
        )

    sections = _result_sections(
        parse_result
    )

    if not sections:
        return []

    chunks = chunk_document(
        sections,
        source=source,
        enable_overlap=enable_overlap,
        config=config,
    )

    return _stamp_chunks(
        chunks=chunks,
        parse_result=parse_result,
        unit=None,
        unit_index=0,
        global_order_start=0,
    )


#--------------------UNIT CHUNKING--------------------

def _chunk_units(
    *,
    parse_result: Any,
    units: list[Any],
    source: str,
    enable_overlap: bool,
    config: ChunkingConfig | None,
) -> list[Chunk]:
    output: list[Chunk] = []

    global_order = 0

    for unit_index, unit in enumerate(
        units
    ):
        sections = _unit_sections(
            unit
        )

        if not sections:
            continue

        unit_chunks = chunk_document(
            sections,
            source=source,
            enable_overlap=enable_overlap,
            config=config,
        )

        stamped = _stamp_chunks(
            chunks=unit_chunks,
            parse_result=parse_result,
            unit=unit,
            unit_index=unit_index,
            global_order_start=global_order,
        )

        output.extend(
            stamped
        )

        global_order += len(
            stamped
        )

    return output


#--------------------METADATA STAMPING--------------------

def _stamp_chunks(
    *,
    chunks: list[Chunk],
    parse_result: Any,
    unit: Any | None,
    unit_index: int,
    global_order_start: int,
) -> list[Chunk]:
    stamped: list[Chunk] = []

    result_doc_id = _as_text(
        getattr(
            parse_result,
            "doc_id",
            "",
        )
    )

    result_strategy = _as_text(
        getattr(
            parse_result,
            "parse_strategy",
            "",
        )
    )

    parser_family = _as_text(
        getattr(
            parse_result,
            "parser_family",
            "",
        )
    )

    parser_variant = _as_text(
        getattr(
            parse_result,
            "parser_variant",
            "",
        )
    )

    result_meta = _safe_dict(
        getattr(
            parse_result,
            "meta",
            {},
        )
    )

    for local_order, chunk in enumerate(
        chunks
    ):
        global_order = (
            global_order_start
            + local_order
        )

        metadata = dict(
            chunk.metadata
        )

        metadata.update(
            {
                "parser_doc_id": result_doc_id,
                "parser_family": parser_family,
                "parser_variant": parser_variant,
                "parse_strategy": result_strategy,
                "chunk_global_order": global_order,
                "chunk_local_order": local_order,
                "chunk_unit_index": unit_index,
            }
        )

        if result_meta:
            metadata[
                "parser_result_meta"
            ] = dict(
                result_meta
            )

        if unit is not None:
            metadata.update(
                _unit_metadata(
                    unit
                )
            )

        metadata = _drop_empty_values(
            metadata
        )

        stamped.append(
            chunk.copy_with(
                metadata=metadata,
                order=global_order,
            )
        )

    return stamped


#--------------------UNIT METADATA--------------------

def _unit_metadata(
    unit: Any,
) -> dict[str, Any]:
    unit_meta = _safe_dict(
        getattr(
            unit,
            "meta",
            {},
        )
    )

    metadata: dict[str, Any] = {
        "parser_unit_id": _as_text(
            getattr(
                unit,
                "unit_id",
                "",
            )
        ),
        "parser_unit_doc_id": _as_text(
            getattr(
                unit,
                "doc_id",
                "",
            )
        ),
        "parser_source_doc_id": _as_text(
            getattr(
                unit,
                "source_doc_id",
                "",
            )
        ),
        "parser_unit_strategy": _as_text(
            getattr(
                unit,
                "parse_strategy",
                "",
            )
        ),
        "parser_unit_section_count": getattr(
            unit,
            "section_count",
            0,
        ),
        "parser_unit_block_count": getattr(
            unit,
            "block_count",
            0,
        ),
    }

    for key in (
        "unit_kind",
        "parser_variant",
        "fragment_id",
        "fragment_index",
        "packed_fragment_id",
        "packed_fragment_index",
        "spine_node_ids",
        "heading_path_prefix",
    ):
        if key in unit_meta:
            metadata[
                f"parser_{key}"
            ] = unit_meta[
                key
            ]

    return _drop_empty_values(
        metadata
    )


#--------------------RESULT ACCESS--------------------

def _result_units(
    parse_result: Any,
) -> list[Any]:
    units = getattr(
        parse_result,
        "units",
        None,
    )

    if not isinstance(
        units,
        list,
    ):
        return []

    return [
        unit
        for unit in units
        if unit is not None
    ]


def _unit_sections(
    unit: Any,
) -> list[dict[str, Any]]:
    sections = getattr(
        unit,
        "sections",
        None,
    )

    if not isinstance(
        sections,
        list,
    ):
        return []

    return [
        section
        for section in sections
        if isinstance(
            section,
            dict,
        )
    ]


def _result_sections(
    parse_result: Any,
) -> list[dict[str, Any]]:
    iter_sections = getattr(
        parse_result,
        "iter_sections",
        None,
    )

    if callable(
        iter_sections
    ):
        return [
            section
            for section in iter_sections()
            if isinstance(
                section,
                dict,
            )
        ]

    sections = getattr(
        parse_result,
        "sections",
        None,
    )

    if not isinstance(
        sections,
        list,
    ):
        return []

    return [
        section
        for section in sections
        if isinstance(
            section,
            dict,
        )
    ]


#--------------------VALUE HELPERS--------------------

def _safe_dict(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        return {}

    return value


def _as_text(
    value: Any,
) -> str:
    return str(
        value
        or ""
    ).strip()


def _drop_empty_values(
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None
        and value != ""
        and value != []
    }