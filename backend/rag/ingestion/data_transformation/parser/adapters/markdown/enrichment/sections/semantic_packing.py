from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Mapping

from rag.ingestion.data_transformation.parser.adapters.markdown.ast.provenance import merge_source_provenance
from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config


#--------------------TEXT--------------------

def _clean_text(
    value: Any,
) -> str:
    return str(
        value
        or ""
    ).strip()


def _word_count(
    value: Any,
) -> int:
    return len(
        _clean_text(
            value
        ).split()
    )


def _block_type(
    block: Mapping[str, Any],
) -> str:
    value = _clean_text(
        block.get(
            "type"
        )
    ).lower()

    if value == "table":
        return "generic_table"

    return value


#--------------------SCOPE--------------------

def _local_scope_key(
    block: Mapping[str, Any],
) -> tuple[str, str]:
    return (
        _clean_text(
            block.get(
                "local_scope_id"
            )
        ),
        _clean_text(
            block.get(
                "local_scope_title"
            )
        ),
    )


def _component_key(
    block: Mapping[str, Any],
) -> str:
    return _clean_text(
        block.get(
            "component_context"
        )
    )


def _same_scope(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    if (
        config.merge_only_same_local_scope
        and _local_scope_key(
            left
        )
        != _local_scope_key(
            right
        )
    ):
        return False

    if (
        config.merge_only_same_component_context
        and _component_key(
            left
        )
        != _component_key(
            right
        )
    ):
        return False

    return True


#--------------------METADATA--------------------

def _dedupe(
    values: list[Any],
) -> list[Any]:
    output: list[Any] = []

    for value in values:
        if value not in output:
            output.append(
                value
            )

    return output


def _merge_list_field(
    blocks: list[dict[str, Any]],
    key: str,
) -> list[Any]:
    values: list[Any] = []

    for block in blocks:
        value = block.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            values.extend(
                deepcopy(
                    value
                )
            )

        elif value not in (
            None,
            "",
        ):
            values.append(
                deepcopy(
                    value
                )
            )

    return _dedupe(
        values
    )


def _source_indexes(
    blocks: list[dict[str, Any]],
) -> list[int]:
    indexes: list[int] = []

    for block in blocks:
        source = block.get(
            "source"
        )

        if not isinstance(
            source,
            Mapping,
        ):
            continue

        fragments = source.get(
            "fragments"
        )

        if not isinstance(
            fragments,
            list,
        ):
            continue

        for fragment in fragments:
            if not isinstance(
                fragment,
                Mapping,
            ):
                continue

            index = fragment.get(
                "block_index"
            )

            if (
                isinstance(
                    index,
                    int,
                )
                and not isinstance(
                    index,
                    bool,
                )
                and index not in indexes
            ):
                indexes.append(
                    index
                )

    return indexes


#--------------------GROUP MERGE--------------------

def _merge_group(
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(
        blocks
    ) == 1:
        return deepcopy(
            blocks[0]
        )

    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    merged = deepcopy(
        blocks[0]
    )

    texts = [
        _clean_text(
            block.get(
                "text"
            )
        )
        for block in blocks
        if _clean_text(
            block.get(
                "text"
            )
        )
    ]

    separator = (
        config.paragraph_separator
        if config.preserve_paragraph_boundaries
        else " "
    )

    merged[
        "text"
    ] = separator.join(
        texts
    ).strip()

    for key in (
        "links",
        "images",
        "inline_code",
        "html_inline",
        "html_block",
        "strong_text",
        "selector_metadata",
    ):
        values = _merge_list_field(
            blocks,
            key,
        )

        if values:
            merged[
                key
            ] = values
        else:
            merged.pop(
                key,
                None,
            )

    source = merge_source_provenance(
        blocks
    )

    if source:
        merged[
            "source"
        ] = source
    else:
        merged.pop(
            "source",
            None,
        )

    source_indexes = _source_indexes(
        blocks
    )

    merged[
        "semantic_packed"
    ] = True

    merged[
        "semantic_packed_block_count"
    ] = len(
        blocks
    )

    if source_indexes:
        merged[
            "semantic_source_indexes"
        ] = source_indexes

    return merged


#--------------------POLICY--------------------

def _eligible(
    block: Mapping[str, Any],
) -> bool:
    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    return (
        _block_type(
            block
        )
        in config.eligible_block_types
    )


def _boundary(
    block: Mapping[str, Any],
) -> bool:
    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    return (
        _block_type(
            block
        )
        in config.boundary_block_types
    )


def _group_words(
    group: list[dict[str, Any]],
) -> int:
    return sum(
        _word_count(
            block.get(
                "text"
            )
        )
        for block in group
    )


def _must_flush_before(
    group: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> bool:
    if not group:
        return False

    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    if not _same_scope(
        group[-1],
        candidate,
    ):
        return True

    if (
        len(
            group
        )
        >= config.maximum_blocks_per_group
    ):
        return True

    current_words = _group_words(
        group
    )

    candidate_words = _word_count(
        candidate.get(
            "text"
        )
    )

    projected_words = (
        current_words
        + candidate_words
    )

    if (
        projected_words
        > config.maximum_target_words
    ):
        return True

    if (
        current_words
        >= config.minimum_target_words
        and projected_words
        > config.preferred_target_words
    ):
        return True

    return False


#--------------------PUBLIC API--------------------

def pack_semantic_paragraphs(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Pack adjacent narrative paragraphs into coherent retrieval units.

    Structural blocks remain atomic at parser level.

    Exact embedding-model token enforcement belongs to chunking.
    """
    if not content:
        return []

    config = (
        load_parser_heuristics_config()
        .markdown
        .semantic_packing
    )

    normalized = [
        deepcopy(
            dict(
                block
            )
        )
        for block in content
        if isinstance(
            block,
            Mapping,
        )
    ]

    if not config.enabled:
        return normalized

    output: list[
        dict[str, Any]
    ] = []

    group: list[
        dict[str, Any]
    ] = []

    def flush() -> None:
        nonlocal group

        if not group:
            return

        output.append(
            _merge_group(
                group
            )
        )

        group = []

    for block in normalized:
        if _boundary(
            block
        ):
            if config.flush_before_structural_block:
                flush()

            output.append(
                block
            )

            if config.flush_after_structural_block:
                flush()

            continue

        if not _eligible(
            block
        ):
            flush()

            output.append(
                block
            )

            continue

        words = _word_count(
            block.get(
                "text"
            )
        )

        if (
            words
            > config.maximum_target_words
            and config.allow_single_oversized_block
        ):
            flush()

            output.append(
                block
            )

            continue

        if _must_flush_before(
            group,
            block,
        ):
            flush()

        group.append(
            block
        )

        if (
            _group_words(
                group
            )
            >= config.preferred_target_words
        ):
            flush()

            continue

        if (
            len(
                group
            )
            >= config.maximum_blocks_per_group
        ):
            flush()

    flush()

    return output
