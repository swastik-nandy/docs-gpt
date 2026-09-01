# rag/ingestion/data_transformation/parser/adapters/markdown/ast/provenance.py

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from rag.ingestion.data_transformation.parser.config_loader import (
    load_parser_heuristics_config,
)


SOURCE_META_KEY = "source"
SOURCE_FRAGMENTS_KEY = "fragments"


def _token_line_map(token: Any) -> tuple[int, int] | None:
    """
    Return the raw Markdown-it line span for a token.

    Markdown-it maps use:

        [line_start, line_end)

    where:
    - line_start is zero-based
    - line_end is exclusive

    Invalid or missing token maps are ignored rather than guessed.
    """

    raw_map = getattr(token, "map", None)

    if not isinstance(raw_map, (list, tuple)):
        return None

    if len(raw_map) != 2:
        return None

    raw_start, raw_end = raw_map

    if isinstance(raw_start, bool) or not isinstance(raw_start, int):
        return None

    if isinstance(raw_end, bool) or not isinstance(raw_end, int):
        return None

    if raw_start < 0:
        return None

    if raw_end < raw_start:
        return None

    return raw_start, raw_end


def _display_line_span(
    raw_start: int,
    raw_end: int,
) -> tuple[int, int]:
    """
    Convert Markdown-it's internal line range into configured output semantics.

    Source extraction always uses the original zero-based, end-exclusive range.
    Only the serialized line numbers are converted.
    """

    config = load_parser_heuristics_config().markdown.source_provenance

    line_start = raw_start
    line_end = raw_end

    if not config.line_end_exclusive:
        line_end = max(raw_start, raw_end - 1)

    if not config.line_numbers_zero_based:
        line_start += 1
        line_end += 1

    return line_start, line_end


def _extract_source_text(
    source_lines: Sequence[str],
    raw_start: int,
    raw_end: int,
) -> str:
    """
    Extract the exact source text covered by a Markdown-it token map.

    `source_lines` must come from:

        source_text.splitlines(keepends=True)

    Keeping line endings allows merged provenance to reconstruct exact source
    fragments without fuzzy text matching.
    """

    if raw_start >= len(source_lines):
        return ""

    bounded_end = min(raw_end, len(source_lines))

    if bounded_end < raw_start:
        return ""

    return "".join(source_lines[raw_start:bounded_end])


def build_source_fragment(
    *,
    token: Any,
    source_lines: Sequence[str],
    block_index: int,
) -> Dict[str, Any] | None:
    """
    Build one source-provenance fragment for a top-level parser atom.

    A fragment may contain:

    - parser block index
    - source line start
    - source line end
    - exact source text

    Every field is controlled by `heuristics.toml`.
    """

    config = load_parser_heuristics_config().markdown.source_provenance

    if not config.enabled:
        return None

    fragment: Dict[str, Any] = {}

    if config.preserve_block_index:
        fragment["block_index"] = block_index

    raw_span = _token_line_map(token)

    if raw_span is not None:
        raw_start, raw_end = raw_span

        if config.preserve_line_spans:
            line_start, line_end = _display_line_span(
                raw_start,
                raw_end,
            )

            fragment["line_start"] = line_start
            fragment["line_end"] = line_end

        if config.preserve_original_text:
            fragment["text"] = _extract_source_text(
                source_lines,
                raw_start,
                raw_end,
            )

    if not fragment:
        return None

    return fragment


def attach_source_provenance(
    element: MutableMapping[str, Any],
    *,
    token: Any,
    source_lines: Sequence[str],
    block_index: int,
) -> MutableMapping[str, Any]:
    """
    Attach source provenance to one AST element in place.

    The representation always uses a fragment list:

        {
            "source": {
                "fragments": [
                    {
                        "block_index": 4,
                        "line_start": 10,
                        "line_end": 13,
                        "text": "..."
                    }
                ]
            }
        }

    Using a list from the beginning means semantic packing can combine source
    atoms without changing the metadata schema.
    """

    fragment = build_source_fragment(
        token=token,
        source_lines=source_lines,
        block_index=block_index,
    )

    if fragment is None:
        return element

    element[SOURCE_META_KEY] = {
        SOURCE_FRAGMENTS_KEY: [fragment],
    }

    return element


def get_source_fragments(
    value: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """
    Return defensive copies of valid source fragments.

    `value` may be either:

    - a complete parser/enrichment block containing `source`
    - the `source` mapping itself
    """

    if not isinstance(value, Mapping):
        return []

    possible_source = value.get(SOURCE_META_KEY)

    if isinstance(possible_source, Mapping):
        source = possible_source
    else:
        source = value

    raw_fragments = source.get(SOURCE_FRAGMENTS_KEY)

    if not isinstance(raw_fragments, list):
        return []

    fragments: List[Dict[str, Any]] = []

    for raw_fragment in raw_fragments:
        if not isinstance(raw_fragment, Mapping):
            continue

        fragment = dict(raw_fragment)

        if fragment:
            fragments.append(deepcopy(fragment))

    return fragments


def merge_source_provenance(
    values: Iterable[Mapping[str, Any] | None],
) -> Dict[str, Any]:
    """
    Merge provenance from multiple parser or enrichment blocks.

    Ordering is preserved because semantic packing must retain source order.

    Duplicate fragments are removed using their complete serialized identity.
    This protects against metadata being copied through multiple enrichment
    stages before blocks are packed.
    """

    merged: List[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for value in values:
        for fragment in get_source_fragments(value):
            identity = (
                fragment.get("block_index"),
                fragment.get("line_start"),
                fragment.get("line_end"),
                fragment.get("text"),
            )

            if identity in seen:
                continue

            seen.add(identity)
            merged.append(fragment)

    if not merged:
        return {}

    return {
        SOURCE_FRAGMENTS_KEY: merged,
    }


def copy_source_provenance(
    source: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """
    Copy the complete normalized `source` metadata mapping.

    This helper should be used by enrichment and IR builders instead of
    manually selecting provenance fields.
    """

    fragments = get_source_fragments(source)

    if not fragments:
        return {}

    return {
        SOURCE_FRAGMENTS_KEY: fragments,
    }