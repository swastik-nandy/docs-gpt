from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer import (
    PackedSpineFragment,
    load_packed_spine_config,
    pack_spine_nodes_with_diagnostics,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import MarkdownParsedUnit
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.text_pipeline import (
    collect_ir_blocks_from_sections,
    parse_markdown_text_to_sections,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.ir.block_builder import (
    refresh_retrieval_text,
)


# --------------- IO --------------------------------

def read_byte_range_as_text(
    file_path: str,
    byte_start: int,
    byte_end: int,
) -> str:
    """
    Read a byte range from the source file as UTF-8 text.

    Invalid bytes are replaced instead of killing the whole parse.
    """

    start = max(0, int(byte_start))
    end = max(start, int(byte_end))

    with Path(file_path).open("rb") as file:
        file.seek(start)
        data = file.read(end - start)

    return data.decode("utf-8", errors="replace")


# --------------- IDS --------------------------------

def safe_fragment_id(index: int, node_id: str) -> str:
    """
    Build a stable, filesystem-safe fragment id.
    """

    cleaned = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(node_id or "")
    ).strip("_")

    if not cleaned:
        cleaned = "fragment"

    return f"f{index:04d}_{cleaned[:80]}"


def safe_packed_fragment_id(fragment: PackedSpineFragment) -> str:
    """
    Build stable packed fragment id.
    """

    return f"{fragment.packed_fragment_id}_{fragment.fragment_index:04d}"


# --------------- PATH HELPERS --------------------------------

def _clean_path(parts: list[Any]) -> list[str]:
    cleaned: list[str] = []

    for part in parts or []:
        value = str(part or "").strip()

        if not value:
            continue

        if cleaned and cleaned[-1] == value:
            continue

        cleaned.append(value)

    return cleaned


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _section_heading(section: dict[str, Any]) -> str:
    return str(section.get("heading") or "").strip()


def _section_parent_path(section: dict[str, Any]) -> list[str]:
    parent_path = section.get("parent_path") or section.get("path") or []

    if not isinstance(parent_path, list):
        return []

    return _clean_path(parent_path)


def _spine_path(spine_node: Any) -> list[str]:
    return _clean_path(list(getattr(spine_node, "path", []) or []))


def _spine_title(spine_node: Any) -> str:
    return str(getattr(spine_node, "title", "") or "").strip()


def _spine_node_id(spine_node: Any) -> str:
    return str(getattr(spine_node, "node_id", "") or "").strip()


# --------------- SPINE OWNER RESOLUTION --------------------------------

def _find_owning_spine_node(
    section: dict[str, Any],
    spine_nodes: tuple[Any, ...],
    cursor: int,
) -> tuple[Any, int]:
    """
    Resolve the most likely owning spine node for a parsed section.

    This is intentionally monotonic:
    once we match a later spine node, the next section search starts there.
    """

    if not spine_nodes:
        return None, cursor

    heading = _norm(_section_heading(section))

    if heading:
        for index in range(cursor, len(spine_nodes)):
            node = spine_nodes[index]

            if _norm(_spine_title(node)) == heading:
                return node, index

    fallback_index = min(cursor, len(spine_nodes) - 1)

    return spine_nodes[fallback_index], fallback_index


# --------------- PATCHING --------------------------------

def _patch_sections_with_spine_context(
    sections: list[dict[str, Any]],
    *,
    source_doc_id: str,
    fragment_id: str,
    spine_node: Any,
    packed_fragment: PackedSpineFragment | None = None,
) -> list[dict[str, Any]]:
    """
    Patch sections/blocks after parsing a fragment.

    A fragment slice can be parsed independently, but it may not contain all
    ancestor headings from the original document. The spine carries that
    missing breadcrumb trail.
    """

    spine_path = _spine_path(spine_node)
    spine_parent_path = spine_path[:-1]
    spine_title = _spine_title(spine_node)

    byte_range = [
        int(getattr(spine_node, "byte_start", 0)),
        int(getattr(spine_node, "byte_end", 0)),
    ]

    line_range = [
        int(getattr(spine_node, "line_start", 0)),
        int(getattr(spine_node, "line_end", 0)),
    ]

    packed_meta = packed_fragment.to_meta() if packed_fragment else {}

    for section in sections:
        if not isinstance(section, dict):
            continue

        heading = _section_heading(section)
        original_parent_path = _section_parent_path(section)

        merged_parent_path = _clean_path(
            spine_parent_path + original_parent_path
        )

        heading_path = _clean_path(
            merged_parent_path + ([heading] if heading else [])
        )

        context = (
            " > ".join(merged_parent_path)
            if merged_parent_path
            else heading or spine_title or "root"
        )

        local_context = (
            heading
            or spine_title
            or (merged_parent_path[-1] if merged_parent_path else "")
        )

        section["source_doc_id"] = source_doc_id
        section["fragment_id"] = fragment_id
        section["spine_node_id"] = _spine_node_id(spine_node)
        section["spine_parent_id"] = getattr(spine_node, "parent_id", "")
        section["spine_level"] = getattr(spine_node, "level", 0)
        section["spine_path"] = spine_path
        section["byte_range"] = byte_range
        section["line_range"] = line_range
        section["parent_path"] = merged_parent_path
        section["heading_path"] = heading_path
        section["context"] = context
        section["local_context"] = local_context

        if packed_fragment:
            section["packed_fragment_id"] = packed_fragment.packed_fragment_id
            section["packed_fragment_index"] = packed_fragment.fragment_index
            section["packed_fragment_byte_range"] = [
                packed_fragment.byte_start,
                packed_fragment.byte_end,
            ]
            section["packed_fragment_line_range"] = [
                packed_fragment.line_start,
                packed_fragment.line_end,
            ]

        for block in section.get("blocks", []) or []:
            if not isinstance(block, dict):
                continue

            block["context"] = context
            block["heading"] = heading or block.get("heading", "")
            block["local_context"] = local_context

            meta = block.setdefault("meta", {})
            meta["source_doc_id"] = source_doc_id
            meta["fragment_id"] = fragment_id
            meta["spine_node_id"] = _spine_node_id(spine_node)
            meta["spine_parent_id"] = getattr(spine_node, "parent_id", "")
            meta["spine_level"] = getattr(spine_node, "level", 0)
            meta["spine_path"] = spine_path
            meta["structural_path"] = heading_path or spine_path
            meta["byte_range"] = byte_range
            meta["line_range"] = line_range
            meta["absolute_byte_range"] = byte_range
            meta["absolute_line_range"] = line_range
            meta["owning_spine_node_id"] = _spine_node_id(spine_node)

            if packed_fragment:
                meta["packed_fragment_id"] = packed_fragment.packed_fragment_id
                meta["packed_fragment_index"] = packed_fragment.fragment_index
                meta["packed_fragment_byte_range"] = [
                    packed_fragment.byte_start,
                    packed_fragment.byte_end,
                ]
                meta["packed_fragment_line_range"] = [
                    packed_fragment.line_start,
                    packed_fragment.line_end,
                ]
                meta["packed_fragment_spine_node_count"] = packed_fragment.spine_node_count

            refresh_retrieval_text(block)

    return sections


def _patch_packed_sections_with_spine_context(
    sections: list[dict[str, Any]],
    *,
    source_doc_id: str,
    fragment_id: str,
    packed_fragment: PackedSpineFragment,
) -> list[dict[str, Any]]:
    patched_sections: list[dict[str, Any]] = []
    cursor = 0

    for section in sections or []:
        if not isinstance(section, dict):
            continue

        owning_node, matched_index = _find_owning_spine_node(
            section=section,
            spine_nodes=packed_fragment.spine_nodes,
            cursor=cursor,
        )

        if owning_node is None:
            continue

        cursor = matched_index

        patched = _patch_sections_with_spine_context(
            [section],
            source_doc_id=source_doc_id,
            fragment_id=fragment_id,
            spine_node=owning_node,
            packed_fragment=packed_fragment,
        )

        patched_sections.extend(patched)

    return patched_sections


# --------------- SINGLE SPINE NODE PARSE --------------------------------

def parse_spine_fragment(
    *,
    file_path: str,
    source_doc_id: str,
    fragment_index: int,
    spine_node: Any,
    variant: str,
) -> MarkdownParsedUnit | None:
    """
    Parse one spine node byte range into one independent parsed unit.

    Kept for compatibility and fallback.
    New large-file execution should prefer packed spine fragments.
    """

    fragment_id = safe_fragment_id(
        fragment_index,
        _spine_node_id(spine_node),
    )

    fragment_text = read_byte_range_as_text(
        file_path=file_path,
        byte_start=int(getattr(spine_node, "byte_start", 0)),
        byte_end=int(getattr(spine_node, "byte_end", 0)),
    )

    if not fragment_text.strip():
        return None

    fragment_doc_id = f"{source_doc_id}:{fragment_id}"

    sections = parse_markdown_text_to_sections(
        fragment_text,
        variant=variant,
        doc_id=fragment_doc_id,
    )

    if not sections:
        return None

    sections = _patch_sections_with_spine_context(
        sections,
        source_doc_id=source_doc_id,
        fragment_id=fragment_id,
        spine_node=spine_node,
    )

    blocks = collect_ir_blocks_from_sections(sections)

    if not blocks:
        return None

    return MarkdownParsedUnit(
        unit_id=fragment_id,
        doc_id=fragment_doc_id,
        source_doc_id=source_doc_id,
        parse_strategy="spine_fragmented",
        sections=sections,
        blocks=blocks,
        meta={
            "unit_kind": "spine_fragment",
            "source_path": str(Path(file_path).resolve()),
            "parser_variant": variant,
            "fragment_index": fragment_index,
            "fragment_id": fragment_id,
            "spine_node_id": _spine_node_id(spine_node),
            "spine_parent_id": getattr(spine_node, "parent_id", ""),
            "spine_path": _spine_path(spine_node),
            "byte_start": int(getattr(spine_node, "byte_start", 0)),
            "byte_end": int(getattr(spine_node, "byte_end", 0)),
            "line_start": int(getattr(spine_node, "line_start", 0)),
            "line_end": int(getattr(spine_node, "line_end", 0)),
            "estimated_tokens": int(getattr(spine_node, "estimated_tokens", 0)),
        },
        warnings=[],
    )


# --------------- PACKED SPINE PARSE --------------------------------

def parse_packed_spine_fragment(
    *,
    file_path: str,
    source_doc_id: str,
    packed_fragment: PackedSpineFragment,
    variant: str,
) -> MarkdownParsedUnit | None:
    """
    Parse one packed spine fragment into one parsed unit.

    This is the new execution granularity:
      many spine nodes -> one AST/enrichment cycle.
    """

    fragment_id = safe_packed_fragment_id(packed_fragment)
    fragment_doc_id = f"{source_doc_id}:{fragment_id}"

    fragment_text = read_byte_range_as_text(
        file_path=file_path,
        byte_start=packed_fragment.byte_start,
        byte_end=packed_fragment.byte_end,
    )

    if not fragment_text.strip():
        return None

    sections = parse_markdown_text_to_sections(
        fragment_text,
        variant=variant,
        doc_id=fragment_doc_id,
    )

    if not sections:
        return None

    sections = _patch_packed_sections_with_spine_context(
        sections,
        source_doc_id=source_doc_id,
        fragment_id=fragment_id,
        packed_fragment=packed_fragment,
    )

    blocks = collect_ir_blocks_from_sections(sections)

    if not blocks:
        return None

    meta = {
        "unit_kind": "packed_spine_fragment",
        "source_path": str(Path(file_path).resolve()),
        "parser_variant": variant,
        "parse_strategy": "spine_fragmented",
        "fragment_index": packed_fragment.fragment_index,
        "fragment_id": fragment_id,
        **packed_fragment.to_meta(),
        "section_count": len(sections),
        "block_count": len(blocks),
    }

    return MarkdownParsedUnit(
        unit_id=fragment_id,
        doc_id=fragment_doc_id,
        source_doc_id=source_doc_id,
        parse_strategy="spine_fragmented",
        sections=sections,
        blocks=blocks,
        meta=meta,
        warnings=[],
    )


def parse_packed_spine_fragments_sequential(
    *,
    file_path: str,
    source_doc_id: str,
    packed_fragments: list[PackedSpineFragment],
    variant: str,
) -> list[MarkdownParsedUnit]:
    """
    Parse packed spine fragments one by one.
    """

    units: list[MarkdownParsedUnit] = []

    for packed_fragment in packed_fragments or []:
        unit = parse_packed_spine_fragment(
            file_path=file_path,
            source_doc_id=source_doc_id,
            packed_fragment=packed_fragment,
            variant=variant,
        )

        if unit:
            units.append(unit)

    return units


def parse_spine_fragments_sequential(
    *,
    file_path: str,
    source_doc_id: str,
    spine_nodes: list[Any],
    variant: str,
) -> list[MarkdownParsedUnit]:
    """
    Parse spine fragments sequentially using packed execution fragments.

    Spine nodes remain structural pointers.
    Packed fragments become AST/enrichment execution units.
    """

    packed_fragments, diagnostics = pack_spine_nodes_with_diagnostics(
        spine_nodes or [],
        source_doc_id=source_doc_id,
        config=load_packed_spine_config(),
        source_path=file_path,
    )

    units = parse_packed_spine_fragments_sequential(
        file_path=file_path,
        source_doc_id=source_doc_id,
        packed_fragments=packed_fragments,
        variant=variant,
    )

    if units:
        units[0].meta["packed_spine_diagnostics"] = diagnostics.to_meta()

    return units
