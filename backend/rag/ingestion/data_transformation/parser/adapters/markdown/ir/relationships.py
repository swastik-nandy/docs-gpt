# rag/ingestion/data_transformation/parser/adapters/markdown/ir/relationships.py

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config

from .block_builder import refresh_retrieval_text


#------------------------ CONFIGURATION -------------------------

def _relationship_config():
    return load_parser_heuristics_config().markdown.relationships


def _signal_config():
    return load_parser_heuristics_config().markdown.signals


#------------------------ TEXT HELPERS -------------------------

def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        _clean_text(value),
    )


def _word_count(value: Any) -> int:
    return len(
        _normalized_text(value).split()
    )


def _looks_like_internal_period(
    text: str,
    index: int,
) -> bool:
    if text[index] != ".":
        return False

    token_start = index

    while (
        token_start > 0
        and not text[token_start - 1].isspace()
    ):
        token_start -= 1

    token = text[token_start:index + 1]

    return token.count(".") > 1


def _sentence_units(text: str) -> List[str]:
    """
    Conservatively divide text into complete sentence-like units.

    This is used only for bounding duplicated parent context. It never changes
    the source block itself.
    """

    value = _normalized_text(text)

    if not value:
        return []

    units: List[str] = []
    start = 0
    index = 0

    while index < len(value):
        char = value[index]

        if char not in {".", "!", "?"}:
            index += 1
            continue

        if (
            char == "."
            and _looks_like_internal_period(
                value,
                index,
            )
        ):
            index += 1
            continue

        end = index + 1

        while (
            end < len(value)
            and value[end] in {'"', "'", ")", "]", "}"}
        ):
            end += 1

        if (
            end < len(value)
            and not value[end].isspace()
        ):
            index += 1
            continue

        sentence = value[start:end].strip()

        if sentence:
            units.append(sentence)

        while (
            end < len(value)
            and value[end].isspace()
        ):
            end += 1

        start = end
        index = end

    tail = value[start:].strip()

    if tail:
        units.append(tail)

    return units


def _bounded_parent_text(
    text: str,
) -> str:
    config = _relationship_config()
    value = _normalized_text(text)

    if not value:
        return ""

    if not config.embed_parent_text:
        return ""

    if (
        _word_count(value)
        <= config.parent_text_max_words
    ):
        return value

    if not config.parent_text_preserve_complete_sentences:
        words = value.split()

        return " ".join(
            words[:config.parent_text_max_words]
        ).strip()

    sentences = _sentence_units(value)

    if not sentences:
        return value

    selected: List[str] = []
    selected_words = 0

    for sentence in sentences:
        sentence_words = _word_count(sentence)

        if not selected:
            selected.append(sentence)
            selected_words = sentence_words

            if (
                selected_words
                >= config.parent_text_max_words
            ):
                break

            continue

        if (
            selected_words + sentence_words
            > config.parent_text_max_words
        ):
            break

        selected.append(sentence)
        selected_words += sentence_words

    return " ".join(selected).strip()


#------------------------ BLOCK METADATA -------------------------

def _meta(
    block: Dict[str, Any],
) -> Dict[str, Any]:
    value = block.get("meta")

    if isinstance(value, dict):
        return value

    value = {}
    block["meta"] = value

    return value


def _meta_value(
    block: Dict[str, Any],
    key: str,
) -> Any:
    direct = block.get(key)

    if direct is not None and direct != "":
        return direct

    block_meta = block.get("meta")

    if isinstance(block_meta, dict):
        return block_meta.get(key)

    return None


def _local_scope_id(
    block: Dict[str, Any],
) -> str:
    return _clean_text(
        _meta_value(
            block,
            "local_scope_id",
        )
    )


def _structural_role(
    block: Dict[str, Any],
) -> str:
    return _clean_text(
        _meta_value(
            block,
            "structural_role",
        )
    ).lower()


#------------------------ RELATIONSHIP HELPERS -------------------------

def _add_relation(
    source: Dict[str, Any],
    target: Dict[str, Any],
    relation_type: str,
    confidence: float,
    reason: str,
) -> None:
    if not source or not target:
        return

    source_id = _clean_text(
        source.get("id")
    )
    target_id = _clean_text(
        target.get("id")
    )

    if not source_id or not target_id:
        return

    if source_id == target_id:
        return

    relation = {
        "type": relation_type,
        "source_block_id": source_id,
        "target_block_id": target_id,
        "confidence": confidence,
        "reason": reason,
    }

    existing = source.setdefault(
        "relationships",
        [],
    )

    for current in existing:
        if (
            current.get("type") == relation_type
            and current.get("target_block_id") == target_id
            and current.get("reason") == reason
        ):
            return

    existing.append(relation)


def _set_meta_parent(
    block: Dict[str, Any],
    parent: Dict[str, Any],
    parent_kind: str,
    overwrite: bool = False,
) -> None:
    if not block or not parent:
        return

    if block.get("id") == parent.get("id"):
        return

    block_meta = _meta(block)

    parent_text_key = f"parent_{parent_kind}"
    parent_id_key = f"parent_{parent_kind}_id"

    if (
        not overwrite
        and (
            block_meta.get(parent_text_key)
            or block_meta.get(parent_id_key)
        )
    ):
        return

    parent_id = _clean_text(
        parent.get("id")
    )

    parent_text = _bounded_parent_text(
        _clean_text(
            parent.get("text")
        )
    )

    if parent_id:
        block_meta[parent_id_key] = parent_id

    if parent_text:
        block_meta[parent_text_key] = parent_text
    else:
        block_meta.pop(
            parent_text_key,
            None,
        )


def _set_section_context_parent(
    block: Dict[str, Any],
) -> None:
    config = _relationship_config()

    if not config.attach_section_context_to_orphan_code:
        return

    block_meta = _meta(block)

    if (
        block_meta.get("parent_instruction")
        or block_meta.get("parent_instruction_id")
        or block_meta.get("parent_context")
        or block_meta.get("parent_context_id")
    ):
        return

    section_context = (
        _clean_text(
            block.get("local_context")
        )
        or _clean_text(
            block.get("heading")
        )
        or _clean_text(
            block.get("context")
        )
    )

    if not section_context:
        return

    parent_text = _bounded_parent_text(
        section_context
    )

    if parent_text:
        block_meta["parent_context"] = parent_text

    section_id = _clean_text(
        block.get("section_id")
    )

    if section_id:
        block_meta["parent_context_id"] = section_id


#------------------------ BLOCK PREDICATES -------------------------

def _is_code(
    block: Dict[str, Any],
) -> bool:
    return block.get("type") == "code"


def _is_non_code(
    block: Dict[str, Any],
) -> bool:
    return not _is_code(block)


def _is_table(
    block: Dict[str, Any],
) -> bool:
    return block.get("type") == "table"


def _is_instruction(
    block: Dict[str, Any],
) -> bool:
    return (
        block.get("function")
        == "instruction"
    )


def _is_contextual(
    block: Dict[str, Any],
) -> bool:
    return block.get("function") in {
        "explanation",
        "reference",
        "configuration",
        "mixed",
    }


def _is_command_like_code(
    block: Dict[str, Any],
) -> bool:
    if not _is_code(block):
        return False

    signals = block.get("signals", {}) or {}

    return bool(
        signals.get("is_command_like")
        or signals.get("has_flags")
    )


def _is_example_like_code(
    block: Dict[str, Any],
) -> bool:
    if not _is_code(block):
        return False

    signals = block.get("signals", {}) or {}
    function = block.get("function")

    if function in {
        "example",
        "configuration",
    }:
        return True

    return bool(
        signals.get("has_symbols")
        and not _is_command_like_code(block)
    )


def _is_short_instruction_label(
    block: Dict[str, Any],
) -> bool:
    if not _is_instruction(block):
        return False

    if _is_code(block):
        return False

    text = _clean_text(
        block.get("text")
    )

    if not text:
        return False

    signals = block.get("signals", {}) or {}
    thresholds = _signal_config().thresholds

    if (
        _word_count(text)
        > thresholds.command_max_words
    ):
        return False

    if text.endswith(":"):
        return True

    return bool(
        signals.get("is_labeled_action")
    )


#------------------------ SCOPE VALIDATION -------------------------

def _same_local_scope(
    child: Dict[str, Any],
    parent: Dict[str, Any],
) -> bool:
    config = _relationship_config()

    if not config.scope_aware:
        return True

    if config.allow_cross_local_scope_parent:
        return True

    return (
        _local_scope_id(child)
        == _local_scope_id(parent)
    )


def _is_valid_parent_candidate(
    child: Dict[str, Any],
    candidate: Dict[str, Any],
) -> bool:
    config = _relationship_config()

    if not child or not candidate:
        return False

    if _is_code(candidate):
        return False

    if child.get("id") == candidate.get("id"):
        return False

    if not _same_local_scope(
        child,
        candidate,
    ):
        return False

    if (
        not config.allow_structural_selector_as_parent
        and _structural_role(candidate) == "selector"
    ):
        return False

    if (
        not config.allow_table_row_as_code_parent
        and _is_table(candidate)
    ):
        return False

    if (
        not config.allow_mixed_short_text_as_parent
        and candidate.get("function") == "mixed"
    ):
        signals = (
            candidate.get("signals", {})
            or {}
        )

        if signals.get("is_short"):
            return False

    return True


#------------------------ CANDIDATE SEARCH -------------------------

ParentPredicate = Callable[
    [Dict[str, Any]],
    bool,
]


def _find_previous_candidate(
    blocks: List[Dict[str, Any]],
    child_index: int,
    predicate: ParentPredicate,
) -> Dict[str, Any] | None:
    config = _relationship_config()

    minimum_index = max(
        0,
        child_index
        - config.maximum_parent_distance,
    )

    for candidate_index in range(
        child_index - 1,
        minimum_index - 1,
        -1,
    ):
        candidate = blocks[candidate_index]
        child = blocks[child_index]

        if not _is_valid_parent_candidate(
            child,
            candidate,
        ):
            continue

        if predicate(candidate):
            return candidate

    return None


def _find_previous_instruction(
    blocks: List[Dict[str, Any]],
    child_index: int,
) -> Dict[str, Any] | None:
    return _find_previous_candidate(
        blocks=blocks,
        child_index=child_index,
        predicate=_is_instruction,
    )


def _find_previous_context(
    blocks: List[Dict[str, Any]],
    child_index: int,
) -> Dict[str, Any] | None:
    return _find_previous_candidate(
        blocks=blocks,
        child_index=child_index,
        predicate=_is_contextual,
    )


#------------------------ SEQUENTIAL RELATIONS -------------------------

def link_following_items(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    config = _relationship_config()

    previous: Dict[str, Any] | None = None

    for block in blocks:
        if previous:
            if (
                not config.scope_aware
                or config.allow_cross_local_scope_parent
                or _same_local_scope(
                    block,
                    previous,
                )
            ):
                _add_relation(
                    source=block,
                    target=previous,
                    relation_type="follows",
                    confidence=config.sequential_relation_confidence,
                    reason="sequential_neighbor_in_scope",
                )

        previous = block

    return blocks


#------------------------ INSTRUCTION PARENTS -------------------------

def link_code_to_previous_instruction(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    config = _relationship_config()

    for index, block in enumerate(blocks):
        if not _is_command_like_code(block):
            continue

        parent = _find_previous_instruction(
            blocks,
            index,
        )

        if not parent:
            continue

        _set_meta_parent(
            block=block,
            parent=parent,
            parent_kind="instruction",
        )

        _add_relation(
            source=block,
            target=parent,
            relation_type="implements",
            confidence=config.command_parent_confidence,
            reason="command_like_code_after_instruction_in_scope",
        )

    return blocks


def link_immediate_code_after_instruction_label(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    config = _relationship_config()

    for index, block in enumerate(blocks):
        if not _is_short_instruction_label(block):
            continue

        next_index = index + 1

        if next_index >= len(blocks):
            continue

        next_block = blocks[next_index]

        if not _is_code(next_block):
            continue

        if not _is_valid_parent_candidate(
            next_block,
            block,
        ):
            continue

        _set_meta_parent(
            block=next_block,
            parent=block,
            parent_kind="instruction",
            overwrite=True,
        )

        _add_relation(
            source=next_block,
            target=block,
            relation_type="implements",
            confidence=config.immediate_parent_confidence,
            reason="code_immediately_after_instruction_label_in_scope",
        )

    return blocks


#------------------------ CONTEXT PARENTS-------------------------

def link_code_to_previous_context(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    config = _relationship_config()

    for index, block in enumerate(blocks):
        if not _is_example_like_code(block):
            continue

        parent = _find_previous_context(
            blocks,
            index,
        )

        if not parent:
            continue

        _set_meta_parent(
            block=block,
            parent=parent,
            parent_kind="context",
        )

        _add_relation(
            source=block,
            target=parent,
            relation_type="supports",
            confidence=config.context_parent_confidence,
            reason="example_like_code_after_context_in_scope",
        )

    return blocks


#------------------------ SECTION FALLBACK-------------------------

def link_unparented_code_to_section_context(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for block in blocks:
        if not _is_code(block):
            continue

        _set_section_context_parent(
            block
        )

    return blocks


#------------------------ RETRIEVAL REFRESH -------------------------

def refresh_all_retrieval_texts(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for block in blocks:
        refresh_retrieval_text(
            block
        )

    return blocks


#------------------------ PUBLIC API -------------------------

def enrich_block_relationships(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich IR blocks after stable block identities exist.

    Parent selection is bounded and local-scope aware. A previous block cannot
    become code context merely because it happened to appear earlier in the
    same Markdown section.
    """

    config = _relationship_config()

    if not blocks:
        return blocks

    if not config.enabled:
        return refresh_all_retrieval_texts(
            blocks
        )

    blocks = link_following_items(
        blocks
    )

    blocks = link_code_to_previous_instruction(
        blocks
    )

    blocks = link_immediate_code_after_instruction_label(
        blocks
    )

    blocks = link_code_to_previous_context(
        blocks
    )

    blocks = link_unparented_code_to_section_context(
        blocks
    )

    return refresh_all_retrieval_texts(
        blocks
    )