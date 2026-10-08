# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/sections/structural_artifacts.py

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from typing import Any, Mapping

from ingestion.data_transformation.parser.markdown.config import get_markdown_config


#------------------------anchor pattern-------------------------

ANCHOR_ONLY_RE = re.compile(
    r"""
    ^\s*
    <a
      \s+
      (?P<attribute>name|id)
      \s*=\s*
      ["'](?P<value>[^"']+)["']
      \s*
    >
    \s*
    </a>
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


#------------------------text helpers-------------------------

def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    text = _clean_text(value)
    return re.sub(r"\s+", " ", text)


def _word_count(value: Any) -> int:
    return len(_normalize_text(value).split())


def _block_type(block: Mapping[str, Any]) -> str:
    block_type = _clean_text(block.get("type")).lower()

    if block_type == "table":
        return "generic_table"

    return block_type


def _structural_role(block: Mapping[str, Any]) -> str:
    return _clean_text(block.get("structural_role")).lower()


#------------------------standalone anchors-------------------------

def clean_anchor(value: Any) -> str:
    return _clean_text(value)


def extract_standalone_anchor(text: Any) -> str:
    value = _clean_text(text)
    match = ANCHOR_ONLY_RE.match(value)

    if not match:
        return ""

    return clean_anchor(match.group("value"))


def is_standalone_anchor_block(
    block: Mapping[str, Any],
) -> bool:
    if not block:
        return False

    if _block_type(block) != "paragraph":
        return False

    return bool(
        extract_standalone_anchor(
            block.get("text", ""),
        )
    )


def _strip_standalone_anchors(
    section: dict[str, Any],
) -> dict[str, Any]:
    content = section.get("content", []) or []

    if not content:
        return section

    anchors: list[str] = []
    clean_content: list[dict[str, Any]] = []

    for raw_block in content:
        if not isinstance(raw_block, Mapping):
            continue

        block = deepcopy(dict(raw_block))
        anchor = extract_standalone_anchor(
            block.get("text", ""),
        )

        if anchor and is_standalone_anchor_block(block):
            if anchor not in anchors:
                anchors.append(anchor)

            continue

        clean_content.append(block)

    if not anchors:
        section["content"] = clean_content
        return section

    existing_anchors = list(
        section.get("anchors", []) or []
    )

    for anchor in anchors:
        if anchor not in existing_anchors:
            existing_anchors.append(anchor)

    section["anchors"] = existing_anchors
    section["content"] = clean_content

    if (
        existing_anchors
        and not section.get("heading_anchor")
    ):
        section["heading_anchor"] = existing_anchors[0]

    return section


#------------------------local scope detection-------------------------

def _strong_text_values(
    block: Mapping[str, Any],
) -> list[str]:
    raw_values = block.get("strong_text", [])

    if not isinstance(raw_values, list):
        return []

    values: list[str] = []

    for raw_value in raw_values:
        value = _normalize_text(raw_value)

        if value:
            values.append(value)

    return values


def _is_strong_only_paragraph(
    block: Mapping[str, Any],
) -> bool:
    if _block_type(block) != "paragraph":
        return False

    text = _normalize_text(block.get("text"))
    strong_values = _strong_text_values(block)

    if not text or not strong_values:
        return False

    combined_strong_text = _normalize_text(
        " ".join(strong_values)
    )

    return combined_strong_text == text


def _has_rejected_terminal_punctuation(
    text: str,
) -> bool:
    config = (
        get_markdown_config()
        .local_scopes
    )

    if not text:
        return True

    terminal = text[-1]

    if terminal == ":" and config.allow_terminal_colon:
        return False

    if (
        terminal == "."
        and config.reject_sentence_terminal_period
    ):
        return True

    if (
        terminal == "?"
        and config.reject_sentence_terminal_question
    ):
        return True

    if (
        terminal == "!"
        and config.reject_sentence_terminal_exclamation
    ):
        return True

    return False


def _has_structural_followers(
    content: list[dict[str, Any]],
    anchor_index: int,
) -> bool:
    config = (
        get_markdown_config()
        .local_scopes
    )

    follower_count = 0
    lookahead_end = min(
        len(content),
        anchor_index + config.max_lookahead_blocks + 1,
    )

    for index in range(
        anchor_index + 1,
        lookahead_end,
    ):
        follower = content[index]

        if not isinstance(follower, Mapping):
            continue

        if (
            _block_type(follower)
            in config.structural_follower_types
        ):
            follower_count += 1

        if (
            follower_count
            >= config.minimum_structural_followers
        ):
            return True

    return False


def _is_local_scope_anchor(
    content: list[dict[str, Any]],
    index: int,
) -> bool:
    config = (
        get_markdown_config()
        .local_scopes
    )

    if not config.enabled:
        return False

    if not config.promote_strong_only_paragraphs:
        return False

    block = content[index]

    if not _is_strong_only_paragraph(block):
        return False

    text = _normalize_text(block.get("text"))

    if _word_count(text) > config.max_anchor_words:
        return False

    if len(text) > config.max_anchor_chars:
        return False

    if _has_rejected_terminal_punctuation(text):
        return False

    return _has_structural_followers(
        content,
        index,
    )


def _section_scope_seed(
    section: Mapping[str, Any],
    anchor_index: int,
    anchor_text: str,
) -> str:
    path = section.get("path", [])

    if isinstance(path, list):
        path_text = " / ".join(
            _normalize_text(item)
            for item in path
            if _normalize_text(item)
        )
    else:
        path_text = _normalize_text(path)

    heading = _normalize_text(
        section.get("heading", "")
    )

    return "|".join(
        (
            path_text,
            heading,
            str(anchor_index),
            anchor_text,
        )
    )


def _build_local_scope_id(
    section: Mapping[str, Any],
    anchor_index: int,
    anchor_text: str,
) -> str:
    seed = _section_scope_seed(
        section,
        anchor_index,
        anchor_text,
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            seed,
        )
    )


def _apply_local_scopes(
    section: dict[str, Any],
) -> dict[str, Any]:
    config = (
        get_markdown_config()
        .local_scopes
    )

    if not config.enabled:
        return section

    raw_content = section.get("content", []) or []

    content = [
        deepcopy(dict(block))
        for block in raw_content
        if isinstance(block, Mapping)
    ]

    if not content:
        section["content"] = []
        return section

    anchor_indexes = {
        index
        for index in range(len(content))
        if _is_local_scope_anchor(content, index)
    }

    if not anchor_indexes:
        section["content"] = content
        return section

    scoped_content: list[dict[str, Any]] = []
    local_scopes: list[dict[str, Any]] = []

    active_scope_id = ""
    active_scope_title = ""

    for index, block in enumerate(content):
        if index in anchor_indexes:
            anchor_text = _normalize_text(
                block.get("text")
            )

            active_scope_id = _build_local_scope_id(
                section,
                index,
                anchor_text,
            )
            active_scope_title = anchor_text

            scope_meta: dict[str, Any] = {
                "scope_id": active_scope_id,
                "title": active_scope_title,
                "anchor_index": index,
            }

            if block.get("source"):
                scope_meta["source"] = deepcopy(
                    block["source"]
                )

            local_scopes.append(scope_meta)

            if config.remove_promoted_anchor_from_retrieval:
                continue

            block["structural_role"] = "local_anchor"
            block["local_scope_id"] = active_scope_id
            block["local_scope_title"] = active_scope_title

            scoped_content.append(block)
            continue

        if active_scope_id:
            block["local_scope_id"] = active_scope_id
            block["local_scope_title"] = active_scope_title

        scoped_content.append(block)

    section["content"] = scoped_content
    section["local_scopes"] = local_scopes

    return section


#------------------------selector detection-------------------------

def _is_sentence_like_selector_candidate(
    text: str,
) -> bool:
    if not text:
        return True

    return text[-1] in {".", "?", "!", ";"}


def _following_selector_targets(
    content: list[dict[str, Any]],
    selector_index: int,
) -> list[int]:
    config = (
        get_markdown_config()
        .selector_blocks
    )

    target_indexes: list[int] = []
    target_type = ""

    lookahead_end = min(
        len(content),
        selector_index + config.max_lookahead_blocks + 1,
    )

    for index in range(
        selector_index + 1,
        lookahead_end,
    ):
        block = content[index]
        block_type = _block_type(block)

        if block_type not in config.eligible_following_types:
            break

        if not target_type:
            target_type = block_type

        if (
            config.require_repeated_following_type
            and block_type != target_type
        ):
            break

        target_indexes.append(index)

    if (
        len(target_indexes)
        < config.minimum_following_blocks
    ):
        return []

    return target_indexes


def _is_selector_candidate(
    content: list[dict[str, Any]],
    index: int,
) -> tuple[bool, list[int]]:
    config = (
        get_markdown_config()
        .selector_blocks
    )

    if not config.enabled:
        return False, []

    block = content[index]

    if _block_type(block) != "paragraph":
        return False, []

    if _structural_role(block) == "local_anchor":
        return False, []

    text = _normalize_text(block.get("text"))

    if not text:
        return False, []

    if config.require_short_text:
        if _word_count(text) > config.max_words:
            return False, []

        if len(text) > config.max_chars:
            return False, []

    if (
        config.require_non_sentence_like
        and _is_sentence_like_selector_candidate(text)
    ):
        return False, []

    target_indexes = _following_selector_targets(
        content,
        index,
    )

    if not target_indexes:
        return False, []

    return True, target_indexes


def _selector_metadata(
    block: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "text": _normalize_text(
            block.get("text")
        ),
        "structural_role": "selector",
    }

    for key in (
        "source",
        "links",
        "inline_code",
        "strong_text",
        "local_scope_id",
        "local_scope_title",
    ):
        value = block.get(key)

        if value:
            metadata[key] = deepcopy(value)

    return metadata


def _merge_selector_metadata(
    block: dict[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    existing = block.get("selector_metadata")

    if isinstance(existing, list):
        selector_values = deepcopy(existing)
    elif isinstance(existing, Mapping):
        selector_values = [deepcopy(dict(existing))]
    else:
        selector_values = []

    normalized_metadata = deepcopy(dict(metadata))

    if normalized_metadata not in selector_values:
        selector_values.append(normalized_metadata)

    block["selector_metadata"] = selector_values


def _apply_selector_blocks(
    section: dict[str, Any],
) -> dict[str, Any]:
    config = (
        get_markdown_config()
        .selector_blocks
    )

    if not config.enabled:
        return section

    raw_content = section.get("content", []) or []

    content = [
        deepcopy(dict(block))
        for block in raw_content
        if isinstance(block, Mapping)
    ]

    if not content:
        section["content"] = []
        return section

    selector_indexes: set[int] = set()
    artifacts: list[dict[str, Any]] = []

    for index in range(len(content)):
        is_selector, target_indexes = (
            _is_selector_candidate(
                content,
                index,
            )
        )

        if not is_selector:
            continue

        selector = content[index]
        metadata = _selector_metadata(selector)

        if config.attach_selector_metadata_to_following_blocks:
            for target_index in target_indexes:
                _merge_selector_metadata(
                    content[target_index],
                    metadata,
                )

        selector["structural_role"] = "selector"

        artifacts.append(
            {
                "type": "selector",
                "content_index": index,
                "target_indexes": target_indexes,
                "metadata": metadata,
            }
        )

        if config.remove_selector_from_retrieval:
            selector_indexes.add(index)

    section["content"] = [
        block
        for index, block in enumerate(content)
        if index not in selector_indexes
    ]

    if artifacts:
        existing_artifacts = list(
            section.get("structural_artifacts", []) or []
        )
        existing_artifacts.extend(artifacts)
        section["structural_artifacts"] = existing_artifacts

    return section


#------------------------public api-------------------------

def attach_anchors_and_remove_noise_blocks(
    section: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply domain-neutral structural cleanup to one section.

    The pass uses parser structure, inline formatting and neighboring block
    shapes. It does not match vendor names or document-specific phrases.
    """

    if not section:
        return section

    patched = deepcopy(section)
    patched = _strip_standalone_anchors(patched)
    patched = _apply_local_scopes(patched)
    patched = _apply_selector_blocks(patched)

    return patched