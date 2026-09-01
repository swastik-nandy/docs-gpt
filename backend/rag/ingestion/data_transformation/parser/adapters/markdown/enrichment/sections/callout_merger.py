from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config


# --------------- CONFIG --------------------------------

_CALLOUT_CONFIG = load_parser_heuristics_config().markdown.callouts

CALLOUT_KIND_MAP = _CALLOUT_CONFIG.kind_map
MAX_CALLOUT_PARAGRAPHS = _CALLOUT_CONFIG.max_plain_callout_paragraphs


# --------------- NORMALIZATION --------------------------------

def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _label_key(text: Any) -> str:
    value = _clean_text(text)
    value = value.strip(":").strip()
    return value.lower()


def _display_label(label: Any) -> str:
    value = _clean_text(label)
    value = value.strip(":").strip()
    return value


def _known_callout_kind(label: Any) -> str:
    key = _label_key(label)
    return CALLOUT_KIND_MAP.get(key, "")


def _is_known_callout_label(label: Any) -> bool:
    return bool(_known_callout_kind(label))


# --------------- BLOCK PREDICATES --------------------------------

def _is_paragraph(block: Dict[str, Any]) -> bool:
    return bool(block and block.get("type") == "paragraph")


def _has_heavy_inline_structure(block: Dict[str, Any]) -> bool:
    """
    A standalone label should be plain and tiny.

    This avoids turning rich paragraphs, image-only blocks, or code-heavy
    blocks into callout labels.
    """

    return bool(
        block.get("inline_code")
        or block.get("images")
        or block.get("code_blocks")
    )


def _is_plain_callout_label(block: Dict[str, Any]) -> bool:
    """
    Detect standalone callout label paragraphs.

    Examples:
        Note
        Warning:
        Important

    Label vocabulary comes from TOML, not from any document/vendor.
    """

    if not _is_paragraph(block):
        return False

    if _has_heavy_inline_structure(block):
        return False

    text = _clean_text(block.get("text", ""))

    if not text:
        return False

    if len(text.split()) > 4:
        return False

    return _is_known_callout_label(text)


def _is_mergeable_body_paragraph(block: Dict[str, Any]) -> bool:
    if not _is_paragraph(block):
        return False

    if _is_plain_callout_label(block):
        return False

    text = _clean_text(block.get("text", ""))

    return bool(text)


# --------------- METADATA MERGE --------------------------------

def _copy_context_fields(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    for key in (
        "component_stack",
        "component_context",
        "component_name",
        "component_props",
        "meta",
    ):
        if source.get(key):
            target[key] = deepcopy(source[key])


def _extend_meta_list(
    target: Dict[str, Any],
    key: str,
    values: Any,
) -> None:
    if not values:
        return

    if not isinstance(values, list):
        values = [values]

    if not values:
        return

    target.setdefault(key, [])
    target[key].extend(deepcopy(values))


def _merge_body_metadata(
    target: Dict[str, Any],
    body_blocks: List[Dict[str, Any]],
) -> None:
    for block in body_blocks:
        for key in (
            "links",
            "images",
            "inline_code",
            "html_inline",
            "html_block",
            "strong_text",
        ):
            _extend_meta_list(target, key, block.get(key))

        meta = block.get("meta")

        if isinstance(meta, dict) and meta:
            target.setdefault("merged_meta", [])
            target["merged_meta"].append(deepcopy(meta))


def _copy_inline_metadata(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    for key in (
        "links",
        "images",
        "inline_code",
        "html_inline",
        "html_block",
        "strong_text",
    ):
        if source.get(key):
            target[key] = deepcopy(source[key])


# --------------- CALLOUT BUILDERS --------------------------------

def _build_callout_block(
    *,
    label: str,
    kind: str,
    body_text: str,
    merged_from: str,
    label_block: Dict[str, Any] | None = None,
    body_blocks: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    callout: Dict[str, Any] = {
        "type": "blockquote",
        "text": body_text,
        "label": label,
        "kind": kind,
        "plain_callout": True,
        "merged_from": merged_from,
    }

    if label_block:
        _copy_context_fields(callout, label_block)
        _copy_inline_metadata(callout, label_block)

    if body_blocks:
        _merge_body_metadata(callout, body_blocks)

    return callout


def _build_label_plus_body_callout(
    label_block: Dict[str, Any],
    body_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label = _display_label(label_block.get("text", ""))
    kind = _known_callout_kind(label) or "note"

    body_text = "\n\n".join(
        _clean_text(block.get("text", ""))
        for block in body_blocks
        if _clean_text(block.get("text", ""))
    ).strip()

    return _build_callout_block(
        label=label,
        kind=kind,
        body_text=body_text,
        merged_from="paragraph_label",
        label_block=label_block,
        body_blocks=body_blocks,
    )


# --------------- STRONG LABEL DETECTION --------------------------------

def _first_strong_text(block: Dict[str, Any]) -> str:
    values = block.get("strong_text", []) or []

    if not isinstance(values, list):
        return ""

    for value in values:
        text = _clean_text(value)

        if text:
            return text

    return ""


def _remove_label_prefix(text: str, label: str) -> str:
    value = _clean_text(text)
    label_value = _clean_text(label)

    if not value or not label_value:
        return value

    if not value.lower().startswith(label_value.lower()):
        return value

    return value[len(label_value):].strip(" :-–—\n\t")


def _split_strong_label_callout(
    block: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Detect structural bold-label callouts.

    Example:
        **Important**
        Body text.

    This needs no product-specific rule. It uses:
    - Markdown strong text metadata
    - TOML callout label vocabulary
    """

    if not _is_paragraph(block):
        return None

    label = _first_strong_text(block)

    if not label:
        return None

    kind = _known_callout_kind(label)

    if not kind:
        return None

    full_text = _clean_text(block.get("text", ""))
    body = _remove_label_prefix(full_text, label)

    if not body:
        return None

    if len(body.split()) < 3:
        return None

    return _build_callout_block(
        label=_display_label(label),
        kind=kind,
        body_text=body,
        merged_from="strong_label_paragraph",
        label_block=block,
        body_blocks=[],
    )


# --------------- PREFIX LABEL DETECTION --------------------------------

def _candidate_prefix_labels() -> List[str]:
    return sorted(
        CALLOUT_KIND_MAP.keys(),
        key=len,
        reverse=True,
    )


def _has_safe_label_boundary(
    text: str,
    label: str,
) -> bool:
    """
    Generic fallback for parsers that already flattened bold text.

    Accepts:
        Important This feature is deprecated.
        Warning: Do not continue.
        Good to know This works on Linux.

    Avoids common prose:
        Note that this setting...
    """

    value = _clean_text(text)
    label_value = _clean_text(label)

    if not value or not label_value:
        return False

    if not value.lower().startswith(label_value.lower()):
        return False

    body = value[len(label_value):]

    if not body:
        return False

    if body[0] in {":", "-", "–", "—"}:
        return True

    if body[0].isspace():
        body_text = body.strip()

        if not body_text:
            return False

        first_body_word = body_text.split()[0].lower().strip(".,:;!?")

        if label_value.lower() == "note" and first_body_word in {"that", "when", "if"}:
            return False

        return True

    return False


def _split_prefixed_label_callout(
    block: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Detect flattened label-prefix callouts.

    This is a fallback for cases where Markdown strong metadata was not
    available upstream.

    Still domain-agnostic:
    - labels come from TOML
    - no vendor/source/heading/file checks
    """

    if not _is_paragraph(block):
        return None

    if _has_heavy_inline_structure(block):
        return None

    text = _clean_text(block.get("text", ""))

    if not text:
        return None

    for label in _candidate_prefix_labels():
        if not _has_safe_label_boundary(text, label):
            continue

        kind = _known_callout_kind(label)

        if not kind:
            continue

        body = _remove_label_prefix(text, label)

        if not body:
            continue

        if len(body.split()) < 3:
            continue

        return _build_callout_block(
            label=_display_label(label),
            kind=kind,
            body_text=body,
            merged_from="prefixed_label_paragraph",
            label_block=block,
            body_blocks=[],
        )

    return None


# --------------- PUBLIC API --------------------------------

def merge_plain_callouts(
    content: List[Dict[str, Any]],
    max_body_paragraphs: int = MAX_CALLOUT_PARAGRAPHS,
) -> List[Dict[str, Any]]:
    """
    Merge documentation callouts represented as plain paragraphs.

    Supported generic shapes:

    1. Standalone label + following paragraph:
        Important
        Body text.

    2. Strong label paragraph:
        **Important**
        Body text.

    3. Flattened label prefix:
        Important Body text.

    The parser remains domain-agnostic because all labels come from
    parser/config/heuristics.toml.
    """

    if not content:
        return []

    merged: List[Dict[str, Any]] = []
    index = 0

    while index < len(content):
        block = content[index]

        strong_label_callout = _split_strong_label_callout(block)

        if strong_label_callout:
            merged.append(strong_label_callout)
            index += 1
            continue

        prefixed_label_callout = _split_prefixed_label_callout(block)

        if prefixed_label_callout:
            merged.append(prefixed_label_callout)
            index += 1
            continue

        if not _is_plain_callout_label(block):
            merged.append(block)
            index += 1
            continue

        body_blocks: List[Dict[str, Any]] = []
        cursor = index + 1

        while cursor < len(content) and len(body_blocks) < max_body_paragraphs:
            candidate = content[cursor]

            if not _is_mergeable_body_paragraph(candidate):
                break

            body_blocks.append(candidate)
            cursor += 1

        if not body_blocks:
            merged.append(block)
            index += 1
            continue

        merged.append(
            _build_label_plus_body_callout(
                label_block=block,
                body_blocks=body_blocks,
            )
        )

        index = cursor

    return merged