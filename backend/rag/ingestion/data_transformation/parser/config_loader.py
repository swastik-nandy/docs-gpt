from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rtoml


CONFIG_FILE_NAME = "heuristics.toml"
CONFIG_ROOT_KEY = "parser_heuristics"


# ---------------------------------------------------------------------------
# TOP-LEVEL FILE TYPE CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParserFileTypeConfig:
    enabled_families: frozenset[str]
    markdown_variants: dict[str, str]
    known_extensions: dict[str, tuple[str, ...]]


# ---------------------------------------------------------------------------
# EXISTING MARKDOWN CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownMdxMarkerConfig:
    prefix: str
    open: str
    close: str
    self: str


@dataclass(frozen=True)
class MarkdownCalloutConfig:
    kind_map: dict[str, str]
    max_plain_callout_paragraphs: int


@dataclass(frozen=True)
class MarkdownSignalThresholdConfig:
    short_text_max_words: int
    long_text_min_words: int
    command_max_words: int
    label_max_words: int
    label_max_chars: int


@dataclass(frozen=True)
class MarkdownSignalConfig:
    thresholds: MarkdownSignalThresholdConfig
    common_sentence_starters: frozenset[str]
    action_starters: frozenset[str]
    directive_starters: frozenset[str]


@dataclass(frozen=True)
class MarkdownMetadataConfig:
    keyword_stopwords: frozenset[str]
    placeholder_headings: frozenset[str]


# ---------------------------------------------------------------------------
# SOURCE PROVENANCE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownSourceProvenanceConfig:
    enabled: bool
    preserve_line_spans: bool
    preserve_block_index: bool
    preserve_original_text: bool
    line_numbers_zero_based: bool
    line_end_exclusive: bool


# ---------------------------------------------------------------------------
# LOCAL SEMANTIC SCOPES
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownLocalScopeConfig:
    enabled: bool

    promote_strong_only_paragraphs: bool
    remove_promoted_anchor_from_retrieval: bool

    max_anchor_words: int
    max_anchor_chars: int
    max_lookahead_blocks: int
    minimum_structural_followers: int

    reject_sentence_terminal_period: bool
    reject_sentence_terminal_question: bool
    reject_sentence_terminal_exclamation: bool
    allow_terminal_colon: bool

    structural_follower_types: frozenset[str]


# ---------------------------------------------------------------------------
# SELECTOR-LIKE STRUCTURAL BLOCKS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownSelectorBlockConfig:
    enabled: bool

    max_words: int
    max_chars: int
    max_lookahead_blocks: int
    minimum_following_blocks: int

    require_short_text: bool
    require_non_sentence_like: bool
    require_repeated_following_type: bool

    eligible_following_types: frozenset[str]

    remove_selector_from_retrieval: bool
    attach_selector_metadata_to_following_blocks: bool


# ---------------------------------------------------------------------------
# SEMANTIC PACKING
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownSemanticPackingConfig:
    enabled: bool

    preserve_paragraph_boundaries: bool
    paragraph_separator: str

    merge_only_adjacent_blocks: bool
    merge_only_same_section: bool
    merge_only_same_local_scope: bool
    merge_only_same_component_context: bool

    minimum_target_words: int
    preferred_target_words: int
    maximum_target_words: int
    maximum_blocks_per_group: int

    allow_single_oversized_block: bool
    flush_before_structural_block: bool
    flush_after_structural_block: bool

    eligible_block_types: frozenset[str]
    boundary_block_types: frozenset[str]

    lexical_dependency_detection_enabled: bool


# ---------------------------------------------------------------------------
# RELATIONSHIPS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownRelationshipConfig:
    enabled: bool

    scope_aware: bool
    maximum_parent_distance: int

    allow_cross_local_scope_parent: bool
    allow_structural_selector_as_parent: bool
    allow_table_row_as_code_parent: bool
    allow_mixed_short_text_as_parent: bool

    attach_section_context_to_orphan_code: bool

    embed_parent_text: bool
    parent_text_max_words: int
    parent_text_preserve_complete_sentences: bool

    sequential_relation_confidence: float
    command_parent_confidence: float
    immediate_parent_confidence: float
    context_parent_confidence: float
    section_fallback_confidence: float


# ---------------------------------------------------------------------------
# RETRIEVAL TEXT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownRetrievalTextConfig:
    include_context: bool
    include_heading: bool
    include_local_context: bool
    include_function: bool
    include_parent_instruction: bool
    include_parent_context: bool
    include_text: bool

    deduplicate_equal_context_fields: bool
    preserve_content_paragraphs: bool


# ---------------------------------------------------------------------------
# COMPOSITE CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkdownHeuristicsConfig:
    mdx_markers: MarkdownMdxMarkerConfig
    heading_permalink_texts: frozenset[str]
    callouts: MarkdownCalloutConfig
    signals: MarkdownSignalConfig
    metadata: MarkdownMetadataConfig

    source_provenance: MarkdownSourceProvenanceConfig
    local_scopes: MarkdownLocalScopeConfig
    selector_blocks: MarkdownSelectorBlockConfig
    semantic_packing: MarkdownSemanticPackingConfig
    relationships: MarkdownRelationshipConfig
    retrieval_text: MarkdownRetrievalTextConfig


@dataclass(frozen=True)
class ParserHeuristicsConfig:
    file_types: ParserFileTypeConfig
    markdown: MarkdownHeuristicsConfig


# ---------------------------------------------------------------------------
# RAW CONFIG LOADING
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / CONFIG_FILE_NAME


def _load_toml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Parser heuristics config not found: {path}")

    try:
        data = rtoml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Failed to parse parser heuristics TOML: {path} | {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise ValueError(
            f"Parser heuristics config must contain a TOML mapping: {path}"
        )

    root = data.get(CONFIG_ROOT_KEY)

    if not isinstance(root, Mapping):
        raise ValueError(
            f"Parser heuristics config is missing root key "
            f"{CONFIG_ROOT_KEY!r}: {path}"
        )

    return root


# ---------------------------------------------------------------------------
# REQUIRED VALUE HELPERS
# ---------------------------------------------------------------------------


def _require_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Mapping[str, Any]:
    value = data.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {path}.{key}")

    return value


def _require_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean at {path}.{key}")

    return value


def _require_string(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str:
    value = data.get(key)

    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}.{key}")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"Expected non-empty string at {path}.{key}")

    return cleaned


def _require_preserved_string(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str:
    """
    Read a string without stripping meaningful whitespace.

    This is required for values such as paragraph separators, where "\\n\\n"
    must remain exactly "\\n\\n".
    """
    value = data.get(key)

    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}.{key}")

    if not value:
        raise ValueError(f"Expected non-empty string at {path}.{key}")

    return value


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    min_value: int | None = None,
) -> int:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer at {path}.{key}")

    if min_value is not None and value < min_value:
        raise ValueError(
            f"Expected {path}.{key} >= {min_value}, got {value}"
        )

    return value


def _require_float(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = data.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected number at {path}.{key}")

    parsed = float(value)

    if min_value is not None and parsed < min_value:
        raise ValueError(
            f"Expected {path}.{key} >= {min_value}, got {parsed}"
        )

    if max_value is not None and parsed > max_value:
        raise ValueError(
            f"Expected {path}.{key} <= {max_value}, got {parsed}"
        )

    return parsed


def _require_string_list(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    allow_empty_items: bool = False,
) -> tuple[str, ...]:
    value = data.get(key)

    if not isinstance(value, list):
        raise ValueError(f"Expected list at {path}.{key}")

    result: list[str] = []

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"Expected string at {path}.{key}[{index}]"
            )

        cleaned = item.strip()

        if not cleaned and not allow_empty_items:
            raise ValueError(
                f"Expected non-empty string at {path}.{key}[{index}]"
            )

        result.append(cleaned)

    return tuple(result)


def _require_string_set(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    lowercase: bool = False,
    allow_empty_items: bool = False,
) -> frozenset[str]:
    values = _require_string_list(
        data,
        key,
        path=path,
        allow_empty_items=allow_empty_items,
    )

    if lowercase:
        return frozenset(value.lower() for value in values)

    return frozenset(values)


# ---------------------------------------------------------------------------
# MAPPING HELPERS
# ---------------------------------------------------------------------------


def _parse_string_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
    lowercase_keys: bool = False,
    lowercase_values: bool = False,
) -> dict[str, str]:
    mapping_path = f"{path}.{key}"
    raw = _require_mapping(data, key, path=path)

    parsed: dict[str, str] = {}

    for raw_key, raw_value in raw.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                f"Expected string key inside {mapping_path}"
            )

        if not isinstance(raw_value, str):
            raise ValueError(
                f"Expected string value at {mapping_path}.{raw_key}"
            )

        clean_key = raw_key.strip()
        clean_value = raw_value.strip()

        if not clean_key:
            raise ValueError(
                f"Expected non-empty key inside {mapping_path}"
            )

        if not clean_value:
            raise ValueError(
                f"Expected non-empty value at {mapping_path}.{raw_key}"
            )

        if lowercase_keys:
            clean_key = clean_key.lower()

        if lowercase_values:
            clean_value = clean_value.lower()

        parsed[clean_key] = clean_value

    return parsed


def _parse_extension_groups(
    data: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> dict[str, tuple[str, ...]]:
    mapping_path = f"{path}.{key}"
    raw = _require_mapping(data, key, path=path)

    parsed: dict[str, tuple[str, ...]] = {}

    for family, extensions in raw.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError(
                f"Expected non-empty family name inside {mapping_path}"
            )

        if not isinstance(extensions, list):
            raise ValueError(
                f"Expected list at {mapping_path}.{family}"
            )

        clean_extensions: list[str] = []

        for index, extension in enumerate(extensions):
            if not isinstance(extension, str):
                raise ValueError(
                    f"Expected string at "
                    f"{mapping_path}.{family}[{index}]"
                )

            value = extension.strip().lower()

            if not value:
                raise ValueError(
                    f"Expected non-empty extension at "
                    f"{mapping_path}.{family}[{index}]"
                )

            if not value.startswith("."):
                raise ValueError(
                    f"Extension must start with '.': "
                    f"{mapping_path}.{family}[{index}]={value!r}"
                )

            clean_extensions.append(value)

        parsed[family.strip().lower()] = tuple(clean_extensions)

    return parsed


# ---------------------------------------------------------------------------
# FILE TYPE PARSING
# ---------------------------------------------------------------------------


def _parse_file_types(
    root: Mapping[str, Any],
) -> ParserFileTypeConfig:
    path = CONFIG_ROOT_KEY
    file_types = _require_mapping(root, "file_types", path=path)
    file_types_path = f"{path}.file_types"

    return ParserFileTypeConfig(
        enabled_families=_require_string_set(
            file_types,
            "enabled_families",
            path=file_types_path,
            lowercase=True,
        ),
        markdown_variants=_parse_string_mapping(
            file_types,
            "markdown_variants",
            path=file_types_path,
            lowercase_keys=True,
            lowercase_values=True,
        ),
        known_extensions=_parse_extension_groups(
            file_types,
            "known_extensions",
            path=file_types_path,
        ),
    )


# ---------------------------------------------------------------------------
# EXISTING MARKDOWN PARSING
# ---------------------------------------------------------------------------


def _parse_mdx_markers(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownMdxMarkerConfig:
    markers = _require_mapping(
        markdown,
        "mdx_markers",
        path=path,
    )
    markers_path = f"{path}.mdx_markers"

    return MarkdownMdxMarkerConfig(
        prefix=_require_string(
            markers,
            "prefix",
            path=markers_path,
        ),
        open=_require_string(
            markers,
            "open",
            path=markers_path,
        ),
        close=_require_string(
            markers,
            "close",
            path=markers_path,
        ),
        self=_require_string(
            markers,
            "self",
            path=markers_path,
        ),
    )


def _parse_heading_permalink_texts(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> frozenset[str]:
    heading = _require_mapping(
        markdown,
        "heading_permalink",
        path=path,
    )
    heading_path = f"{path}.heading_permalink"

    return _require_string_set(
        heading,
        "texts",
        path=heading_path,
        lowercase=True,
        allow_empty_items=True,
    )


def _parse_callouts(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownCalloutConfig:
    callouts = _require_mapping(
        markdown,
        "callouts",
        path=path,
    )
    callouts_path = f"{path}.callouts"

    return MarkdownCalloutConfig(
        kind_map=_parse_string_mapping(
            callouts,
            "kind_map",
            path=callouts_path,
            lowercase_keys=True,
            lowercase_values=True,
        ),
        max_plain_callout_paragraphs=_require_int(
            callouts,
            "max_plain_callout_paragraphs",
            path=callouts_path,
            min_value=1,
        ),
    )


def _parse_signal_thresholds(
    signals: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownSignalThresholdConfig:
    thresholds = _require_mapping(
        signals,
        "thresholds",
        path=path,
    )
    thresholds_path = f"{path}.thresholds"

    parsed = MarkdownSignalThresholdConfig(
        short_text_max_words=_require_int(
            thresholds,
            "short_text_max_words",
            path=thresholds_path,
            min_value=1,
        ),
        long_text_min_words=_require_int(
            thresholds,
            "long_text_min_words",
            path=thresholds_path,
            min_value=1,
        ),
        command_max_words=_require_int(
            thresholds,
            "command_max_words",
            path=thresholds_path,
            min_value=1,
        ),
        label_max_words=_require_int(
            thresholds,
            "label_max_words",
            path=thresholds_path,
            min_value=1,
        ),
        label_max_chars=_require_int(
            thresholds,
            "label_max_chars",
            path=thresholds_path,
            min_value=1,
        ),
    )

    if parsed.short_text_max_words >= parsed.long_text_min_words:
        raise ValueError(
            f"{thresholds_path}.short_text_max_words must be lower than "
            f"{thresholds_path}.long_text_min_words"
        )

    return parsed


def _parse_signals(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownSignalConfig:
    signals = _require_mapping(
        markdown,
        "signals",
        path=path,
    )
    signals_path = f"{path}.signals"

    common_sentence = _require_mapping(
        signals,
        "common_sentence_starters",
        path=signals_path,
    )
    actions = _require_mapping(
        signals,
        "action_starters",
        path=signals_path,
    )
    directives = _require_mapping(
        signals,
        "directive_starters",
        path=signals_path,
    )

    return MarkdownSignalConfig(
        thresholds=_parse_signal_thresholds(
            signals,
            path=signals_path,
        ),
        common_sentence_starters=_require_string_set(
            common_sentence,
            "words",
            path=f"{signals_path}.common_sentence_starters",
            lowercase=True,
        ),
        action_starters=_require_string_set(
            actions,
            "words",
            path=f"{signals_path}.action_starters",
            lowercase=True,
        ),
        directive_starters=_require_string_set(
            directives,
            "phrases",
            path=f"{signals_path}.directive_starters",
            lowercase=True,
        ),
    )


def _parse_metadata(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownMetadataConfig:
    metadata = _require_mapping(
        markdown,
        "metadata",
        path=path,
    )
    metadata_path = f"{path}.metadata"

    return MarkdownMetadataConfig(
        keyword_stopwords=_require_string_set(
            metadata,
            "keyword_stopwords",
            path=metadata_path,
            lowercase=True,
        ),
        placeholder_headings=_require_string_set(
            metadata,
            "placeholder_headings",
            path=metadata_path,
            lowercase=True,
            allow_empty_items=True,
        ),
    )


# ---------------------------------------------------------------------------
# SOURCE PROVENANCE PARSING
# ---------------------------------------------------------------------------


def _parse_source_provenance(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownSourceProvenanceConfig:
    source = _require_mapping(
        markdown,
        "source_provenance",
        path=path,
    )
    source_path = f"{path}.source_provenance"

    return MarkdownSourceProvenanceConfig(
        enabled=_require_bool(
            source,
            "enabled",
            path=source_path,
        ),
        preserve_line_spans=_require_bool(
            source,
            "preserve_line_spans",
            path=source_path,
        ),
        preserve_block_index=_require_bool(
            source,
            "preserve_block_index",
            path=source_path,
        ),
        preserve_original_text=_require_bool(
            source,
            "preserve_original_text",
            path=source_path,
        ),
        line_numbers_zero_based=_require_bool(
            source,
            "line_numbers_zero_based",
            path=source_path,
        ),
        line_end_exclusive=_require_bool(
            source,
            "line_end_exclusive",
            path=source_path,
        ),
    )


# ---------------------------------------------------------------------------
# LOCAL SCOPE PARSING
# ---------------------------------------------------------------------------


def _parse_local_scopes(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownLocalScopeConfig:
    local_scopes = _require_mapping(
        markdown,
        "local_scopes",
        path=path,
    )
    local_path = f"{path}.local_scopes"

    return MarkdownLocalScopeConfig(
        enabled=_require_bool(
            local_scopes,
            "enabled",
            path=local_path,
        ),
        promote_strong_only_paragraphs=_require_bool(
            local_scopes,
            "promote_strong_only_paragraphs",
            path=local_path,
        ),
        remove_promoted_anchor_from_retrieval=_require_bool(
            local_scopes,
            "remove_promoted_anchor_from_retrieval",
            path=local_path,
        ),
        max_anchor_words=_require_int(
            local_scopes,
            "max_anchor_words",
            path=local_path,
            min_value=1,
        ),
        max_anchor_chars=_require_int(
            local_scopes,
            "max_anchor_chars",
            path=local_path,
            min_value=1,
        ),
        max_lookahead_blocks=_require_int(
            local_scopes,
            "max_lookahead_blocks",
            path=local_path,
            min_value=1,
        ),
        minimum_structural_followers=_require_int(
            local_scopes,
            "minimum_structural_followers",
            path=local_path,
            min_value=1,
        ),
        reject_sentence_terminal_period=_require_bool(
            local_scopes,
            "reject_sentence_terminal_period",
            path=local_path,
        ),
        reject_sentence_terminal_question=_require_bool(
            local_scopes,
            "reject_sentence_terminal_question",
            path=local_path,
        ),
        reject_sentence_terminal_exclamation=_require_bool(
            local_scopes,
            "reject_sentence_terminal_exclamation",
            path=local_path,
        ),
        allow_terminal_colon=_require_bool(
            local_scopes,
            "allow_terminal_colon",
            path=local_path,
        ),
        structural_follower_types=_require_string_set(
            local_scopes,
            "structural_follower_types",
            path=local_path,
            lowercase=True,
        ),
    )


# ---------------------------------------------------------------------------
# SELECTOR BLOCK PARSING
# ---------------------------------------------------------------------------


def _parse_selector_blocks(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownSelectorBlockConfig:
    selector = _require_mapping(
        markdown,
        "selector_blocks",
        path=path,
    )
    selector_path = f"{path}.selector_blocks"

    return MarkdownSelectorBlockConfig(
        enabled=_require_bool(
            selector,
            "enabled",
            path=selector_path,
        ),
        max_words=_require_int(
            selector,
            "max_words",
            path=selector_path,
            min_value=1,
        ),
        max_chars=_require_int(
            selector,
            "max_chars",
            path=selector_path,
            min_value=1,
        ),
        max_lookahead_blocks=_require_int(
            selector,
            "max_lookahead_blocks",
            path=selector_path,
            min_value=1,
        ),
        minimum_following_blocks=_require_int(
            selector,
            "minimum_following_blocks",
            path=selector_path,
            min_value=1,
        ),
        require_short_text=_require_bool(
            selector,
            "require_short_text",
            path=selector_path,
        ),
        require_non_sentence_like=_require_bool(
            selector,
            "require_non_sentence_like",
            path=selector_path,
        ),
        require_repeated_following_type=_require_bool(
            selector,
            "require_repeated_following_type",
            path=selector_path,
        ),
        eligible_following_types=_require_string_set(
            selector,
            "eligible_following_types",
            path=selector_path,
            lowercase=True,
        ),
        remove_selector_from_retrieval=_require_bool(
            selector,
            "remove_selector_from_retrieval",
            path=selector_path,
        ),
        attach_selector_metadata_to_following_blocks=_require_bool(
            selector,
            "attach_selector_metadata_to_following_blocks",
            path=selector_path,
        ),
    )


# ---------------------------------------------------------------------------
# SEMANTIC PACKING PARSING
# ---------------------------------------------------------------------------


def _parse_semantic_packing(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownSemanticPackingConfig:
    packing = _require_mapping(
        markdown,
        "semantic_packing",
        path=path,
    )
    packing_path = f"{path}.semantic_packing"

    parsed = MarkdownSemanticPackingConfig(
        enabled=_require_bool(
            packing,
            "enabled",
            path=packing_path,
        ),
        preserve_paragraph_boundaries=_require_bool(
            packing,
            "preserve_paragraph_boundaries",
            path=packing_path,
        ),
        paragraph_separator=_require_preserved_string(
            packing,
            "paragraph_separator",
            path=packing_path,
        ),
        merge_only_adjacent_blocks=_require_bool(
            packing,
            "merge_only_adjacent_blocks",
            path=packing_path,
        ),
        merge_only_same_section=_require_bool(
            packing,
            "merge_only_same_section",
            path=packing_path,
        ),
        merge_only_same_local_scope=_require_bool(
            packing,
            "merge_only_same_local_scope",
            path=packing_path,
        ),
        merge_only_same_component_context=_require_bool(
            packing,
            "merge_only_same_component_context",
            path=packing_path,
        ),
        minimum_target_words=_require_int(
            packing,
            "minimum_target_words",
            path=packing_path,
            min_value=1,
        ),
        preferred_target_words=_require_int(
            packing,
            "preferred_target_words",
            path=packing_path,
            min_value=1,
        ),
        maximum_target_words=_require_int(
            packing,
            "maximum_target_words",
            path=packing_path,
            min_value=1,
        ),
        maximum_blocks_per_group=_require_int(
            packing,
            "maximum_blocks_per_group",
            path=packing_path,
            min_value=1,
        ),
        allow_single_oversized_block=_require_bool(
            packing,
            "allow_single_oversized_block",
            path=packing_path,
        ),
        flush_before_structural_block=_require_bool(
            packing,
            "flush_before_structural_block",
            path=packing_path,
        ),
        flush_after_structural_block=_require_bool(
            packing,
            "flush_after_structural_block",
            path=packing_path,
        ),
        eligible_block_types=_require_string_set(
            packing,
            "eligible_block_types",
            path=packing_path,
            lowercase=True,
        ),
        boundary_block_types=_require_string_set(
            packing,
            "boundary_block_types",
            path=packing_path,
            lowercase=True,
        ),
        lexical_dependency_detection_enabled=_require_bool(
            packing,
            "lexical_dependency_detection_enabled",
            path=packing_path,
        ),
    )

    if not (
        parsed.minimum_target_words
        <= parsed.preferred_target_words
        <= parsed.maximum_target_words
    ):
        raise ValueError(
            f"Expected {packing_path} word budgets to satisfy "
            f"minimum_target_words <= preferred_target_words "
            f"<= maximum_target_words"
        )

    overlap = (
        parsed.eligible_block_types
        & parsed.boundary_block_types
    )

    if overlap:
        raise ValueError(
            f"{packing_path}.eligible_block_types and "
            f"{packing_path}.boundary_block_types overlap: "
            f"{sorted(overlap)}"
        )

    return parsed


# ---------------------------------------------------------------------------
# RELATIONSHIP PARSING
# ---------------------------------------------------------------------------


def _parse_relationships(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownRelationshipConfig:
    relationships = _require_mapping(
        markdown,
        "relationships",
        path=path,
    )
    relationships_path = f"{path}.relationships"

    confidence_args = {
        "path": relationships_path,
        "min_value": 0.0,
        "max_value": 1.0,
    }

    return MarkdownRelationshipConfig(
        enabled=_require_bool(
            relationships,
            "enabled",
            path=relationships_path,
        ),
        scope_aware=_require_bool(
            relationships,
            "scope_aware",
            path=relationships_path,
        ),
        maximum_parent_distance=_require_int(
            relationships,
            "maximum_parent_distance",
            path=relationships_path,
            min_value=1,
        ),
        allow_cross_local_scope_parent=_require_bool(
            relationships,
            "allow_cross_local_scope_parent",
            path=relationships_path,
        ),
        allow_structural_selector_as_parent=_require_bool(
            relationships,
            "allow_structural_selector_as_parent",
            path=relationships_path,
        ),
        allow_table_row_as_code_parent=_require_bool(
            relationships,
            "allow_table_row_as_code_parent",
            path=relationships_path,
        ),
        allow_mixed_short_text_as_parent=_require_bool(
            relationships,
            "allow_mixed_short_text_as_parent",
            path=relationships_path,
        ),
        attach_section_context_to_orphan_code=_require_bool(
            relationships,
            "attach_section_context_to_orphan_code",
            path=relationships_path,
        ),
        embed_parent_text=_require_bool(
            relationships,
            "embed_parent_text",
            path=relationships_path,
        ),
        parent_text_max_words=_require_int(
            relationships,
            "parent_text_max_words",
            path=relationships_path,
            min_value=1,
        ),
        parent_text_preserve_complete_sentences=_require_bool(
            relationships,
            "parent_text_preserve_complete_sentences",
            path=relationships_path,
        ),
        sequential_relation_confidence=_require_float(
            relationships,
            "sequential_relation_confidence",
            **confidence_args,
        ),
        command_parent_confidence=_require_float(
            relationships,
            "command_parent_confidence",
            **confidence_args,
        ),
        immediate_parent_confidence=_require_float(
            relationships,
            "immediate_parent_confidence",
            **confidence_args,
        ),
        context_parent_confidence=_require_float(
            relationships,
            "context_parent_confidence",
            **confidence_args,
        ),
        section_fallback_confidence=_require_float(
            relationships,
            "section_fallback_confidence",
            **confidence_args,
        ),
    )


# ---------------------------------------------------------------------------
# RETRIEVAL TEXT PARSING
# ---------------------------------------------------------------------------


def _parse_retrieval_text(
    markdown: Mapping[str, Any],
    *,
    path: str,
) -> MarkdownRetrievalTextConfig:
    retrieval = _require_mapping(
        markdown,
        "retrieval_text",
        path=path,
    )
    retrieval_path = f"{path}.retrieval_text"

    return MarkdownRetrievalTextConfig(
        include_context=_require_bool(
            retrieval,
            "include_context",
            path=retrieval_path,
        ),
        include_heading=_require_bool(
            retrieval,
            "include_heading",
            path=retrieval_path,
        ),
        include_local_context=_require_bool(
            retrieval,
            "include_local_context",
            path=retrieval_path,
        ),
        include_function=_require_bool(
            retrieval,
            "include_function",
            path=retrieval_path,
        ),
        include_parent_instruction=_require_bool(
            retrieval,
            "include_parent_instruction",
            path=retrieval_path,
        ),
        include_parent_context=_require_bool(
            retrieval,
            "include_parent_context",
            path=retrieval_path,
        ),
        include_text=_require_bool(
            retrieval,
            "include_text",
            path=retrieval_path,
        ),
        deduplicate_equal_context_fields=_require_bool(
            retrieval,
            "deduplicate_equal_context_fields",
            path=retrieval_path,
        ),
        preserve_content_paragraphs=_require_bool(
            retrieval,
            "preserve_content_paragraphs",
            path=retrieval_path,
        ),
    )


# ---------------------------------------------------------------------------
# COMPOSITE PARSING
# ---------------------------------------------------------------------------


def _parse_markdown(
    root: Mapping[str, Any],
) -> MarkdownHeuristicsConfig:
    root_path = CONFIG_ROOT_KEY
    markdown = _require_mapping(
        root,
        "markdown",
        path=root_path,
    )
    markdown_path = f"{root_path}.markdown"

    return MarkdownHeuristicsConfig(
        mdx_markers=_parse_mdx_markers(
            markdown,
            path=markdown_path,
        ),
        heading_permalink_texts=_parse_heading_permalink_texts(
            markdown,
            path=markdown_path,
        ),
        callouts=_parse_callouts(
            markdown,
            path=markdown_path,
        ),
        signals=_parse_signals(
            markdown,
            path=markdown_path,
        ),
        metadata=_parse_metadata(
            markdown,
            path=markdown_path,
        ),
        source_provenance=_parse_source_provenance(
            markdown,
            path=markdown_path,
        ),
        local_scopes=_parse_local_scopes(
            markdown,
            path=markdown_path,
        ),
        selector_blocks=_parse_selector_blocks(
            markdown,
            path=markdown_path,
        ),
        semantic_packing=_parse_semantic_packing(
            markdown,
            path=markdown_path,
        ),
        relationships=_parse_relationships(
            markdown,
            path=markdown_path,
        ),
        retrieval_text=_parse_retrieval_text(
            markdown,
            path=markdown_path,
        ),
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_parser_heuristics_config() -> ParserHeuristicsConfig:
    root = _load_toml(_config_path())

    return ParserHeuristicsConfig(
        file_types=_parse_file_types(root),
        markdown=_parse_markdown(root),
    )


def reload_parser_heuristics_config() -> ParserHeuristicsConfig:
    load_parser_heuristics_config.cache_clear()
    return load_parser_heuristics_config()