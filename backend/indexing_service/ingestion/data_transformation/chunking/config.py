from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from typing import TypeVar

import tomllib


#--------------------ERRORS--------------------

class ChunkingConfigError(ValueError):
    pass


#--------------------TOKENIZATION--------------------

@dataclass(frozen=True)
class TokenizationConfig:
    enabled: bool
    role: str
    add_special_tokens: bool
    model_max_input_tokens: int
    require_local_assets: bool
    require_exact_tokenizer: bool
    allow_network: bool
    allow_runtime_download: bool
    allow_approximate_counting: bool
    fail_if_unavailable: bool


#--------------------SEMANTIC CHUNKING--------------------

@dataclass(frozen=True)
class SemanticChunkingConfig:
    enabled: bool
    minimum_chunk_tokens: int
    preferred_chunk_tokens: int
    strict_chunk_tokens: int
    preserve_complete_sentences: bool
    preserve_paragraph_boundaries: bool
    preserve_ir_boundaries: bool
    allow_complete_sentence_above_strict: bool
    split_only_when_required: bool


#--------------------BLOCK BUDGET--------------------

@dataclass(frozen=True)
class BlockBudgetConfig:
    enabled: bool
    default_ratio: float
    paragraph_ratio: float
    blockquote_ratio: float
    list_ratio: float
    code_ratio: float
    table_ratio: float
    component_ratio: float
    low_density_ratio: float
    medium_density_ratio: float
    high_density_ratio: float
    minimum_ratio: float

    def ratio_for(
        self,
        subtype: str,
        density: str | None = None,
    ) -> float:
        if not self.enabled:
            return 1.0

        subtype_ratios = {
            "paragraph": self.paragraph_ratio,
            "blockquote": self.blockquote_ratio,
            "list": self.list_ratio,
            "code": self.code_ratio,
            "table": self.table_ratio,
            "component": self.component_ratio,
        }

        density_ratios = {
            "low": self.low_density_ratio,
            "medium": self.medium_density_ratio,
            "high": self.high_density_ratio,
        }

        subtype_ratio = subtype_ratios.get(
            str(
                subtype
                or ""
            ).strip().lower(),
            self.default_ratio,
        )

        density_ratio = density_ratios.get(
            str(
                density
                or ""
            ).strip().lower(),
            1.0,
        )

        return max(
            self.minimum_ratio,
            min(
                1.0,
                subtype_ratio * density_ratio,
            ),
        )


#--------------------ATOMIC POLICY--------------------

@dataclass(frozen=True)
class AtomicPolicyConfig:
    enabled: bool
    subtypes: frozenset[str]
    preserve_within_hard_limit: bool
    never_sentence_split: bool
    never_word_split: bool
    never_character_split: bool
    quarantine_if_above_hard_limit: bool


#--------------------SPLITTABLE POLICY--------------------

@dataclass(frozen=True)
class SplittablePolicyConfig:
    enabled: bool
    subtypes: frozenset[str]
    sentence_boundary_only: bool
    preserve_source_order: bool
    preserve_blank_line_boundaries: bool


#--------------------SENTENCE SPLITTING--------------------

@dataclass(frozen=True)
class SentenceSplittingConfig:
    enabled: bool
    preserve_complete_sentences: bool
    allow_sentence_above_preferred: bool
    allow_sentence_above_strict: bool
    allow_sentence_above_hard_limit: bool
    word_fallback: bool
    character_fallback: bool
    preserve_terminal_punctuation: bool
    preserve_internal_whitespace: bool


#--------------------HARD LIMIT--------------------

@dataclass(frozen=True)
class HardLimitPolicyConfig:
    enabled: bool
    limit_source: str
    truncate: bool
    drop: bool
    quarantine: bool
    allow_atomic_above_strict: bool
    allow_sentence_above_strict: bool
    allow_atomic_above_hard_limit: bool
    allow_sentence_above_hard_limit: bool
    fail_if_hard_limit_unknown: bool


#--------------------SPLIT IDENTITY--------------------

@dataclass(frozen=True)
class SplitIdentityConfig:
    enabled: bool
    generate_unique_child_ids: bool
    preserve_parent_chunk_id: bool
    preserve_source_ir_id: bool
    split_index_start: int
    child_id_template: str


#--------------------SPLIT PROVENANCE--------------------

@dataclass(frozen=True)
class SplitProvenanceConfig:
    enabled: bool
    copy_source_metadata: bool
    record_parent_chunk_id: bool
    record_source_ir_id: bool
    record_split_index: bool
    record_split_count: bool
    record_original_token_count: bool
    record_split_token_count: bool
    record_split_reason: bool


#--------------------VALIDATION--------------------

@dataclass(frozen=True)
class LightweightValidationConfig:
    enabled: bool
    run_before_embedding: bool
    check_empty_text: bool
    check_identity: bool
    check_token_count: bool
    check_hard_limit: bool


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool
    reject_empty_text: bool
    reject_missing_identity: bool
    validate_token_counts: bool
    validate_split_identity: bool
    validate_split_provenance: bool
    enforce_model_hard_limit: bool
    strict_limit_is_error: bool
    hard_limit_is_error: bool
    deduplicate: bool
    deduplicate_by_ir_id: bool
    lightweight: LightweightValidationConfig


#--------------------DEDUPLICATION--------------------

@dataclass(frozen=True)
class TextDeduplicationConfig:
    enabled: bool
    scope: str
    exact_text_only: bool
    normalize_outer_whitespace: bool
    normalize_internal_whitespace: bool
    case_sensitive: bool
    preserve_distinct_split_children: bool
    preserve_distinct_source_spans: bool


#--------------------PREFIX BUDGET--------------------

@dataclass(frozen=True)
class RetrievalPrefixBudgetConfig:
    enabled: bool
    reserved_tokens: int
    clamp_semantic_budget_to_model_limit: bool
    never_reduce_below_minimum: bool
    fail_if_effective_budget_non_positive: bool


#--------------------QUALITY DIAGNOSTICS--------------------

@dataclass(frozen=True)
class QualityDiagnosticsConfig:
    enabled: bool
    record_token_count: bool
    record_preferred_limit: bool
    record_strict_limit: bool
    record_effective_hard_limit: bool
    record_budget_ratio: bool
    record_budget_density: bool
    record_oversized_sentence: bool
    record_oversized_atomic_block: bool
    record_split_reason: bool
    record_split_count: bool
    record_tokenizer_model_id: bool
    record_tokenizer_revision: bool


#--------------------QUALITY SCORING--------------------

@dataclass(frozen=True)
class QualityScoringConfig:
    enabled: bool
    minimum_score: float
    maximum_score: float
    base_score: float
    under_minimum_penalty: float
    above_preferred_penalty: float
    above_strict_penalty: float
    oversized_sentence_penalty: float
    oversized_atomic_penalty: float
    split_penalty: float
    empty_context_penalty: float
    missing_heading_penalty: float
    source_provenance_bonus: float
    local_scope_bonus: float
    parent_context_bonus: float


#--------------------ROOT CONFIG--------------------

@dataclass(frozen=True)
class ChunkingConfig:
    tokenization: TokenizationConfig
    semantic_chunking: SemanticChunkingConfig
    block_budget: BlockBudgetConfig
    atomic_policy: AtomicPolicyConfig
    splittable_policy: SplittablePolicyConfig
    sentence_splitting: SentenceSplittingConfig
    hard_limit_policy: HardLimitPolicyConfig
    split_identity: SplitIdentityConfig
    split_provenance: SplitProvenanceConfig
    validation: ValidationConfig
    text_deduplication: TextDeduplicationConfig
    retrieval_prefix_budget: RetrievalPrefixBudgetConfig
    quality_diagnostics: QualityDiagnosticsConfig
    quality_scoring: QualityScoringConfig


#--------------------TOKEN BUDGET--------------------

@dataclass(frozen=True)
class ChunkTokenBudget:
    minimum_tokens: int
    preferred_tokens: int
    strict_tokens: int
    model_max_input_tokens: int
    reserved_prefix_tokens: int
    effective_chunk_hard_tokens: int

    @property
    def available_input_tokens(
        self,
    ) -> int:
        return self.effective_chunk_hard_tokens

    @property
    def has_prefix_reserve(
        self,
    ) -> bool:
        return self.reserved_prefix_tokens > 0

    def contains_within_preferred(
        self,
        token_count: int,
    ) -> bool:
        return (
            token_count
            <= self.preferred_tokens
        )

    def contains_within_strict(
        self,
        token_count: int,
    ) -> bool:
        return (
            token_count
            <= self.strict_tokens
        )

    def contains_within_chunk_hard_limit(
        self,
        token_count: int,
    ) -> bool:
        return (
            token_count
            <= self.effective_chunk_hard_tokens
        )

    def contains_within_model_limit(
        self,
        token_count: int,
    ) -> bool:
        return (
            token_count
            <= self.model_max_input_tokens
        )


#--------------------CONSTANTS--------------------

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config.toml"
)

T = TypeVar(
    "T"
)


#--------------------READING--------------------

def _read_toml(
    path: str | Path,
) -> Mapping[str, Any]:
    config_path = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not config_path.is_file():
        raise ChunkingConfigError(
            "Chunking config does not exist: "
            f"{config_path}"
        )

    try:
        text = config_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ChunkingConfigError(
            "Unable to read chunking config: "
            f"{config_path}"
        ) from exc

    try:
        value = tomllib.loads(
            text
        )
    except Exception as exc:
        raise ChunkingConfigError(
            "Invalid chunking TOML: "
            f"{config_path}"
        ) from exc

    if not isinstance(
        value,
        Mapping,
    ):
        raise ChunkingConfigError(
            "Chunking TOML root must be a table"
        )

    return value


def _section(
    root: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = root.get(
        name
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise ChunkingConfigError(
            f"{name} must be a TOML table"
        )

    return value


#--------------------VALUE PARSING--------------------

def _bool(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> bool:
    value = data.get(
        key
    )

    if not isinstance(
        value,
        bool,
    ):
        raise ChunkingConfigError(
            f"{section}.{key} must be boolean"
        )

    return value


def _int(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> int:
    value = data.get(
        key
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise ChunkingConfigError(
            f"{section}.{key} must be integer"
        )

    return value


def _float(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> float:
    value = data.get(
        key
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise ChunkingConfigError(
            f"{section}.{key} must be numeric"
        )

    return float(
        value
    )


def _str(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> str:
    value = data.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):
        raise ChunkingConfigError(
            f"{section}.{key} must be string"
        )

    value = value.strip()

    if not value:
        raise ChunkingConfigError(
            f"{section}.{key} cannot be empty"
        )

    return value


def _strset(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> frozenset[str]:
    value = data.get(
        key
    )

    if not isinstance(
        value,
        list,
    ):
        raise ChunkingConfigError(
            f"{section}.{key} must be an array"
        )

    output: set[str] = set()

    for index, item in enumerate(
        value
    ):
        if not isinstance(
            item,
            str,
        ):
            raise ChunkingConfigError(
                f"{section}.{key}[{index}] "
                "must be string"
            )

        item = item.strip()

        if not item:
            raise ChunkingConfigError(
                f"{section}.{key}[{index}] "
                "cannot be empty"
            )

        output.add(
            item
        )

    return frozenset(
        output
    )


#--------------------SECTION BUILDER--------------------

def _build(
    *,
    root: Mapping[str, Any],
    section_name: str,
    cls: Callable[..., T],
    bools: tuple[str, ...] = (),
    ints: tuple[str, ...] = (),
    floats: tuple[str, ...] = (),
    strings: tuple[str, ...] = (),
    string_sets: tuple[str, ...] = (),
) -> T:
    data = _section(
        root,
        section_name,
    )

    expected = (
        set(
            bools
        )
        | set(
            ints
        )
        | set(
            floats
        )
        | set(
            strings
        )
        | set(
            string_sets
        )
    )

    unknown = (
        set(
            data.keys()
        )
        - expected
    )

    if unknown:
        raise ChunkingConfigError(
            f"Unknown key(s) in {section_name}: "
            + ", ".join(
                sorted(
                    str(
                        value
                    )
                    for value in unknown
                )
            )
        )

    values: dict[
        str,
        Any,
    ] = {}

    for key in bools:
        values[
            key
        ] = _bool(
            data,
            key,
            section_name,
        )

    for key in ints:
        values[
            key
        ] = _int(
            data,
            key,
            section_name,
        )

    for key in floats:
        values[
            key
        ] = _float(
            data,
            key,
            section_name,
        )

    for key in strings:
        values[
            key
        ] = _str(
            data,
            key,
            section_name,
        )

    for key in string_sets:
        values[
            key
        ] = _strset(
            data,
            key,
            section_name,
        )

    return cls(
        **values
    )


#--------------------PARSING--------------------

def _parse_config(
    root: Mapping[str, Any],
) -> ChunkingConfig:
    tokenization = _build(
        root=root,
        section_name="tokenization",
        cls=TokenizationConfig,
        bools=(
            "enabled",
            "add_special_tokens",
            "require_local_assets",
            "require_exact_tokenizer",
            "allow_network",
            "allow_runtime_download",
            "allow_approximate_counting",
            "fail_if_unavailable",
        ),
        ints=(
            "model_max_input_tokens",
        ),
        strings=(
            "role",
        ),
    )

    semantic = _build(
        root=root,
        section_name="semantic_chunking",
        cls=SemanticChunkingConfig,
        bools=(
            "enabled",
            "preserve_complete_sentences",
            "preserve_paragraph_boundaries",
            "preserve_ir_boundaries",
            "allow_complete_sentence_above_strict",
            "split_only_when_required",
        ),
        ints=(
            "minimum_chunk_tokens",
            "preferred_chunk_tokens",
            "strict_chunk_tokens",
        ),
    )

    block_budget = _build(
        root=root,
        section_name="block_budget",
        cls=BlockBudgetConfig,
        bools=(
            "enabled",
        ),
        floats=(
            "default_ratio",
            "paragraph_ratio",
            "blockquote_ratio",
            "list_ratio",
            "code_ratio",
            "table_ratio",
            "component_ratio",
            "low_density_ratio",
            "medium_density_ratio",
            "high_density_ratio",
            "minimum_ratio",
        ),
    )

    atomic = _build(
        root=root,
        section_name="atomic_policy",
        cls=AtomicPolicyConfig,
        bools=(
            "enabled",
            "preserve_within_hard_limit",
            "never_sentence_split",
            "never_word_split",
            "never_character_split",
            "quarantine_if_above_hard_limit",
        ),
        string_sets=(
            "subtypes",
        ),
    )

    splittable = _build(
        root=root,
        section_name="splittable_policy",
        cls=SplittablePolicyConfig,
        bools=(
            "enabled",
            "sentence_boundary_only",
            "preserve_source_order",
            "preserve_blank_line_boundaries",
        ),
        string_sets=(
            "subtypes",
        ),
    )

    sentence = _build(
        root=root,
        section_name="sentence_splitting",
        cls=SentenceSplittingConfig,
        bools=(
            "enabled",
            "preserve_complete_sentences",
            "allow_sentence_above_preferred",
            "allow_sentence_above_strict",
            "allow_sentence_above_hard_limit",
            "word_fallback",
            "character_fallback",
            "preserve_terminal_punctuation",
            "preserve_internal_whitespace",
        ),
    )

    hard = _build(
        root=root,
        section_name="hard_limit_policy",
        cls=HardLimitPolicyConfig,
        bools=(
            "enabled",
            "truncate",
            "drop",
            "quarantine",
            "allow_atomic_above_strict",
            "allow_sentence_above_strict",
            "allow_atomic_above_hard_limit",
            "allow_sentence_above_hard_limit",
            "fail_if_hard_limit_unknown",
        ),
        strings=(
            "limit_source",
        ),
    )

    split_identity = _build(
        root=root,
        section_name="split_identity",
        cls=SplitIdentityConfig,
        bools=(
            "enabled",
            "generate_unique_child_ids",
            "preserve_parent_chunk_id",
            "preserve_source_ir_id",
        ),
        ints=(
            "split_index_start",
        ),
        strings=(
            "child_id_template",
        ),
    )

    split_provenance = _build(
        root=root,
        section_name="split_provenance",
        cls=SplitProvenanceConfig,
        bools=(
            "enabled",
            "copy_source_metadata",
            "record_parent_chunk_id",
            "record_source_ir_id",
            "record_split_index",
            "record_split_count",
            "record_original_token_count",
            "record_split_token_count",
            "record_split_reason",
        ),
    )

    validation_data = _section(
        root,
        "validation",
    )

    lightweight_data = validation_data.get(
        "lightweight"
    )

    if not isinstance(
        lightweight_data,
        Mapping,
    ):
        raise ChunkingConfigError(
            "validation.lightweight must be a TOML table"
        )

    lightweight = LightweightValidationConfig(
        enabled=_bool(
            lightweight_data,
            "enabled",
            "validation.lightweight",
        ),
        run_before_embedding=_bool(
            lightweight_data,
            "run_before_embedding",
            "validation.lightweight",
        ),
        check_empty_text=_bool(
            lightweight_data,
            "check_empty_text",
            "validation.lightweight",
        ),
        check_identity=_bool(
            lightweight_data,
            "check_identity",
            "validation.lightweight",
        ),
        check_token_count=_bool(
            lightweight_data,
            "check_token_count",
            "validation.lightweight",
        ),
        check_hard_limit=_bool(
            lightweight_data,
            "check_hard_limit",
            "validation.lightweight",
        ),
    )

    validation = ValidationConfig(
        enabled=_bool(
            validation_data,
            "enabled",
            "validation",
        ),
        reject_empty_text=_bool(
            validation_data,
            "reject_empty_text",
            "validation",
        ),
        reject_missing_identity=_bool(
            validation_data,
            "reject_missing_identity",
            "validation",
        ),
        validate_token_counts=_bool(
            validation_data,
            "validate_token_counts",
            "validation",
        ),
        validate_split_identity=_bool(
            validation_data,
            "validate_split_identity",
            "validation",
        ),
        validate_split_provenance=_bool(
            validation_data,
            "validate_split_provenance",
            "validation",
        ),
        enforce_model_hard_limit=_bool(
            validation_data,
            "enforce_model_hard_limit",
            "validation",
        ),
        strict_limit_is_error=_bool(
            validation_data,
            "strict_limit_is_error",
            "validation",
        ),
        hard_limit_is_error=_bool(
            validation_data,
            "hard_limit_is_error",
            "validation",
        ),
        deduplicate=_bool(
            validation_data,
            "deduplicate",
            "validation",
        ),
        deduplicate_by_ir_id=_bool(
            validation_data,
            "deduplicate_by_ir_id",
            "validation",
        ),
        lightweight=lightweight,
    )

    dedup = _build(
        root=root,
        section_name="text_deduplication",
        cls=TextDeduplicationConfig,
        bools=(
            "enabled",
            "exact_text_only",
            "normalize_outer_whitespace",
            "normalize_internal_whitespace",
            "case_sensitive",
            "preserve_distinct_split_children",
            "preserve_distinct_source_spans",
        ),
        strings=(
            "scope",
        ),
    )

    prefix = _build(
        root=root,
        section_name="retrieval_prefix_budget",
        cls=RetrievalPrefixBudgetConfig,
        bools=(
            "enabled",
            "clamp_semantic_budget_to_model_limit",
            "never_reduce_below_minimum",
            "fail_if_effective_budget_non_positive",
        ),
        ints=(
            "reserved_tokens",
        ),
    )

    diagnostics = _build(
        root=root,
        section_name="quality_diagnostics",
        cls=QualityDiagnosticsConfig,
        bools=(
            "enabled",
            "record_token_count",
            "record_preferred_limit",
            "record_strict_limit",
            "record_effective_hard_limit",
            "record_budget_ratio",
            "record_budget_density",
            "record_oversized_sentence",
            "record_oversized_atomic_block",
            "record_split_reason",
            "record_split_count",
            "record_tokenizer_model_id",
            "record_tokenizer_revision",
        ),
    )

    scoring = _build(
        root=root,
        section_name="quality_scoring",
        cls=QualityScoringConfig,
        bools=(
            "enabled",
        ),
        floats=(
            "minimum_score",
            "maximum_score",
            "base_score",
            "under_minimum_penalty",
            "above_preferred_penalty",
            "above_strict_penalty",
            "oversized_sentence_penalty",
            "oversized_atomic_penalty",
            "split_penalty",
            "empty_context_penalty",
            "missing_heading_penalty",
            "source_provenance_bonus",
            "local_scope_bonus",
            "parent_context_bonus",
        ),
    )

    config = ChunkingConfig(
        tokenization=tokenization,
        semantic_chunking=semantic,
        block_budget=block_budget,
        atomic_policy=atomic,
        splittable_policy=splittable,
        sentence_splitting=sentence,
        hard_limit_policy=hard,
        split_identity=split_identity,
        split_provenance=split_provenance,
        validation=validation,
        text_deduplication=dedup,
        retrieval_prefix_budget=prefix,
        quality_diagnostics=diagnostics,
        quality_scoring=scoring,
    )

    _validate_config(
        config
    )

    return config


#--------------------VALIDATION--------------------

def _validate_ratio(
    name: str,
    value: float,
) -> None:
    if not (
        0.0
        < value
        <= 1.0
    ):
        raise ChunkingConfigError(
            f"{name} must be in (0, 1]"
        )


def _validate_config(
    config: ChunkingConfig,
) -> None:
    tokenization = config.tokenization

    if tokenization.role != "dense":
        raise ChunkingConfigError(
            "tokenization.role must be dense"
        )

    if tokenization.model_max_input_tokens <= 0:
        raise ChunkingConfigError(
            "tokenization.model_max_input_tokens "
            "must be positive"
        )

    if (
        tokenization.require_exact_tokenizer
        and tokenization.allow_approximate_counting
    ):
        raise ChunkingConfigError(
            "Exact tokenizer mode cannot use "
            "approximate token counting"
        )

    if (
        tokenization.require_local_assets
        and tokenization.allow_runtime_download
    ):
        raise ChunkingConfigError(
            "Offline tokenizer mode cannot allow "
            "runtime downloads"
        )

    if (
        tokenization.require_local_assets
        and tokenization.allow_network
    ):
        raise ChunkingConfigError(
            "Offline tokenizer mode cannot allow network"
        )

    semantic = config.semantic_chunking

    if semantic.minimum_chunk_tokens <= 0:
        raise ChunkingConfigError(
            "minimum_chunk_tokens must be positive"
        )

    if not (
        semantic.minimum_chunk_tokens
        <= semantic.preferred_chunk_tokens
        <= semantic.strict_chunk_tokens
    ):
        raise ChunkingConfigError(
            "Chunk token limits must satisfy "
            "minimum <= preferred <= strict"
        )

    block_budget = config.block_budget

    for name in (
        "default_ratio",
        "paragraph_ratio",
        "blockquote_ratio",
        "list_ratio",
        "code_ratio",
        "table_ratio",
        "component_ratio",
        "low_density_ratio",
        "medium_density_ratio",
        "high_density_ratio",
        "minimum_ratio",
    ):
        _validate_ratio(
            f"block_budget.{name}",
            getattr(
                block_budget,
                name,
            ),
        )

    overlap = (
        config.atomic_policy.subtypes
        & config.splittable_policy.subtypes
    )

    if overlap:
        raise ChunkingConfigError(
            "Atomic and splittable subtype sets overlap: "
            + ", ".join(
                sorted(
                    overlap
                )
            )
        )

    sentence = config.sentence_splitting

    if sentence.word_fallback:
        raise ChunkingConfigError(
            "sentence_splitting.word_fallback "
            "must remain false"
        )

    if sentence.character_fallback:
        raise ChunkingConfigError(
            "sentence_splitting.character_fallback "
            "must remain false"
        )

    hard = config.hard_limit_policy

    if hard.limit_source != "tokenizer_contract":
        raise ChunkingConfigError(
            "hard_limit_policy.limit_source must be "
            "'tokenizer_contract'"
        )

    if hard.truncate:
        raise ChunkingConfigError(
            "Hard-limit policy cannot silently truncate"
        )

    if hard.drop:
        raise ChunkingConfigError(
            "Hard-limit policy cannot silently drop chunks"
        )

    identity = config.split_identity

    if identity.split_index_start < 1:
        raise ChunkingConfigError(
            "split_identity.split_index_start "
            "must be at least 1"
        )

    template = identity.child_id_template

    if "{parent_id}" not in template:
        raise ChunkingConfigError(
            "child_id_template must contain {parent_id}"
        )

    if "{split_index}" not in template:
        raise ChunkingConfigError(
            "child_id_template must contain {split_index}"
        )

    if config.text_deduplication.scope != "document":
        raise ChunkingConfigError(
            "text_deduplication.scope currently "
            "supports document only"
        )

    prefix = config.retrieval_prefix_budget

    if prefix.reserved_tokens < 0:
        raise ChunkingConfigError(
            "reserved_tokens cannot be negative"
        )

    scoring = config.quality_scoring

    if scoring.minimum_score > scoring.maximum_score:
        raise ChunkingConfigError(
            "quality score minimum exceeds maximum"
        )

    if not (
        scoring.minimum_score
        <= scoring.base_score
        <= scoring.maximum_score
    ):
        raise ChunkingConfigError(
            "quality base_score must lie within "
            "score bounds"
        )


#--------------------MODEL LIMIT--------------------

def resolve_model_max_input_tokens(
    config: ChunkingConfig,
) -> int:
    max_input_tokens = (
        config
        .tokenization
        .model_max_input_tokens
    )

    if (
        isinstance(
            max_input_tokens,
            bool,
        )
        or not isinstance(
            max_input_tokens,
            int,
        )
    ):
        raise ChunkingConfigError(
            "tokenization.model_max_input_tokens "
            "must be integer"
        )

    if max_input_tokens <= 0:
        raise ChunkingConfigError(
            "tokenization.model_max_input_tokens "
            "must be positive"
        )

    return max_input_tokens


#--------------------TOKEN BUDGET--------------------

def build_chunk_token_budget(
    config: ChunkingConfig,
) -> ChunkTokenBudget:
    semantic = config.semantic_chunking
    prefix = config.retrieval_prefix_budget

    model_max = (
        resolve_model_max_input_tokens(config)
    )

    reserved = (
        prefix.reserved_tokens
        if prefix.enabled
        else 0
    )

    if (
        prefix.enabled
        and prefix.clamp_semantic_budget_to_model_limit
    ):
        effective_hard = (
            model_max
            - reserved
        )
    else:
        effective_hard = model_max

    if effective_hard <= 0:
        raise ChunkingConfigError(
            "Retrieval prefix reserve leaves no "
            "chunk-content capacity"
        )

    if (
        effective_hard
        < semantic.minimum_chunk_tokens
        and prefix.never_reduce_below_minimum
    ):
        raise ChunkingConfigError(
            "Effective chunk budget is below "
            "minimum_chunk_tokens"
        )

    minimum = min(
        semantic.minimum_chunk_tokens,
        effective_hard,
    )

    preferred = min(
        semantic.preferred_chunk_tokens,
        effective_hard,
    )

    strict = min(
        semantic.strict_chunk_tokens,
        effective_hard,
    )

    preferred = max(
        minimum,
        preferred,
    )

    strict = max(
        preferred,
        strict,
    )

    return ChunkTokenBudget(
        minimum_tokens=minimum,
        preferred_tokens=preferred,
        strict_tokens=strict,
        model_max_input_tokens=model_max,
        reserved_prefix_tokens=reserved,
        effective_chunk_hard_tokens=effective_hard,
    )


#--------------------PUBLIC API--------------------

def load_chunking_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ChunkingConfig:
    return _parse_config(
        _read_toml(
            path
        )
    )


@lru_cache(maxsize=1)
def get_chunking_config(
) -> ChunkingConfig:
    return load_chunking_config(
        DEFAULT_CONFIG_PATH
    )


def clear_chunking_config_cache(
) -> None:
    get_chunking_config.cache_clear()


def get_chunk_token_budget(
    config: ChunkingConfig | None = None,
) -> ChunkTokenBudget:
    active = (
        config
        if config is not None
        else get_chunking_config()
    )

    return build_chunk_token_budget(
        active
    )


__all__ = [
    "AtomicPolicyConfig",
    "BlockBudgetConfig",
    "ChunkingConfig",
    "ChunkingConfigError",
    "ChunkTokenBudget",
    "HardLimitPolicyConfig",
    "LightweightValidationConfig",
    "QualityDiagnosticsConfig",
    "QualityScoringConfig",
    "RetrievalPrefixBudgetConfig",
    "SemanticChunkingConfig",
    "SentenceSplittingConfig",
    "SplitIdentityConfig",
    "SplitProvenanceConfig",
    "SplittablePolicyConfig",
    "TextDeduplicationConfig",
    "TokenizationConfig",
    "ValidationConfig",
    "build_chunk_token_budget",
    "clear_chunking_config_cache",
    "get_chunk_token_budget",
    "get_chunking_config",
    "load_chunking_config",
    "resolve_model_max_input_tokens",
]
