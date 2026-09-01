from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rtoml

from rag.retrieval.contracts.errors import RetrievalConfigurationError
from backend.rag.retrieval.contracts.enums import ConfidenceBand, RetrievalMode


Payload = dict[str, Any]

DEFAULT_RETRIEVAL_CONFIG_PATH = Path(__file__).with_name("retrieval.toml")


@dataclass(frozen=True)
class GraphConfig:
    recursion_limit: int
    max_corrective_retries: int
    include_debug_metadata: bool

    def validate(self) -> None:
        if self.recursion_limit <= 0:
            raise ValueError("graph.recursion_limit must be greater than zero.")

        if self.max_corrective_retries < 0:
            raise ValueError("graph.max_corrective_retries cannot be negative.")


@dataclass(frozen=True)
class RequestConfig:
    default_mode: RetrievalMode
    default_top_k: int
    max_top_k: int

    def validate(self) -> None:
        if self.default_top_k <= 0:
            raise ValueError("request.default_top_k must be greater than zero.")

        if self.max_top_k < self.default_top_k:
            raise ValueError("request.max_top_k must be >= request.default_top_k.")


@dataclass(frozen=True)
class QueryConfig:
    min_chars: int
    max_chars: int
    low_information_max_terms: int
    too_short_max_chars: int
    too_long_min_chars: int
    max_extracted_terms: int
    max_rewritten_queries: int

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("query.min_chars must be greater than zero.")

        if self.max_chars < self.min_chars:
            raise ValueError("query.max_chars must be >= query.min_chars.")

        if self.low_information_max_terms < 0:
            raise ValueError("query.low_information_max_terms cannot be negative.")

        if self.too_short_max_chars < self.min_chars:
            raise ValueError("query.too_short_max_chars must be >= query.min_chars.")

        if self.too_long_min_chars > self.max_chars:
            raise ValueError("query.too_long_min_chars must be <= query.max_chars.")

        if self.max_extracted_terms <= 0:
            raise ValueError("query.max_extracted_terms must be greater than zero.")

        if self.max_rewritten_queries < 0:
            raise ValueError("query.max_rewritten_queries cannot be negative.")


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    max_chat_turns: int
    max_user_turns: int
    max_assistant_turns: int
    preserve_recent_turns: int

    def validate(self) -> None:
        if self.max_chat_turns < 0:
            raise ValueError("memory.max_chat_turns cannot be negative.")

        if self.max_user_turns < 0:
            raise ValueError("memory.max_user_turns cannot be negative.")

        if self.max_assistant_turns < 0:
            raise ValueError("memory.max_assistant_turns cannot be negative.")

        if self.preserve_recent_turns < 0:
            raise ValueError("memory.preserve_recent_turns cannot be negative.")

        if self.preserve_recent_turns > self.max_chat_turns:
            raise ValueError("memory.preserve_recent_turns must be <= memory.max_chat_turns.")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model_name: str
    vector_size: int
    normalize_vectors: bool
    max_query_tokens: int
    timeout_seconds: float
    max_retries: int

    def validate(self) -> None:
        _validate_non_empty("embedding.provider", self.provider)
        _validate_non_empty("embedding.model_name", self.model_name)

        if self.vector_size <= 0:
            raise ValueError("embedding.vector_size must be greater than zero.")

        if self.max_query_tokens <= 0:
            raise ValueError("embedding.max_query_tokens must be greater than zero.")

        if self.timeout_seconds <= 0:
            raise ValueError("embedding.timeout_seconds must be greater than zero.")

        if self.max_retries < 0:
            raise ValueError("embedding.max_retries cannot be negative.")


@dataclass(frozen=True)
class SearchConfig:
    enabled: bool
    top_k: int
    score_threshold: float

    def validate(self, section_name: str) -> None:
        if self.top_k <= 0:
            raise ValueError(f"{section_name}.top_k must be greater than zero.")

        if self.score_threshold < 0:
            raise ValueError(f"{section_name}.score_threshold cannot be negative.")


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool
    strategy: str
    rrf_k: int
    fused_top_k: int

    def validate(self) -> None:
        _validate_non_empty("fusion.strategy", self.strategy)

        if self.rrf_k <= 0:
            raise ValueError("fusion.rrf_k must be greater than zero.")

        if self.fused_top_k <= 0:
            raise ValueError("fusion.fused_top_k must be greater than zero.")


@dataclass(frozen=True)
class StructureConfig:
    enabled: bool
    sibling_window: int
    parent_window: int
    child_window: int
    max_expanded_candidates: int

    def validate(self) -> None:
        if self.sibling_window < 0:
            raise ValueError("structure.sibling_window cannot be negative.")

        if self.parent_window < 0:
            raise ValueError("structure.parent_window cannot be negative.")

        if self.child_window < 0:
            raise ValueError("structure.child_window cannot be negative.")

        if self.max_expanded_candidates <= 0:
            raise ValueError("structure.max_expanded_candidates must be greater than zero.")


@dataclass(frozen=True)
class RerankingConfig:
    enabled: bool
    provider: str
    model_name: str
    input_top_k: int
    output_top_k: int
    timeout_seconds: float
    max_retries: int

    def validate(self) -> None:
        _validate_non_empty("reranking.provider", self.provider)
        _validate_non_empty("reranking.model_name", self.model_name)

        if self.input_top_k <= 0:
            raise ValueError("reranking.input_top_k must be greater than zero.")

        if self.output_top_k <= 0:
            raise ValueError("reranking.output_top_k must be greater than zero.")

        if self.output_top_k > self.input_top_k:
            raise ValueError("reranking.output_top_k must be <= reranking.input_top_k.")

        if self.timeout_seconds <= 0:
            raise ValueError("reranking.timeout_seconds must be greater than zero.")

        if self.max_retries < 0:
            raise ValueError("reranking.max_retries cannot be negative.")


@dataclass(frozen=True)
class GradingConfig:
    min_high_confidence_score: float
    min_medium_confidence_score: float
    retry_below_confidence: ConfidenceBand
    retry_when_no_candidates: bool

    def validate(self) -> None:
        _validate_probability(
            "grading.min_high_confidence_score",
            self.min_high_confidence_score,
        )
        _validate_probability(
            "grading.min_medium_confidence_score",
            self.min_medium_confidence_score,
        )

        if self.min_high_confidence_score < self.min_medium_confidence_score:
            raise ValueError(
                "grading.min_high_confidence_score must be >= "
                "grading.min_medium_confidence_score."
            )


@dataclass(frozen=True)
class ContextConfig:
    max_context_blocks: int
    max_total_tokens: int
    max_block_tokens: int
    reserve_tokens_for_answer: int
    deduplicate_by_chunk_id: bool
    deduplicate_by_text_hash: bool

    def validate(self) -> None:
        if self.max_context_blocks <= 0:
            raise ValueError("context.max_context_blocks must be greater than zero.")

        if self.max_total_tokens <= 0:
            raise ValueError("context.max_total_tokens must be greater than zero.")

        if self.max_block_tokens <= 0:
            raise ValueError("context.max_block_tokens must be greater than zero.")

        if self.reserve_tokens_for_answer < 0:
            raise ValueError("context.reserve_tokens_for_answer cannot be negative.")

        if self.max_block_tokens > self.max_total_tokens:
            raise ValueError("context.max_block_tokens must be <= context.max_total_tokens.")

        if self.reserve_tokens_for_answer >= self.max_total_tokens:
            raise ValueError(
                "context.reserve_tokens_for_answer must be less than context.max_total_tokens."
            )


@dataclass(frozen=True)
class SemanticCacheConfig:
    enabled: bool
    provider: str
    model_name: str
    vector_size: int
    similarity_threshold: float
    ttl_seconds: int

    def validate(self) -> None:
        _validate_non_empty("semantic_cache.provider", self.provider)
        _validate_non_empty("semantic_cache.model_name", self.model_name)

        if self.vector_size <= 0:
            raise ValueError("semantic_cache.vector_size must be greater than zero.")

        _validate_probability(
            "semantic_cache.similarity_threshold",
            self.similarity_threshold,
        )

        if self.ttl_seconds <= 0:
            raise ValueError("semantic_cache.ttl_seconds must be greater than zero.")


@dataclass(frozen=True)
class TimeoutsConfig:
    dense_retrieval_seconds: float
    sparse_retrieval_seconds: float
    fusion_seconds: float
    structure_expansion_seconds: float
    context_packing_seconds: float

    def validate(self) -> None:
        for key, value in (
            ("timeouts.dense_retrieval_seconds", self.dense_retrieval_seconds),
            ("timeouts.sparse_retrieval_seconds", self.sparse_retrieval_seconds),
            ("timeouts.fusion_seconds", self.fusion_seconds),
            ("timeouts.structure_expansion_seconds", self.structure_expansion_seconds),
            ("timeouts.context_packing_seconds", self.context_packing_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{key} must be greater than zero.")


@dataclass(frozen=True)
class ObservabilityConfig:
    emit_events: bool
    include_trace: bool
    include_candidate_debug: bool
    include_payload_debug: bool


@dataclass(frozen=True)
class RetrievalConfig:
    graph: GraphConfig
    request: RequestConfig
    query: QueryConfig
    memory: MemoryConfig
    embedding: EmbeddingConfig
    dense: SearchConfig
    sparse: SearchConfig
    fusion: FusionConfig
    structure: StructureConfig
    reranking: RerankingConfig
    grading: GradingConfig
    context: ContextConfig
    semantic_cache: SemanticCacheConfig
    timeouts: TimeoutsConfig
    observability: ObservabilityConfig

    def validate(self) -> None:
        self.graph.validate()
        self.request.validate()
        self.query.validate()
        self.memory.validate()
        self.embedding.validate()
        self.dense.validate("dense")
        self.sparse.validate("sparse")
        self.fusion.validate()
        self.structure.validate()
        self.reranking.validate()
        self.grading.validate()
        self.context.validate()
        self.semantic_cache.validate()
        self.timeouts.validate()


def load_retrieval_config(path: str | Path | None = None) -> RetrievalConfig:
    config_path = Path(path) if path is not None else DEFAULT_RETRIEVAL_CONFIG_PATH

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RetrievalConfigurationError(
            f"Could not read retrieval config: {config_path}",
            cause=exc,
            metadata={
                "path": str(config_path),
            },
        ) from exc

    try:
        raw_config = rtoml.loads(raw_text)
    except Exception as exc:
        raise RetrievalConfigurationError(
            f"Could not parse retrieval TOML config: {config_path}",
            cause=exc,
            metadata={
                "path": str(config_path),
            },
        ) from exc

    try:
        config = parse_retrieval_config(raw_config)
        config.validate()
        return config

    except RetrievalConfigurationError:
        raise

    except Exception as exc:
        raise RetrievalConfigurationError(
            f"Invalid retrieval config: {config_path}",
            cause=exc,
            metadata={
                "path": str(config_path),
            },
        ) from exc


def parse_retrieval_config(raw_config: Mapping[str, Any]) -> RetrievalConfig:
    if not isinstance(raw_config, Mapping):
        raise RetrievalConfigurationError("Retrieval config root must be a TOML table.")

    return RetrievalConfig(
        graph=_parse_graph_config(_required_section(raw_config, "graph")),
        request=_parse_request_config(_required_section(raw_config, "request")),
        query=_parse_query_config(_required_section(raw_config, "query")),
        memory=_parse_memory_config(_required_section(raw_config, "memory")),
        embedding=_parse_embedding_config(_required_section(raw_config, "embedding")),
        dense=_parse_search_config(_required_section(raw_config, "dense")),
        sparse=_parse_search_config(_required_section(raw_config, "sparse")),
        fusion=_parse_fusion_config(_required_section(raw_config, "fusion")),
        structure=_parse_structure_config(_required_section(raw_config, "structure")),
        reranking=_parse_reranking_config(_required_section(raw_config, "reranking")),
        grading=_parse_grading_config(_required_section(raw_config, "grading")),
        context=_parse_context_config(_required_section(raw_config, "context")),
        semantic_cache=_parse_semantic_cache_config(
            _required_section(raw_config, "semantic_cache")
        ),
        timeouts=_parse_timeouts_config(_required_section(raw_config, "timeouts")),
        observability=_parse_observability_config(
            _required_section(raw_config, "observability")
        ),
    )


def _parse_graph_config(section: Mapping[str, Any]) -> GraphConfig:
    return GraphConfig(
        recursion_limit=_required_int(section, "recursion_limit"),
        max_corrective_retries=_required_int(section, "max_corrective_retries"),
        include_debug_metadata=_required_bool(section, "include_debug_metadata"),
    )


def _parse_request_config(section: Mapping[str, Any]) -> RequestConfig:
    return RequestConfig(
        default_mode=_required_enum(
            section,
            "default_mode",
            enum_type=RetrievalMode,
        ),
        default_top_k=_required_int(section, "default_top_k"),
        max_top_k=_required_int(section, "max_top_k"),
    )


def _parse_query_config(section: Mapping[str, Any]) -> QueryConfig:
    return QueryConfig(
        min_chars=_required_int(section, "min_chars"),
        max_chars=_required_int(section, "max_chars"),
        low_information_max_terms=_required_int(section, "low_information_max_terms"),
        too_short_max_chars=_required_int(section, "too_short_max_chars"),
        too_long_min_chars=_required_int(section, "too_long_min_chars"),
        max_extracted_terms=_required_int(section, "max_extracted_terms"),
        max_rewritten_queries=_required_int(section, "max_rewritten_queries"),
    )


def _parse_memory_config(section: Mapping[str, Any]) -> MemoryConfig:
    return MemoryConfig(
        enabled=_required_bool(section, "enabled"),
        max_chat_turns=_required_int(section, "max_chat_turns"),
        max_user_turns=_required_int(section, "max_user_turns"),
        max_assistant_turns=_required_int(section, "max_assistant_turns"),
        preserve_recent_turns=_required_int(section, "preserve_recent_turns"),
    )


def _parse_embedding_config(section: Mapping[str, Any]) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=_required_str(section, "provider"),
        model_name=_required_str(section, "model_name"),
        vector_size=_required_int(section, "vector_size"),
        normalize_vectors=_required_bool(section, "normalize_vectors"),
        max_query_tokens=_required_int(section, "max_query_tokens"),
        timeout_seconds=_required_float(section, "timeout_seconds"),
        max_retries=_required_int(section, "max_retries"),
    )


def _parse_search_config(section: Mapping[str, Any]) -> SearchConfig:
    return SearchConfig(
        enabled=_required_bool(section, "enabled"),
        top_k=_required_int(section, "top_k"),
        score_threshold=_required_float(section, "score_threshold"),
    )


def _parse_fusion_config(section: Mapping[str, Any]) -> FusionConfig:
    return FusionConfig(
        enabled=_required_bool(section, "enabled"),
        strategy=_required_str(section, "strategy"),
        rrf_k=_required_int(section, "rrf_k"),
        fused_top_k=_required_int(section, "fused_top_k"),
    )


def _parse_structure_config(section: Mapping[str, Any]) -> StructureConfig:
    return StructureConfig(
        enabled=_required_bool(section, "enabled"),
        sibling_window=_required_int(section, "sibling_window"),
        parent_window=_required_int(section, "parent_window"),
        child_window=_required_int(section, "child_window"),
        max_expanded_candidates=_required_int(section, "max_expanded_candidates"),
    )


def _parse_reranking_config(section: Mapping[str, Any]) -> RerankingConfig:
    return RerankingConfig(
        enabled=_required_bool(section, "enabled"),
        provider=_required_str(section, "provider"),
        model_name=_required_str(section, "model_name"),
        input_top_k=_required_int(section, "input_top_k"),
        output_top_k=_required_int(section, "output_top_k"),
        timeout_seconds=_required_float(section, "timeout_seconds"),
        max_retries=_required_int(section, "max_retries"),
    )


def _parse_grading_config(section: Mapping[str, Any]) -> GradingConfig:
    return GradingConfig(
        min_high_confidence_score=_required_float(
            section,
            "min_high_confidence_score",
        ),
        min_medium_confidence_score=_required_float(
            section,
            "min_medium_confidence_score",
        ),
        retry_below_confidence=_required_enum(
            section,
            "retry_below_confidence",
            enum_type=ConfidenceBand,
        ),
        retry_when_no_candidates=_required_bool(section, "retry_when_no_candidates"),
    )


def _parse_context_config(section: Mapping[str, Any]) -> ContextConfig:
    return ContextConfig(
        max_context_blocks=_required_int(section, "max_context_blocks"),
        max_total_tokens=_required_int(section, "max_total_tokens"),
        max_block_tokens=_required_int(section, "max_block_tokens"),
        reserve_tokens_for_answer=_required_int(section, "reserve_tokens_for_answer"),
        deduplicate_by_chunk_id=_required_bool(section, "deduplicate_by_chunk_id"),
        deduplicate_by_text_hash=_required_bool(section, "deduplicate_by_text_hash"),
    )


def _parse_semantic_cache_config(section: Mapping[str, Any]) -> SemanticCacheConfig:
    return SemanticCacheConfig(
        enabled=_required_bool(section, "enabled"),
        provider=_required_str(section, "provider"),
        model_name=_required_str(section, "model_name"),
        vector_size=_required_int(section, "vector_size"),
        similarity_threshold=_required_float(section, "similarity_threshold"),
        ttl_seconds=_required_int(section, "ttl_seconds"),
    )


def _parse_timeouts_config(section: Mapping[str, Any]) -> TimeoutsConfig:
    return TimeoutsConfig(
        dense_retrieval_seconds=_required_float(
            section,
            "dense_retrieval_seconds",
        ),
        sparse_retrieval_seconds=_required_float(
            section,
            "sparse_retrieval_seconds",
        ),
        fusion_seconds=_required_float(section, "fusion_seconds"),
        structure_expansion_seconds=_required_float(
            section,
            "structure_expansion_seconds",
        ),
        context_packing_seconds=_required_float(section, "context_packing_seconds"),
    )


def _parse_observability_config(section: Mapping[str, Any]) -> ObservabilityConfig:
    return ObservabilityConfig(
        emit_events=_required_bool(section, "emit_events"),
        include_trace=_required_bool(section, "include_trace"),
        include_candidate_debug=_required_bool(section, "include_candidate_debug"),
        include_payload_debug=_required_bool(section, "include_payload_debug"),
    )


def _required_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)

    if value is None:
        raise RetrievalConfigurationError(f"Missing required config section: {key}")

    if not isinstance(value, Mapping):
        raise RetrievalConfigurationError(f"Config section must be a table: {key}")

    return value


def _required_str(section: Mapping[str, Any], key: str) -> str:
    value = _required_value(section, key)

    if not isinstance(value, str):
        raise RetrievalConfigurationError(
            f"Config value {key!r} must be a string, got {type(value).__name__}."
        )

    cleaned = value.strip()
    if not cleaned:
        raise RetrievalConfigurationError(f"Config value {key!r} cannot be empty.")

    return cleaned


def _required_int(section: Mapping[str, Any], key: str) -> int:
    value = _required_value(section, key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrievalConfigurationError(
            f"Config value {key!r} must be an integer, got {type(value).__name__}."
        )

    return value


def _required_float(section: Mapping[str, Any], key: str) -> float:
    value = _required_value(section, key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalConfigurationError(
            f"Config value {key!r} must be numeric, got {type(value).__name__}."
        )

    return float(value)


def _required_bool(section: Mapping[str, Any], key: str) -> bool:
    value = _required_value(section, key)

    if not isinstance(value, bool):
        raise RetrievalConfigurationError(
            f"Config value {key!r} must be boolean, got {type(value).__name__}."
        )

    return value


def _required_enum(
    section: Mapping[str, Any],
    key: str,
    *,
    enum_type: type[RetrievalMode] | type[ConfidenceBand],
) -> RetrievalMode | ConfidenceBand:
    raw_value = _required_str(section, key)

    try:
        return enum_type(raw_value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise RetrievalConfigurationError(
            f"Invalid value for config key {key!r}: {raw_value!r}. "
            f"Allowed values: {allowed}."
        ) from exc


def _required_value(section: Mapping[str, Any], key: str) -> Any:
    if key not in section:
        raise RetrievalConfigurationError(f"Missing required config key: {key}")

    return section[key]


def _validate_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def _validate_probability(name: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")