from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, TypeVar

from rag.retrieval.contracts.config.models import (
    ContextDefaultsConfig,
    ContextLimitsConfig,
    ContractSchemaConfig,
    MemoryDefaultsConfig,
    MemoryLimitsConfig,
    QueryProcessingDefaultsConfig,
    QueryProcessingLimitsConfig,
    QueryTextLimitsConfig,
    RequestDefaultsConfig,
    RequestLimitsConfig,
    RetrievalContractConfig,
    SourceConstraintDefaultsConfig,
    ValidationConfig,
)
from rag.retrieval.contracts.enums import (
    MemoryScope,
    QueryProcessingMode,
    RetrievalMode,
)


EnumT = TypeVar("EnumT", bound=Enum)


class RetrievalContractConfigParseError(ValueError):
    """Raised when raw retrieval contract config cannot be parsed."""


def parse_retrieval_contract_config(
    raw: Mapping[str, Any],
) -> RetrievalContractConfig:
    schema = _table(raw, "schema")

    request = _table(raw, "request")
    request_defaults = _table(request, "request.defaults", "defaults")
    request_limits = _table(request, "request.limits", "limits")

    source_constraints = _table(raw, "source_constraints")
    source_constraint_defaults = _table(
        source_constraints,
        "source_constraints.defaults",
        "defaults",
    )

    memory = _table(raw, "memory")
    memory_defaults = _table(memory, "memory.defaults", "defaults")
    memory_limits = _table(memory, "memory.limits", "limits")

    query_processing = _table(raw, "query_processing")
    query_processing_defaults = _table(
        query_processing,
        "query_processing.defaults",
        "defaults",
    )
    query_processing_limits = _table(
        query_processing,
        "query_processing.limits",
        "limits",
    )

    query_text = _table(raw, "query_text")
    query_text_limits = _table(query_text, "query_text.limits", "limits")

    context = _table(raw, "context")
    context_defaults = _table(context, "context.defaults", "defaults")
    context_limits = _table(context, "context.limits", "limits")

    validation = _table(raw, "validation")

    config = RetrievalContractConfig(
        schema=ContractSchemaConfig(
            version=_str(schema, "schema.version", "version"),
        ),
        request_defaults=RequestDefaultsConfig(
            mode=_enum(
                request_defaults,
                "request.defaults.mode",
                "mode",
                RetrievalMode,
            ),
            top_k=_int(request_defaults, "request.defaults.top_k", "top_k"),
        ),
        request_limits=RequestLimitsConfig(
            max_top_k=_int(
                request_limits,
                "request.limits.max_top_k",
                "max_top_k",
            ),
            max_chat_history_turns=_int(
                request_limits,
                "request.limits.max_chat_history_turns",
                "max_chat_history_turns",
            ),
            max_metadata_keys=_int(
                request_limits,
                "request.limits.max_metadata_keys",
                "max_metadata_keys",
            ),
        ),
        source_constraints=SourceConstraintDefaultsConfig(
            allow_global_search=_bool(
                source_constraint_defaults,
                "source_constraints.defaults.allow_global_search",
                "allow_global_search",
            ),
            allow_payload_filters=_bool(
                source_constraint_defaults,
                "source_constraints.defaults.allow_payload_filters",
                "allow_payload_filters",
            ),
            max_heading_path_depth=_int(
                source_constraint_defaults,
                "source_constraints.defaults.max_heading_path_depth",
                "max_heading_path_depth",
            ),
            max_payload_filter_keys=_int(
                source_constraint_defaults,
                "source_constraints.defaults.max_payload_filter_keys",
                "max_payload_filter_keys",
            ),
        ),
        memory_defaults=MemoryDefaultsConfig(
            enabled=_bool(
                memory_defaults,
                "memory.defaults.enabled",
                "enabled",
            ),
            scope=_enum(
                memory_defaults,
                "memory.defaults.scope",
                "scope",
                MemoryScope,
            ),
            include_short_term=_bool(
                memory_defaults,
                "memory.defaults.include_short_term",
                "include_short_term",
            ),
            include_long_term=_bool(
                memory_defaults,
                "memory.defaults.include_long_term",
                "include_long_term",
            ),
            max_chat_turns=_int(
                memory_defaults,
                "memory.defaults.max_chat_turns",
                "max_chat_turns",
            ),
            max_user_turns=_int(
                memory_defaults,
                "memory.defaults.max_user_turns",
                "max_user_turns",
            ),
            max_assistant_turns=_int(
                memory_defaults,
                "memory.defaults.max_assistant_turns",
                "max_assistant_turns",
            ),
            preserve_recent_turns=_int(
                memory_defaults,
                "memory.defaults.preserve_recent_turns",
                "preserve_recent_turns",
            ),
            memory_top_k=_int(
                memory_defaults,
                "memory.defaults.memory_top_k",
                "memory_top_k",
            ),
            token_budget=_int(
                memory_defaults,
                "memory.defaults.token_budget",
                "token_budget",
            ),
        ),
        memory_limits=MemoryLimitsConfig(
            max_chat_turns=_int(
                memory_limits,
                "memory.limits.max_chat_turns",
                "max_chat_turns",
            ),
            max_user_turns=_int(
                memory_limits,
                "memory.limits.max_user_turns",
                "max_user_turns",
            ),
            max_assistant_turns=_int(
                memory_limits,
                "memory.limits.max_assistant_turns",
                "max_assistant_turns",
            ),
            max_preserve_recent_turns=_int(
                memory_limits,
                "memory.limits.max_preserve_recent_turns",
                "max_preserve_recent_turns",
            ),
            max_memory_top_k=_int(
                memory_limits,
                "memory.limits.max_memory_top_k",
                "max_memory_top_k",
            ),
            max_token_budget=_int(
                memory_limits,
                "memory.limits.max_token_budget",
                "max_token_budget",
            ),
        ),
        query_processing_defaults=QueryProcessingDefaultsConfig(
            mode=_enum(
                query_processing_defaults,
                "query_processing.defaults.mode",
                "mode",
                QueryProcessingMode,
            ),
            allow_query_rewrite=_bool(
                query_processing_defaults,
                "query_processing.defaults.allow_query_rewrite",
                "allow_query_rewrite",
            ),
            allow_filter_extraction=_bool(
                query_processing_defaults,
                "query_processing.defaults.allow_filter_extraction",
                "allow_filter_extraction",
            ),
            allow_intent_classification=_bool(
                query_processing_defaults,
                "query_processing.defaults.allow_intent_classification",
                "allow_intent_classification",
            ),
            allow_hypothetical_answer=_bool(
                query_processing_defaults,
                "query_processing.defaults.allow_hypothetical_answer",
                "allow_hypothetical_answer",
            ),
            max_rewritten_queries=_int(
                query_processing_defaults,
                "query_processing.defaults.max_rewritten_queries",
                "max_rewritten_queries",
            ),
            max_extracted_terms=_int(
                query_processing_defaults,
                "query_processing.defaults.max_extracted_terms",
                "max_extracted_terms",
            ),
        ),
        query_processing_limits=QueryProcessingLimitsConfig(
            max_rewritten_queries=_int(
                query_processing_limits,
                "query_processing.limits.max_rewritten_queries",
                "max_rewritten_queries",
            ),
            max_extracted_terms=_int(
                query_processing_limits,
                "query_processing.limits.max_extracted_terms",
                "max_extracted_terms",
            ),
            max_search_queries=_int(
                query_processing_limits,
                "query_processing.limits.max_search_queries",
                "max_search_queries",
            ),
        ),
        query_text_limits=QueryTextLimitsConfig(
            min_chars=_int(
                query_text_limits,
                "query_text.limits.min_chars",
                "min_chars",
            ),
            max_chars=_int(
                query_text_limits,
                "query_text.limits.max_chars",
                "max_chars",
            ),
            too_short_max_chars=_int(
                query_text_limits,
                "query_text.limits.too_short_max_chars",
                "too_short_max_chars",
            ),
            too_long_min_chars=_int(
                query_text_limits,
                "query_text.limits.too_long_min_chars",
                "too_long_min_chars",
            ),
            low_information_max_terms=_int(
                query_text_limits,
                "query_text.limits.low_information_max_terms",
                "low_information_max_terms",
            ),
        ),
        context_defaults=ContextDefaultsConfig(
            max_context_blocks=_int(
                context_defaults,
                "context.defaults.max_context_blocks",
                "max_context_blocks",
            ),
            max_total_tokens=_int(
                context_defaults,
                "context.defaults.max_total_tokens",
                "max_total_tokens",
            ),
            max_block_tokens=_int(
                context_defaults,
                "context.defaults.max_block_tokens",
                "max_block_tokens",
            ),
            reserve_tokens_for_answer=_int(
                context_defaults,
                "context.defaults.reserve_tokens_for_answer",
                "reserve_tokens_for_answer",
            ),
            deduplicate_by_chunk_id=_bool(
                context_defaults,
                "context.defaults.deduplicate_by_chunk_id",
                "deduplicate_by_chunk_id",
            ),
            deduplicate_by_text_hash=_bool(
                context_defaults,
                "context.defaults.deduplicate_by_text_hash",
                "deduplicate_by_text_hash",
            ),
        ),
        context_limits=ContextLimitsConfig(
            max_context_blocks=_int(
                context_limits,
                "context.limits.max_context_blocks",
                "max_context_blocks",
            ),
            max_total_tokens=_int(
                context_limits,
                "context.limits.max_total_tokens",
                "max_total_tokens",
            ),
            max_block_tokens=_int(
                context_limits,
                "context.limits.max_block_tokens",
                "max_block_tokens",
            ),
            max_reserve_tokens_for_answer=_int(
                context_limits,
                "context.limits.max_reserve_tokens_for_answer",
                "max_reserve_tokens_for_answer",
            ),
        ),
        validation=ValidationConfig(
            strict_enums=_bool(
                validation,
                "validation.strict_enums",
                "strict_enums",
            ),
            reject_blank_optional_strings=_bool(
                validation,
                "validation.reject_blank_optional_strings",
                "reject_blank_optional_strings",
            ),
            reject_non_finite_scores=_bool(
                validation,
                "validation.reject_non_finite_scores",
                "reject_non_finite_scores",
            ),
            reject_empty_context_blocks=_bool(
                validation,
                "validation.reject_empty_context_blocks",
                "reject_empty_context_blocks",
            ),
            reject_empty_candidates=_bool(
                validation,
                "validation.reject_empty_candidates",
                "reject_empty_candidates",
            ),
        ),
    )

    config.validate()
    return config


def _table(
    data: Mapping[str, Any],
    dotted_name: str,
    key: str | None = None,
) -> Mapping[str, Any]:
    actual_key = key or dotted_name
    value = data.get(actual_key)

    if value is None:
        raise RetrievalContractConfigParseError(
            f"Missing required config table: {dotted_name}"
        )

    if not isinstance(value, Mapping):
        raise RetrievalContractConfigParseError(
            f"Config value must be a table: {dotted_name}"
        )

    return value


def _str(data: Mapping[str, Any], dotted_name: str, key: str) -> str:
    value = _required(data, dotted_name, key)

    if not isinstance(value, str):
        raise RetrievalContractConfigParseError(
            f"Config value must be a string: {dotted_name}"
        )

    if not value.strip():
        raise RetrievalContractConfigParseError(
            f"Config string cannot be empty: {dotted_name}"
        )

    return value


def _bool(data: Mapping[str, Any], dotted_name: str, key: str) -> bool:
    value = _required(data, dotted_name, key)

    if not isinstance(value, bool):
        raise RetrievalContractConfigParseError(
            f"Config value must be boolean: {dotted_name}"
        )

    return value


def _int(data: Mapping[str, Any], dotted_name: str, key: str) -> int:
    value = _required(data, dotted_name, key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrievalContractConfigParseError(
            f"Config value must be integer: {dotted_name}"
        )

    return value


def _enum(
    data: Mapping[str, Any],
    dotted_name: str,
    key: str,
    enum_type: type[EnumT],
) -> EnumT:
    value = _str(data, dotted_name, key)

    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise RetrievalContractConfigParseError(
            f"Invalid enum value for {dotted_name}: {value!r}. "
            f"Allowed values: {allowed}."
        ) from exc


def _required(data: Mapping[str, Any], dotted_name: str, key: str) -> Any:
    if key not in data:
        raise RetrievalContractConfigParseError(
            f"Missing required config value: {dotted_name}"
        )

    return data[key]