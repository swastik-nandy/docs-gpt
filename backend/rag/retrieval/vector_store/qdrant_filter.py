from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from qdrant_client.http import models

from rag.retrieval.contracts.query import ExtractedFilter, NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest


Payload = dict[str, Any]


@dataclass(frozen=True)
class QdrantPayloadKeys:
    tenant_id: str = "tenant_id"
    user_id: str = "user_id"
    source_id: str = "source_id"
    job_id: str = "job_id"
    source_doc_id: str = "source_doc_id"
    doc_id: str = "doc_id"
    file_id: str = "file_id"
    file_name: str = "file_name"
    file_path: str = "file_path"
    file_ext: str = "file_ext"
    parser_family: str = "parser_family"
    language: str = "language"
    chunk_id: str = "chunk_id"
    section_id: str = "section_id"
    heading_path: str = "heading_path"

    def source_identity_keys(self) -> tuple[str, ...]:
        return (
            self.source_id,
            self.source_doc_id,
            self.doc_id,
            self.file_id,
            self.file_name,
            self.file_path,
        )


@dataclass(frozen=True)
class QdrantFilterBuilderConfig:
    payload_keys: QdrantPayloadKeys = field(default_factory=QdrantPayloadKeys)
    always_filter_by_tenant: bool = True
    include_user_filter: bool = False
    allow_extracted_filters: bool = True
    allow_source_constraints: bool = True
    max_filter_values: int = 128

    def validate(self) -> None:
        if not isinstance(self.always_filter_by_tenant, bool):
            raise ValueError("always_filter_by_tenant must be boolean.")

        if not isinstance(self.include_user_filter, bool):
            raise ValueError("include_user_filter must be boolean.")

        if not isinstance(self.allow_extracted_filters, bool):
            raise ValueError("allow_extracted_filters must be boolean.")

        if not isinstance(self.allow_source_constraints, bool):
            raise ValueError("allow_source_constraints must be boolean.")

        if self.max_filter_values <= 0:
            raise ValueError("max_filter_values must be greater than zero.")


@dataclass(frozen=True)
class QdrantFilterBuildResult:
    filter: models.Filter | None
    applied_constraints: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: Payload = field(default_factory=dict)

    @property
    def has_filter(self) -> bool:
        return self.filter is not None


class QdrantFilterBuilder:
    """
    Builds Qdrant payload filters from retrieval request constraints.

    This is the source/file gatekeeper.

    It supports:
    - tenant isolation
    - optional user isolation
    - explicit source/file constraints
    - agent/query-extracted filters
    - metadata payload filters

    Dense, sparse, hybrid, and agentic retrieval should all use this same builder.
    """

    def __init__(
        self,
        config: QdrantFilterBuilderConfig | None = None,
    ) -> None:
        self.config = config or QdrantFilterBuilderConfig()
        self.config.validate()

    def build(
        self,
        *,
        request: RetrievalRequest,
        normalized_query: NormalizedQuery | None = None,
    ) -> QdrantFilterBuildResult:
        request.validate()
        if normalized_query is not None:
            normalized_query.validate()

        must: list[models.Condition] = []
        should: list[models.Condition] = []
        must_not: list[models.Condition] = []
        applied: list[str] = []
        warnings: list[str] = []

        self._add_tenant_filter(
            request=request,
            must=must,
            applied=applied,
            warnings=warnings,
        )

        self._add_user_filter(
            request=request,
            must=must,
            applied=applied,
        )

        if self.config.allow_source_constraints:
            self._add_source_constraints(
                request=request,
                must=must,
                should=should,
                must_not=must_not,
                applied=applied,
                warnings=warnings,
            )

        if self.config.allow_extracted_filters and normalized_query is not None:
            self._add_extracted_filters(
                normalized_query=normalized_query,
                must=must,
                should=should,
                must_not=must_not,
                applied=applied,
                warnings=warnings,
            )

        filter_obj = _build_filter_or_none(
            must=must,
            should=should,
            must_not=must_not,
        )

        return QdrantFilterBuildResult(
            filter=filter_obj,
            applied_constraints=tuple(applied),
            warnings=tuple(warnings),
            metadata={
                "must_count": len(must),
                "should_count": len(should),
                "must_not_count": len(must_not),
                "has_normalized_query": normalized_query is not None,
            },
        )

    def _add_tenant_filter(
        self,
        *,
        request: RetrievalRequest,
        must: list[models.Condition],
        applied: list[str],
        warnings: list[str],
    ) -> None:
        tenant_id = _clean_optional(getattr(request.tenant, "tenant_id", None))

        if not tenant_id:
            if self.config.always_filter_by_tenant:
                raise ValueError("tenant_id is required for tenant-isolated retrieval.")

            warnings.append("tenant_id missing; tenant filter skipped")
            return

        must.append(_match_value(self.config.payload_keys.tenant_id, tenant_id))
        applied.append(self.config.payload_keys.tenant_id)

    def _add_user_filter(
        self,
        *,
        request: RetrievalRequest,
        must: list[models.Condition],
        applied: list[str],
    ) -> None:
        if not self.config.include_user_filter:
            return

        user_id = _clean_optional(getattr(request.tenant, "user_id", None))
        if not user_id:
            return

        must.append(_match_value(self.config.payload_keys.user_id, user_id))
        applied.append(self.config.payload_keys.user_id)

    def _add_source_constraints(
        self,
        *,
        request: RetrievalRequest,
        must: list[models.Condition],
        should: list[models.Condition],
        must_not: list[models.Condition],
        applied: list[str],
        warnings: list[str],
    ) -> None:
        constraints = getattr(request, "source_constraints", None)
        if constraints is None:
            return

        payload_filters = _extract_payload_filters(constraints)
        for key, value in payload_filters.items():
            condition = _condition_from_key_value(
                key=key,
                value=value,
                max_values=self.config.max_filter_values,
            )
            if condition is None:
                warnings.append(f"source payload filter skipped: {key}")
                continue

            must.append(condition)
            applied.append(key)

        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("source_ids", "source_id"),
            payload_key=self.config.payload_keys.source_id,
            must=must,
            applied=applied,
        )

        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("job_ids", "job_id"),
            payload_key=self.config.payload_keys.job_id,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("source_doc_ids", "source_doc_id"),
            payload_key=self.config.payload_keys.source_doc_id,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("doc_ids", "doc_id"),
            payload_key=self.config.payload_keys.doc_id,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("file_ids", "file_id"),
            payload_key=self.config.payload_keys.file_id,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("file_names", "file_name"),
            payload_key=self.config.payload_keys.file_name,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("file_paths", "file_path"),
            payload_key=self.config.payload_keys.file_path,
            must=must,
            applied=applied,
        )
        self._add_named_constraint_values(
            constraints=constraints,
            field_names=("file_exts", "file_ext"),
            payload_key=self.config.payload_keys.file_ext,
            must=must,
            applied=applied,
        )

        excluded_sources = _first_existing_values(
            constraints,
            ("excluded_source_ids", "exclude_source_ids"),
        )
        if excluded_sources:
            must_not.append(
                _match_any_or_value(
                    self.config.payload_keys.source_id,
                    excluded_sources,
                    max_values=self.config.max_filter_values,
                )
            )
            applied.append(f"exclude:{self.config.payload_keys.source_id}")

        preferred_sources = _first_existing_values(
            constraints,
            ("preferred_source_ids", "prefer_source_ids"),
        )
        if preferred_sources:
            should.append(
                _match_any_or_value(
                    self.config.payload_keys.source_id,
                    preferred_sources,
                    max_values=self.config.max_filter_values,
                )
            )
            applied.append(f"prefer:{self.config.payload_keys.source_id}")

    def _add_named_constraint_values(
        self,
        *,
        constraints: Any,
        field_names: tuple[str, ...],
        payload_key: str,
        must: list[models.Condition],
        applied: list[str],
    ) -> None:
        values = _first_existing_values(constraints, field_names)
        if not values:
            return

        must.append(
            _match_any_or_value(
                payload_key,
                values,
                max_values=self.config.max_filter_values,
            )
        )
        applied.append(payload_key)

    def _add_extracted_filters(
        self,
        *,
        normalized_query: NormalizedQuery,
        must: list[models.Condition],
        should: list[models.Condition],
        must_not: list[models.Condition],
        applied: list[str],
        warnings: list[str],
    ) -> None:
        for extracted_filter in normalized_query.extracted_filters:
            condition = _condition_from_extracted_filter(
                extracted_filter,
                max_values=self.config.max_filter_values,
            )

            if condition is None:
                warnings.append(f"extracted filter skipped: {extracted_filter.key}")
                continue

            operator = extracted_filter.operator.strip().lower()

            if operator in {"not", "!=", "not_eq", "exclude"}:
                must_not.append(condition)
                applied.append(f"exclude:{extracted_filter.key}")
                continue

            if operator in {"prefer", "should"}:
                should.append(condition)
                applied.append(f"prefer:{extracted_filter.key}")
                continue

            must.append(condition)
            applied.append(extracted_filter.key)


def build_qdrant_filter(
    *,
    request: RetrievalRequest,
    normalized_query: NormalizedQuery | None = None,
    config: QdrantFilterBuilderConfig | None = None,
) -> QdrantFilterBuildResult:
    return QdrantFilterBuilder(config).build(
        request=request,
        normalized_query=normalized_query,
    )


def build_exact_file_filter(
    *,
    tenant_id: str,
    file_name: str | None = None,
    file_path: str | None = None,
    source_doc_id: str | None = None,
    payload_keys: QdrantPayloadKeys | None = None,
) -> models.Filter:
    keys = payload_keys or QdrantPayloadKeys()
    must: list[models.Condition] = [
        _match_value(keys.tenant_id, tenant_id),
    ]

    if file_name:
        must.append(_match_value(keys.file_name, file_name))

    if file_path:
        must.append(_match_value(keys.file_path, file_path))

    if source_doc_id:
        must.append(_match_value(keys.source_doc_id, source_doc_id))

    return models.Filter(must=must)


def _build_filter_or_none(
    *,
    must: list[models.Condition],
    should: list[models.Condition],
    must_not: list[models.Condition],
) -> models.Filter | None:
    if not must and not should and not must_not:
        return None

    kwargs: Payload = {}

    if must:
        kwargs["must"] = must

    if should:
        kwargs["should"] = should

    if must_not:
        kwargs["must_not"] = must_not

    return models.Filter(**kwargs)


def _condition_from_extracted_filter(
    extracted_filter: ExtractedFilter,
    *,
    max_values: int,
) -> models.Condition | None:
    key = extracted_filter.key.strip()
    if not key:
        return None

    return _condition_from_key_operator_value(
        key=key,
        operator=extracted_filter.operator,
        value=extracted_filter.value,
        max_values=max_values,
    )


def _condition_from_key_value(
    *,
    key: str,
    value: Any,
    max_values: int,
) -> models.Condition | None:
    return _condition_from_key_operator_value(
        key=key,
        operator="eq",
        value=value,
        max_values=max_values,
    )


def _condition_from_key_operator_value(
    *,
    key: str,
    operator: str,
    value: Any,
    max_values: int,
) -> models.Condition | None:
    cleaned_key = key.strip()
    if not cleaned_key:
        return None

    normalized_operator = operator.strip().lower()

    if normalized_operator in {"eq", "=", "==", "not", "!=", "not_eq", "exclude"}:
        if _is_sequence_value(value):
            return _match_any_or_value(
                cleaned_key,
                _clean_values(value),
                max_values=max_values,
            )

        cleaned = _clean_scalar(value)
        if cleaned is None:
            return None

        return _match_value(cleaned_key, cleaned)

    if normalized_operator in {"in", "any", "one_of", "prefer", "should"}:
        values = _clean_values(value)
        if not values:
            return None

        return _match_any_or_value(
            cleaned_key,
            values,
            max_values=max_values,
        )

    if normalized_operator in {"gt", "gte", "lt", "lte", "range"}:
        range_obj = _range_from_value(
            operator=normalized_operator,
            value=value,
        )
        if range_obj is None:
            return None

        return models.FieldCondition(
            key=cleaned_key,
            range=range_obj,
        )

    if normalized_operator in {"text", "match_text"}:
        cleaned = _clean_scalar(value)
        if cleaned is None:
            return None

        return models.FieldCondition(
            key=cleaned_key,
            match=models.MatchText(text=str(cleaned)),
        )

    cleaned = _clean_scalar(value)
    if cleaned is None:
        return None

    return _match_value(cleaned_key, cleaned)


def _range_from_value(
    *,
    operator: str,
    value: Any,
) -> models.Range | None:
    if isinstance(value, Mapping):
        kwargs: Payload = {}

        for key in ("gt", "gte", "lt", "lte"):
            if key in value and value[key] is not None:
                kwargs[key] = float(value[key])

        if not kwargs:
            return None

        return models.Range(**kwargs)

    if value is None:
        return None

    number = float(value)

    if operator == "gt":
        return models.Range(gt=number)

    if operator == "gte":
        return models.Range(gte=number)

    if operator == "lt":
        return models.Range(lt=number)

    if operator == "lte":
        return models.Range(lte=number)

    return None


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(
        key=key,
        match=models.MatchValue(value=value),
    )


def _match_any_or_value(
    key: str,
    values: Sequence[Any],
    *,
    max_values: int,
) -> models.FieldCondition:
    cleaned_values = tuple(_limit_values(_clean_values(values), max_values=max_values))

    if len(cleaned_values) == 1:
        return _match_value(key, cleaned_values[0])

    return models.FieldCondition(
        key=key,
        match=models.MatchAny(any=list(cleaned_values)),
    )


def _extract_payload_filters(constraints: Any) -> Mapping[str, Any]:
    for attr_name in ("payload_filters", "metadata_filters", "filters"):
        value = getattr(constraints, attr_name, None)
        if isinstance(value, Mapping):
            return value

    return {}


def _first_existing_values(
    obj: Any,
    names: tuple[str, ...],
) -> tuple[Any, ...]:
    for name in names:
        value = getattr(obj, name, None)
        values = _clean_values(value)
        if values:
            return values

    return ()


def _clean_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()

    if isinstance(value, Mapping):
        return tuple(value.values())

    if isinstance(value, Iterable):
        cleaned_values: list[Any] = []
        for item in value:
            cleaned = _clean_scalar(item)
            if cleaned is not None:
                cleaned_values.append(cleaned)

        return tuple(cleaned_values)

    cleaned = _clean_scalar(value)
    return (cleaned,) if cleaned is not None else ()


def _limit_values(
    values: Sequence[Any],
    *,
    max_values: int,
) -> tuple[Any, ...]:
    if len(values) > max_values:
        raise ValueError(f"filter value count cannot exceed {max_values}.")

    return tuple(values)


def _clean_scalar(value: Any) -> Any | None:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None

    return value


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    )
