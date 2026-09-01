from __future__ import annotations

from dataclasses import dataclass, replace

from rag.retrieval.contracts.request import SourceConstraints


_DATE_FILTER_KEYS = frozenset(
    {
        "date",
        "date_from",
        "date_to",
        "start_date",
        "end_date",
        "created_at",
        "updated_at",
        "published_at",
        "modified_at",
        "year",
        "date_range",
    }
)

_DOCUMENT_TYPE_FILTER_KEYS = frozenset(
    {
        "doc_type",
        "document_type",
        "file_type",
        "file_ext",
        "mime_type",
        "parser_family",
    }
)


@dataclass(frozen=True)
class ScopeRelaxation:
    constraints: SourceConstraints
    changed: bool
    removed_constraint: str | None = None


def has_relaxable_scope(constraints: SourceConstraints) -> bool:
    return relax_next_scope(constraints).changed


def relax_next_scope(constraints: SourceConstraints) -> ScopeRelaxation:
    """
    Remove exactly one constraint layer.

    Tenant and user isolation are not represented by SourceConstraints and are
    therefore never touched here.
    """
    payload_filters = dict(constraints.payload_filters)

    date_keys = tuple(
        key
        for key in payload_filters
        if str(key).strip().lower() in _DATE_FILTER_KEYS
    )
    if date_keys:
        for key in date_keys:
            payload_filters.pop(key, None)

        return ScopeRelaxation(
            constraints=replace(
                constraints,
                payload_filters=payload_filters,
            ),
            changed=True,
            removed_constraint=f"date_filters:{','.join(map(str, date_keys))}",
        )

    document_type_keys = tuple(
        key
        for key in payload_filters
        if str(key).strip().lower() in _DOCUMENT_TYPE_FILTER_KEYS
    )
    if document_type_keys:
        for key in document_type_keys:
            payload_filters.pop(key, None)

        return ScopeRelaxation(
            constraints=replace(
                constraints,
                payload_filters=payload_filters,
            ),
            changed=True,
            removed_constraint=(
                f"document_type_filters:{','.join(map(str, document_type_keys))}"
            ),
        )

    # Narrow document-local constraints before broad source constraints.
    if constraints.section_id:
        return ScopeRelaxation(
            constraints=replace(constraints, section_id=None),
            changed=True,
            removed_constraint="section_id",
        )

    if constraints.heading_path:
        return ScopeRelaxation(
            constraints=replace(constraints, heading_path=()),
            changed=True,
            removed_constraint="heading_path",
        )

    if constraints.ir_id:
        return ScopeRelaxation(
            constraints=replace(constraints, ir_id=None),
            changed=True,
            removed_constraint="ir_id",
        )

    if constraints.parser_doc_id:
        return ScopeRelaxation(
            constraints=replace(constraints, parser_doc_id=None),
            changed=True,
            removed_constraint="parser_doc_id",
        )

    if constraints.job_id:
        return ScopeRelaxation(
            constraints=replace(constraints, job_id=None),
            changed=True,
            removed_constraint="job_id",
        )

    if constraints.source_id:
        return ScopeRelaxation(
            constraints=replace(constraints, source_id=None),
            changed=True,
            removed_constraint="source_id",
        )

    return ScopeRelaxation(
        constraints=constraints,
        changed=False,
        removed_constraint=None,
    )