from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.chunking.config import build_chunk_token_budget
from rag.ingestion.data_transformation.chunking.config import get_chunking_config
from rag.ingestion.data_transformation.chunking.tokenization import count_tokens
from rag.ingestion.data_transformation.chunking.tokenization import tokenizer_runtime_metadata
from rag.ingestion.data_transformation.parse_chunk_pipeline import run_parse_chunk_pipeline


#--------------------CONSTANTS--------------------

SUPPORTED_SUFFIXES = {
    ".md",
    ".mdx",
    ".markdown",
    ".mdown",
    ".mkd",
    ".mdwn",
}

LEGACY_SUBTYPES = {
    "text_block",
    "text_atom",
    "text_split",
    "note",
    "code_block",
    "table_row",
    "list_item",
    "component_block",
}

DEPENDENT_ENDINGS = (
    "as follows",
    "as follows.",
    "shown below",
    "shown below.",
    "looks like this",
    "looks like this.",
    "in the following example",
    "in the following example.",
    "in the following examples",
    "in the following examples.",
)


#--------------------DISCOVERY--------------------

def discover_files(
    inputs: list[str],
) -> list[Path]:
    requested = (
        inputs
        or [
            "test_data/qdrant"
        ]
    )

    found: set[Path] = set()

    for raw in requested:
        path = (
            Path(
                raw
            )
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Input does not exist: {path}"
            )

        if path.is_file():
            if (
                path.suffix.lower()
                not in SUPPORTED_SUFFIXES
            ):
                raise ValueError(
                    "Unsupported Markdown file: "
                    f"{path}"
                )

            found.add(
                path
            )

            continue

        for candidate in path.rglob(
            "*"
        ):
            if (
                candidate.is_file()
                and candidate.suffix.lower()
                in SUPPORTED_SUFFIXES
            ):
                found.add(
                    candidate.resolve()
                )

    if not found:
        raise FileNotFoundError(
            "No Markdown/MDX files discovered"
        )

    return sorted(
        found,
        key=lambda value: str(
            value
        ).lower(),
    )


#--------------------PARSE ACCESS--------------------

def collect_sections(
    parse_result: Any,
) -> list[dict[str, Any]]:
    """
    Resolve exactly one canonical section source.

    Important:
    do not combine top-level sections and unit sections,
    otherwise parser blocks can be counted twice.
    """
    units = _value(
        parse_result,
        "units",
    )

    if (
        isinstance(
            units,
            list,
        )
        and units
    ):
        output: list[
            dict[str, Any]
        ] = []

        for unit in units:
            sections = _value(
                unit,
                "sections",
            )

            if not isinstance(
                sections,
                list,
            ):
                continue

            output.extend(
                section
                for section in sections
                if isinstance(
                    section,
                    dict,
                )
            )

        if output:
            return output

    iterator = getattr(
        parse_result,
        "iter_sections",
        None,
    )

    if callable(
        iterator
    ):
        output = [
            section
            for section in iterator()
            if isinstance(
                section,
                dict,
            )
        ]

        if output:
            return output

    sections = _value(
        parse_result,
        "sections",
    )

    if isinstance(
        sections,
        list,
    ):
        return [
            section
            for section in sections
            if isinstance(
                section,
                dict,
            )
        ]

    unit = _value(
        parse_result,
        "unit",
    )

    sections = _value(
        unit,
        "sections",
    )

    if isinstance(
        sections,
        list,
    ):
        return [
            section
            for section in sections
            if isinstance(
                section,
                dict,
            )
        ]

    return []


def collect_blocks(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[
        dict[str, Any]
    ] = []

    for section in sections:
        blocks = section.get(
            "blocks"
        )

        if not isinstance(
            blocks,
            list,
        ):
            continue

        output.extend(
            block
            for block in blocks
            if isinstance(
                block,
                dict,
            )
        )

    return output


#--------------------IDENTITY--------------------

def parser_ir_id(
    block: dict[str, Any],
) -> str:
    meta = _dict(
        block.get(
            "meta"
        )
    )

    return (
        _text(
            block.get(
                "id"
            )
        )
        or _text(
            block.get(
                "ir_id"
            )
        )
        or _text(
            meta.get(
                "ir_id"
            )
        )
    )


def chunk_ir_id(
    chunk: Any,
) -> str:
    metadata = _dict(
        chunk.metadata
    )

    return (
        _text(
            metadata.get(
                "source_ir_id"
            )
        )
        or _text(
            metadata.get(
                "ir_id"
            )
        )
    )


#--------------------LOCAL SCOPE--------------------

def parser_scope_id(
    block: dict[str, Any],
) -> str:
    meta = _dict(
        block.get(
            "meta"
        )
    )

    return (
        _text(
            block.get(
                "local_scope_id"
            )
        )
        or _text(
            meta.get(
                "local_scope_id"
            )
        )
    )


def chunk_ir_ids(
    chunk: Any,
) -> set[str]:
    metadata = _dict(
        chunk.metadata
    )

    output: set[str] = set()

    values = metadata.get(
        "source_ir_ids"
    )

    if isinstance(
        values,
        list,
    ):
        for value in values:
            normalized = _text(
                value
            )

            if normalized:
                output.add(
                    normalized
                )

    for key in (
        "source_ir_id",
        "ir_id",
    ):
        value = _text(
            metadata.get(
                key
            )
        )

        if value:
            output.add(
                value
            )

    return output


def chunk_scope_id(
    chunk: Any,
) -> str:
    return _text(
        _dict(
            chunk.metadata
        ).get(
            "local_scope_id"
        )
    )


def collect_parser_scope_ids(
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> set[str]:
    output = {
        value
        for block in blocks
        if (
            value
            := parser_scope_id(
                block
            )
        )
    }

    for section in sections:
        scopes = section.get(
            "local_scopes"
        )

        if not isinstance(
            scopes,
            list,
        ):
            continue

        for scope in scopes:
            if not isinstance(
                scope,
                dict,
            ):
                continue

            scope_id = _text(
                scope.get(
                    "scope_id"
                )
            )

            if scope_id:
                output.add(
                    scope_id
                )

    return output


#--------------------STRUCTURAL ARTIFACTS--------------------

def collect_selector_texts(
    sections: list[dict[str, Any]],
) -> set[str]:
    """
    Detect selectors using parser evidence rather than
    hardcoding Qdrant/vendor-specific text.
    """
    output: set[str] = set()

    for section in sections:
        artifacts = section.get(
            "structural_artifacts"
        )

        if not isinstance(
            artifacts,
            list,
        ):
            continue

        for artifact in artifacts:
            if not isinstance(
                artifact,
                dict,
            ):
                continue

            if (
                _text(
                    artifact.get(
                        "type"
                    )
                ).lower()
                != "selector"
            ):
                continue

            metadata = _dict(
                artifact.get(
                    "metadata"
                )
            )

            text = _normalize(
                metadata.get(
                    "text"
                )
            )

            if text:
                output.add(
                    text
                )

    return output


#--------------------RETRIEVAL KEYS--------------------

def retrieval_key(
    chunk: Any,
) -> tuple[Any, ...]:
    metadata = _dict(
        chunk.metadata
    )

    source_start = _first_not_none(
        metadata.get(
            "source_start"
        ),
        metadata.get(
            "start_offset"
        ),
    )

    source_end = _first_not_none(
        metadata.get(
            "source_end"
        ),
        metadata.get(
            "end_offset"
        ),
    )

    return (
        _normalize(
            chunk.text
        ),
        _text(
            chunk.heading
        ),
        _text(
            chunk.context
        ),
        source_start,
        source_end,
        chunk_ir_id(
            chunk
        ),
    )


#--------------------DEPENDENCY CHECK--------------------

def is_dependent(
    chunk: Any,
) -> bool:
    metadata = _dict(
        chunk.metadata
    )

    diagnostics = _dict(
        metadata.get(
            "quality_diagnostics"
        )
    )

    explicit = diagnostics.get(
        "dependent_fragment"
    )

    if isinstance(
        explicit,
        bool,
    ):
        return explicit

    subtype = _text(
        chunk.subtype
    ).lower()

    if subtype not in {
        "paragraph",
        "blockquote",
    }:
        return False

    text = _normalize(
        chunk.text
    )

    if not text:
        return True

    if text.endswith(
        ":"
    ):
        return True

    return text.lower().endswith(
        DEPENDENT_ENDINGS
    )


#--------------------DECORATION CHECK--------------------

def is_decorated(
    text: str,
) -> bool:
    value = text.lstrip()

    return (
        value.startswith(
            "Context:"
        )
        and "\nHeading:"
        in value
        and "\nText:"
        in value
    )


#--------------------SUSPICIOUS LABEL CHECK--------------------

def is_suspicious_compact_label(
    chunk: Any,
) -> bool:
    """
    Catch flattened UI/tab-selector style garbage such as
    long concatenated lowercase labels without relying on
    any specific vendor vocabulary.
    """
    if (
        _text(
            chunk.subtype
        ).lower()
        != "paragraph"
    ):
        return False

    text = _normalize(
        chunk.text
    )

    return (
        len(
            text
        )
        >= 24
        and " "
        not in text
        and text.isalpha()
        and text.islower()
    )


#--------------------ANALYSIS--------------------

def analyze_file(
    path: Path,
    inspect_limit: int,
) -> tuple[
    dict[str, Any],
    list[Any],
    list[Any],
]:
    config = get_chunking_config()

    budget = build_chunk_token_budget(
        config
    )

    result = run_parse_chunk_pipeline(
        path,
        enable_overlap=False,
        chunking_config=config,
    )

    sections = collect_sections(
        result.parse_result
    )

    blocks = collect_blocks(
        sections
    )

    chunks = list(
        result.chunks
    )

    parser_counts = [
        count_tokens(
            text,
            role=config.tokenization.role,
            add_special_tokens=(
                config
                .tokenization
                .add_special_tokens
            ),
        )
        for block in blocks
        if (
            text
            := _text(
                block.get(
                    "text"
                )
            )
        )
    ]

    exact_counts = [
        count_tokens(
            chunk.text,
            role=config.tokenization.role,
            add_special_tokens=(
                config
                .tokenization
                .add_special_tokens
            ),
        )
        for chunk in chunks
    ]

    token_mismatches = sum(
        chunk.tokens
        != exact
        for chunk, exact in zip(
            chunks,
            exact_counts,
        )
    )

    duplicate_ids = _duplicate_count(
        [
            _text(
                chunk.chunk_id
            )
            for chunk in chunks
        ]
    )

    duplicate_keys = _duplicate_count(
        [
            retrieval_key(
                chunk
            )
            for chunk in chunks
            if _text(
                chunk.text
            )
        ]
    )

    subtype_counts = Counter(
        _text(
            chunk.subtype
        )
        for chunk in chunks
    )

    parser_ids = {
        value
        for block in blocks
        if (
            value
            := parser_ir_id(
                block
            )
        )
    }

    represented_ids: set[str] = set()

    for chunk in chunks:
        represented_ids.update(
            chunk_ir_ids(
                chunk
            )
        )

    missing_ids = (
        parser_ids
        - represented_ids
    )

    parser_scopes = (
        collect_parser_scope_ids(
            sections,
            blocks,
        )
    )

    chunk_scopes = {
        value
        for chunk in chunks
        if (
            value
            := chunk_scope_id(
                chunk
            )
        )
    }

    lost_scopes = (
        parser_scopes
        - chunk_scopes
    )

    selector_texts = (
        collect_selector_texts(
            sections
        )
    )

    selector_text_leaks = 0
    selector_context_leaks = 0

    quality_scores: list[
        float
    ] = []

    quality_issues: Counter[
        str
    ] = Counter()

    for chunk in chunks:
        metadata = _dict(
            chunk.metadata
        )

        normalized_text = _normalize(
            chunk.text
        )

        if (
            normalized_text
            and normalized_text
            in selector_texts
        ):
            selector_text_leaks += 1

        contexts = {
            _normalize(
                chunk.heading
            ),
            _normalize(
                chunk.context
            ),
            _normalize(
                metadata.get(
                    "local_context"
                )
            ),
            _normalize(
                metadata.get(
                    "parent_context"
                )
            ),
            _normalize(
                metadata.get(
                    "parent_instruction"
                )
            ),
        }

        contexts.discard(
            ""
        )

        if selector_texts.intersection(
            contexts
        ):
            selector_context_leaks += 1

        score = metadata.get(
            "quality_score"
        )

        if (
            isinstance(
                score,
                (
                    int,
                    float,
                ),
            )
            and not isinstance(
                score,
                bool,
            )
        ):
            quality_scores.append(
                float(
                    score
                )
            )

        issues = metadata.get(
            "quality_issues"
        )

        if isinstance(
            issues,
            list,
        ):
            quality_issues.update(
                str(
                    issue
                )
                for issue in issues
                if str(
                    issue
                ).strip()
            )

    above_hard = sum(
        count
        > budget.effective_chunk_hard_tokens
        for count in exact_counts
    )

    fallback_chunks = sum(
        _dict(
            chunk.metadata
        ).get(
            "fallback_chunking"
        )
        is True
        for chunk in chunks
    )

    decorated_chunks = sum(
        is_decorated(
            chunk.text
        )
        for chunk in chunks
    )

    dependent_chunks = [
        chunk
        for chunk in chunks
        if is_dependent(
            chunk
        )
    ]

    suspicious_labels = [
        chunk
        for chunk in chunks
        if is_suspicious_compact_label(
            chunk
        )
    ]

    missing_heading = sum(
        not _text(
            chunk.heading
        )
        for chunk in chunks
    )

    missing_context = sum(
        not _text(
            chunk.context
        )
        for chunk in chunks
    )

    legacy_subtypes = sum(
        count
        for subtype, count
        in subtype_counts.items()
        if subtype
        in LEGACY_SUBTYPES
    )

    hard_failures: list[
        str
    ] = []

    warnings: list[
        str
    ] = []

    if not sections:
        hard_failures.append(
            "parser_produced_no_sections"
        )

    if not blocks:
        hard_failures.append(
            "parser_produced_no_ir_blocks"
        )

    if not chunks:
        hard_failures.append(
            "chunker_produced_no_chunks"
        )

    if duplicate_ids:
        hard_failures.append(
            f"duplicate_chunk_ids={duplicate_ids}"
        )

    if token_mismatches:
        hard_failures.append(
            f"stale_token_counts={token_mismatches}"
        )

    if above_hard:
        hard_failures.append(
            f"above_hard={above_hard}"
        )

    if legacy_subtypes:
        hard_failures.append(
            f"legacy_subtypes={legacy_subtypes}"
        )

    if selector_text_leaks:
        hard_failures.append(
            "selector_text_leaks="
            f"{selector_text_leaks}"
        )

    if selector_context_leaks:
        hard_failures.append(
            "selector_context_leaks="
            f"{selector_context_leaks}"
        )

    if fallback_chunks:
        hard_failures.append(
            f"fallback_chunks={fallback_chunks}"
        )

    if decorated_chunks:
        hard_failures.append(
            "decorated_chunk_texts="
            f"{decorated_chunks}"
        )

    if duplicate_keys:
        warnings.append(
            "duplicate_retrieval_keys="
            f"{duplicate_keys}"
        )

    if missing_ids:
        warnings.append(
            "unrepresented_parser_ir="
            f"{len(missing_ids)}"
        )

    if lost_scopes:
        warnings.append(
            "lost_local_scope_ids="
            f"{len(lost_scopes)}"
        )

    if dependent_chunks:
        warnings.append(
            "dependent_fragments="
            f"{len(dependent_chunks)}"
        )

    if suspicious_labels:
        warnings.append(
            "suspicious_compact_labels="
            f"{len(suspicious_labels)}"
        )

    if missing_heading:
        warnings.append(
            f"missing_heading={missing_heading}"
        )

    if missing_context:
        warnings.append(
            f"missing_context={missing_context}"
        )

    report = {
        "path": str(
            path
        ),
        "parser_sections": len(
            sections
        ),
        "parser_blocks": len(
            blocks
        ),
        "parser_tiny_under_50": sum(
            count < 50
            for count in parser_counts
        ),
        "final_chunks": len(
            chunks
        ),
        "final_tiny_under_50": sum(
            count < 50
            for count in exact_counts
        ),
        "minimum_tokens": min(
            exact_counts,
            default=0,
        ),
        "median_tokens": _median(
            exact_counts
        ),
        "average_tokens": _average(
            exact_counts
        ),
        "p90_tokens": _percentile(
            exact_counts,
            0.90,
        ),
        "maximum_tokens": max(
            exact_counts,
            default=0,
        ),
        "below_minimum": sum(
            count
            < budget.minimum_tokens
            for count in exact_counts
        ),
        "above_preferred": sum(
            count
            > budget.preferred_tokens
            for count in exact_counts
        ),
        "above_strict": sum(
            count
            > budget.strict_tokens
            for count in exact_counts
        ),
        "above_hard": above_hard,
        "duplicate_chunk_ids": duplicate_ids,
        "duplicate_retrieval_keys": duplicate_keys,
        "token_count_mismatches": token_mismatches,
        "legacy_subtypes": legacy_subtypes,
        "fallback_chunks": fallback_chunks,
        "parser_ir_ids": len(
            parser_ids
        ),
        "represented_ir_ids": len(
            represented_ids
        ),
        "missing_ir_ids": len(
            missing_ids
        ),
        "parser_local_scopes": len(
            parser_scopes
        ),
        "chunk_local_scopes": len(
            chunk_scopes
        ),
        "lost_local_scopes": len(
            lost_scopes
        ),
        "selector_artifacts": len(
            selector_texts
        ),
        "selector_text_leaks": (
            selector_text_leaks
        ),
        "selector_context_leaks": (
            selector_context_leaks
        ),
        "decorated_chunk_texts": (
            decorated_chunks
        ),
        "dependent_fragments": len(
            dependent_chunks
        ),
        "suspicious_compact_labels": len(
            suspicious_labels
        ),
        "missing_heading": missing_heading,
        "missing_context": missing_context,
        "quality_average": _average(
            quality_scores
        ),
        "quality_issues": dict(
            quality_issues
        ),
        "subtypes": dict(
            subtype_counts
        ),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "timings": result.timings.to_dict(),
    }

    low_density = sorted(
        chunks,
        key=lambda chunk: (
            chunk.tokens,
            len(
                chunk.text
            ),
        ),
    )[:inspect_limit]

    problematic = (
        dependent_chunks
        + suspicious_labels
        + [
            chunk
            for chunk in chunks
            if _dict(
                chunk.metadata
            ).get(
                "quality_issues"
            )
        ]
    )

    problematic = _unique_chunks(
        problematic
    )[:inspect_limit]

    return (
        report,
        low_density,
        problematic,
    )


#--------------------RUNTIME INFO--------------------

def print_runtime(
) -> None:
    config = get_chunking_config()

    budget = build_chunk_token_budget(
        config
    )

    runtime = tokenizer_runtime_metadata(
        role=config.tokenization.role
    )

    print(
        "TOKENIZER / BUDGET"
    )

    print(
        f"  role          {runtime['role']}"
    )

    print(
        f"  model         {runtime['model_id']}"
    )

    print(
        f"  revision      {runtime['revision']}"
    )

    print(
        f"  offline       {runtime['offline_runtime']}"
    )

    print(
        f"  minimum       {budget.minimum_tokens}"
    )

    print(
        f"  preferred     {budget.preferred_tokens}"
    )

    print(
        f"  strict        {budget.strict_tokens}"
    )

    print(
        "  content hard  "
        f"{budget.effective_chunk_hard_tokens}"
    )

    print(
        f"  model max     {budget.model_max_input_tokens}"
    )


#--------------------REPORT OUTPUT--------------------

def print_report(
    report: dict[str, Any],
) -> None:
    verdict = (
        "FAIL"
        if report[
            "hard_failures"
        ]
        else (
            "WARN"
            if report[
                "warnings"
            ]
            else "PASS"
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        report[
            "path"
        ]
    )
    print(
        "-" * 100
    )

    print(
        f"VERDICT: {verdict}"
    )

    print(
        "parser:",
        f"sections={report['parser_sections']}",
        f"blocks={report['parser_blocks']}",
        f"tiny<50={report['parser_tiny_under_50']}",
        f"local_scopes={report['parser_local_scopes']}",
        f"selectors={report['selector_artifacts']}",
    )

    print(
        "chunks:",
        f"count={report['final_chunks']}",
        f"tiny<50={report['final_tiny_under_50']}",
        f"subtypes={report['subtypes']}",
    )

    print(
        "tokens:",
        f"min={report['minimum_tokens']}",
        f"median={report['median_tokens']:.1f}",
        f"avg={report['average_tokens']:.1f}",
        f"p90={report['p90_tokens']:.1f}",
        f"max={report['maximum_tokens']}",
    )

    print(
        "limits:",
        f"below_min={report['below_minimum']}",
        f"above_preferred={report['above_preferred']}",
        f"above_strict={report['above_strict']}",
        f"above_hard={report['above_hard']}",
    )

    print(
        "identity:",
        f"parser_ir={report['parser_ir_ids']}",
        f"represented={report['represented_ir_ids']}",
        f"missing={report['missing_ir_ids']}",
        f"duplicate_ids={report['duplicate_chunk_ids']}",
        f"token_mismatch={report['token_count_mismatches']}",
    )

    print(
        "structure:",
        f"chunk_scopes={report['chunk_local_scopes']}",
        f"lost_scopes={report['lost_local_scopes']}",
        f"selector_text_leaks={report['selector_text_leaks']}",
        f"selector_context_leaks={report['selector_context_leaks']}",
        f"fallback={report['fallback_chunks']}",
    )

    print(
        "retrieval:",
        f"dependent={report['dependent_fragments']}",
        f"suspicious_labels={report['suspicious_compact_labels']}",
        f"duplicate_keys={report['duplicate_retrieval_keys']}",
        f"decorated={report['decorated_chunk_texts']}",
        f"legacy={report['legacy_subtypes']}",
        f"quality_avg={report['quality_average']:.4f}",
    )

    if report[
        "quality_issues"
    ]:
        print(
            "quality issues:",
            report[
                "quality_issues"
            ],
        )

    if report[
        "hard_failures"
    ]:
        print(
            "HARD FAILURES:"
        )

        for issue in report[
            "hard_failures"
        ]:
            print(
                f"  - {issue}"
            )

    if report[
        "warnings"
    ]:
        print(
            "WARNINGS:"
        )

        for issue in report[
            "warnings"
        ]:
            print(
                f"  - {issue}"
            )


#--------------------CHUNK OUTPUT--------------------

def print_chunks(
    title: str,
    chunks: list[Any],
) -> None:
    if not chunks:
        return

    print()
    print(
        title
    )

    for chunk in chunks:
        preview = " ".join(
            chunk.text.split()
        )

        if len(
            preview
        ) > 220:
            preview = (
                preview[:217]
                + "..."
            )

        issues = _dict(
            chunk.metadata
        ).get(
            "quality_issues"
        )

        print(
            f"{chunk.tokens:>5} | "
            f"{chunk.subtype:<12} | "
            f"heading={chunk.heading!r} | "
            f"id={chunk.chunk_id}"
        )

        if issues:
            print(
                f"      issues={issues}"
            )

        print(
            f"      {preview}"
        )


#--------------------AGGREGATION--------------------

def aggregate(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = (
        "parser_blocks",
        "final_chunks",
        "parser_tiny_under_50",
        "final_tiny_under_50",
        "above_hard",
        "duplicate_chunk_ids",
        "token_count_mismatches",
        "fallback_chunks",
        "selector_text_leaks",
        "selector_context_leaks",
        "lost_local_scopes",
        "dependent_fragments",
        "suspicious_compact_labels",
    )

    output = {
        "files": len(
            reports
        ),
        "hard_failed_files": sum(
            bool(
                report[
                    "hard_failures"
                ]
            )
            for report in reports
        ),
        "warned_files": sum(
            bool(
                report[
                    "warnings"
                ]
            )
            for report in reports
        ),
    }

    for key in keys:
        output[
            key
        ] = sum(
            report[
                key
            ]
            for report in reports
        )

    output[
        "average_quality_score"
    ] = _average(
        [
            report[
                "quality_average"
            ]
            for report in reports
            if report[
                "quality_average"
            ] > 0
        ]
    )

    return output


#--------------------HELPERS--------------------

def _value(
    obj: Any,
    key: str,
) -> Any:
    if isinstance(
        obj,
        dict,
    ):
        return obj.get(
            key
        )

    return getattr(
        obj,
        key,
        None,
    )


def _dict(
    value: Any,
) -> dict[str, Any]:
    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


def _text(
    value: Any,
) -> str:
    return str(
        value
        or ""
    ).strip()


def _normalize(
    value: Any,
) -> str:
    return " ".join(
        _text(
            value
        ).split()
    )


def _first_not_none(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def _duplicate_count(
    values: list[Any],
) -> int:
    counts = Counter(
        values
    )

    return sum(
        count - 1
        for count in counts.values()
        if count > 1
    )


def _average(
    values: list[int] | list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        statistics.fmean(
            values
        )
    )


def _median(
    values: list[int],
) -> float:
    if not values:
        return 0.0

    return float(
        statistics.median(
            values
        )
    )


def _percentile(
    values: list[int],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    index = round(
        (
            len(
                ordered
            )
            - 1
        )
        * percentile
    )

    return float(
        ordered[
            index
        ]
    )


def _unique_chunks(
    chunks: list[Any],
) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()

    for chunk in chunks:
        identity = _text(
            chunk.chunk_id
        )

        if identity in seen:
            continue

        seen.add(
            identity
        )

        output.append(
            chunk
        )

    return output


#--------------------CLI--------------------

def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end real Markdown/MDX "
            "parse + chunk quality gate."
        )
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Markdown files or directories. "
            "Defaults to test_data/qdrant."
        ),
    )

    parser.add_argument(
        "--inspect",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "/tmp/parse_chunk_quality.json"
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return non-zero for warnings "
            "as well as hard failures."
        ),
    )

    return parser.parse_args()


#--------------------MAIN--------------------

def main(
) -> int:
    args = parse_args()

    files = discover_files(
        args.paths
    )

    print_runtime()

    print()
    print(
        "Markdown files discovered:",
        len(
            files
        ),
    )

    reports: list[
        dict[str, Any]
    ] = []

    for index, path in enumerate(
        files,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(files)}] "
            f"Processing {path}"
        )

        try:
            report, low_density, problematic = (
                analyze_file(
                    path,
                    inspect_limit=max(
                        0,
                        args.inspect,
                    ),
                )
            )
        except Exception as exc:
            print(
                "PIPELINE CRASH:",
                repr(
                    exc
                ),
            )

            return 2

        reports.append(
            report
        )

        print_report(
            report
        )

        if args.inspect > 0:
            print_chunks(
                "LOWEST-DENSITY CHUNKS",
                low_density,
            )

            print_chunks(
                "PROBLEMATIC CHUNKS",
                problematic,
            )

    summary = aggregate(
        reports
    )

    print()
    print(
        "#" * 100
    )
    print(
        "PARSE / CHUNK QUALITY SUMMARY"
    )
    print(
        "#" * 100
    )

    for key, value in summary.items():
        print(
            f"{key:<32} {value}"
        )

    args.json_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_out.write_text(
        json.dumps(
            {
                "summary": summary,
                "files": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"JSON report: {args.json_out}"
    )

    hard_failed = any(
        report[
            "hard_failures"
        ]
        for report in reports
    )

    warned = any(
        report[
            "warnings"
        ]
        for report in reports
    )

    if hard_failed:
        return 1

    if (
        args.strict
        and warned
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
