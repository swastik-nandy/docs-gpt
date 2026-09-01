from __future__ import annotations

import hashlib
from pathlib import Path

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.fragments import (
    parse_spine_fragments_sequential,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.full_document import (
    parse_full_markdown_document,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.policy import (
    build_markdown_execution_plan,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.process_pool import (
    parse_spine_fragments_process_pool,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import (
    MarkdownExecutionPolicyContext,
    MarkdownParseExecutionResult,
    MarkdownParsedUnit,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.timing import (
    TimingCollector,
    timed,
    use_timing_collector,
)
from rag.ingestion.data_transformation.parser.inspection.markdown import inspect_markdown_file


# --------------- DOC ID --------------------------------

def _stable_doc_id_from_path(path: str) -> str:
    """
    Deterministic fallback doc id.

    Used only when inspection/source hash is unavailable.
    """

    resolved = str(Path(path).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()

    return f"doc_{digest[:16]}"


def _doc_id_from_inspection(file_path: str, inspection: object) -> str:
    """
    Build stable source document id from inspection result.

    Preferred:
        source.sha256

    Fallback:
        absolute path hash
    """

    source = getattr(inspection, "source", None)
    sha256 = str(getattr(source, "sha256", "") or "").strip()

    if sha256:
        return f"doc_{sha256[:16]}"

    return _stable_doc_id_from_path(file_path)


# --------------- STRATEGY EXECUTION --------------------------------

def _parse_full_document_unit(
    *,
    file_path: str,
    doc_id: str,
    variant: str,
    parse_strategy: str,
) -> list[MarkdownParsedUnit]:
    """
    Execute full-document parse strategy.
    """

    with timed(
        "strategy.full_document",
        doc_id=doc_id,
        parse_strategy=parse_strategy,
    ):
        return [
            parse_full_markdown_document(
                file_path=file_path,
                doc_id=doc_id,
                variant=variant,
                parse_strategy=parse_strategy,
            )
        ]


def _parse_spine_fragmented_units(
    *,
    file_path: str,
    doc_id: str,
    variant: str,
    spine_nodes: list[object],
    execution_mode: str,
    max_workers: int,
) -> list[MarkdownParsedUnit]:
    """
    Execute spine-fragmented parse strategy.

    The same logical strategy can run sequentially or through a process pool
    depending on the execution plan.
    """

    with timed(
        "strategy.spine_fragmented",
        doc_id=doc_id,
        execution_mode=execution_mode,
        max_workers=max_workers,
        spine_node_count=len(spine_nodes or []),
    ):
        if execution_mode == "process_pool":
            return parse_spine_fragments_process_pool(
                file_path=file_path,
                source_doc_id=doc_id,
                spine_nodes=spine_nodes,
                variant=variant,
                max_workers=max_workers,
            )

        return parse_spine_fragments_sequential(
            file_path=file_path,
            source_doc_id=doc_id,
            spine_nodes=spine_nodes,
            variant=variant,
        )


# --------------- UNIT EXECUTION METRICS --------------------------------

def _collect_unit_execution_metrics(
    units: list[MarkdownParsedUnit],
) -> dict[str, object]:
    metrics: dict[str, object] = {}

    for unit in units or []:
        meta = unit.meta or {}

        packed_diagnostics = meta.get("packed_spine_diagnostics")

        if isinstance(packed_diagnostics, dict):
            metrics["packed_spine_diagnostics"] = packed_diagnostics
            metrics["packed_fragment_count"] = packed_diagnostics.get("packed_fragment_count", 0)
            metrics["packed_close_reason_counts"] = packed_diagnostics.get(
                "close_reason_counts",
                {},
            )
            metrics["avg_spine_nodes_per_packed_fragment"] = packed_diagnostics.get(
                "avg_spine_nodes_per_fragment",
                0,
            )
            metrics["max_spine_nodes_per_packed_fragment"] = packed_diagnostics.get(
                "max_spine_nodes_per_fragment",
                0,
            )
            metrics["max_packed_fragment_bytes"] = packed_diagnostics.get(
                "max_fragment_bytes",
                0,
            )

        process_pool = meta.get("process_pool")

        if isinstance(process_pool, dict):
            metrics["process_pool"] = process_pool

        worker_timing = meta.get("process_pool_worker_timing")

        if isinstance(worker_timing, dict):
            totals = worker_timing.get("totals_ms", {}) or {}
            max_ms = worker_timing.get("max_ms", {}) or {}

            metrics["process_pool_worker_timing"] = worker_timing
            metrics["worker_cpu_sum_ms"] = float(totals.get("text_pipeline.total", 0.0) or 0.0)
            metrics["worker_text_pipeline_cpu_sum_ms"] = float(
                totals.get("text_pipeline.total", 0.0) or 0.0
            )
            metrics["worker_ast_cpu_sum_ms"] = float(
                totals.get("text_pipeline.ast", 0.0) or 0.0
            )
            metrics["worker_enrich_cpu_sum_ms"] = float(
                totals.get("text_pipeline.enrich", 0.0) or 0.0
            )
            metrics["worker_clean_cpu_sum_ms"] = float(
                totals.get("text_pipeline.clean", 0.0) or 0.0
            )
            metrics["worker_fragment_wall_sum_ms"] = float(
                totals.get("worker.fragment", 0.0) or 0.0
            )
            metrics["worker_fragment_wall_max_ms"] = float(
                max_ms.get("worker.fragment", 0.0) or 0.0
            )

    return metrics


# --------------- PUBLIC API --------------------------------

def execute_markdown_parse(
    file_path: str,
    *,
    policy_context: MarkdownExecutionPolicyContext | None = None,
) -> MarkdownParseExecutionResult:
    """
    Strategy-aware Markdown/MDX parser execution entrypoint.

    This function writes nothing.

    It does:
      1. Inspect Markdown/MDX file
      2. Build resource-aware execution plan
      3. Parse using the selected execution path
      4. Return IR-ready parsed units

    Runners decide what happens next:
      - quality runner prints diagnostics
      - artifact runner writes .ir files
      - future orchestrator dispatches jobs
    """

    timing = TimingCollector()

    with use_timing_collector(timing):
        with timed("execution.total", file_path=str(Path(file_path).resolve())):
            resolved_path = str(Path(file_path).resolve())

            with timed("inspection.total", file_path=resolved_path):
                inspection = inspect_markdown_file(resolved_path)

            with timed("execution.plan", file_path=resolved_path):
                execution_plan = build_markdown_execution_plan(
                    inspection=inspection,
                    policy_context=policy_context,
                )

            doc_id = _doc_id_from_inspection(resolved_path, inspection)

            variant = str(getattr(inspection, "variant", "") or "markdown")
            parse_strategy = execution_plan.parse_strategy

            warnings = list(getattr(inspection, "warnings", []) or [])
            units: list[MarkdownParsedUnit] = []

            if not execution_plan.should_parse:
                warnings.append("execution_plan_skipped_parse")

            elif parse_strategy in {"single_document", "full_parse_then_chunk_fanout"}:
                units = _parse_full_document_unit(
                    file_path=resolved_path,
                    doc_id=doc_id,
                    variant=variant,
                    parse_strategy=parse_strategy,
                )

            elif parse_strategy == "spine_fragmented":
                spine_nodes = list(getattr(inspection, "spine_nodes", []) or [])

                units = _parse_spine_fragmented_units(
                    file_path=resolved_path,
                    doc_id=doc_id,
                    variant=variant,
                    spine_nodes=spine_nodes,
                    execution_mode=execution_plan.execution_mode,
                    max_workers=execution_plan.max_workers,
                )

                if not units:
                    warnings.append("spine_fragmented_strategy_produced_no_units")

            elif parse_strategy == "quarantine":
                warnings.append("document_not_parsed_due_to_quarantine_strategy")

            else:
                raise ValueError(f"Unsupported markdown parse strategy: {parse_strategy!r}")

            unit_execution_metrics = _collect_unit_execution_metrics(units)

            result_payload = {
                "doc_id": doc_id,
                "source_path": resolved_path,
                "parser_family": "markdown",
                "parser_variant": variant,
                "parse_strategy": parse_strategy,
                "inspection": inspection,
                "execution_plan": execution_plan,
                "units": units,
                "warnings": warnings,
                "base_metrics": {
                    "size_class": getattr(inspection, "size_class", ""),
                    "risk_level": getattr(inspection, "risk_level", ""),
                    "estimated_tokens": getattr(inspection, "estimated_tokens", 0),
                    "line_count": getattr(inspection, "line_count", 0),
                    "heading_count": getattr(inspection, "heading_count", 0),
                    "spine_node_count": len(getattr(inspection, "spine_nodes", []) or []),
                    "unit_count": len(units),
                    "section_count": sum(unit.section_count for unit in units),
                    "block_count": sum(unit.block_count for unit in units),
                    "execution_mode": execution_plan.execution_mode,
                    "resource_class": execution_plan.resource_class,
                    "max_workers": execution_plan.max_workers,
                    "execution_reasons": execution_plan.reasons,
                    **(getattr(inspection, "metrics", {}) or {}),
                    **unit_execution_metrics,
                },
            }

    timing_data = timing.to_dict()
    metrics = dict(result_payload["base_metrics"])
    metrics["timing"] = timing_data
    metrics["timing_totals_ms"] = timing_data["totals_ms"]

    return MarkdownParseExecutionResult(
        doc_id=result_payload["doc_id"],
        source_path=result_payload["source_path"],
        parser_family=result_payload["parser_family"],
        parser_variant=result_payload["parser_variant"],
        parse_strategy=result_payload["parse_strategy"],
        inspection=result_payload["inspection"],
        execution_plan=result_payload["execution_plan"],
        units=result_payload["units"],
        warnings=result_payload["warnings"],
        metrics=metrics,
    )
