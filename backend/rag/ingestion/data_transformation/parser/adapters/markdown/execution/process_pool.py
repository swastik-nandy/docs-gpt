from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.fragments import (
    parse_packed_spine_fragment,
    parse_spine_fragments_sequential,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.spine_packer import (
    PackedSpineDiagnostics,
    PackedSpineFragment,
    load_packed_spine_config,
    pack_spine_nodes_with_diagnostics,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import MarkdownParsedUnit
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.timing import (
    TimingCollector,
    timed,
    use_timing_collector,
)


# --------------- WORKER --------------------------------

def _parse_packed_fragment_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Process-pool worker wrapper.

    Keep this top-level so multiprocessing can pickle it.

    Returns:
      - parsed unit
      - child-process timing data
    """

    timing = TimingCollector()

    with use_timing_collector(timing):
        with timed(
            "worker.fragment",
            fragment_index=payload["fragment_index"],
            packed_fragment_id=payload["packed_fragment"].packed_fragment_id,
        ):
            unit = parse_packed_spine_fragment(
                file_path=payload["file_path"],
                source_doc_id=payload["source_doc_id"],
                packed_fragment=payload["packed_fragment"],
                variant=payload["variant"],
            )

    return {
        "fragment_index": payload["fragment_index"],
        "unit": unit,
        "timing": timing.to_dict(),
    }


# --------------- TIMING MERGE --------------------------------

def _merge_worker_timings(
    timing_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    totals_ms: dict[str, float] = {}
    counts: dict[str, int] = {}
    max_ms: dict[str, float] = {}

    for timing in timing_payloads:
        if not isinstance(timing, dict):
            continue

        for name, duration in (timing.get("totals_ms") or {}).items():
            if not isinstance(duration, (int, float)):
                continue

            totals_ms[name] = round(totals_ms.get(name, 0.0) + float(duration), 3)

        for name, count in (timing.get("counts") or {}).items():
            if not isinstance(count, int):
                continue

            counts[name] = counts.get(name, 0) + count

        for event in timing.get("events", []) or []:
            if not isinstance(event, dict):
                continue

            name = str(event.get("name") or "")
            duration = event.get("duration_ms")

            if not name or not isinstance(duration, (int, float)):
                continue

            max_ms[name] = round(max(max_ms.get(name, 0.0), float(duration)), 3)

    return {
        "totals_ms": totals_ms,
        "counts": counts,
        "max_ms": max_ms,
        "worker_result_count": len(timing_payloads),
    }


# --------------- META ATTACHMENT --------------------------------

def _attach_process_pool_meta(
    units: list[MarkdownParsedUnit],
    *,
    diagnostics: PackedSpineDiagnostics,
    worker_timing: dict[str, Any],
    requested_workers: int,
    active_workers: int,
    submitted_task_count: int,
) -> list[MarkdownParsedUnit]:
    """
    Attach process-pool execution metadata to the first unit.

    MarkdownParsedUnit is frozen, but its meta dict is intentionally mutable.
    This keeps the public return type unchanged while exposing execution metrics.
    """

    if not units:
        return units

    carrier = units[0]

    carrier.meta["packed_spine_diagnostics"] = diagnostics.to_meta()
    carrier.meta["process_pool_worker_timing"] = worker_timing
    carrier.meta["process_pool"] = {
        "requested_workers": requested_workers,
        "active_workers": active_workers,
        "submitted_task_count": submitted_task_count,
        "execution_unit": "packed_spine_fragment",
    }

    return units


# --------------- BOUNDED SUBMISSION --------------------------------

def _submit_next(
    *,
    executor: ProcessPoolExecutor,
    queue: deque[dict[str, Any]],
    in_flight: dict[Future[dict[str, Any]], int],
) -> None:
    if not queue:
        return

    payload = queue.popleft()
    future = executor.submit(_parse_packed_fragment_worker, payload)
    in_flight[future] = payload["fragment_index"]


def _parse_packed_fragments_process_pool(
    *,
    file_path: str,
    source_doc_id: str,
    packed_fragments: list[PackedSpineFragment],
    diagnostics: PackedSpineDiagnostics,
    variant: str,
    max_workers: int,
) -> list[MarkdownParsedUnit]:
    if not packed_fragments:
        return []

    active_workers = max(1, min(max_workers, len(packed_fragments)))

    if active_workers <= 1:
        return []

    payload_queue: deque[dict[str, Any]] = deque(
        {
            "file_path": file_path,
            "source_doc_id": source_doc_id,
            "fragment_index": fragment.fragment_index,
            "packed_fragment": fragment,
            "variant": variant,
        }
        for fragment in packed_fragments
    )

    max_in_flight = max(active_workers, active_workers * 4)
    units_by_index: dict[int, MarkdownParsedUnit] = {}
    timing_payloads: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=active_workers) as executor:
        in_flight: dict[Future[dict[str, Any]], int] = {}

        while payload_queue and len(in_flight) < max_in_flight:
            _submit_next(
                executor=executor,
                queue=payload_queue,
                in_flight=in_flight,
            )

        while in_flight:
            done, _pending = wait(
                in_flight,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                index = in_flight.pop(future)
                result = future.result()

                timing = result.get("timing")

                if isinstance(timing, dict):
                    timing_payloads.append(timing)

                unit = result.get("unit")

                if unit is not None:
                    units_by_index[index] = unit

                while payload_queue and len(in_flight) < max_in_flight:
                    _submit_next(
                        executor=executor,
                        queue=payload_queue,
                        in_flight=in_flight,
                    )

    units = [
        units_by_index[index]
        for index in sorted(units_by_index)
    ]

    worker_timing = _merge_worker_timings(timing_payloads)

    return _attach_process_pool_meta(
        units,
        diagnostics=diagnostics,
        worker_timing=worker_timing,
        requested_workers=max_workers,
        active_workers=active_workers,
        submitted_task_count=len(packed_fragments),
    )


# --------------- PUBLIC API --------------------------------

def parse_spine_fragments_process_pool(
    *,
    file_path: str,
    source_doc_id: str,
    spine_nodes: list[Any],
    variant: str,
    max_workers: int,
) -> list[MarkdownParsedUnit]:
    """
    Parse spine-fragmented documents using packed fragments in a process pool.

    Old behavior:
      one spine node -> one process-pool task -> one AST/enrichment cycle

    New behavior:
      many spine nodes -> one packed process-pool task -> one AST/enrichment cycle

    This reduces process scheduling overhead, AST invocations, enrichment cycles,
    and parent memory pressure from thousands of submitted futures.
    """

    if not spine_nodes:
        return []

    config = load_packed_spine_config()

    packed_fragments, diagnostics = pack_spine_nodes_with_diagnostics(
        spine_nodes,
        source_doc_id=source_doc_id,
        config=config,
        source_path=file_path,
    )

    if max_workers <= 1 or len(packed_fragments) <= 1:
        units = parse_spine_fragments_sequential(
            file_path=file_path,
            source_doc_id=source_doc_id,
            spine_nodes=spine_nodes,
            variant=variant,
        )

        if units:
            units[0].meta["packed_spine_diagnostics"] = diagnostics.to_meta()

        return units

    return _parse_packed_fragments_process_pool(
        file_path=file_path,
        source_doc_id=source_doc_id,
        packed_fragments=packed_fragments,
        diagnostics=diagnostics,
        variant=variant,
        max_workers=max_workers,
    )