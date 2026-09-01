from __future__ import annotations

from pathlib import Path

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config
from rag.ingestion.data_transformation.parser.inspection.detector import read_extension
from rag.ingestion.data_transformation.parser.inspection.schema import (
    MarkdownInspection,
    MarkdownVariant,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.spine.config_loader import (
    MarkdownSpineThresholds,
    load_markdown_spine_thresholds,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.spine.scanner import (
    scan_markdown_spine,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.spine.schema import (
    MarkdownSpineScan,
)


# --------------- SIZE ORDER --------------------------------

SIZE_CLASS_ORDER = {
    "small": 0,
    "normal": 1,
    "large": 2,
    "huge": 3,
    "pathological": 4,
}


# --------------- VARIANT DETECTION --------------------------------

def detect_markdown_variant(file_path: str) -> MarkdownVariant:
    """
    Detect markdown-family variant from TOML config.

    This is not a trust decision.
    Deep structure validation happens in the inspector/spine scanner.
    """

    extension = read_extension(file_path)
    config = load_parser_heuristics_config()

    return config.file_types.markdown_variants.get(
        extension,
        "unknown",
    )  # type: ignore[return-value]


# --------------- SIZE CLASSIFICATION --------------------------------

def _classify_value_by_thresholds(
    *,
    value: int,
    small: int,
    normal: int,
    large: int,
    huge: int,
) -> str:
    if value <= small:
        return "small"

    if value <= normal:
        return "normal"

    if value <= large:
        return "large"

    if value <= huge:
        return "huge"

    return "pathological"


def _worse_size_class(left: str, right: str) -> str:
    left_rank = SIZE_CLASS_ORDER.get(left, SIZE_CLASS_ORDER["pathological"])
    right_rank = SIZE_CLASS_ORDER.get(right, SIZE_CLASS_ORDER["pathological"])

    if left_rank >= right_rank:
        return left

    return right


def _classify_size(
    scan: MarkdownSpineScan,
    thresholds: MarkdownSpineThresholds,
) -> str:
    """
    Classify size by computing byte class and token class separately.

    Do not require bytes and tokens to fit the same bucket via AND logic.
    That incorrectly marks normal text-heavy Markdown as pathological.

    Final size class is the worse of:
      - byte_size class
      - estimated_token class
    """

    byte_size = scan.source.byte_size
    tokens = scan.estimated_tokens

    byte_class = _classify_value_by_thresholds(
        value=byte_size,
        small=thresholds.small_bytes,
        normal=thresholds.normal_bytes,
        large=thresholds.large_bytes,
        huge=thresholds.huge_bytes,
    )

    token_class = _classify_value_by_thresholds(
        value=tokens,
        small=thresholds.small_tokens,
        normal=thresholds.normal_tokens,
        large=thresholds.large_tokens,
        huge=thresholds.huge_tokens,
    )

    return _worse_size_class(byte_class, token_class)


def _size_class_diagnostics(
    scan: MarkdownSpineScan,
    thresholds: MarkdownSpineThresholds,
) -> dict[str, str | int]:
    byte_class = _classify_value_by_thresholds(
        value=scan.source.byte_size,
        small=thresholds.small_bytes,
        normal=thresholds.normal_bytes,
        large=thresholds.large_bytes,
        huge=thresholds.huge_bytes,
    )

    token_class = _classify_value_by_thresholds(
        value=scan.estimated_tokens,
        small=thresholds.small_tokens,
        normal=thresholds.normal_tokens,
        large=thresholds.large_tokens,
        huge=thresholds.huge_tokens,
    )

    return {
        "byte_size": scan.source.byte_size,
        "estimated_tokens": scan.estimated_tokens,
        "byte_size_class": byte_class,
        "token_size_class": token_class,
        "final_size_class": _worse_size_class(byte_class, token_class),
    }


# --------------- RISK CLASSIFICATION --------------------------------

def _largest_section_tokens(scan: MarkdownSpineScan) -> int:
    return int(scan.metrics.get("largest_section_estimated_tokens", 0))


def _has_useful_spine(scan: MarkdownSpineScan) -> bool:
    """
    A useful spine means we have real semantic split points.

    One root node means the file is basically one huge blob from the scanner's
    perspective, so safe section fragmentation is not available yet.
    """

    if len(scan.spine_nodes) <= 1:
        return False

    if "no_heading_spine_detected" in set(scan.warnings or []):
        return False

    return True


def _risk_level(
    scan: MarkdownSpineScan,
    size_class: str,
    thresholds: MarkdownSpineThresholds,
) -> str:
    """
    Estimate operational risk separately from size class.

    Size class means how big the document is.
    Risk means how dangerous normal parsing may be.
    """

    if scan.unclosed_fence:
        if size_class in {"small", "normal"}:
            return "medium"

        return "high"

    if _largest_section_tokens(scan) > thresholds.large_section_tokens:
        return "high"

    if size_class == "pathological":
        return "high"

    if size_class == "huge":
        return "high"

    if size_class == "large":
        return "medium"

    return "low"


# --------------- STRATEGY SELECTION --------------------------------

def _choose_strategy_and_queue(
    scan: MarkdownSpineScan,
    size_class: str,
    risk_level: str,
    thresholds: MarkdownSpineThresholds,
) -> tuple[str, str]:
    """
    Decide final parse strategy and target queue using priority order.

    Priority:
      1. Reject stays quarantine.
      2. Huge/pathological without useful spine is quarantine.
      3. Small docs use single document.
      4. Normal docs use single document unless high-risk and splittable.
      5. Large docs full-parse when manageable, spine-fragment when risky.
      6. Huge/pathological docs use spine-fragmented when splittable.
    """

    useful_spine = _has_useful_spine(scan)
    largest_section_tokens = _largest_section_tokens(scan)

    if risk_level == "reject":
        return "quarantine", thresholds.routing.quarantine_queue

    if size_class == "small":
        return "single_document", thresholds.routing.normal_queue

    if size_class == "normal":
        if risk_level == "high" and useful_spine:
            return "spine_fragmented", thresholds.routing.spine_queue

        return "single_document", thresholds.routing.normal_queue

    if size_class == "large":
        if risk_level == "high" and useful_spine:
            return "spine_fragmented", thresholds.routing.spine_queue

        if largest_section_tokens <= thresholds.large_section_tokens:
            return "full_parse_then_chunk_fanout", thresholds.routing.normal_queue

        if useful_spine:
            return "spine_fragmented", thresholds.routing.spine_queue

        return "quarantine", thresholds.routing.quarantine_queue

    if size_class in {"huge", "pathological"}:
        if useful_spine:
            return "spine_fragmented", thresholds.routing.spine_queue

        return "quarantine", thresholds.routing.quarantine_queue

    return "quarantine", thresholds.routing.quarantine_queue


# --------------- INSPECTION ENTRYPOINT --------------------------------

def inspect_markdown_file(file_path: str) -> MarkdownInspection:
    """
    Inspect a Markdown/MDX file and produce a routing decision.

    This is the pre-parse decision layer.
    It classifies size/risk, selects parse strategy, and chooses target queue.
    It does not allocate workers or execute parsing.
    """

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"Markdown inspection failed. File not found: {file_path}")

    variant = detect_markdown_variant(file_path)

    if variant == "unknown":
        raise ValueError(f"Unsupported markdown-family extension: {file_path}")

    thresholds = load_markdown_spine_thresholds()

    scan = scan_markdown_spine(
        file_path=file_path,
        variant=variant,  # type: ignore[arg-type]
    )

    size_class = _classify_size(
        scan=scan,
        thresholds=thresholds,
    )

    risk_level = _risk_level(
        scan=scan,
        size_class=size_class,
        thresholds=thresholds,
    )

    parse_strategy, target_queue = _choose_strategy_and_queue(
        scan=scan,
        size_class=size_class,
        risk_level=risk_level,
        thresholds=thresholds,
    )

    warnings = list(scan.warnings or [])

    if parse_strategy == "quarantine":
        warnings.append("document_requires_manual_or_hardened_handling")

    if parse_strategy == "spine_fragmented":
        warnings.append("document_will_be_parsed_as_spine_fragments")

    if parse_strategy == "full_parse_then_chunk_fanout":
        warnings.append("document_should_fanout_after_ir_generation")

    metrics = dict(scan.metrics or {})
    metrics.update(_size_class_diagnostics(scan, thresholds))
    metrics["has_useful_spine"] = _has_useful_spine(scan)
    metrics["largest_section_estimated_tokens"] = _largest_section_tokens(scan)

    return MarkdownInspection(
        source=scan.source,
        family="markdown",
        variant=variant,

        size_class=size_class,          # type: ignore[arg-type]
        risk_level=risk_level,          # type: ignore[arg-type]
        parse_strategy=parse_strategy,  # type: ignore[arg-type]
        target_queue=target_queue,

        estimated_tokens=scan.estimated_tokens,
        line_count=scan.line_count,
        heading_count=scan.heading_count,
        max_heading_level=scan.max_heading_level,
        table_like_line_count=scan.table_like_line_count,
        fenced_code_block_count=scan.fenced_code_block_count,
        unclosed_fence=scan.unclosed_fence,
        mdx_component_hint_count=scan.mdx_component_hint_count,

        spine_nodes=scan.spine_nodes,
        warnings=warnings,
        metrics=metrics,
    )