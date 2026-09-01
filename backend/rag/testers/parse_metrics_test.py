from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


# --------------- CONFIG --------------------------------

DEFAULT_TEST_DIR = "test_data/large"

DEFAULT_TEST_FILES = [
    "test_data/large/aws-ecs-realistic-5mb.md",
    # "test_data/aws/ecs/ecs-anywhere.md",
]

RECURSIVE_DIRECTORY_SCAN = False
ALLOWED_EXTENSIONS = {".md", ".mdx", ".markdown", ".mdown", ".mkd", ".mdwn"}

TOP_N_DISTRIBUTION_ITEMS = 12


# --------------- PATH BOOTSTRAP --------------------------------

THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = THIS_FILE.parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from rag.ingestion.data_transformation.parser.adapters.markdown.execution import (  # noqa: E402
    execute_markdown_parse,
)


# --------------- FILE DISCOVERY --------------------------------

def resolve_backend_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return (BACKEND_ROOT / path).resolve()


def discover_files() -> list[Path]:
    explicit_files = [
        resolve_backend_path(path)
        for path in DEFAULT_TEST_FILES
        if str(path).strip()
    ]

    if explicit_files:
        return explicit_files

    test_dir = resolve_backend_path(DEFAULT_TEST_DIR)

    if not test_dir.exists():
        raise FileNotFoundError(f"DEFAULT_TEST_DIR does not exist: {test_dir}")

    iterator = test_dir.rglob("*") if RECURSIVE_DIRECTORY_SCAN else test_dir.glob("*")

    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


# --------------- PRINT HELPERS --------------------------------

def line(char: str = "-", width: int = 132) -> None:
    print(char * width)


def kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<34} {value}")


def fmt_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "0.000"

    return f"{float(value):,.3f}"


def fmt_num(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "0"

    if isinstance(value, float):
        return f"{value:,.2f}"

    return f"{value:,}"


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def scalar_attrs(obj: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in keys:
        value = getattr(obj, key, None)

        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple, set)):
            result[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            result[f"{key}_count"] = len(value)

    return result


def print_scalar_group(title: str, values: dict[str, Any]) -> None:
    print(f"\n{title}")

    if not values:
        kv("available", "no")
        return

    for key, value in values.items():
        kv(key, value)


def print_counter(
    title: str,
    counts: Counter,
    limit: int = TOP_N_DISTRIBUTION_ITEMS,
) -> None:
    print(f"\n{title}")

    if not counts:
        kv("available", "no")
        return

    for key, value in counts.most_common(limit):
        kv(key, value)

    remaining = len(counts) - limit

    if remaining > 0:
        kv("more_keys", remaining)


# --------------- RESULT ACCESS --------------------------------

def get_metrics(result: Any) -> dict[str, Any]:
    return safe_dict(getattr(result, "metrics", {}) or {})


def get_timing_totals(result: Any) -> dict[str, float]:
    metrics = get_metrics(result)

    totals = metrics.get("timing_totals_ms")

    if not isinstance(totals, dict):
        timing = safe_dict(metrics.get("timing", {}) or {})
        totals = timing.get("totals_ms", {})

    if not isinstance(totals, dict):
        return {}

    return {
        str(key): float(value)
        for key, value in totals.items()
        if isinstance(value, (int, float))
    }


def timing_ms(result: Any, key: str) -> float:
    return get_timing_totals(result).get(key, 0.0)


def iter_sections(result: Any) -> list[dict[str, Any]]:
    try:
        return [
            section
            for section in result.iter_sections()
            if isinstance(section, dict)
        ]
    except Exception:
        return []


def iter_blocks(result: Any) -> list[dict[str, Any]]:
    try:
        return [
            block
            for block in result.iter_blocks()
            if isinstance(block, dict)
        ]
    except Exception:
        return []


def collect_content_blocks(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []

    for section in sections:
        for item in section.get("content", []) or []:
            if isinstance(item, dict):
                content_blocks.append(item)

    return content_blocks


# --------------- NUMERIC COUNTS --------------------------------

def count_blocks_with_meta(blocks: list[dict[str, Any]], key: str) -> int:
    count = 0

    for block in blocks:
        meta = safe_dict(block.get("meta") or {})

        if meta.get(key):
            count += 1

    return count


def count_content_with_key(content_blocks: list[dict[str, Any]], key: str) -> int:
    return sum(1 for item in content_blocks if item.get(key))


def duplicate_block_ids(blocks: list[dict[str, Any]]) -> int:
    ids = [
        str(block.get("id", ""))
        for block in blocks
        if block.get("id")
    ]

    counts = Counter(ids)

    return sum(1 for _, value in counts.items() if value > 1)


def empty_block_text_count(blocks: list[dict[str, Any]]) -> int:
    return sum(
        1
        for block in blocks
        if not str(block.get("text", "") or "").strip()
    )


def empty_retrieval_text_count(blocks: list[dict[str, Any]]) -> int:
    return sum(
        1
        for block in blocks
        if not str(block.get("retrieval_text", "") or "").strip()
    )


def sections_without_blocks_count(sections: list[dict[str, Any]]) -> int:
    return sum(1 for section in sections if not section.get("blocks"))


def relationship_count(blocks: list[dict[str, Any]]) -> int:
    return sum(
        len(block.get("relationships", []) or [])
        for block in blocks
    )


def counter(values: list[Any]) -> Counter:
    return Counter(str(value or "unknown") for value in values)


# --------------- SUMMARIES --------------------------------

def source_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    return {
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "byte_size": path.stat().st_size,
        "char_count": len(text),
        "line_count": len(lines),
    }


def result_summary(
    result: Any,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = get_metrics(result)

    return {
        "doc_id": getattr(result, "doc_id", ""),
        "source_path": getattr(result, "source_path", ""),
        "parser_family": getattr(result, "parser_family", ""),
        "parser_variant": getattr(result, "parser_variant", ""),
        "parse_strategy": getattr(result, "parse_strategy", ""),
        "unit_count": getattr(result, "unit_count", 0),
        "section_count": len(sections),
        "block_count": len(blocks),
        "relationship_count": relationship_count(blocks),
        "warnings_count": len(getattr(result, "warnings", []) or []),
        "size_class": metrics.get("size_class", ""),
        "risk_level": metrics.get("risk_level", ""),
        "estimated_tokens": metrics.get("estimated_tokens", 0),
        "inspection_line_count": metrics.get("line_count", 0),
        "inspection_heading_count": metrics.get("heading_count", 0),
        "spine_node_count": metrics.get("spine_node_count", 0),
        "execution_mode": metrics.get("execution_mode", ""),
        "resource_class": metrics.get("resource_class", ""),
        "max_workers": metrics.get("max_workers", ""),
    }


def timing_summary(result: Any) -> dict[str, str]:
    totals = get_timing_totals(result)
    metrics = get_metrics(result)

    summary: dict[str, str] = {}

    wall_keys = (
        "execution.total",
        "inspection.total",
        "execution.plan",
        "strategy.full_document",
        "strategy.spine_fragmented",
        "text_pipeline.total",
        "text_pipeline.mdx_normalize",
        "text_pipeline.clean",
        "text_pipeline.ast",
        "text_pipeline.enrich",
    )

    for key in wall_keys:
        value = totals.get(key, 0.0)

        if value > 0:
            summary[f"{key}_ms"] = fmt_ms(value)

    worker_keys = (
        "worker_cpu_sum_ms",
        "worker_fragment_wall_sum_ms",
        "worker_fragment_wall_max_ms",
        "worker_text_pipeline_cpu_sum_ms",
        "worker_ast_cpu_sum_ms",
        "worker_enrich_cpu_sum_ms",
        "worker_clean_cpu_sum_ms",
    )

    for key in worker_keys:
        value = metrics.get(key, 0.0)

        if isinstance(value, (int, float)) and value > 0:
            summary[key] = fmt_ms(value)

    return summary


def throughput_summary(
    path: Path,
    result: Any,
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    total_ms = timing_ms(result, "execution.total")
    file_mb = path.stat().st_size / (1024 * 1024)
    block_count = len(blocks)

    mb_per_sec = 0.0
    blocks_per_sec = 0.0

    if total_ms > 0:
        seconds = total_ms / 1000.0
        mb_per_sec = file_mb / seconds
        blocks_per_sec = block_count / seconds

    return {
        "mb_per_sec": round(mb_per_sec, 3),
        "blocks_per_sec": round(blocks_per_sec, 1),
        "ms_per_mb": round(total_ms / file_mb, 3) if file_mb > 0 else 0.0,
        "ms_per_block": round(total_ms / block_count, 4) if block_count > 0 else 0.0,
    }


def quality_numeric_summary(
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "duplicate_block_id_count": duplicate_block_ids(blocks),
        "empty_block_text_count": empty_block_text_count(blocks),
        "empty_retrieval_text_count": empty_retrieval_text_count(blocks),
        "sections_without_blocks_count": sections_without_blocks_count(sections),
    }


def block_metadata_counts(blocks: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "links",
        "images",
        "inline_code",
        "html_inline",
        "html_block",
        "strong_text",
        "language",
        "table_profile",
        "raw_rows",
        "kind",
        "label",
        "component_context",
        "parent_instruction",
        "parent_context",
        "heading_anchor",
        "anchors",
        "children",
        "code_blocks",
        "packed_fragment_id",
        "owning_spine_node_id",
    )

    return {
        f"blocks_with_{key}": count_blocks_with_meta(blocks, key)
        for key in keys
    }


def content_metadata_counts(content_blocks: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "links",
        "images",
        "inline_code",
        "html_inline",
        "html_block",
        "strong_text",
        "cell_meta",
        "table_profile",
        "raw_items",
        "code_blocks",
        "children",
    )

    return {
        f"content_with_{key}": count_content_with_key(content_blocks, key)
        for key in keys
    }


def execution_plan_summary(result: Any) -> dict[str, Any]:
    execution_plan = getattr(result, "execution_plan", None)

    values = scalar_attrs(
        execution_plan,
        (
            "parse_strategy",
            "execution_mode",
            "resource_class",
            "max_workers",
            "should_parse",
            "reasons",
        ),
    )

    plan_meta = getattr(execution_plan, "meta", {}) or {}

    if isinstance(plan_meta, dict):
        for key, value in plan_meta.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[f"meta.{key}"] = value
            elif isinstance(value, (list, tuple, set)):
                values[f"meta.{key}"] = ", ".join(str(item) for item in value)

    reasons = list(getattr(execution_plan, "reasons", []) or [])

    for index, reason in enumerate(reasons):
        values[f"reason_{index + 1}"] = reason

    return values


def inspection_summary(result: Any) -> dict[str, Any]:
    inspection = getattr(result, "inspection", None)

    return scalar_attrs(
        inspection,
        (
            "family",
            "variant",
            "size_class",
            "risk_level",
            "parse_strategy",
            "target_queue",
            "estimated_tokens",
            "line_count",
            "heading_count",
            "spine_nodes",
            "warnings",
        ),
    )


# --------------- ROW BUILDING --------------------------------

def build_summary_row(
    path: Path,
    result: Any,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = get_metrics(result)

    worker_ast_ms = float(metrics.get("worker_ast_cpu_sum_ms", 0.0) or 0.0)
    worker_enrich_ms = float(metrics.get("worker_enrich_cpu_sum_ms", 0.0) or 0.0)
    worker_clean_ms = float(metrics.get("worker_clean_cpu_sum_ms", 0.0) or 0.0)
    worker_cpu_sum_ms = float(metrics.get("worker_cpu_sum_ms", 0.0) or 0.0)
    worker_max_ms = float(metrics.get("worker_fragment_wall_max_ms", 0.0) or 0.0)

    parent_ast_ms = timing_ms(result, "text_pipeline.ast")
    parent_enrich_ms = timing_ms(result, "text_pipeline.enrich")
    parent_clean_ms = timing_ms(result, "text_pipeline.clean")

    return {
        "file": path.name,
        "mb": path.stat().st_size / (1024 * 1024),
        "wall_ms": timing_ms(result, "execution.total"),
        "inspect_ms": timing_ms(result, "inspection.total"),
        "strategy_ms": (
            timing_ms(result, "strategy.spine_fragmented")
            or timing_ms(result, "strategy.full_document")
        ),
        "ast_cpu_ms": worker_ast_ms or parent_ast_ms,
        "enrich_cpu_ms": worker_enrich_ms or parent_enrich_ms,
        "clean_cpu_ms": worker_clean_ms or parent_clean_ms,
        "worker_sum_ms": worker_cpu_sum_ms,
        "worker_max_ms": worker_max_ms,
        "sections": len(sections),
        "blocks": len(blocks),
        "warnings": len(getattr(result, "warnings", []) or []),
    }


# --------------- PRINT ONE FILE --------------------------------

def print_file_report(path: Path, result: Any) -> dict[str, Any]:
    sections = iter_sections(result)
    blocks = iter_blocks(result)
    content_blocks = collect_content_blocks(sections)

    line("=")
    print(f"FILE: {path.name}")
    line("=")

    print_scalar_group("SOURCE", source_summary(path))
    print_scalar_group("RESULT", result_summary(result, sections, blocks))
    print_scalar_group("TIMING", timing_summary(result))
    print_scalar_group("THROUGHPUT", throughput_summary(path, result, blocks))
    print_scalar_group("QUALITY_NUMERIC", quality_numeric_summary(sections, blocks))
    print_scalar_group("BLOCK_METADATA_COUNTS", block_metadata_counts(blocks))
    print_scalar_group("CONTENT_METADATA_COUNTS", content_metadata_counts(content_blocks))

    print_counter(
        "SECTION_TYPE_COUNTS",
        counter([section.get("section_type") or section.get("type") for section in sections]),
    )
    print_counter(
        "CONTENT_TYPE_COUNTS",
        counter([item.get("type") for item in content_blocks]),
    )
    print_counter(
        "IR_TYPE_COUNTS",
        counter([block.get("type") for block in blocks]),
    )
    print_counter(
        "IR_FORM_COUNTS",
        counter([block.get("form") for block in blocks]),
    )
    print_counter(
        "IR_FUNCTION_COUNTS",
        counter([block.get("function") for block in blocks]),
    )

    print_scalar_group("INSPECTION_OBJECT", inspection_summary(result))
    print_scalar_group("EXECUTION_PLAN_OBJECT", execution_plan_summary(result))

    return build_summary_row(
        path=path,
        result=result,
        sections=sections,
        blocks=blocks,
    )


# --------------- FINAL TABLE --------------------------------

def print_final_table(rows: list[dict[str, Any]]) -> None:
    line("=")
    print("ALL FILE SUMMARY")
    line("=")

    header = (
        f"{'file':<42}"
        f"{'mb':>8}"
        f"{'wall_ms':>12}"
        f"{'inspect':>10}"
        f"{'strategy':>10}"
        f"{'ast_cpu':>10}"
        f"{'enrich_cpu':>12}"
        f"{'clean_cpu':>10}"
        f"{'worker_sum':>12}"
        f"{'worker_max':>12}"
        f"{'secs':>7}"
        f"{'blocks':>8}"
        f"{'warn':>6}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['file']:<42}"
            f"{row['mb']:>8.2f}"
            f"{fmt_ms(row['wall_ms']):>12}"
            f"{fmt_ms(row['inspect_ms']):>10}"
            f"{fmt_ms(row['strategy_ms']):>10}"
            f"{fmt_ms(row['ast_cpu_ms']):>10}"
            f"{fmt_ms(row['enrich_cpu_ms']):>12}"
            f"{fmt_ms(row['clean_cpu_ms']):>10}"
            f"{fmt_ms(row['worker_sum_ms']):>12}"
            f"{fmt_ms(row['worker_max_ms']):>12}"
            f"{row['sections']:>7}"
            f"{row['blocks']:>8}"
            f"{row['warnings']:>6}"
        )

    print("-" * len(header))

    print(
        f"{'TOTAL':<42}"
        f"{sum(row['mb'] for row in rows):>8.2f}"
        f"{fmt_ms(sum(row['wall_ms'] for row in rows)):>12}"
        f"{fmt_ms(sum(row['inspect_ms'] for row in rows)):>10}"
        f"{fmt_ms(sum(row['strategy_ms'] for row in rows)):>10}"
        f"{fmt_ms(sum(row['ast_cpu_ms'] for row in rows)):>10}"
        f"{fmt_ms(sum(row['enrich_cpu_ms'] for row in rows)):>12}"
        f"{fmt_ms(sum(row['clean_cpu_ms'] for row in rows)):>10}"
        f"{fmt_ms(sum(row['worker_sum_ms'] for row in rows)):>12}"
        f"{fmt_ms(max((row['worker_max_ms'] for row in rows), default=0.0)):>12}"
        f"{sum(row['sections'] for row in rows):>7}"
        f"{sum(row['blocks'] for row in rows):>8}"
        f"{sum(row['warnings'] for row in rows):>6}"
    )


# --------------- RUNNER --------------------------------

def parse_one(path: Path) -> dict[str, Any]:
    result = execute_markdown_parse(str(path))

    return print_file_report(path, result)


def main() -> int:
    files = discover_files()

    if not files:
        print("No Markdown/MDX files found.")
        return 2

    rows: list[dict[str, Any]] = []
    failed = 0

    for path in files:
        try:
            rows.append(parse_one(path))
        except Exception:
            failed += 1
            line("=")
            print(f"FAILED: {path}")
            line("=")
            traceback.print_exc()

    if rows:
        print_final_table(rows)

    line("=")
    print(f"files: {len(files)} | passed: {len(rows)} | failed: {failed}")
    line("=")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
