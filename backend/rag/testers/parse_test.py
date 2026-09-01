from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TEST_FILES = [
    "test_data/aws/ecs/",
]

ALLOWED_EXTENSIONS = {".md", ".mdx", ".markdown", ".mdown", ".mkd", ".mdwn"}

PREVIEW_LIMIT = 180
DEFAULT_IR_SAMPLE_LIMIT = 14
DEFAULT_HEADING_PREVIEW_LIMIT = 12


THIS_FILE = Path(__file__).resolve()


def find_backend_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "rag").is_dir():
            return candidate

    raise RuntimeError(f"Could not locate backend root from: {start}")


BACKEND_ROOT = find_backend_root(THIS_FILE)

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from rag.ingestion.data_transformation.parser.adapters.markdown.execution import execute_markdown_parse  # noqa: E402


HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\([^)]+\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
CALLOUT_LABEL_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"(note|info|tip|warning|caution|danger|important|good to know|heads up|technical details)"
    r"(?:\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceProfile:
    byte_size: int
    line_count: int
    heading_count: int
    code_fence_count: int
    table_count_hint: int
    link_count: int
    image_count: int
    callout_label_count: int
    unclosed_fence: bool
    headings: list[str]


def preview(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    text = str(value or "")
    text = text.replace("\n", "\\n")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= limit:
        return text

    return text[: max(0, limit - 3)] + "..."


def print_line(char: str = "-", width: int = 100) -> None:
    print(char * width)


def print_kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<34} {value}")


def print_counter(title: str, counter: Counter, *, limit: int = 12) -> None:
    print(f"  {title}")

    if not counter:
        print("    <empty>")
        return

    for key, count in counter.most_common(limit):
        print(f"    {str(key):<30} {count}")

    if len(counter) > limit:
        print(f"    ... +{len(counter) - limit} more")


def normalize_heading(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"`+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*[¶#🔗§∞]+\s*$", "", value).strip()

    return value


def profile_source(path: Path) -> SourceProfile:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    headings: list[str] = []
    code_fence_count = 0
    table_count_hint = 0
    callout_label_count = 0

    in_fence = False
    fence_char = ""

    for line in lines:
        stripped = line.strip()
        fence_match = FENCE_RE.match(stripped)

        if fence_match:
            char = fence_match.group("fence")[0]

            if not in_fence:
                in_fence = True
                fence_char = char
                code_fence_count += 1
            elif char == fence_char:
                in_fence = False
                fence_char = ""

            continue

        if in_fence:
            continue

        heading_match = HEADING_RE.match(stripped)

        if heading_match:
            heading = re.sub(r"\s+", " ", heading_match.group("title")).strip()

            if heading:
                headings.append(heading)

        if TABLE_SEPARATOR_RE.match(stripped):
            table_count_hint += 1

        if CALLOUT_LABEL_RE.match(stripped):
            callout_label_count += 1

    return SourceProfile(
        byte_size=path.stat().st_size,
        line_count=len(lines),
        heading_count=len(headings),
        code_fence_count=code_fence_count,
        table_count_hint=table_count_hint,
        link_count=len(LINK_RE.findall(text)),
        image_count=len(IMAGE_RE.findall(text)),
        callout_label_count=callout_label_count,
        unclosed_fence=in_fence,
        headings=headings,
    )


def iter_sections(result: Any) -> list[dict[str, Any]]:
    return [
        section
        for section in result.iter_sections()
        if isinstance(section, dict)
    ]


def iter_blocks(result: Any) -> list[dict[str, Any]]:
    return [
        block
        for block in result.iter_blocks()
        if isinstance(block, dict)
    ]


def collect_content_blocks(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for section in sections
        for item in section.get("content", []) or []
        if isinstance(item, dict)
    ]


def count_values(values: Iterable[Any]) -> Counter:
    return Counter(str(value or "unknown") for value in values)


def duplicate_block_ids(blocks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for block in blocks:
        block_id = str(block.get("id") or "")

        if not block_id:
            continue

        if block_id in seen:
            duplicates.add(block_id)

        seen.add(block_id)

    return sorted(duplicates)


def empty_text_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        block
        for block in blocks
        if not str(block.get("text", "") or "").strip()
    ]


def empty_retrieval_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        block
        for block in blocks
        if not str(block.get("retrieval_text", "") or "").strip()
    ]


def blocks_without_context(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for block in blocks:
        if block.get("context") or block.get("local_context") or block.get("heading"):
            continue

        result.append(block)

    return result


def low_confidence_blocks(
    blocks: list[dict[str, Any]],
    *,
    threshold: float = 0.55,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for block in blocks:
        confidence = block.get("confidence")

        if not isinstance(confidence, (int, float)):
            continue

        if confidence < threshold:
            result.append(block)

    return result


def orphan_code_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for block in blocks:
        if block.get("type") != "code":
            continue

        meta = block.get("meta") or {}

        has_context = bool(
            block.get("heading")
            or block.get("context")
            or block.get("local_context")
            or meta.get("parent_instruction")
            or meta.get("parent_context")
        )

        if not has_context:
            result.append(block)

    return result


def heading_coverage(
    source: SourceProfile,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    source_headings = {
        normalize_heading(heading)
        for heading in source.headings
        if normalize_heading(heading)
    }

    parsed_headings = {
        normalize_heading(section.get("heading", ""))
        for section in sections
        if normalize_heading(section.get("heading", ""))
    }

    matched = source_headings & parsed_headings
    missing = sorted(source_headings - parsed_headings)

    return {
        "source_unique": len(source_headings),
        "parsed_unique": len(parsed_headings),
        "matched": len(matched),
        "ratio": round(len(matched) / max(1, len(source_headings)), 4),
        "missing": missing,
    }


def print_source_report(source: SourceProfile) -> None:
    print("\nSOURCE PROFILE")
    print_kv("byte_size", source.byte_size)
    print_kv("line_count", source.line_count)
    print_kv("heading_count", source.heading_count)
    print_kv("code_fence_count", source.code_fence_count)
    print_kv("table_count_hint", source.table_count_hint)
    print_kv("link_count", source.link_count)
    print_kv("image_count", source.image_count)
    print_kv("callout_label_count", source.callout_label_count)
    print_kv("unclosed_fence", source.unclosed_fence)


def print_decision_report(result: Any) -> None:
    inspection = getattr(result, "inspection", None)
    plan = getattr(result, "execution_plan", None)

    print("\nPARSER DECISION")
    print_kv("parser_family", getattr(result, "parser_family", ""))
    print_kv("parser_variant", getattr(result, "parser_variant", ""))
    print_kv("parse_strategy", getattr(result, "parse_strategy", ""))

    if inspection is not None:
        print_kv("size_class", getattr(inspection, "size_class", ""))
        print_kv("risk_level", getattr(inspection, "risk_level", ""))
        print_kv("estimated_tokens", getattr(inspection, "estimated_tokens", 0))
        print_kv("inspection_headings", getattr(inspection, "heading_count", 0))
        print_kv("spine_nodes", len(getattr(inspection, "spine_nodes", []) or []))
        print_kv("target_queue", getattr(inspection, "target_queue", ""))

    if plan is not None:
        print_kv("execution_mode", getattr(plan, "execution_mode", ""))
        print_kv("resource_class", getattr(plan, "resource_class", ""))
        print_kv("max_workers", getattr(plan, "max_workers", ""))
        print_kv("should_parse", getattr(plan, "should_parse", ""))

        reasons = list(getattr(plan, "reasons", []) or [])

        if reasons:
            print_kv("policy_reasons", ", ".join(reasons))


def print_structure_report(
    source: SourceProfile,
    result: Any,
    sections: list[dict[str, Any]],
    content_blocks: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> None:
    coverage = heading_coverage(source, sections)

    print("\nPARSE STRUCTURE")
    print_kv("unit_count", getattr(result, "unit_count", 0))
    print_kv("section_count", len(sections))
    print_kv("content_block_count", len(content_blocks))
    print_kv("ir_block_count", len(blocks))
    print_kv("heading_match_ratio", coverage["ratio"])
    print_kv("heading_matched", f"{coverage['matched']}/{coverage['source_unique']}")

    if coverage["missing"]:
        print_kv("missing_heading_preview", len(coverage["missing"]))

        for heading in coverage["missing"][:DEFAULT_HEADING_PREVIEW_LIMIT]:
            print(f"    - {preview(heading)}")

    print_counter(
        "section_type_counts",
        count_values(section.get("section_type") or section.get("type") for section in sections),
    )
    print_counter(
        "content_type_counts",
        count_values(item.get("type") for item in content_blocks),
    )
    print_counter(
        "ir_type_counts",
        count_values(f"{block.get('type')}/{block.get('form')}" for block in blocks),
    )


def print_units(result: Any) -> None:
    units = list(getattr(result, "units", []) or [])

    print("\nUNITS")

    if not units:
        print_kv("units", "none")
        return

    for index, unit in enumerate(units[:10]):
        print(
            f"  [{index}] "
            f"unit_id={getattr(unit, 'unit_id', '')} "
            f"sections={getattr(unit, 'section_count', 0)} "
            f"blocks={getattr(unit, 'block_count', 0)} "
            f"strategy={getattr(unit, 'parse_strategy', '')}"
        )

    if len(units) > 10:
        print(f"  ... +{len(units) - 10} more")


def block_reason(block: dict[str, Any]) -> str:
    meta = block.get("meta") or {}

    if not str(block.get("retrieval_text", "") or "").strip():
        return "empty_retrieval"

    if not str(block.get("text", "") or "").strip():
        return "empty_text"

    confidence = block.get("confidence")

    if isinstance(confidence, (int, float)) and confidence < 0.55:
        return "low_confidence"

    if block.get("type") == "code":
        return "code"

    if meta.get("table_profile"):
        return "table"

    if meta.get("kind") or meta.get("label"):
        return "callout"

    if meta.get("links") or meta.get("images") or meta.get("inline_code"):
        return "rich_inline_meta"

    if meta.get("component_context"):
        return "component_context"

    if block.get("type") == "list":
        return "list"

    return "representative"


def select_interesting_blocks(
    blocks: list[dict[str, Any]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    buckets: list[tuple[str, list[dict[str, Any]]]] = [
        ("empty_retrieval", empty_retrieval_blocks(blocks)),
        ("empty_text", empty_text_blocks(blocks)),
        ("low_confidence", low_confidence_blocks(blocks)),
        ("no_context", blocks_without_context(blocks)),
        ("orphan_code", orphan_code_blocks(blocks)),
        ("table", [block for block in blocks if (block.get("meta") or {}).get("table_profile")]),
        ("code", [block for block in blocks if block.get("type") == "code"]),
        ("callout", [block for block in blocks if (block.get("meta") or {}).get("kind") or (block.get("meta") or {}).get("label")]),
        ("rich_inline_meta", [block for block in blocks if any((block.get("meta") or {}).get(key) for key in ("links", "images", "inline_code"))]),
        ("component_context", [block for block in blocks if (block.get("meta") or {}).get("component_context")]),
        ("representative", blocks[: max(3, limit)]),
    ]

    selected: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()

    for reason, bucket in buckets:
        for block in bucket:
            block_id = str(block.get("id") or id(block))

            if block_id in seen_ids:
                continue

            selected.append((reason, block))
            seen_ids.add(block_id)

            if len(selected) >= limit:
                return selected

    return selected


def print_ir_block_samples(blocks: list[dict[str, Any]], *, limit: int) -> None:
    print("\nSELECTED IR BLOCKS")

    if not blocks:
        print_kv("blocks", "none")
        return

    selected = select_interesting_blocks(blocks, limit=limit)

    for index, (reason, block) in enumerate(selected):
        meta = block.get("meta") or {}

        print(
            f"  [{index}] reason={reason} "
            f"id={block.get('id', '')} "
            f"type={block.get('type', '')}/{block.get('form', '')} "
            f"fn={block.get('function', '')} "
            f"conf={block.get('confidence', '')}"
        )
        print(f"      heading       : {preview(block.get('heading', ''))}")
        print(f"      context       : {preview(block.get('context', ''))}")
        print(f"      local_context : {preview(block.get('local_context', ''))}")
        print(f"      text          : {preview(block.get('text', ''), 220)}")
        print(f"      retrieval     : {preview(block.get('retrieval_text', ''), 260)}")

        meta_keys = [
            key
            for key in (
                "source_type",
                "language",
                "links",
                "images",
                "inline_code",
                "table_profile",
                "kind",
                "label",
                "component_context",
                "parent_instruction",
            )
            if meta.get(key)
        ]

        if meta_keys:
            print(f"      meta_keys     : {', '.join(meta_keys)}")


def print_quality_gates(
    result: Any,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> bool:
    duplicates = duplicate_block_ids(blocks)
    empty_text = empty_text_blocks(blocks)
    empty_retrieval = empty_retrieval_blocks(blocks)
    no_context = blocks_without_context(blocks)
    low_confidence = low_confidence_blocks(blocks)
    orphan_code = orphan_code_blocks(blocks)

    hard_gates = {
        "inspection_present": getattr(result, "inspection", None) is not None,
        "execution_plan_present": getattr(result, "execution_plan", None) is not None,
        "unit_count_positive": int(getattr(result, "unit_count", 0) or 0) > 0,
        "sections_present": bool(sections),
        "ir_blocks_present": bool(blocks),
        "duplicate_block_ids": not duplicates,
        "empty_text_blocks": not empty_text,
        "empty_retrieval_blocks": not empty_retrieval,
    }

    print("\nQUALITY GATES")

    for name, passed in hard_gates.items():
        print_kv(name, "PASS" if passed else "FAIL")

    print_kv("blocks_without_context", "PASS" if not no_context else f"WARN {len(no_context)}")
    print_kv("low_confidence_blocks", "PASS" if not low_confidence else f"WARN {len(low_confidence)}")
    print_kv("orphan_code_blocks", "PASS" if not orphan_code else f"WARN {len(orphan_code)}")

    warnings = list(getattr(result, "warnings", []) or [])
    print_kv("parser_warnings", "PASS" if not warnings else f"WARN {len(warnings)}")

    if warnings:
        for warning in warnings[:8]:
            print(f"    - {preview(warning, 240)}")

    return all(hard_gates.values())


def resolve_backend_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return (BACKEND_ROOT / path).resolve()


def discover_files(args: argparse.Namespace) -> list[Path]:
    raw_paths = args.paths or DEFAULT_TEST_FILES
    files: list[Path] = []

    for raw_path in raw_paths:
        path = resolve_backend_path(raw_path)

        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path.resolve())
            continue

        if path.is_dir():
            iterator = path.rglob("*") if args.recursive else path.glob("*")
            files.extend(
                item.resolve()
                for item in iterator
                if item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS
            )

    return sorted(dict.fromkeys(files))


def run_one_file(path: Path, args: argparse.Namespace) -> bool:
    print_line("=")
    print(f"FILE: {path}")
    print_line("=")

    try:
        source = profile_source(path)
    except Exception:
        print("\nSOURCE PROFILE FAILED")
        traceback.print_exc()
        return False

    print_source_report(source)

    try:
        result = execute_markdown_parse(str(path))
    except Exception:
        print("\nPARSER CRASHED")
        traceback.print_exc()
        return False

    sections = iter_sections(result)
    content_blocks = collect_content_blocks(sections)
    blocks = iter_blocks(result)

    print_decision_report(result)
    print_structure_report(source, result, sections, content_blocks, blocks)
    print_units(result)
    quality_ok = print_quality_gates(result, sections, blocks)
    print_ir_block_samples(blocks, limit=args.ir_samples)

    return quality_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Markdown parser quality inspection harness.",
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help="Markdown files or directories. Defaults to DEFAULT_TEST_FILES.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories.",
    )

    parser.add_argument(
        "--ir-samples",
        type=int,
        default=DEFAULT_IR_SAMPLE_LIMIT,
        help="Maximum selected IR blocks to print.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_line("=")
    print("MARKDOWN PARSER QUALITY INSPECTION")
    print_line("=")

    print_kv("backend_root", BACKEND_ROOT)
    print_kv("ir_sample_limit", args.ir_samples)

    try:
        files = discover_files(args)
    except Exception:
        print("\nFILE DISCOVERY FAILED")
        traceback.print_exc()
        return 2

    print_kv("files_discovered", len(files))

    if not files:
        print("\nNo Markdown/MDX files found.")
        return 2

    passed = 0
    failed = 0

    for path in files:
        ok = run_one_file(path, args)

        if ok:
            passed += 1
        else:
            failed += 1

    print_line("=")
    print("FINAL SUMMARY")
    print_line("=")

    print_kv("passed", passed)
    print_kv("failed", failed)
    print_kv("total", passed + failed)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())