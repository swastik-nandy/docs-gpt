from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.chunking.parser_result_adapter import chunk_parse_result
from rag.ingestion.data_transformation.parser.adapters.markdown.execution import execute_markdown_parse


DEFAULT_TEST_FILES = [
    "test_data/aws/ecs/ecs-anywhere.md",
]

ALLOWED_EXTENSIONS = {".md", ".mdx", ".markdown", ".mdown", ".mkd", ".mdwn"}


THIS_FILE = Path(__file__).resolve()


def find_backend_root(start: Path) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "rag").is_dir():
            return candidate

    raise RuntimeError(f"Could not locate backend root from: {start}")


BACKEND_ROOT = find_backend_root(THIS_FILE)

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def print_line(char: str = "-", width: int = 100) -> None:
    print(char * width)


def print_kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<34} {value}")


def preview(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", "\\n").strip()

    if len(text) <= limit:
        return text

    return text[: max(0, limit - 3)] + "..."


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
                candidate.resolve()
                for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in ALLOWED_EXTENSIONS
            )

    return sorted(dict.fromkeys(files))


def chunk_meta(chunk: Any) -> dict[str, Any]:
    return chunk.metadata if isinstance(chunk.metadata, dict) else {}


def count_values(values: list[Any]) -> Counter:
    return Counter(str(value or "unknown") for value in values)


def duplicate_chunk_ids(chunks: list[Any]) -> list[str]:
    ids = [str(chunk.chunk_id or "") for chunk in chunks if chunk.chunk_id]
    counts = Counter(ids)

    return sorted(chunk_id for chunk_id, count in counts.items() if count > 1)


def empty_chunk_texts(chunks: list[Any]) -> list[Any]:
    return [
        chunk
        for chunk in chunks
        if not str(chunk.text or "").strip()
    ]


def missing_parser_metadata(chunks: list[Any]) -> list[Any]:
    required_keys = {
        "parser_doc_id",
        "parse_strategy",
        "parser_family",
        "parser_variant",
        "chunk_global_order",
    }

    missing: list[Any] = []

    for chunk in chunks:
        meta = chunk_meta(chunk)

        if any(key not in meta for key in required_keys):
            missing.append(chunk)

    return missing


def missing_ir_metadata(chunks: list[Any]) -> list[Any]:
    missing: list[Any] = []

    for chunk in chunks:
        meta = chunk_meta(chunk)

        if not meta.get("ir_id") and not meta.get("ir_type") and not meta.get("ir_form"):
            missing.append(chunk)

    return missing


def not_using_retrieval_text(chunks: list[Any]) -> list[Any]:
    offenders: list[Any] = []

    for chunk in chunks:
        meta = chunk_meta(chunk)

        if meta.get("retrieval_text") and meta.get("chunk_text_source") != "retrieval_text":
            offenders.append(chunk)

    return offenders


def oversized_chunks(chunks: list[Any], max_tokens: int) -> list[Any]:
    return [
        chunk
        for chunk in chunks
        if int(chunk.tokens or 0) > max_tokens
    ]



def print_timing_report(
    *,
    parse_seconds: float,
    chunk_seconds: float,
    total_seconds: float,
    chunk_count: int,
) -> None:
    print("\nTIMING")
    print_kv("parse_seconds", round(parse_seconds, 4))
    print_kv("chunk_seconds", round(chunk_seconds, 4))
    print_kv("total_seconds", round(total_seconds, 4))
    print_kv("chunks_per_second", round(chunk_count / max(total_seconds, 0.0001), 2))
    print_kv("chunker_chunks_per_second", round(chunk_count / max(chunk_seconds, 0.0001), 2))


def print_parser_report(parse_result: Any) -> None:
    inspection = getattr(parse_result, "inspection", None)
    plan = getattr(parse_result, "execution_plan", None)

    print("\nPARSER RESULT")
    print_kv("doc_id", getattr(parse_result, "doc_id", ""))
    print_kv("parser_family", getattr(parse_result, "parser_family", ""))
    print_kv("parser_variant", getattr(parse_result, "parser_variant", ""))
    print_kv("parse_strategy", getattr(parse_result, "parse_strategy", ""))
    print_kv("unit_count", getattr(parse_result, "unit_count", 0))

    if inspection is not None:
        print_kv("size_class", getattr(inspection, "size_class", ""))
        print_kv("risk_level", getattr(inspection, "risk_level", ""))
        print_kv("estimated_tokens", getattr(inspection, "estimated_tokens", 0))
        print_kv("heading_count", getattr(inspection, "heading_count", 0))

    if plan is not None:
        print_kv("execution_mode", getattr(plan, "execution_mode", ""))
        print_kv("resource_class", getattr(plan, "resource_class", ""))
        print_kv("max_workers", getattr(plan, "max_workers", ""))


def print_chunk_report(chunks: list[Any], max_tokens: int) -> None:
    print("\nCHUNK OUTPUT")
    print_kv("chunk_count", len(chunks))
    print_kv("total_tokens", sum(int(chunk.tokens or 0) for chunk in chunks))
    print_kv("max_chunk_tokens", max((int(chunk.tokens or 0) for chunk in chunks), default=0))
    print_kv("avg_chunk_tokens", round(sum(int(chunk.tokens or 0) for chunk in chunks) / max(1, len(chunks)), 2))

    print("\nCHUNK TYPE COUNTS")

    type_counts = count_values([
        f"{chunk.chunk_type}/{chunk.subtype}"
        for chunk in chunks
    ])

    for key, count in type_counts.most_common(14):
        print(f"  {key:<32} {count}")

    text_source_counts = count_values([
        chunk_meta(chunk).get("chunk_text_source")
        for chunk in chunks
    ])

    print("\nTEXT SOURCE COUNTS")

    for key, count in text_source_counts.most_common(8):
        print(f"  {key:<32} {count}")

    strategy_counts = count_values([
        chunk_meta(chunk).get("parse_strategy")
        for chunk in chunks
    ])

    print("\nSTRATEGY COUNTS")

    for key, count in strategy_counts.most_common(8):
        print(f"  {key:<32} {count}")

    duplicated = duplicate_chunk_ids(chunks)
    empty_text = empty_chunk_texts(chunks)
    missing_parser = missing_parser_metadata(chunks)
    missing_ir = missing_ir_metadata(chunks)
    retrieval_text_offenders = not_using_retrieval_text(chunks)
    oversized = oversized_chunks(chunks, max_tokens=max_tokens)

    gates = {
        "chunks_present": bool(chunks),
        "duplicate_chunk_ids": not duplicated,
        "empty_chunk_text": not empty_text,
        "parser_metadata_present": not missing_parser,
        "ir_metadata_present": not missing_ir,
        "uses_retrieval_text_when_available": not retrieval_text_offenders,
        "under_token_limit_or_marked_atomic": not [
            chunk
            for chunk in oversized
            if not chunk_meta(chunk).get("oversized_atomic")
        ],
    }

    print("\nCHUNK QUALITY GATES")

    for name, passed in gates.items():
        print_kv(name, "PASS" if passed else "FAIL")

    if oversized:
        print_kv("oversized_chunks", len(oversized))


def print_chunk_samples(chunks: list[Any], limit: int) -> None:
    print("\nSELECTED CHUNKS")

    if not chunks:
        print_kv("chunks", "none")
        return

    selected = select_interesting_chunks(chunks, limit=limit)

    for index, chunk in enumerate(selected):
        meta = chunk_meta(chunk)

        print(
            f"  [{index}] "
            f"id={chunk.chunk_id} "
            f"type={chunk.chunk_type}/{chunk.subtype} "
            f"tokens={chunk.tokens} "
            f"order={chunk.order}"
        )
        print(f"      heading       : {preview(chunk.heading)}")
        print(f"      context       : {preview(chunk.context)}")
        print(f"      strategy      : {preview(meta.get('parse_strategy'))}")
        print(f"      unit          : {preview(meta.get('parser_unit_id'))}")
        print(f"      text_source   : {preview(meta.get('chunk_text_source'))}")
        print(f"      text          : {preview(chunk.text, 360)}")

        meta_keys = [
            key
            for key in (
                "ir_id",
                "ir_type",
                "ir_form",
                "function",
                "parent_instruction",
                "table_profile",
                "relationships",
                "quality_score",
                "validated",
                "oversized_atomic",
            )
            if meta.get(key)
        ]

        if meta_keys:
            print(f"      meta_keys     : {', '.join(meta_keys)}")


def select_interesting_chunks(chunks: list[Any], limit: int) -> list[Any]:
    buckets = [
        [chunk for chunk in chunks if chunk_meta(chunk).get("oversized_atomic")],
        [chunk for chunk in chunks if chunk.subtype == "code_block"],
        [chunk for chunk in chunks if chunk.subtype == "table_row"],
        [chunk for chunk in chunks if chunk.subtype in {"note", "callout"}],
        [chunk for chunk in chunks if chunk.subtype == "list_item"],
        [chunk for chunk in chunks if chunk_meta(chunk).get("parent_instruction")],
        chunks[: max(3, limit)],
    ]

    selected: list[Any] = []
    seen: set[str] = set()

    for bucket in buckets:
        for chunk in bucket:
            key = str(chunk.chunk_id or id(chunk))

            if key in seen:
                continue

            selected.append(chunk)
            seen.add(key)

            if len(selected) >= limit:
                return selected

    return selected


def run_one_file(path: Path, args: argparse.Namespace) -> bool:
    print_line("=")
    print(f"FILE: {path}")
    print_line("=")

    total_started = time.perf_counter()

    try:
        parse_started = time.perf_counter()
        parse_result = execute_markdown_parse(str(path))
        parse_seconds = time.perf_counter() - parse_started
    except Exception:
        print("\nPARSER CRASHED")
        traceback.print_exc()
        return False

    print_parser_report(parse_result)

    try:
        chunk_started = time.perf_counter()
        chunks = chunk_parse_result(
            parse_result,
            source=str(path),
            max_tokens=args.max_tokens,
            enable_overlap=args.enable_overlap,
        )
        chunk_seconds = time.perf_counter() - chunk_started
    except Exception:
        print("\nCHUNKER CRASHED")
        traceback.print_exc()
        return False

    total_seconds = time.perf_counter() - total_started

    print_timing_report(
        parse_seconds=parse_seconds,
        chunk_seconds=chunk_seconds,
        total_seconds=total_seconds,
        chunk_count=len(chunks),
    )

    print_chunk_report(chunks, max_tokens=args.max_tokens)
    print_chunk_samples(chunks, limit=args.samples)

    failures = [
        duplicate_chunk_ids(chunks),
        empty_chunk_texts(chunks),
        missing_parser_metadata(chunks),
        missing_ir_metadata(chunks),
        not_using_retrieval_text(chunks),
    ]

    return bool(chunks) and not any(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse + chunk wiring smoke test. No persistence.",
    )

    parser.add_argument("paths", nargs="*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--enable-overlap", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_line("=")
    print("PARSE → CHUNK WIRING TEST")
    print_line("=")

    print_kv("backend_root", BACKEND_ROOT)
    print_kv("max_tokens", args.max_tokens)
    print_kv("samples", args.samples)
    print_kv("enable_overlap", args.enable_overlap)

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