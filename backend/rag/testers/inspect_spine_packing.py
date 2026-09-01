from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# --------------- CONFIG --------------------------------

DEFAULT_TEST_FILES = [
    "test_data/large/aws-ecs-realistic-5mb.md",
]

TOP_N_FRAGMENT_SAMPLES = 8


# --------------- PATH BOOTSTRAP --------------------------------

THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = THIS_FILE.parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from rag.ingestion.data_transformation.parser.adapters.markdown.execution.packed_spine import (  # noqa: E402
    load_packed_spine_config,
    pack_spine_nodes_with_diagnostics,
)
from rag.ingestion.data_transformation.parser.inspection.markdown import inspect_markdown_file  # noqa: E402


# --------------- HELPERS --------------------------------

def line(char: str = "-", width: int = 120) -> None:
    print(char * width)


def kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<42} {value}")


def node_title(node: Any) -> str:
    return str(getattr(node, "title", "") or "").strip()


def node_level(node: Any) -> int:
    return int(getattr(node, "level", 0) or 0)


def node_id(node: Any) -> str:
    return str(getattr(node, "node_id", "") or "").strip()


def pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"

    return f"{(part / total) * 100:.2f}%"


# --------------- REPORT --------------------------------

def print_fragment_samples(fragments: list[Any]) -> None:
    print("\nPACKED FRAGMENT SAMPLES")
    line("-")

    for fragment in fragments[:TOP_N_FRAGMENT_SAMPLES]:
        first = fragment.spine_nodes[0] if fragment.spine_nodes else None
        last = fragment.spine_nodes[-1] if fragment.spine_nodes else None

        kv(
            f"[{fragment.fragment_index}]",
            (
                f"id={fragment.packed_fragment_id} "
                f"nodes={fragment.spine_node_count} "
                f"bytes={fragment.byte_size} "
                f"lines={fragment.line_count} "
                f"levels={fragment.min_heading_level}-{fragment.max_heading_level} "
                f"variance={fragment.heading_depth_variance} "
                f"tokens={fragment.estimated_tokens} "
                f"close={fragment.close_reason}"
            ),
        )

        if first:
            kv("first", f"L{node_level(first)} {node_id(first)} {node_title(first)}", indent=4)

        if last:
            kv("last", f"L{node_level(last)} {node_id(last)} {node_title(last)}", indent=4)

    remaining = len(fragments) - TOP_N_FRAGMENT_SAMPLES

    if remaining > 0:
        kv("more_fragments", remaining)


def print_report(path: Path) -> None:
    inspection = inspect_markdown_file(str(path))
    spine_nodes = list(getattr(inspection, "spine_nodes", []) or [])

    config = load_packed_spine_config()

    fragments, diagnostics = pack_spine_nodes_with_diagnostics(
        spine_nodes,
        source_doc_id="inspection_only",
        config=config,
        source_path=str(path),
    )

    meta = diagnostics.to_meta()

    line("=")
    print(f"SPINE PACKING INSPECTION: {path.name}")
    line("=")

    print("\nINSPECTION")
    kv("source_path", path)
    kv("size_class", getattr(inspection, "size_class", ""))
    kv("risk_level", getattr(inspection, "risk_level", ""))
    kv("parse_strategy", getattr(inspection, "parse_strategy", ""))
    kv("line_count", getattr(inspection, "line_count", 0))
    kv("heading_count", getattr(inspection, "heading_count", 0))
    kv("estimated_tokens", getattr(inspection, "estimated_tokens", 0))
    kv("spine_node_count", len(spine_nodes))

    print("\nPACKING CONFIG")
    for key, value in meta["config"].items():
        kv(key, value)

    print("\nPACKING SUMMARY")
    kv("packed_fragment_count", meta["packed_fragment_count"])
    kv("node_reduction_ratio", f"{meta['spine_node_count']} -> {meta['packed_fragment_count']}")
    kv("avg_spine_nodes_per_fragment", meta["avg_spine_nodes_per_fragment"])
    kv("max_spine_nodes_per_fragment", meta["max_spine_nodes_per_fragment"])
    kv("avg_fragment_bytes", meta["avg_fragment_bytes"])
    kv("max_fragment_bytes", meta["max_fragment_bytes"])
    kv("avg_fragment_lines", meta["avg_fragment_lines"])
    kv("max_fragment_lines", meta["max_fragment_lines"])
    kv("avg_heading_depth_variance", meta["avg_heading_depth_variance"])
    kv("max_heading_depth_variance", meta["max_heading_depth_variance"])

    print("\nCLOSE REASONS")
    total_fragments = meta["packed_fragment_count"]

    for reason, count in sorted(
        meta["close_reason_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        kv(reason, f"{count} ({pct(count, total_fragments)})")

    print_fragment_samples(fragments)


# --------------- MAIN --------------------------------

def main() -> int:
    paths = [
        (BACKEND_ROOT / path).resolve()
        for path in DEFAULT_TEST_FILES
    ]

    for path in paths:
        if not path.exists():
            print(f"Missing file: {path}")
            return 2

        print_report(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
