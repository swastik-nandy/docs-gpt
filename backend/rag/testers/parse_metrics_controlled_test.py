from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Literal


# --------------- HARD-CODED TEST KNOBS --------------------------------

TestMode = Literal["sequential", "pool"]

TEST_MODE: TestMode = "sequential"
TEST_CPU_COUNT = 0
TEST_WORKERS = 0


# --------------- PATH BOOTSTRAP --------------------------------

THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = THIS_FILE.parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from rag.ingestion.data_transformation.parser.adapters.markdown.execution import (  # noqa: E402
    MarkdownExecutionPolicyContext,
    execute_markdown_parse,
)
from rag.testers.parse_metrics_test import (  # noqa: E402
    discover_files,
    line,
    print_file_report,
    print_final_table,
)


# --------------- CONTROLLED POLICY --------------------------------

def detected_cpu_count() -> int:
    if TEST_CPU_COUNT > 0:
        return max(1, TEST_CPU_COUNT)

    return max(1, os.cpu_count() or 1)


def auto_worker_count(cpu_count: int) -> int:
    return max(1, cpu_count - 1)


def controlled_policy_context() -> MarkdownExecutionPolicyContext:
    cpu_count = detected_cpu_count()

    if TEST_MODE == "sequential":
        return MarkdownExecutionPolicyContext(
            user_tier="internal",
            demand_level="low",
            local_cpu_count=cpu_count,
            max_local_workers=1,
            allow_process_pool=False,
            available_resource_classes=("tiny", "small", "medium", "large", "xlarge"),
            prefer_fast_execution=False,
        )

    if TEST_MODE == "pool":
        workers = TEST_WORKERS if TEST_WORKERS > 0 else auto_worker_count(cpu_count)

        return MarkdownExecutionPolicyContext(
            user_tier="internal",
            demand_level="low",
            local_cpu_count=cpu_count,
            max_local_workers=max(1, workers),
            allow_process_pool=True,
            available_resource_classes=("tiny", "small", "medium", "large", "xlarge"),
            prefer_fast_execution=True,
        )

    raise ValueError(f"Unsupported TEST_MODE: {TEST_MODE!r}")


# --------------- RUNNER --------------------------------

def print_control_header() -> None:
    cpu_count = detected_cpu_count()
    workers = TEST_WORKERS if TEST_WORKERS > 0 else auto_worker_count(cpu_count)

    line("=")
    print("CONTROLLED MARKDOWN PARSER METRICS TEST")
    line("=")
    print(f"test_mode   : {TEST_MODE}")
    print(f"cpu_count   : {cpu_count}")
    print(f"workers     : {1 if TEST_MODE == 'sequential' else workers}")
    print(f"policy      : hardcoded tester override")
    line("-")


def parse_one(path: Path) -> dict:
    result = execute_markdown_parse(
        str(path),
        policy_context=controlled_policy_context(),
    )

    return print_file_report(path, result)


def main() -> int:
    print_control_header()

    files = discover_files()

    if not files:
        print("No Markdown/MDX files found.")
        return 2

    rows: list[dict] = []
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