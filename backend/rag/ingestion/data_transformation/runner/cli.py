from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.chunking.config import build_chunk_token_budget
from rag.ingestion.data_transformation.chunking.config import get_chunking_config
from rag.ingestion.data_transformation.runner.discovery import DEFAULT_OUTPUT_ROOT
from rag.ingestion.data_transformation.runner.discovery import DEFAULT_SOURCE_ROOT
from rag.ingestion.data_transformation.runner.discovery import discover_ready_jobs
from rag.ingestion.data_transformation.runner.service import PollingWorkerOptions
from rag.ingestion.data_transformation.runner.service import run_polling_worker
from rag.ingestion.data_transformation.runner.transform import TransformOptions
from rag.ingestion.data_transformation.runner.transform import transform_jobs


#--------------------LOGGING--------------------

def configure_logging(
    level: str,
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            level.upper(),
            logging.INFO,
        ),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s | %(message)s"
        ),
        stream=sys.stdout,
    )


#--------------------OUTPUT--------------------

def print_line(
    char: str = "-",
    width: int = 100,
) -> None:
    print(
        char
        * width
    )


def print_kv(
    key: str,
    value: Any,
    indent: int = 2,
) -> None:
    print(
        f"{' ' * indent}"
        f"{key:<34} "
        f"{value}"
    )


#--------------------CLI--------------------

def parse_args(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse and chunk ready Markdown/MDX "
            "artifacts into JSONL retrieval chunks."
        )
    )

    parser.add_argument(
        "--source-root",
        default=str(
            DEFAULT_SOURCE_ROOT
        ),
        help=(
            "Artifact source root. Default: "
            "rag/ingestion/artifacts/source"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
        help=(
            "Artifact output root. Default: "
            "rag/ingestion/artifacts/chunked"
        ),
    )

    parser.add_argument(
        "--enable-overlap",
        action="store_true",
        help=(
            "Enable configured chunk overlap. "
            "Disabled by default."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop the current job after the first "
            "source transformation failure."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover ready jobs and files without "
            "running the parser/chunker."
        ),
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Continuously poll artifacts/source "
            "for newly ready jobs."
        ),
    )

    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--idle-jitter-seconds",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
    )

    return parser.parse_args()


#--------------------MAIN--------------------

def main(
) -> int:
    args = parse_args()

    configure_logging(
        args.log_level
    )

    source_root = (
        Path(
            args.source_root
        )
        .expanduser()
        .resolve()
    )

    output_root = (
        Path(
            args.output_root
        )
        .expanduser()
        .resolve()
    )

    if not source_root.exists():
        print(
            "Source artifact root does not exist:"
        )

        print(
            source_root
        )

        return 2

    if not source_root.is_dir():
        print(
            "Source artifact root is not a directory:"
        )

        print(
            source_root
        )

        return 2

    config = get_chunking_config()

    budget = build_chunk_token_budget(
        config
    )

    if args.watch:
        options = PollingWorkerOptions(
            source_root=source_root,
            output_root=output_root,
            enable_overlap=args.enable_overlap,
            continue_on_error=(
                not args.fail_fast
            ),
            poll_interval_seconds=(
                args.poll_interval_seconds
            ),
            idle_jitter_seconds=(
                args.idle_jitter_seconds
            ),
            run_once=False,
        )

        run_polling_worker(
            options
        )

        return 0

    print_line(
        "="
    )

    print(
        "DATA TRANSFORMATION ARTIFACT RUNNER"
    )

    print_line(
        "="
    )

    print_kv(
        "source_root",
        source_root,
    )

    print_kv(
        "output_root",
        output_root,
    )

    print_kv(
        "minimum_chunk_tokens",
        budget.minimum_tokens,
    )

    print_kv(
        "preferred_chunk_tokens",
        budget.preferred_tokens,
    )

    print_kv(
        "strict_chunk_tokens",
        budget.strict_tokens,
    )

    print_kv(
        "effective_hard_tokens",
        budget.effective_chunk_hard_tokens,
    )

    print_kv(
        "enable_overlap",
        args.enable_overlap,
    )

    print_kv(
        "fail_fast",
        args.fail_fast,
    )

    print_kv(
        "dry_run",
        args.dry_run,
    )

    try:
        jobs = discover_ready_jobs(
            source_root=source_root,
            output_root=output_root,
        )

    except Exception:
        print(
            "\nDISCOVERY FAILED"
        )

        traceback.print_exc()

        return 2

    print(
        "\nDISCOVERY"
    )

    print_kv(
        "ready_jobs",
        len(
            jobs
        ),
    )

    total_sources = 0

    for job in jobs:
        total_sources += len(
            job.files
        )

        print(
            f"\n  {job.job_key}"
        )

        print(
            f"    output: "
            f"{job.output_job_root}"
        )

        for source in job.files:
            chunks_path = (
                job.output_job_root
                / "chunks"
                / (
                    f"{source.source_id}"
                    ".chunks.jsonl"
                )
            )

            print(
                f"    - {source.relative_path}"
            )

            print(
                f"      source_id: "
                f"{source.source_id}"
            )

            print(
                f"      chunks: "
                f"{chunks_path}"
            )

    print_kv(
        "ready_sources",
        total_sources,
    )

    if args.dry_run:
        return 0

    if not jobs:
        print(
            "\nNo ready jobs found."
        )

        return 0

    options = TransformOptions(
        enable_overlap=args.enable_overlap,
        continue_on_error=(
            not args.fail_fast
        ),
    )

    try:
        reports = transform_jobs(
            jobs,
            options=options,
        )

    except Exception:
        print(
            "\nTRANSFORMATION RUNNER CRASHED"
        )

        traceback.print_exc()

        return 1

    print(
        "\nRESULTS"
    )

    failed_jobs = 0
    succeeded_sources = 0
    failed_sources = 0
    chunks_written = 0

    for report in reports:
        if report.status != "succeeded":
            failed_jobs += 1

        print(
            f"\n  {report.tenant_id}/"
            f"{report.job_id}"
        )

        print(
            f"    status: "
            f"{report.status}"
        )

        job_root = (
            output_root
            / report.tenant_id
            / report.job_id
        )

        for source_report in report.source_reports:
            if (
                source_report.status
                == "succeeded"
            ):
                succeeded_sources += 1
                chunks_written += (
                    source_report.chunk_count
                )

                chunks_path = (
                    job_root
                    / "chunks"
                    / (
                        f"{source_report.source_id}"
                        ".chunks.jsonl"
                    )
                )

                print(
                    f"    ✓ "
                    f"{source_report.source_id}"
                )

                print(
                    f"      chunks: "
                    f"{source_report.chunk_count}"
                )

                print(
                    f"      tokens: "
                    f"{source_report.total_chunk_tokens}"
                )

                print(
                    f"      output: "
                    f"{chunks_path}"
                )

            else:
                failed_sources += 1

                print(
                    f"    ✗ "
                    f"{source_report.source_id}"
                )

                print(
                    f"      error: "
                    f"{source_report.error}"
                )

    print_line(
        "="
    )

    print(
        "FINAL SUMMARY"
    )

    print_line(
        "="
    )

    print_kv(
        "jobs_processed",
        len(
            reports
        ),
    )

    print_kv(
        "jobs_succeeded",
        len(
            reports
        )
        - failed_jobs,
    )

    print_kv(
        "jobs_failed",
        failed_jobs,
    )

    print_kv(
        "sources_succeeded",
        succeeded_sources,
    )

    print_kv(
        "sources_failed",
        failed_sources,
    )

    print_kv(
        "chunks_written",
        chunks_written,
    )

    return (
        0
        if failed_jobs
        == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
