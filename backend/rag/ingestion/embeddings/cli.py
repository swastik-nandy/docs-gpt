from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import time
from typing import Any

from rag.ingestion.embeddings.artifact_reader import discover_chunk_artifact_jobs
from rag.ingestion.embeddings.config import get_embedding_pipeline_config
from rag.ingestion.embeddings.service import ChunkArtifactIndexingService


LOGGER = logging.getLogger(__name__)


class StopFlag:
    def __init__(self) -> None:
        self.stop_requested = False

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True


def configure_logging(level: str) -> None:
    active_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    logging.getLogger("rag.ingestion.embeddings").setLevel(active_level)

    noisy_loggers = [
        "httpx",
        "httpcore",
        "qdrant_client",
        "urllib3",
        "transformers",
        "sentence_transformers",
        "FlagEmbedding",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index transformed chunk artifacts into Qdrant.",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one indexing iteration and exit.",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll chunked artifacts and index ready jobs.",
    )

    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only list discovered chunk artifact jobs. Do not embed or upsert.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    config = get_embedding_pipeline_config()

    if args.discover_only:
        jobs = discover_chunk_artifact_jobs(config)

        print("Chunk artifact discovery")
        print("------------------------")
        print(f"chunked_root: {config.artifacts.chunked_root}")
        print(f"indexed_root: {config.artifacts.indexed_root}")
        print(f"jobs:         {len(jobs)}")

        for job in jobs:
            print(
                f"- tenant={job.tenant_id} "
                f"job={job.job_id} "
                f"files={job.file_count}"
            )

        return 0

    if not args.once and not args.watch:
        args.once = True

    stop_flag = StopFlag()
    signal.signal(signal.SIGTERM, stop_flag.request_stop)
    signal.signal(signal.SIGINT, stop_flag.request_stop)

    service = ChunkArtifactIndexingService(config)

    if args.once:
        reports = service.index_available_jobs()
        _print_reports(reports)
        return _exit_code(reports)

    LOGGER.info(
        "Starting embeddings indexing worker | chunked_root=%s indexed_root=%s poll_interval=%s",
        config.artifacts.chunked_root,
        config.artifacts.indexed_root,
        config.indexing.poll_interval_seconds,
    )

    while not stop_flag.stop_requested:
        reports = service.index_available_jobs()

        if reports:
            _print_reports(reports)
        else:
            sleep_seconds = (
                config.indexing.poll_interval_seconds
                + random.uniform(0.0, config.indexing.idle_jitter_seconds)
            )
            time.sleep(sleep_seconds)

    LOGGER.info("Embeddings indexing worker stopped.")
    return 0


def _print_reports(reports: list[Any]) -> None:
    if not reports:
        print("No chunk artifact jobs ready for indexing.")
        return

    print("\nIndexing results")
    print("----------------")

    for report in reports:
        print(
            f"- {report.tenant_id}/{report.job_id} "
            f"status={report.status} "
            f"sources={report.source_count} "
            f"chunks={report.chunk_count} "
            f"embedded={report.embedded_count} "
            f"upserted={report.upserted_count} "
            f"skipped={report.skipped_count} "
            f"failed={report.failed}"
        )


def _exit_code(reports: list[Any]) -> int:
    if not reports:
        return 0

    return 0 if all(report.status == "succeeded" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())