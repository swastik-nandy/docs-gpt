from __future__ import annotations

import logging
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable

from rag.ingestion.data_transformation.runner.discovery import DEFAULT_OUTPUT_ROOT
from rag.ingestion.data_transformation.runner.discovery import DEFAULT_SOURCE_ROOT
from rag.ingestion.data_transformation.runner.discovery import discover_ready_jobs
from rag.ingestion.data_transformation.runner.schema import JobTransformReport
from rag.ingestion.data_transformation.runner.schema import TransformationJob
from rag.ingestion.data_transformation.runner.transform import TransformOptions
from rag.ingestion.data_transformation.runner.transform import transform_jobs


LOGGER = logging.getLogger(
    __name__
)


#--------------------OPTIONS--------------------

@dataclass(frozen=True)
class PollingWorkerOptions:
    source_root: Path = DEFAULT_SOURCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    enable_overlap: bool = False
    continue_on_error: bool = True
    poll_interval_seconds: float = 5.0
    idle_jitter_seconds: float = 0.75
    run_once: bool = False


#--------------------ITERATION RESULT--------------------

@dataclass(frozen=True)
class PollingIterationResult:
    discovered_job_count: int
    processed_job_count: int
    failed_job_count: int
    reports: tuple[
        JobTransformReport,
        ...
    ]

    @property
    def succeeded_job_count(
        self,
    ) -> int:
        return (
            self.processed_job_count
            - self.failed_job_count
        )


#--------------------WORKER--------------------

class DataTransformationPollingWorker:
    def __init__(
        self,
        options: PollingWorkerOptions,
        *,
        sleep_fn: Callable[
            [float],
            None,
        ] = time.sleep,
        stop_requested_fn: Callable[
            [],
            bool,
        ]
        | None = None,
    ) -> None:
        self.options = options
        self._sleep = sleep_fn
        self._external_stop_requested = (
            stop_requested_fn
        )
        self._stopping = False

    def request_stop(
        self,
    ) -> None:
        self._stopping = True

    def run_forever(
        self,
    ) -> None:
        self._install_signal_handlers()

        LOGGER.info(
            "Starting data transformation worker | "
            "source_root=%s output_root=%s "
            "poll_interval=%s",
            self.options.source_root,
            self.options.output_root,
            self.options.poll_interval_seconds,
        )

        while not self._should_stop():
            result = self.run_once()

            if (
                result.discovered_job_count
                == 0
            ):
                self._sleep(
                    self._next_idle_sleep_seconds()
                )

            if self.options.run_once:
                break

        LOGGER.info(
            "Data transformation worker stopped."
        )

    def run_once(
        self,
    ) -> PollingIterationResult:
        jobs = self._discover_jobs()

        if not jobs:
            LOGGER.debug(
                "No ready transformation jobs found."
            )

            return PollingIterationResult(
                discovered_job_count=0,
                processed_job_count=0,
                failed_job_count=0,
                reports=(),
            )

        LOGGER.info(
            "Discovered %s ready transformation job(s).",
            len(
                jobs
            ),
        )

        reports = self._transform_jobs(
            jobs
        )

        failed = sum(
            1
            for report in reports
            if report.status
            != "succeeded"
        )

        LOGGER.info(
            "Transformation iteration finished | "
            "processed=%s succeeded=%s failed=%s",
            len(
                reports
            ),
            len(
                reports
            )
            - failed,
            failed,
        )

        return PollingIterationResult(
            discovered_job_count=len(
                jobs
            ),
            processed_job_count=len(
                reports
            ),
            failed_job_count=failed,
            reports=tuple(
                reports
            ),
        )

    def _discover_jobs(
        self,
    ) -> list[TransformationJob]:
        try:
            return discover_ready_jobs(
                source_root=self.options.source_root,
                output_root=self.options.output_root,
            )

        except Exception:
            LOGGER.exception(
                "Transformation job discovery failed."
            )

            return []

    def _transform_jobs(
        self,
        jobs: Iterable[
            TransformationJob
        ],
    ) -> list[JobTransformReport]:
        options = TransformOptions(
            enable_overlap=(
                self.options.enable_overlap
            ),
            continue_on_error=(
                self.options.continue_on_error
            ),
        )

        try:
            return transform_jobs(
                jobs,
                options=options,
            )

        except Exception:
            LOGGER.exception(
                "Transformation job batch crashed."
            )

            return []

    def _next_idle_sleep_seconds(
        self,
    ) -> float:
        base = max(
            0.1,
            float(
                self.options.poll_interval_seconds
            ),
        )

        jitter = max(
            0.0,
            float(
                self.options.idle_jitter_seconds
            ),
        )

        if jitter <= 0:
            return base

        return (
            base
            + random.uniform(
                0.0,
                jitter,
            )
        )

    def _should_stop(
        self,
    ) -> bool:
        if self._stopping:
            return True

        if (
            self._external_stop_requested
            and self._external_stop_requested()
        ):
            return True

        return False

    def _install_signal_handlers(
        self,
    ) -> None:
        def handle_stop(
            _signum: int,
            _frame: object,
        ) -> None:
            LOGGER.info(
                "Stop signal received. "
                "Worker exits after current iteration."
            )

            self.request_stop()

        try:
            signal.signal(
                signal.SIGTERM,
                handle_stop,
            )

            signal.signal(
                signal.SIGINT,
                handle_stop,
            )

        except ValueError:
            LOGGER.debug(
                "Signal handlers unavailable "
                "outside the main thread."
            )


#--------------------PUBLIC API--------------------

def run_polling_worker(
    options: PollingWorkerOptions,
) -> None:
    worker = DataTransformationPollingWorker(
        options
    )

    worker.run_forever()
