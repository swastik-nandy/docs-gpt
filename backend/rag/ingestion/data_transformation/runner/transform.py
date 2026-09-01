from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rag.ingestion.data_transformation.chunking.config import get_chunking_config
from rag.ingestion.data_transformation.parse_chunk_pipeline import run_parse_chunk_pipeline
from rag.ingestion.data_transformation.runner.schema import JobTransformReport
from rag.ingestion.data_transformation.runner.schema import SourceArtifact
from rag.ingestion.data_transformation.runner.schema import SourceTransformReport
from rag.ingestion.data_transformation.runner.schema import TransformationJob
from rag.ingestion.data_transformation.runner.writer import write_job_report
from rag.ingestion.data_transformation.runner.writer import write_source_failure
from rag.ingestion.data_transformation.runner.writer import write_source_success


#--------------------SOURCE TYPES--------------------

MARKDOWN_SUFFIXES = {
    ".md",
    ".mdx",
    ".markdown",
    ".mdown",
    ".mkd",
    ".mdwn",
}


#--------------------OPTIONS--------------------

@dataclass(frozen=True)
class TransformOptions:
    enable_overlap: bool = False
    continue_on_error: bool = True


#--------------------ERRORS--------------------

class UnsupportedSourceTypeError(
    ValueError
):
    pass


#--------------------JOB TRANSFORMATION--------------------

def transform_job(
    job: TransformationJob,
    *,
    options: TransformOptions | None = None,
) -> JobTransformReport:
    """
    Transform every Markdown source in one ready artifact job.
    """
    active_options = (
        options
        or TransformOptions()
    )

    source_reports: list[
        SourceTransformReport
    ] = []

    for source in job.files:
        try:
            report = transform_source(
                job=job,
                source=source,
                options=active_options,
            )

        except Exception as exc:
            report = write_source_failure(
                job=job,
                source=source,
                error=exc,
            )

            source_reports.append(
                report
            )

            if not active_options.continue_on_error:
                break

            continue

        source_reports.append(
            report
        )

    return write_job_report(
        job=job,
        source_reports=source_reports,
    )


#--------------------SOURCE TRANSFORMATION--------------------

def transform_source(
    *,
    job: TransformationJob,
    source: SourceArtifact,
    options: TransformOptions,
) -> SourceTransformReport:
    """
    Run the canonical parse -> chunk pipeline for one source.

    Token limits come exclusively from chunking/config.toml and
    the configured embedding model. The runner never overrides them.
    """
    _validate_supported_source(
        source
    )

    config = get_chunking_config()

    result = run_parse_chunk_pipeline(
        source.path,
        enable_overlap=options.enable_overlap,
        chunking_config=config,
    )

    return write_source_success(
        job=job,
        source=source,
        parse_result=result.parse_result,
        chunks=result.chunks,
        summary=result.summary(),
    )


#--------------------BATCH TRANSFORMATION--------------------

def transform_jobs(
    jobs: Iterable[
        TransformationJob
    ],
    *,
    options: TransformOptions | None = None,
) -> list[JobTransformReport]:
    active_options = (
        options
        or TransformOptions()
    )

    reports: list[
        JobTransformReport
    ] = []

    for job in jobs:
        reports.append(
            transform_job(
                job,
                options=active_options,
            )
        )

    return reports


#--------------------VALIDATION--------------------

def _validate_supported_source(
    source: SourceArtifact,
) -> None:
    suffix = source.suffix

    if suffix in MARKDOWN_SUFFIXES:
        return

    raise UnsupportedSourceTypeError(
        "Unsupported data-transformation source: "
        f"source_id={source.source_id} "
        f"suffix={suffix} "
        f"media_type={source.media_type}. "
        "Only Markdown and MDX are enabled."
    )
