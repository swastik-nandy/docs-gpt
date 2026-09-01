from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.runner.discovery import DONE_MARKER
from rag.ingestion.data_transformation.runner.discovery import FAILED_MARKER
from rag.ingestion.data_transformation.runner.schema import JobTransformReport
from rag.ingestion.data_transformation.runner.schema import SourceArtifact
from rag.ingestion.data_transformation.runner.schema import SourceTransformReport
from rag.ingestion.data_transformation.runner.schema import TransformationJob


#--------------------OUTPUT LAYOUT--------------------

def prepare_output_job_root(
    job: TransformationJob,
) -> Path:
    root = job.output_job_root

    (
        root
        / "parser"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        root
        / "chunks"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        root
        / "reports"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return root


#--------------------SOURCE SUCCESS--------------------

def write_source_success(
    *,
    job: TransformationJob,
    source: SourceArtifact,
    parse_result: Any,
    chunks: list[Chunk],
    summary: dict[str, Any],
) -> SourceTransformReport:
    root = prepare_output_job_root(
        job
    )

    parser_path = (
        root
        / "parser"
        / f"{source.source_id}.ir.json"
    )

    chunks_path = (
        root
        / "chunks"
        / f"{source.source_id}.chunks.jsonl"
    )

    report_path = (
        root
        / "reports"
        / (
            f"{source.source_id}"
            ".transform_report.json"
        )
    )

    _write_json(
        parser_path,
        _serialize_parse_result(
            parse_result
        ),
    )

    _write_chunks_jsonl(
        chunks_path,
        chunks,
    )

    report = SourceTransformReport(
        source_id=source.source_id,
        source_path=str(
            source.path
        ),
        status="succeeded",
        doc_id=str(
            summary.get(
                "doc_id",
                "",
            )
            or ""
        ),
        parser_family=str(
            summary.get(
                "parser_family",
                "",
            )
            or ""
        ),
        parser_variant=str(
            summary.get(
                "parser_variant",
                "",
            )
            or ""
        ),
        parse_strategy=str(
            summary.get(
                "parse_strategy",
                "",
            )
            or ""
        ),
        unit_count=int(
            summary.get(
                "unit_count",
                0,
            )
            or 0
        ),
        chunk_count=int(
            summary.get(
                "chunk_count",
                0,
            )
            or 0
        ),
        total_chunk_tokens=int(
            summary.get(
                "total_chunk_tokens",
                0,
            )
            or 0
        ),
        timings=_safe_dict(
            summary.get(
                "timings"
            )
        ),
    )

    _write_json(
        report_path,
        report.to_dict(),
    )

    return report


#--------------------SOURCE FAILURE--------------------

def write_source_failure(
    *,
    job: TransformationJob,
    source: SourceArtifact,
    error: BaseException,
) -> SourceTransformReport:
    root = prepare_output_job_root(
        job
    )

    report = SourceTransformReport(
        source_id=source.source_id,
        source_path=str(
            source.path
        ),
        status="failed",
        error=(
            f"{type(error).__name__}: "
            f"{error}"
        ),
    )

    report_path = (
        root
        / "reports"
        / (
            f"{source.source_id}"
            ".transform_report.json"
        )
    )

    _write_json(
        report_path,
        report.to_dict(),
    )

    return report


#--------------------JOB REPORT--------------------

def write_job_report(
    *,
    job: TransformationJob,
    source_reports: Iterable[
        SourceTransformReport
    ],
) -> JobTransformReport:
    root = prepare_output_job_root(
        job
    )

    reports = tuple(
        source_reports
    )

    succeeded = (
        bool(
            reports
        )
        and all(
            report.status
            == "succeeded"
            for report in reports
        )
    )

    status = (
        "succeeded"
        if succeeded
        else "failed"
    )

    job_report = JobTransformReport(
        tenant_id=job.tenant_id,
        job_id=job.job_id,
        status=status,
        source_reports=reports,
    )

    _write_json(
        root
        / "manifest.json",
        _build_output_manifest(
            job,
            job_report,
        ),
    )

    _write_json(
        root
        / "reports"
        / "job_report.json",
        job_report.to_dict(),
    )

    if succeeded:
        _remove_marker(
            root
            / FAILED_MARKER
        )

        _touch_marker(
            root
            / DONE_MARKER
        )

    else:
        _remove_marker(
            root
            / DONE_MARKER
        )

        _touch_marker(
            root
            / FAILED_MARKER
        )

    return job_report


#--------------------OUTPUT MANIFEST--------------------

def _build_output_manifest(
    job: TransformationJob,
    report: JobTransformReport,
) -> dict[str, Any]:
    return {
        "tenant_id": job.tenant_id,
        "job_id": job.job_id,
        "status": report.status,
        "source_count": len(
            job.files
        ),
        "input_manifest": job.manifest,
        "outputs": {
            "parser_dir": "parser",
            "chunks_dir": "chunks",
            "reports_dir": "reports",
        },
        "sources": [
            {
                "source_id": source.source_id,
                "relative_path": source.relative_path,
                "media_type": source.media_type,
                "chunks_path": (
                    "chunks/"
                    f"{source.source_id}"
                    ".chunks.jsonl"
                ),
                "parser_path": (
                    "parser/"
                    f"{source.source_id}"
                    ".ir.json"
                ),
            }
            for source in job.files
        ],
    }


#--------------------PARSE SERIALIZATION--------------------

def _serialize_parse_result(
    parse_result: Any,
) -> dict[str, Any]:
    to_dict = getattr(
        parse_result,
        "to_dict",
        None,
    )

    if callable(
        to_dict
    ):
        value = to_dict()

        if isinstance(
            value,
            dict,
        ):
            return value

    if is_dataclass(
        parse_result
    ):
        value = asdict(
            parse_result
        )

        if isinstance(
            value,
            dict,
        ):
            return value

    return {
        "doc_id": str(
            getattr(
                parse_result,
                "doc_id",
                "",
            )
            or ""
        ),
        "parser_family": str(
            getattr(
                parse_result,
                "parser_family",
                "",
            )
            or ""
        ),
        "parser_variant": str(
            getattr(
                parse_result,
                "parser_variant",
                "",
            )
            or ""
        ),
        "parse_strategy": str(
            getattr(
                parse_result,
                "parse_strategy",
                "",
            )
            or ""
        ),
        "unit_count": int(
            getattr(
                parse_result,
                "unit_count",
                0,
            )
            or 0
        ),
        "warnings": list(
            getattr(
                parse_result,
                "warnings",
                [],
            )
            or []
        ),
        "metrics": _safe_dict(
            getattr(
                parse_result,
                "metrics",
                {},
            )
        ),
        "units": [
            _serialize_unit(
                unit
            )
            for unit in list(
                getattr(
                    parse_result,
                    "units",
                    [],
                )
                or []
            )
        ],
    }


def _serialize_unit(
    unit: Any,
) -> dict[str, Any]:
    to_dict = getattr(
        unit,
        "to_dict",
        None,
    )

    if callable(
        to_dict
    ):
        value = to_dict()

        if isinstance(
            value,
            dict,
        ):
            return value

    if is_dataclass(
        unit
    ):
        value = asdict(
            unit
        )

        if isinstance(
            value,
            dict,
        ):
            return value

    return {
        "unit_id": str(
            getattr(
                unit,
                "unit_id",
                "",
            )
            or ""
        ),
        "doc_id": str(
            getattr(
                unit,
                "doc_id",
                "",
            )
            or ""
        ),
        "source_doc_id": str(
            getattr(
                unit,
                "source_doc_id",
                "",
            )
            or ""
        ),
        "parse_strategy": str(
            getattr(
                unit,
                "parse_strategy",
                "",
            )
            or ""
        ),
        "sections": list(
            getattr(
                unit,
                "sections",
                [],
            )
            or []
        ),
        "blocks": list(
            getattr(
                unit,
                "blocks",
                [],
            )
            or []
        ),
        "meta": _safe_dict(
            getattr(
                unit,
                "meta",
                {},
            )
        ),
        "warnings": list(
            getattr(
                unit,
                "warnings",
                [],
            )
            or []
        ),
    }


#--------------------ATOMIC WRITES--------------------

def _write_chunks_jsonl(
    path: Path,
    chunks: list[Chunk],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for chunk in chunks:
                payload = chunk.to_dict()

                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                    )
                )

                handle.write(
                    "\n"
                )

            handle.flush()

        temp_path.replace(
            path
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True
        )

        raise


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    try:
        temp_path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        temp_path.replace(
            path
        )

    except Exception:
        temp_path.unlink(
            missing_ok=True
        )

        raise


#--------------------MARKERS--------------------

def _touch_marker(
    path: Path,
) -> None:
    path.write_text(
        "",
        encoding="utf-8",
    )


def _remove_marker(
    path: Path,
) -> None:
    path.unlink(
        missing_ok=True
    )


#--------------------HELPERS--------------------

def _safe_dict(
    value: Any,
) -> dict[str, Any]:
    if isinstance(
        value,
        dict,
    ):
        return value

    return {}
