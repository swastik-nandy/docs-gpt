from __future__ import annotations

import hashlib
import json
import shutil
import time

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    chunked_job_ready,
    chunked_job_root,
    load_source_job,
    read_json,
    write_json_atomic,
)

from .chunking.config import (
    get_chunking_config,
)

from .chunking.parser_result_adapter import (
    chunk_parse_result,
)

from .parser.markdown import (
    parse_markdown,
)


# --------- RESULT ------------------------

@dataclass(frozen=True)
class TransformResult:
    job_id: str
    status: str
    source_count: int
    chunk_count: int
    total_tokens: int
    total_seconds: float
    output_root: Path


# --------- TRANSFORM JOB ------------------------

def transform_job(
    job_id: str,
    *,
    force: bool = False,
    enable_overlap: bool = False,
) -> TransformResult:
    started = time.perf_counter()

    output_root = chunked_job_root(
        job_id
    )

    if (
        not force
        and chunked_job_ready(
            job_id
        )
    ):
        manifest = read_json(
            output_root
            / "manifest.json"
        )

        return TransformResult(
            job_id=job_id,
            status="ready",
            source_count=int(
                manifest.get(
                    "source_count"
                )
                or 0
            ),
            chunk_count=int(
                manifest.get(
                    "chunk_count"
                )
                or 0
            ),
            total_tokens=int(
                manifest.get(
                    "total_tokens"
                )
                or 0
            ),
            total_seconds=(
                time.perf_counter()
                - started
            ),
            output_root=output_root,
        )

    job = load_source_job(
        job_id
    )

    # --------- RESET OUTPUT ------------------------

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks_path = (
        output_root
        / "chunks.jsonl"
    )

    chunks_temp_path = Path(
        f"{chunks_path}.tmp"
    )

    config = get_chunking_config()

    source_reports: list[
        dict[str, Any]
    ] = []

    total_chunks = 0
    total_tokens = 0

    try:
        with chunks_temp_path.open(
            "w",
            encoding="utf-8",
        ) as chunks_file:

            for source in job.files:
                source_started = (
                    time.perf_counter()
                )

                # --------- SOURCE HASH ------------------------

                content_sha256 = (
                    _sha256_file(
                        source.path
                    )
                )

                # --------- PARSE ------------------------

                parse_started = (
                    time.perf_counter()
                )

                parse_result = (
                    parse_markdown(
                        source.path
                    )
                )

                parse_seconds = (
                    time.perf_counter()
                    - parse_started
                )

                # --------- CHUNK ------------------------

                chunk_started = (
                    time.perf_counter()
                )

                chunks = (
                    chunk_parse_result(
                        parse_result,
                        source=source.source_id,
                        enable_overlap=enable_overlap,
                        config=config,
                    )
                )

                chunk_seconds = (
                    time.perf_counter()
                    - chunk_started
                )

                # --------- SERIALIZE FINAL CHUNKS ------------------------

                for chunk in chunks:
                    payload = (
                        chunk.to_dict()
                    )

                    artifact_metadata = {
                        "job_id": job.job_id,
                        "tenant_id": job.tenant_id,
                        "source_id": source.source_id,
                        "file_id": source.file_id,
                        "file_name": source.file_name,
                        "media_type": source.media_type,
                        "source_sha256": content_sha256,
                    }

                    metadata = dict(
                        payload.get(
                            "metadata"
                        )
                        or {}
                    )

                    metadata.update(
                        artifact_metadata
                    )

                    payload.update(
                        artifact_metadata
                    )

                    payload[
                        "metadata"
                    ] = metadata

                    chunks_file.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(
                                ",",
                                ":",
                            ),
                            default=str,
                        )
                    )

                    chunks_file.write(
                        "\n"
                    )

                source_chunk_count = len(
                    chunks
                )

                source_token_count = sum(
                    chunk.tokens
                    for chunk in chunks
                )

                total_chunks += (
                    source_chunk_count
                )

                total_tokens += (
                    source_token_count
                )

                source_reports.append(
                    {
                        "source_id": (
                            source.source_id
                        ),
                        "file_id": (
                            source.file_id
                        ),
                        "file_name": (
                            source.file_name
                        ),
                        "media_type": (
                            source.media_type
                        ),
                        "source_path": (
                            source.relative_path
                        ),
                        "source_sha256": (
                            content_sha256
                        ),
                        "size_bytes": (
                            source.path
                            .stat()
                            .st_size
                        ),
                        "doc_id": (
                            parse_result.doc_id
                        ),
                        "unit_count": (
                            parse_result.unit_count
                        ),
                        "section_count": (
                            parse_result.section_count
                        ),
                        "block_count": (
                            parse_result.block_count
                        ),
                        "chunk_count": (
                            source_chunk_count
                        ),
                        "total_tokens": (
                            source_token_count
                        ),
                        "timings": {
                            "parse_seconds": round(
                                parse_seconds,
                                6,
                            ),
                            "chunk_seconds": round(
                                chunk_seconds,
                                6,
                            ),
                            "total_seconds": round(
                                (
                                    time.perf_counter()
                                    - source_started
                                ),
                                6,
                            ),
                        },
                    }
                )

        # Atomic promotion only after all sources finish.
        chunks_temp_path.replace(
            chunks_path
        )

        total_seconds = (
            time.perf_counter()
            - started
        )

        manifest = {
            "schema_version": 1,
            "stage": "chunked",
            "status": "ready",
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "source_count": len(
                job.files
            ),
            "chunk_count": total_chunks,
            "total_tokens": total_tokens,
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "timings": {
                "total_seconds": round(
                    total_seconds,
                    6,
                ),
            },
            "sources": source_reports,
        }

        write_json_atomic(
            output_root
            / "manifest.json",
            manifest,
        )

        # _READY is the commit marker.
        # The embedder will only consume jobs
        # containing this marker.
        (
            output_root
            / "_READY"
        ).touch()

        return TransformResult(
            job_id=job.job_id,
            status="ready",
            source_count=len(
                job.files
            ),
            chunk_count=total_chunks,
            total_tokens=total_tokens,
            total_seconds=total_seconds,
            output_root=output_root,
        )

    except Exception as exc:
        chunks_temp_path.unlink(
            missing_ok=True
        )

        (
            output_root
            / "_READY"
        ).unlink(
            missing_ok=True
        )

        failure = {
            "schema_version": 1,
            "stage": "chunked",
            "status": "failed",
            "job_id": job.job_id,
            "tenant_id": job.tenant_id,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "sources": source_reports,
        }

        write_json_atomic(
            output_root
            / "manifest.json",
            failure,
        )

        (
            output_root
            / "_FAILED"
        ).touch()

        raise


# --------- HASHING ------------------------

def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()
