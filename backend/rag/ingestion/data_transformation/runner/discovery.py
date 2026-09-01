from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.runner.schema import SourceArtifact
from rag.ingestion.data_transformation.runner.schema import TransformationJob


#--------------------PATHS--------------------

DEFAULT_SOURCE_ROOT = Path(
    "rag/ingestion/artifacts/source"
)

DEFAULT_OUTPUT_ROOT = Path(
    "rag/ingestion/artifacts/chunked"
)


#--------------------MARKERS--------------------

READY_MARKER = "_READY"
DONE_MARKER = "_DONE"
FAILED_MARKER = "_FAILED"


#--------------------SOURCE TYPES--------------------

SUPPORTED_SUFFIXES = {
    ".md": "text/markdown",
    ".mdx": "text/mdx",
    ".markdown": "text/markdown",
    ".mdown": "text/markdown",
    ".mkd": "text/markdown",
    ".mdwn": "text/markdown",
}


#--------------------PUBLIC DISCOVERY--------------------

def discover_ready_jobs(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> list[TransformationJob]:
    """
    Discover artifact jobs explicitly marked ready.

    Expected source layout:

        artifacts/source/
          tenant_id/
            job_id/
              _READY
              manifest.json
              files/
                **/*.md
                **/*.mdx

    Only completed jobs carrying _READY are transformed.
    """
    source_root_path = (
        Path(
            source_root
        )
        .expanduser()
        .resolve()
    )

    output_root_path = (
        Path(
            output_root
        )
        .expanduser()
        .resolve()
    )

    if not source_root_path.exists():
        return []

    if not source_root_path.is_dir():
        raise NotADirectoryError(
            f"Source root is not a directory: "
            f"{source_root_path}"
        )

    jobs: list[TransformationJob] = []

    markers = sorted(
        source_root_path.glob(
            f"*/*/{READY_MARKER}"
        ),
        key=lambda path: str(
            path
        ).lower(),
    )

    for ready_marker in markers:
        if not ready_marker.is_file():
            continue

        job_root = ready_marker.parent

        if _is_already_done(
            job_root=job_root,
            source_root=source_root_path,
            output_root=output_root_path,
        ):
            continue

        job = _build_job_from_root(
            job_root=job_root,
            source_root=source_root_path,
            output_root=output_root_path,
        )

        if job.files:
            jobs.append(
                job
            )

    return jobs


#--------------------JOB STATE--------------------

def _is_already_done(
    *,
    job_root: Path,
    source_root: Path,
    output_root: Path,
) -> bool:
    relative = job_root.relative_to(
        source_root
    )

    output_job_root = (
        output_root
        / relative
    )

    return (
        output_job_root
        / DONE_MARKER
    ).is_file()


#--------------------JOB BUILDING--------------------

def _build_job_from_root(
    *,
    job_root: Path,
    source_root: Path,
    output_root: Path,
) -> TransformationJob:
    relative_parts = (
        job_root
        .relative_to(
            source_root
        )
        .parts
    )

    if len(
        relative_parts
    ) != 2:
        raise ValueError(
            f"Invalid artifact job root: "
            f"{job_root}"
        )

    tenant_id = relative_parts[
        0
    ]

    job_id = relative_parts[
        1
    ]

    manifest = _load_manifest(
        job_root
        / "manifest.json"
    )

    files = tuple(
        _discover_files(
            job_root=job_root,
            tenant_id=tenant_id,
            job_id=job_id,
            manifest=manifest,
        )
    )

    return TransformationJob(
        tenant_id=tenant_id,
        job_id=job_id,
        source_root=source_root,
        output_root=output_root,
        files=files,
        manifest=manifest,
    )


#--------------------MANIFEST--------------------

def _load_manifest(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    if not path.is_file():
        raise ValueError(
            f"Manifest path is not a file: "
            f"{path}"
        )

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid manifest JSON: "
            f"{path}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Manifest must contain a JSON object: "
            f"{path}"
        )

    return value


#--------------------FILE DISCOVERY--------------------

def _discover_files(
    *,
    job_root: Path,
    tenant_id: str,
    job_id: str,
    manifest: dict[str, Any],
) -> list[SourceArtifact]:
    manifest_files = manifest.get(
        "files"
    )

    if (
        isinstance(
            manifest_files,
            list,
        )
        and manifest_files
    ):
        return _discover_manifest_files(
            job_root=job_root,
            tenant_id=tenant_id,
            job_id=job_id,
            manifest_files=manifest_files,
        )

    return _discover_directory_files(
        job_root=job_root,
        tenant_id=tenant_id,
        job_id=job_id,
    )


#--------------------MANIFEST FILES--------------------

def _discover_manifest_files(
    *,
    job_root: Path,
    tenant_id: str,
    job_id: str,
    manifest_files: list[Any],
) -> list[SourceArtifact]:
    files: list[
        SourceArtifact
    ] = []

    seen_source_ids: set[
        str
    ] = set()

    for index, item in enumerate(
        manifest_files
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        relative_path = str(
            item.get(
                "path"
            )
            or ""
        ).strip()

        if not relative_path:
            continue

        path = _resolve_inside_job(
            job_root=job_root,
            relative_path=relative_path,
        )

        if (
            not path.exists()
            or not path.is_file()
        ):
            continue

        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            continue

        requested_source_id = str(
            item.get(
                "source_id"
            )
            or ""
        ).strip()

        if requested_source_id:
            source_id = _safe_source_id(
                requested_source_id
            )
        else:
            source_id = _source_id_from_relative_path(
                relative_path,
                fallback=f"source_{index}",
            )

        source_id = _unique_source_id(
            source_id=source_id,
            relative_path=relative_path,
            seen_source_ids=seen_source_ids,
        )

        media_type = str(
            item.get(
                "media_type"
            )
            or SUPPORTED_SUFFIXES[
                suffix
            ]
        ).strip()

        files.append(
            SourceArtifact(
                tenant_id=tenant_id,
                job_id=job_id,
                source_id=source_id,
                path=path,
                relative_path=relative_path,
                media_type=media_type,
            )
        )

    return files


#--------------------DIRECTORY FILES--------------------

def _discover_directory_files(
    *,
    job_root: Path,
    tenant_id: str,
    job_id: str,
) -> list[SourceArtifact]:
    files_root = (
        job_root
        / "files"
    )

    if not files_root.exists():
        files_root = job_root

    if not files_root.is_dir():
        return []

    candidates = sorted(
        files_root.rglob(
            "*"
        ),
        key=lambda path: str(
            path
        ).lower(),
    )

    files: list[
        SourceArtifact
    ] = []

    seen_source_ids: set[
        str
    ] = set()

    for path in candidates:
        if not path.is_file():
            continue

        if path.name in {
            READY_MARKER,
            DONE_MARKER,
            FAILED_MARKER,
            "manifest.json",
        }:
            continue

        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            continue

        relative_to_job = path.relative_to(
            job_root
        )

        relative_to_files = path.relative_to(
            files_root
        )

        relative_path = str(
            relative_to_job
        )

        source_id = _source_id_from_relative_path(
            str(
                relative_to_files
            ),
            fallback=path.stem,
        )

        source_id = _unique_source_id(
            source_id=source_id,
            relative_path=relative_path,
            seen_source_ids=seen_source_ids,
        )

        files.append(
            SourceArtifact(
                tenant_id=tenant_id,
                job_id=job_id,
                source_id=source_id,
                path=path.resolve(),
                relative_path=relative_path,
                media_type=SUPPORTED_SUFFIXES[
                    suffix
                ],
            )
        )

    return files


#--------------------PATH SAFETY--------------------

def _resolve_inside_job(
    *,
    job_root: Path,
    relative_path: str,
) -> Path:
    root = job_root.resolve()

    path = (
        root
        / relative_path
    ).resolve()

    try:
        path.relative_to(
            root
        )
    except ValueError as exc:
        raise ValueError(
            "Manifest source escapes artifact job root: "
            f"{relative_path}"
        ) from exc

    return path


#--------------------SOURCE ID--------------------

def _source_id_from_relative_path(
    relative_path: str,
    *,
    fallback: str,
) -> str:
    path = Path(
        relative_path
    )

    without_suffix = path.with_suffix(
        ""
    )

    parts = list(
        without_suffix.parts
    )

    if (
        parts
        and parts[
            0
        ].lower()
        == "files"
    ):
        parts = parts[
            1:
        ]

    candidate = "__".join(
        parts
    )

    return _safe_source_id(
        candidate
        or fallback
    )


def _safe_source_id(
    value: str,
) -> str:
    cleaned = "".join(
        char
        if (
            char.isalnum()
            or char
            in {
                "-",
                "_",
            }
        )
        else "_"
        for char in str(
            value
            or ""
        ).strip()
    )

    while "__" in cleaned:
        cleaned = cleaned.replace(
            "__",
            "_",
        )

    cleaned = cleaned.strip(
        "_"
    )

    return (
        cleaned
        or "source"
    )


def _unique_source_id(
    *,
    source_id: str,
    relative_path: str,
    seen_source_ids: set[str],
) -> str:
    if source_id not in seen_source_ids:
        seen_source_ids.add(
            source_id
        )

        return source_id

    digest = hashlib.sha256(
        relative_path.encode(
            "utf-8"
        )
    ).hexdigest()[
        :8
    ]

    candidate = (
        f"{source_id}_"
        f"{digest}"
    )

    if candidate in seen_source_ids:
        raise ValueError(
            "Unable to create unique source_id for "
            f"{relative_path}"
        )

    seen_source_ids.add(
        candidate
    )

    return candidate
