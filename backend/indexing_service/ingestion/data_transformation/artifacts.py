from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------- PATHS ------------------------

BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ARTIFACTS_ROOT = (
    BACKEND_ROOT
    / "artifacts"
)

SOURCE_ROOT = (
    ARTIFACTS_ROOT
    / "source"
)

CHUNKED_ROOT = (
    ARTIFACTS_ROOT
    / "chunked"
)

INDEXED_ROOT = (
    ARTIFACTS_ROOT
    / "indexed"
)


# --------- MODELS ------------------------

@dataclass(frozen=True)
class SourceFile:
    source_id: str
    file_id: str
    file_name: str
    media_type: str
    relative_path: str
    path: Path


@dataclass(frozen=True)
class SourceJob:
    job_id: str
    tenant_id: str
    root: Path
    files: tuple[SourceFile, ...]


# --------- DIRECTORY SETUP ------------------------

def ensure_artifact_roots() -> None:
    for root in (
        SOURCE_ROOT,
        CHUNKED_ROOT,
        INDEXED_ROOT,
    ):
        root.mkdir(
            parents=True,
            exist_ok=True,
        )


# --------- JOB PATHS ------------------------

def source_job_root(
    job_id: str,
) -> Path:
    return (
        SOURCE_ROOT
        / _safe_job_id(job_id)
    )


def chunked_job_root(
    job_id: str,
) -> Path:
    return (
        CHUNKED_ROOT
        / _safe_job_id(job_id)
    )


# --------- SOURCE LOADING ------------------------

def load_source_job(
    job_id: str,
) -> SourceJob:
    ensure_artifact_roots()

    root = (
        source_job_root(job_id)
        .resolve()
    )

    manifest_path = (
        root
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Source manifest not found: "
            f"{manifest_path}"
        )

    manifest = read_json(
        manifest_path
    )

    manifest_job_id = _required_text(
        manifest,
        "job_id",
    )

    if manifest_job_id != job_id:
        raise ValueError(
            "Manifest job_id does not match "
            f"directory: {manifest_job_id!r}"
        )

    if (
        _required_text(
            manifest,
            "stage",
        )
        != "source"
    ):
        raise ValueError(
            "Expected source-stage manifest"
        )

    if (
        _required_text(
            manifest,
            "status",
        )
        != "ready"
    ):
        raise ValueError(
            f"Source job {job_id!r} is not ready"
        )

    tenant_id = str(
        manifest.get(
            "tenant_id"
        )
        or ""
    ).strip()

    raw_files = manifest.get(
        "files"
    )

    if (
        not isinstance(
            raw_files,
            list,
        )
        or not raw_files
    ):
        raise ValueError(
            "Source manifest must contain files"
        )

    files: list[SourceFile] = []

    for raw in raw_files:
        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError(
                "Each source file must be an object"
            )

        source_id = _required_text(
            raw,
            "source_id",
        )

        file_name = _required_text(
            raw,
            "file_name",
        )

        relative_path = _required_text(
            raw,
            "path",
        )

        file_id = str(
            raw.get(
                "file_id"
            )
            or source_id
        ).strip()

        media_type = str(
            raw.get(
                "media_type"
            )
            or "text/markdown"
        ).strip()

        path = (
            root
            / relative_path
        ).resolve()

        if (
            path != root
            and root
            not in path.parents
        ):
            raise ValueError(
                "Source path escapes job directory"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Source file not found: {path}"
            )

        if path.suffix.lower() not in {
            ".md",
            ".markdown",
        }:
            raise ValueError(
                "Only Markdown sources are "
                f"supported right now: {path.name}"
            )

        files.append(
            SourceFile(
                source_id=source_id,
                file_id=file_id,
                file_name=file_name,
                media_type=media_type,
                relative_path=relative_path,
                path=path,
            )
        )

    return SourceJob(
        job_id=job_id,
        tenant_id=tenant_id,
        root=root,
        files=tuple(
            files
        ),
    )


# --------- JSON IO ------------------------

def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = Path(
        f"{path}.tmp"
    )

    temp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(
        path
    )


# --------- STATUS ------------------------

def chunked_job_ready(
    job_id: str,
) -> bool:
    root = chunked_job_root(
        job_id
    )

    return (
        (
            root
            / "_READY"
        ).is_file()
        and (
            root
            / "manifest.json"
        ).is_file()
        and (
            root
            / "chunks.jsonl"
        ).is_file()
    )


# --------- VALIDATION ------------------------

def _safe_job_id(
    job_id: str,
) -> str:
    value = str(
        job_id
        or ""
    ).strip()

    if not value:
        raise ValueError(
            "job_id cannot be empty"
        )

    if (
        value in {
            ".",
            "..",
        }
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            "job_id must be one path-safe segment"
        )

    return value


def _required_text(
    value: dict[str, Any],
    key: str,
) -> str:
    result = str(
        value.get(
            key
        )
        or ""
    ).strip()

    if not result:
        raise ValueError(
            f"Missing required field: {key}"
        )

    return result
