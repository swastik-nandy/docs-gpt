from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.settings import Settings


class ArtifactJobError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class StoredUpload:
    source_id: str
    filename: str
    relative_path: str
    media_type: str
    size_bytes: int
    content_sha256: str


@dataclass(frozen=True)
class DuplicateUpload:
    filename: str
    duplicate_of_source_id: str
    content_sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CreatedArtifactJob:
    tenant_id: str
    job_id: str
    job_root: Path
    upload_count: int
    sources: tuple[StoredUpload, ...]
    duplicates: tuple[DuplicateUpload, ...]


@dataclass(frozen=True)
class _WrittenUpload:
    size_bytes: int
    content_sha256: str


class ArtifactJobWriter:
    """
    Writes multipart uploads into the source artifact contract.

    Shape:
        artifacts/source/{tenant_id}/{job_id}/
          manifest.json
          _READY
          files/*.md|*.mdx

    Dedupe:
        - content SHA-256 is computed while streaming upload bytes
        - duplicate content within the same job is skipped
        - manifest records duplicates separately
        - _READY is written last
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.source_root = settings.artifact_source_root
        self.chunked_root = settings.artifact_chunked_root
        self.allowed_suffixes = settings.uploads.allowed_suffixes

    async def create_job(
        self,
        *,
        tenant_id: str,
        files: list[UploadFile],
    ) -> CreatedArtifactJob:
        safe_tenant_id = _safe_path_segment(
            tenant_id,
            fallback=self.settings.default_tenant_id,
        )
        prepared_files = self._validate_uploads(files)

        job_id = uuid.uuid4().hex
        job_root = self.source_root / safe_tenant_id / job_id
        files_root = job_root / "files"

        stored: list[StoredUpload] = []
        duplicates: list[DuplicateUpload] = []

        total_bytes = 0
        hash_to_source: dict[str, StoredUpload] = {}

        try:
            files_root.mkdir(parents=True, exist_ok=False)

            used_filenames: set[str] = set()
            used_source_ids: set[str] = set()

            for index, upload in enumerate(prepared_files):
                safe_filename = _unique_name(
                    _safe_filename(
                        upload.filename or f"source_{index}.md",
                        allowed_suffixes=self.allowed_suffixes,
                    ),
                    used_filenames,
                )

                suffix = Path(safe_filename).suffix.lower()
                media_type = self.allowed_suffixes[suffix]

                source_id = _unique_name(
                    _safe_source_id(Path(safe_filename).stem or f"source_{index}"),
                    used_source_ids,
                )

                relative_path = f"files/{safe_filename}"
                destination = files_root / safe_filename

                written = await self._write_upload_to_disk(
                    upload=upload,
                    destination=destination,
                )

                if written.size_bytes <= 0:
                    raise ArtifactJobError(
                        status_code=400,
                        detail=f"Uploaded file is empty: {upload.filename}",
                    )

                total_bytes += written.size_bytes

                if total_bytes > self.settings.max_upload_total_bytes:
                    raise ArtifactJobError(
                        status_code=413,
                        detail=(
                            "Upload batch is too large. "
                            f"Max total bytes: {self.settings.max_upload_total_bytes}"
                        ),
                    )

                existing = hash_to_source.get(written.content_sha256)

                if self.settings.dedupe.enabled and existing:
                    destination.unlink(missing_ok=True)

                    duplicates.append(
                        DuplicateUpload(
                            filename=safe_filename,
                            duplicate_of_source_id=existing.source_id,
                            content_sha256=written.content_sha256,
                            size_bytes=written.size_bytes,
                        )
                    )
                    continue

                source = StoredUpload(
                    source_id=source_id,
                    filename=safe_filename,
                    relative_path=relative_path,
                    media_type=media_type,
                    size_bytes=written.size_bytes,
                    content_sha256=written.content_sha256,
                )

                stored.append(source)
                hash_to_source[written.content_sha256] = source

            if not stored:
                raise ArtifactJobError(
                    status_code=409,
                    detail="All uploaded files were duplicates. No new source files were created.",
                )

            self._write_manifest(
                job_root=job_root,
                tenant_id=safe_tenant_id,
                job_id=job_id,
                upload_count=len(prepared_files),
                sources=stored,
                duplicates=duplicates,
            )

            self._touch_ready(job_root)

        except Exception:
            shutil.rmtree(job_root, ignore_errors=True)
            raise

        return CreatedArtifactJob(
            tenant_id=safe_tenant_id,
            job_id=job_id,
            job_root=job_root,
            upload_count=len(prepared_files),
            sources=tuple(stored),
            duplicates=tuple(duplicates),
        )

    def read_status(
        self,
        *,
        tenant_id: str,
        job_id: str,
    ) -> dict[str, Any]:
        safe_tenant_id = _safe_path_segment(
            tenant_id,
            fallback=self.settings.default_tenant_id,
        )
        safe_job_id = _safe_path_segment(job_id, fallback="")

        if not safe_job_id:
            raise ArtifactJobError(status_code=400, detail="Invalid job_id")

        source_job_root = self.source_root / safe_tenant_id / safe_job_id
        output_job_root = self.chunked_root / safe_tenant_id / safe_job_id

        source_exists = source_job_root.exists()
        output_exists = output_job_root.exists()

        if (output_job_root / self.settings.markers.done).exists():
            status = "succeeded"
        elif (output_job_root / self.settings.markers.failed).exists():
            status = "failed"
        elif (source_job_root / self.settings.markers.ready).exists():
            status = "queued"
        else:
            status = "not_found"

        report_path = output_job_root / "reports" / "job_report.json"
        report = _read_json(report_path)

        return {
            "tenant_id": safe_tenant_id,
            "job_id": safe_job_id,
            "status": status,
            "source_exists": source_exists,
            "output_exists": output_exists,
            "source_count": _source_count(
                source_job_root,
                allowed_suffixes=self.allowed_suffixes,
            ),
            "output_root": str(output_job_root) if output_exists else "",
            "report": report,
        }

    def _validate_uploads(self, files: list[UploadFile]) -> list[UploadFile]:
        if not files:
            raise ArtifactJobError(
                status_code=400,
                detail="At least one Markdown or MDX file is required.",
            )

        if len(files) > self.settings.max_upload_files:
            raise ArtifactJobError(
                status_code=413,
                detail=f"Too many files. Max allowed: {self.settings.max_upload_files}",
            )

        for upload in files:
            filename = upload.filename or ""
            suffix = Path(filename).suffix.lower()

            if suffix not in self.allowed_suffixes:
                raise ArtifactJobError(
                    status_code=415,
                    detail=(
                        f"Unsupported file type: {filename!r}. "
                        f"Allowed suffixes: {sorted(self.allowed_suffixes)}"
                    ),
                )

        return files

    async def _write_upload_to_disk(
        self,
        *,
        upload: UploadFile,
        destination: Path,
    ) -> _WrittenUpload:
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        bytes_written = 0
        sha256 = hashlib.sha256()

        try:
            with temp_path.open("wb") as handle:
                while True:
                    chunk = await upload.read(self.settings.upload_stream_chunk_bytes)

                    if not chunk:
                        break

                    bytes_written += len(chunk)

                    if bytes_written > self.settings.max_upload_file_bytes:
                        raise ArtifactJobError(
                            status_code=413,
                            detail=(
                                f"File too large: {upload.filename}. "
                                f"Max bytes: {self.settings.max_upload_file_bytes}"
                            ),
                        )

                    sha256.update(chunk)
                    handle.write(chunk)

                handle.flush()

            temp_path.replace(destination)

            return _WrittenUpload(
                size_bytes=bytes_written,
                content_sha256=sha256.hexdigest(),
            )

        finally:
            await upload.close()

            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _write_manifest(
        self,
        *,
        job_root: Path,
        tenant_id: str,
        job_id: str,
        upload_count: int,
        sources: list[StoredUpload],
        duplicates: list[DuplicateUpload],
    ) -> None:
        manifest = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "upload_count": upload_count,
            "source_count": len(sources),
            "duplicate_count": len(duplicates),
            "dedupe": {
                "enabled": self.settings.dedupe.enabled,
                "algorithm": self.settings.dedupe.algorithm,
                "scope": self.settings.dedupe.scope,
            },
            "files": [
                {
                    "source_id": source.source_id,
                    "path": source.relative_path,
                    "media_type": source.media_type,
                    "filename": source.filename,
                    "size_bytes": source.size_bytes,
                    "content_sha256": source.content_sha256,
                }
                for source in sources
            ],
            "duplicates": [
                {
                    "filename": duplicate.filename,
                    "duplicate_of_source_id": duplicate.duplicate_of_source_id,
                    "content_sha256": duplicate.content_sha256,
                    "size_bytes": duplicate.size_bytes,
                }
                for duplicate in duplicates
            ],
        }

        _write_json_atomic(job_root / "manifest.json", manifest)

    def _touch_ready(self, job_root: Path) -> None:
        ready_path = job_root / self.settings.markers.ready
        temp_path = job_root / f".{self.settings.markers.ready}.tmp"
        temp_path.write_text("", encoding="utf-8")
        temp_path.replace(ready_path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return value if isinstance(value, dict) else {}


def _source_count(
    source_job_root: Path,
    *,
    allowed_suffixes: dict[str, str],
) -> int:
    manifest = _read_json(source_job_root / "manifest.json")
    files = manifest.get("files")

    if isinstance(files, list):
        return len(files)

    files_root = source_job_root / "files"

    if not files_root.exists():
        return 0

    return sum(
        1
        for path in files_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )


def _safe_path_segment(value: str, *, fallback: str) -> str:
    cleaned = _safe_source_id(value)
    return cleaned or fallback


def _safe_filename(
    filename: str,
    *,
    allowed_suffixes: dict[str, str],
) -> str:
    name = Path(filename or "").name.strip()

    if not name:
        name = "source.md"

    suffix = Path(name).suffix.lower()
    stem = Path(name).stem

    if suffix not in allowed_suffixes:
        suffix = ".md"

    safe_stem = _safe_source_id(stem)[:100] or "source"

    return f"{safe_stem}{suffix}"


def _safe_source_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned


def _unique_name(value: str, used: set[str]) -> str:
    if value not in used:
        used.add(value)
        return value

    path = Path(value)
    stem = path.stem
    suffix = path.suffix

    index = 2

    while True:
        candidate = f"{stem}_{index}{suffix}" if suffix else f"{value}_{index}"

        if candidate not in used:
            used.add(candidate)
            return candidate

        index += 1
