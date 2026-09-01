from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class UploadedSource(BaseModel):
    source_id: str
    filename: str
    relative_path: str
    media_type: str
    size_bytes: int
    content_sha256: str


class DuplicateUpload(BaseModel):
    filename: str
    duplicate_of_source_id: str
    content_sha256: str
    size_bytes: int


class CreateIngestionJobResponse(BaseModel):
    tenant_id: str
    job_id: str
    status: Literal["queued"]
    upload_count: int
    source_count: int
    duplicate_count: int
    source_root: str
    sources: list[UploadedSource]
    duplicates: list[DuplicateUpload] = Field(default_factory=list)


class IngestionJobStatusResponse(BaseModel):
    tenant_id: str
    job_id: str
    status: Literal["not_found", "queued", "succeeded", "failed"]
    source_exists: bool
    output_exists: bool
    source_count: int = 0
    output_root: str = ""
    report: dict[str, Any] = Field(default_factory=dict)
