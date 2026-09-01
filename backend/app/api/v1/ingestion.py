from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.settings import Settings, get_settings
from app.core.tenancy import TenantResolutionError, resolve_tenant_context
from app.schemas.ingestion import (
    CreateIngestionJobResponse,
    DuplicateUpload,
    IngestionJobStatusResponse,
    UploadedSource,
)
from app.services.artifact_writer import ArtifactJobError, ArtifactJobWriter


router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def get_artifact_writer(
    settings: Settings = Depends(get_settings),
) -> ArtifactJobWriter:
    return ArtifactJobWriter(settings)


@router.post(
    "/jobs",
    response_model=CreateIngestionJobResponse,
    status_code=202,
)
async def create_ingestion_job(
    files: list[UploadFile] = File(
        ...,
        description="One or more Markdown/MDX files.",
    ),
    settings: Settings = Depends(get_settings),
    writer: ArtifactJobWriter = Depends(get_artifact_writer),
) -> CreateIngestionJobResponse:
    """
    Accept Markdown/MDX files as multipart/form-data and create a source artifact job.

    Tenant identity is resolved internally.
    No human names, emails, or client-provided tenant IDs are used in artifact paths.
    """

    try:
        tenant = resolve_tenant_context(
            requested_tenant_id=None,
            settings=settings,
        )

        job = await writer.create_job(
            tenant_id=tenant.tenant_id,
            files=files,
        )

    except TenantResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ArtifactJobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return CreateIngestionJobResponse(
        tenant_id=job.tenant_id,
        job_id=job.job_id,
        status="queued",
        upload_count=job.upload_count,
        source_count=len(job.sources),
        duplicate_count=len(job.duplicates),
        source_root=str(job.job_root),
        sources=[
            UploadedSource(
                source_id=source.source_id,
                filename=source.filename,
                relative_path=source.relative_path,
                media_type=source.media_type,
                size_bytes=source.size_bytes,
                content_sha256=source.content_sha256,
            )
            for source in job.sources
        ],
        duplicates=[
            DuplicateUpload(
                filename=duplicate.filename,
                duplicate_of_source_id=duplicate.duplicate_of_source_id,
                content_sha256=duplicate.content_sha256,
                size_bytes=duplicate.size_bytes,
            )
            for duplicate in job.duplicates
        ],
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobStatusResponse,
)
def get_ingestion_job_status(
    job_id: str,
    settings: Settings = Depends(get_settings),
    writer: ArtifactJobWriter = Depends(get_artifact_writer),
) -> IngestionJobStatusResponse:
    """
    Read job status from artifact markers.

    Without auth, the local default tenant from TOML is used.
    Later this should resolve tenant from authenticated user/session.
    """

    try:
        tenant = resolve_tenant_context(
            requested_tenant_id=None,
            settings=settings,
        )

        payload = writer.read_status(
            tenant_id=tenant.tenant_id,
            job_id=job_id,
        )

    except TenantResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ArtifactJobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return IngestionJobStatusResponse(**payload)
