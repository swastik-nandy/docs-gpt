from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.config import EmbeddingPipelineConfig


@dataclass(frozen=True)
class ChunkArtifactFile:
    tenant_id: str
    job_id: str
    source_id: str
    path: Path
    report_path: Path | None = None

    @property
    def file_name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class ChunkArtifactJob:
    tenant_id: str
    job_id: str
    root: Path
    chunks_dir: Path
    reports_dir: Path
    manifest_path: Path
    files: tuple[ChunkArtifactFile, ...] = field(default_factory=tuple)

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class LoadedChunkArtifact:
    artifact_file: ChunkArtifactFile
    chunks: tuple[Chunk, ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def discover_chunk_artifact_jobs(
    config: EmbeddingPipelineConfig,
) -> list[ChunkArtifactJob]:
    """
    Discover transformed chunk artifact jobs ready for vector indexing.

    Input shape:
        artifacts/chunked/{tenant_id}/{job_id}/
          _DONE
          chunks/*.chunks.jsonl
          reports/*.transform_report.json
          manifest.json

    Skips:
        - jobs without _DONE
        - jobs already marked _INDEXED
        - jobs already marked _INDEX_FAILED
    """

    root = config.artifacts.chunked_root

    if not root.exists():
        return []

    jobs: list[ChunkArtifactJob] = []

    for done_marker in sorted(root.glob(f"*/*/{config.markers.done}")):
        job_root = done_marker.parent

        if _is_already_indexed(
            job_root=job_root,
            config=config,
        ):
            continue

        job = _build_job_from_root(
            job_root=job_root,
            chunked_root=root,
        )

        if job.files:
            jobs.append(job)

    return jobs


def load_chunk_artifact_file(
    artifact_file: ChunkArtifactFile,
) -> LoadedChunkArtifact:
    chunks: list[Chunk] = []

    with artifact_file.path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {artifact_file.path}:{line_number}"
                ) from exc

            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"Expected JSON object at {artifact_file.path}:{line_number}"
                )

            chunks.append(
                chunk_from_jsonl_payload(
                    payload,
                    artifact_file=artifact_file,
                    line_number=line_number,
                )
            )

    return LoadedChunkArtifact(
        artifact_file=artifact_file,
        chunks=tuple(chunks),
    )


def iter_loaded_chunks(
    jobs: Iterable[ChunkArtifactJob],
) -> Iterator[LoadedChunkArtifact]:
    for job in jobs:
        for artifact_file in job.files:
            yield load_chunk_artifact_file(artifact_file)


def chunk_from_jsonl_payload(
    payload: Mapping[str, Any],
    *,
    artifact_file: ChunkArtifactFile,
    line_number: int,
) -> Chunk:
    """
    Rebuild Chunk from chunk JSONL output.

    The chunk artifact writer serializes Chunk.to_dict() like:
        {
          "id": "...",
          "text": "...",
          "heading": "...",
          "context": "...",
          "type": "...",
          "subtype": "...",
          "order": 0,
          "tokens": 42,
          "source": "...",
          "metadata": {...}
        }
    """

    metadata = _safe_dict(payload.get("metadata"))

    metadata.setdefault("tenant_id", artifact_file.tenant_id)
    metadata.setdefault("job_id", artifact_file.job_id)
    metadata.setdefault("source_id", artifact_file.source_id)
    metadata.setdefault("chunk_artifact_path", artifact_file.path.as_posix())
    metadata.setdefault("chunk_artifact_line", line_number)

    chunk_id = _as_text(payload.get("id")) or _as_text(metadata.get("chunk_id"))

    return Chunk(
        text=_as_text(payload.get("text")),
        heading=_as_text(payload.get("heading")),
        context=_as_text(payload.get("context")),
        chunk_type=_as_text(payload.get("type")) or "text",
        subtype=_as_text(payload.get("subtype")),
        metadata=metadata,
        tokens=_as_int(payload.get("tokens")),
        source=_as_text(payload.get("source")) or artifact_file.path.as_posix(),
        order=_as_int(payload.get("order")),
        chunk_id=chunk_id or None,
    )


def _is_already_indexed(
    *,
    job_root: Path,
    config: EmbeddingPipelineConfig,
) -> bool:
    indexed_root = (
        config.artifacts.indexed_root
        / job_root.parent.name
        / job_root.name
    )

    return (
        indexed_root
        / config.markers.indexed
    ).exists()


def _build_job_from_root(
    *,
    job_root: Path,
    chunked_root: Path,
) -> ChunkArtifactJob:
    relative_parts = job_root.relative_to(chunked_root).parts

    if len(relative_parts) != 2:
        raise ValueError(f"Invalid chunked job root shape: {job_root}")

    tenant_id, job_id = relative_parts

    chunks_dir = job_root / "chunks"
    reports_dir = job_root / "reports"
    manifest_path = job_root / "manifest.json"

    files = tuple(
        _discover_chunk_files(
            tenant_id=tenant_id,
            job_id=job_id,
            chunks_dir=chunks_dir,
            reports_dir=reports_dir,
        )
    )

    return ChunkArtifactJob(
        tenant_id=tenant_id,
        job_id=job_id,
        root=job_root,
        chunks_dir=chunks_dir,
        reports_dir=reports_dir,
        manifest_path=manifest_path,
        files=files,
    )


def _discover_chunk_files(
    *,
    tenant_id: str,
    job_id: str,
    chunks_dir: Path,
    reports_dir: Path,
) -> list[ChunkArtifactFile]:
    if not chunks_dir.exists():
        return []

    files: list[ChunkArtifactFile] = []

    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        if not path.is_file():
            continue

        source_id = path.name.removesuffix(".chunks.jsonl")
        report_path = reports_dir / f"{source_id}.transform_report.json"

        files.append(
            ChunkArtifactFile(
                tenant_id=tenant_id,
                job_id=job_id,
                source_id=source_id,
                path=path,
                report_path=report_path if report_path.exists() else None,
            )
        )

    return files


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
