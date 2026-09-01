from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceArtifact:
    tenant_id: str
    job_id: str
    source_id: str
    path: Path
    relative_path: str
    media_type: str = "text/markdown"

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


@dataclass(frozen=True)
class TransformationJob:
    tenant_id: str
    job_id: str
    source_root: Path
    output_root: Path
    files: tuple[SourceArtifact, ...]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def job_key(self) -> str:
        return f"{self.tenant_id}/{self.job_id}"

    @property
    def output_job_root(self) -> Path:
        return self.output_root / self.tenant_id / self.job_id


@dataclass(frozen=True)
class SourceTransformReport:
    source_id: str
    source_path: str
    status: str
    doc_id: str = ""
    parser_family: str = ""
    parser_variant: str = ""
    parse_strategy: str = ""
    unit_count: int = 0
    chunk_count: int = 0
    total_chunk_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "status": self.status,
            "doc_id": self.doc_id,
            "parser_family": self.parser_family,
            "parser_variant": self.parser_variant,
            "parse_strategy": self.parse_strategy,
            "unit_count": self.unit_count,
            "chunk_count": self.chunk_count,
            "total_chunk_tokens": self.total_chunk_tokens,
            "timings": self.timings,
            "error": self.error,
        }


@dataclass(frozen=True)
class JobTransformReport:
    tenant_id: str
    job_id: str
    status: str
    source_reports: tuple[SourceTransformReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "status": self.status,
            "source_count": len(self.source_reports),
            "succeeded": sum(1 for item in self.source_reports if item.status == "succeeded"),
            "failed": sum(1 for item in self.source_reports if item.status == "failed"),
            "sources": [item.to_dict() for item in self.source_reports],
        }
