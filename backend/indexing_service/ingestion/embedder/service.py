from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cloudflare import CloudflareEmbedder
from .config import get_embedding_config
from .input_builder import build_embedding_text
from .qdrant import QdrantCloud


# --------- PATHS ------------------------

BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# --------- RESULT ------------------------

@dataclass(frozen=True)
class IndexResult:
    job_id: str
    status: str
    chunk_count: int
    embedding_seconds: float
    qdrant_seconds: float
    total_seconds: float
    output_root: Path


# --------- SERVICE ------------------------

class EmbeddingService:
    def __init__(self) -> None:
        self._config = (
            get_embedding_config()
        )

        self._embedder = (
            CloudflareEmbedder()
        )

        self._qdrant = (
            QdrantCloud()
        )

    # --------- LIFECYCLE ------------------------

    async def __aenter__(
        self,
    ) -> "EmbeddingService":
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        await self.close()

    async def close(
        self,
    ) -> None:
        await self._embedder.close()
        await self._qdrant.close()

    # --------- PROCESS JOB ------------------------

    async def process_job(
        self,
        job_id: str,
        *,
        force: bool = False,
    ) -> IndexResult:
        started = time.perf_counter()

        chunked_root = (
            self._artifact_root(
                self._config
                .artifacts
                .chunked_directory
            )
            / job_id
        )

        indexed_root = (
            self._artifact_root(
                self._config
                .artifacts
                .indexed_directory
            )
            / job_id
        )

        ready_marker = (
            chunked_root
            / self._config
            .artifacts
            .ready_marker
        )

        chunks_path = (
            chunked_root
            / "chunks.jsonl"
        )

        if not ready_marker.is_file():
            raise RuntimeError(
                f"Chunked job is not ready: {job_id}"
            )

        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"Missing chunks file: {chunks_path}"
            )

        indexed_ready = (
            indexed_root
            / "_READY"
        )

        # --------- IDEMPOTENT FAST PATH ------------------------

        if (
            indexed_ready.is_file()
            and not force
        ):
            manifest = self._read_json(
                indexed_root
                / "manifest.json"
            )

            return IndexResult(
                job_id=job_id,
                status="ready",
                chunk_count=int(
                    manifest.get(
                        "chunk_count",
                        0,
                    )
                ),
                embedding_seconds=0.0,
                qdrant_seconds=0.0,
                total_seconds=(
                    time.perf_counter()
                    - started
                ),
                output_root=indexed_root,
            )

        # --------- RESET OUTPUT ------------------------

        if indexed_root.exists():
            shutil.rmtree(
                indexed_root
            )

        indexed_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks = self._read_chunks(
            chunks_path
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks found for job: {job_id}"
            )

        embedding_seconds = 0.0
        qdrant_seconds = 0.0
        processed = 0

        try:
            # One schema verification per service instance.
            await self._qdrant.ensure_collection()

            embed_batch_size = (
                self._config
                .embedder
                .request
                .batch_size
            )

            qdrant_batch_size = (
                self._config
                .qdrant
                .upsert_batch_size
            )

            # We immediately upsert each embedding batch.
            # Therefore use the smaller operational limit.
            batch_size = min(
                embed_batch_size,
                qdrant_batch_size,
            )

            for offset in range(
                0,
                len(chunks),
                batch_size,
            ):
                batch = chunks[
                    offset:
                    offset + batch_size
                ]

                # --------- BUILD EMBEDDING INPUT ------------------------

                texts = [
                    build_embedding_text(
                        chunk
                    )
                    for chunk in batch
                ]

                # --------- CLOUDFLARE ------------------------

                embedding_result = (
                    await self
                    ._embedder
                    .embed(
                        texts
                    )
                )

                embedding_seconds += (
                    embedding_result
                    .elapsed_seconds
                )

                # --------- BUILD QDRANT POINTS ------------------------

                points: list[
                    dict[str, Any]
                ] = []

                for (
                    chunk,
                    vector,
                ) in zip(
                    batch,
                    embedding_result.vectors,
                    strict=True,
                ):
                    chunk_id = str(
                        chunk.get(
                            "chunk_id"
                        )
                        or chunk.get(
                            "id"
                        )
                        or ""
                    ).strip()

                    if not chunk_id:
                        raise ValueError(
                            "Chunk has no chunk_id"
                        )

                    points.append(
                        {
                            "id": self._point_id(
                                chunk_id
                            ),
                            "vector": vector,
                            "payload": (
                                self._build_payload(
                                    chunk
                                )
                            ),
                        }
                    )

                # --------- QDRANT ------------------------

                qdrant_result = (
                    await self
                    ._qdrant
                    .upsert(
                        points
                    )
                )

                qdrant_seconds += (
                    qdrant_result
                    .elapsed_seconds
                )

                processed += len(
                    batch
                )

            # --------- WRITE INDEX MANIFEST ------------------------

            total_seconds = (
                time.perf_counter()
                - started
            )

            manifest = {
                "schema_version": 1,
                "stage": "indexed",
                "status": "ready",
                "job_id": job_id,
                "collection": (
                    self._qdrant
                    .collection_name
                ),
                "embedding_model": (
                    self._config
                    .embedder
                    .model
                    .name
                ),
                "vector_size": (
                    self._config
                    .embedder
                    .model
                    .vector_size
                ),
                "chunk_count": processed,
                "chunks_sha256": (
                    self._sha256_file(
                        chunks_path
                    )
                ),
                "timings": {
                    "embedding_seconds": round(
                        embedding_seconds,
                        6,
                    ),
                    "qdrant_seconds": round(
                        qdrant_seconds,
                        6,
                    ),
                    "total_seconds": round(
                        total_seconds,
                        6,
                    ),
                },
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

            self._write_json_atomic(
                indexed_root
                / "manifest.json",
                manifest,
            )

            # Commit marker is always written last.
            indexed_ready.touch()

            return IndexResult(
                job_id=job_id,
                status="ready",
                chunk_count=processed,
                embedding_seconds=(
                    embedding_seconds
                ),
                qdrant_seconds=(
                    qdrant_seconds
                ),
                total_seconds=(
                    total_seconds
                ),
                output_root=indexed_root,
            )

        except Exception as exc:
            indexed_ready.unlink(
                missing_ok=True
            )

            failure = {
                "schema_version": 1,
                "stage": "indexed",
                "status": "failed",
                "job_id": job_id,
                "processed_chunks": processed,
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

            self._write_json_atomic(
                indexed_root
                / "manifest.json",
                failure,
            )

            (
                indexed_root
                / "_FAILED"
            ).touch()

            raise

    # --------- PROCESS READY JOBS ------------------------

    async def process_available_jobs_once(
        self,
    ) -> list[IndexResult]:
        chunked_root = self._artifact_root(
            self._config
            .artifacts
            .chunked_directory
        )

        indexed_root = self._artifact_root(
            self._config
            .artifacts
            .indexed_directory
        )

        chunked_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        indexed_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        results: list[IndexResult] = []

        marker_name = (
            self._config
            .artifacts
            .ready_marker
        )

        for marker in sorted(
            chunked_root.glob(
                f"*/{marker_name}"
            )
        ):
            job_id = (
                marker
                .parent
                .name
            )

            if (
                indexed_root
                / job_id
                / "_READY"
            ).is_file():
                continue

            result = (
                await self.process_job(
                    job_id
                )
            )

            results.append(
                result
            )

        return results

    # --------- QDRANT PAYLOAD ------------------------

    def _build_payload(
        self,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(
            chunk.get(
                "metadata"
            )
            or {}
        )

        heading_path = (
            metadata.get(
                "heading_path"
            )
            or []
        )

        if not isinstance(
            heading_path,
            list,
        ):
            heading_path = []

        return {
            "chunk_id": (
                chunk.get(
                    "chunk_id"
                )
                or chunk.get(
                    "id"
                )
            ),
            "job_id": (
                chunk.get(
                    "job_id"
                )
            ),
            "tenant_id": (
                chunk.get(
                    "tenant_id"
                )
            ),
            "source_id": (
                chunk.get(
                    "source_id"
                )
            ),
            "file_id": (
                chunk.get(
                    "file_id"
                )
            ),
            "file_name": (
                chunk.get(
                    "file_name"
                )
            ),
            "source_sha256": (
                chunk.get(
                    "source_sha256"
                )
            ),
            "text": (
                chunk.get(
                    "text"
                )
            ),
            "heading": (
                chunk.get(
                    "heading"
                )
            ),
            "heading_path": (
                heading_path
            ),
            "context": (
                chunk.get(
                    "context"
                )
            ),
            "chunk_type": (
                chunk.get(
                    "chunk_type"
                )
                or chunk.get(
                    "type"
                )
            ),
            "subtype": (
                chunk.get(
                    "subtype"
                )
            ),
            "tokens": (
                chunk.get(
                    "tokens"
                )
            ),
            "order": (
                chunk.get(
                    "order"
                )
            ),
            "section_id": (
                metadata.get(
                    "section_id"
                )
            ),
        }

    # --------- DETERMINISTIC POINT ID ------------------------

    def _point_id(
        self,
        chunk_id: str,
    ) -> str:
        identity = (
            f"{self._qdrant.collection_name}:"
            f"{chunk_id}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                identity,
            )
        )

    # --------- CHUNK IO ------------------------

    def _read_chunks(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        chunks: list[
            dict[str, Any]
        ] = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for (
                line_number,
                line,
            ) in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                value = json.loads(
                    line
                )

                if not isinstance(
                    value,
                    dict,
                ):
                    raise TypeError(
                        "Chunk line "
                        f"{line_number} "
                        "is not a JSON object"
                    )

                chunks.append(
                    value
                )

        return chunks

    # --------- ARTIFACT PATH ------------------------

    def _artifact_root(
        self,
        configured: str,
    ) -> Path:
        path = Path(
            configured
        )

        if not path.is_absolute():
            path = (
                BACKEND_ROOT
                / path
            )

        return path.resolve()

    # --------- JSON IO ------------------------

    def _read_json(
        self,
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

    def _write_json_atomic(
        self,
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

    # --------- HASHING ------------------------

    def _sha256_file(
        self,
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
