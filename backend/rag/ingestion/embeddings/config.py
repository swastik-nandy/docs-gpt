from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import rtoml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "settings" / "config.toml"


class QdrantMode(str, Enum):
    REMOTE = "remote"
    MEMORY = "memory"


class VectorDistance(str, Enum):
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"


@dataclass(frozen=True)
class SchemaConfig:
    version: str


@dataclass(frozen=True)
class ArtifactConfig:
    chunked_root: Path
    indexed_root: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    sparse_model_name: str
    device: Optional[str]
    batch_size: int
    dense_vector_size: int
    normalize_embeddings: bool
    document_prefix: str
    query_prefix: str
    trust_remote_code: bool
    use_fp16: bool
    max_length: int
    enable_sparse: bool
    sparse_min_abs_value: float
    sparse_max_nonzero: int
    return_colbert_vecs: bool

    @property
    def normalized_device(self) -> str:
        return (self.device or "auto").strip().lower()

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("embedding.model_name cannot be empty.")

        if not self.sparse_model_name.strip():
            raise ValueError("embedding.sparse_model_name cannot be empty.")

        if self.enable_sparse and self.sparse_model_name != self.model_name:
            raise ValueError(
                "This BGEM3FlagModel setup uses one model for dense+sparse. "
                "Set embedding.sparse_model_name equal to embedding.model_name."
            )

        if self.batch_size <= 0:
            raise ValueError("embedding.batch_size must be greater than zero.")

        if self.dense_vector_size <= 0:
            raise ValueError("embedding.dense_vector_size must be greater than zero.")

        if self.max_length <= 0:
            raise ValueError("embedding.max_length must be greater than zero.")

        if self.sparse_min_abs_value < 0:
            raise ValueError("embedding.sparse_min_abs_value cannot be negative.")

        if self.sparse_max_nonzero <= 0:
            raise ValueError("embedding.sparse_max_nonzero must be greater than zero.")

        device = self.normalized_device
        if not (
            device in {"auto", "cpu", "cuda", "mps"}
            or device.startswith("cuda:")
        ):
            raise ValueError(
                f"Invalid embedding.device={self.device!r}. "
                "Use auto, cpu, cuda, cuda:0, or mps."
            )


@dataclass(frozen=True)
class QdrantConfig:
    collection_name: str
    mode: QdrantMode
    url: Optional[str]
    api_key: Optional[str]
    timeout_seconds: int
    prefer_grpc: bool
    distance: VectorDistance
    recreate_collection: bool
    create_payload_indexes: bool
    upsert_batch_size: int

    dense_vector_name: str
    sparse_vector_name: str
    enable_sparse_vectors: bool

    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_full_scan_threshold: int

    dense_on_disk: bool
    sparse_on_disk: bool

    hybrid_prefetch_limit: int
    hybrid_fusion: str

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("qdrant.collection_name cannot be empty.")

        if self.mode == QdrantMode.REMOTE and not self.url:
            raise ValueError("qdrant.url is required when qdrant.mode='remote'.")

        if self.timeout_seconds <= 0:
            raise ValueError("qdrant.timeout_seconds must be greater than zero.")

        if self.upsert_batch_size <= 0:
            raise ValueError("qdrant.upsert_batch_size must be greater than zero.")

        if not self.dense_vector_name.strip():
            raise ValueError("qdrant.dense_vector_name cannot be empty.")

        if not self.sparse_vector_name.strip():
            raise ValueError("qdrant.sparse_vector_name cannot be empty.")

        if self.hnsw_m <= 0:
            raise ValueError("qdrant.hnsw_m must be greater than zero.")

        if self.hnsw_ef_construct <= 0:
            raise ValueError("qdrant.hnsw_ef_construct must be greater than zero.")

        if self.hnsw_full_scan_threshold <= 0:
            raise ValueError("qdrant.hnsw_full_scan_threshold must be greater than zero.")

        if self.hybrid_prefetch_limit <= 0:
            raise ValueError("qdrant.hybrid_prefetch_limit must be greater than zero.")

        if self.hybrid_fusion not in {"RRF", "DBSF"}:
            raise ValueError("qdrant.hybrid_fusion must be RRF or DBSF.")


@dataclass(frozen=True)
class IndexingConfig:
    poll_interval_seconds: float
    idle_jitter_seconds: float
    run_once: bool
    continue_on_error: bool
    max_jobs_per_iteration: int
    max_files_per_job: int

    def validate(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("indexing.poll_interval_seconds must be greater than zero.")

        if self.idle_jitter_seconds < 0:
            raise ValueError("indexing.idle_jitter_seconds cannot be negative.")

        if self.max_jobs_per_iteration <= 0:
            raise ValueError("indexing.max_jobs_per_iteration must be greater than zero.")

        if self.max_files_per_job <= 0:
            raise ValueError("indexing.max_files_per_job must be greater than zero.")


@dataclass(frozen=True)
class MarkerConfig:
    done: str
    failed: str
    indexed: str
    index_failed: str


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    schema: SchemaConfig
    artifacts: ArtifactConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    indexing: IndexingConfig
    markers: MarkerConfig

    @property
    def schema_version(self) -> str:
        return self.schema.version

    def validate(self) -> None:
        if not self.schema.version.strip():
            raise ValueError("schema.version cannot be empty.")

        self.embedding.validate()
        self.qdrant.validate()
        self.indexing.validate()

        markers = {
            self.markers.done,
            self.markers.failed,
            self.markers.indexed,
            self.markers.index_failed,
        }
        if any(not marker.strip() for marker in markers):
            raise ValueError("All marker values must be non-empty.")


def load_embedding_pipeline_config(
    config_path: str | Path | None = None,
) -> EmbeddingPipelineConfig:
    active_path = Path(
        config_path
        or os.getenv("DOCS_GPT_EMBEDDINGS_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    )

    data = _load_toml(active_path)

    schema = _require_mapping(data, "schema")
    artifacts = _require_mapping(data, "artifacts")
    embedding = _require_mapping(data, "embedding")
    qdrant = _require_mapping(data, "qdrant")
    indexing = _require_mapping(data, "indexing")
    markers = _require_mapping(data, "markers")

    model_name = os.getenv("EMBEDDING_MODEL", _require_str(embedding, "model_name")).strip()
    sparse_model_name = os.getenv(
        "SPARSE_EMBEDDING_MODEL",
        _optional_str(embedding, "sparse_model_name", default=model_name) or model_name,
    ).strip()

    qdrant_url = os.getenv("QDRANT_URL", _optional_str(qdrant, "url", default=None) or "").strip() or None
    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY",
        _optional_str(qdrant, "api_key", default=None) or "",
    ).strip() or None

    config = EmbeddingPipelineConfig(
        schema=SchemaConfig(
            version=_require_str(schema, "version"),
        ),
        artifacts=ArtifactConfig(
            chunked_root=Path(_require_str(artifacts, "chunked_root")),
            indexed_root=Path(_require_str(artifacts, "indexed_root")),
        ),
        embedding=EmbeddingConfig(
            model_name=model_name,
            sparse_model_name=sparse_model_name,
            device=_optional_str(embedding, "device", default="auto"),
            batch_size=_require_int(embedding, "batch_size"),
            dense_vector_size=_require_int(embedding, "dense_vector_size"),
            normalize_embeddings=_require_bool(embedding, "normalize_embeddings"),
            document_prefix=_optional_str(embedding, "document_prefix", default="") or "",
            query_prefix=_optional_str(embedding, "query_prefix", default="") or "",
            trust_remote_code=_require_bool(embedding, "trust_remote_code"),
            use_fp16=_require_bool(embedding, "use_fp16"),
            max_length=_require_int(embedding, "max_length"),
            enable_sparse=_require_bool(embedding, "enable_sparse"),
            sparse_min_abs_value=_require_float(embedding, "sparse_min_abs_value"),
            sparse_max_nonzero=_require_int(embedding, "sparse_max_nonzero"),
            return_colbert_vecs=_require_bool(embedding, "return_colbert_vecs"),
        ),
        qdrant=QdrantConfig(
            collection_name=_require_str(qdrant, "collection_name"),
            mode=_parse_qdrant_mode(_require_str(qdrant, "mode")),
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout_seconds=_require_int(qdrant, "timeout_seconds"),
            prefer_grpc=_require_bool(qdrant, "prefer_grpc"),
            distance=_parse_distance(_require_str(qdrant, "distance")),
            recreate_collection=_require_bool(qdrant, "recreate_collection"),
            create_payload_indexes=_require_bool(qdrant, "create_payload_indexes"),
            upsert_batch_size=_require_int(qdrant, "upsert_batch_size"),
            dense_vector_name=_require_str(qdrant, "dense_vector_name"),
            sparse_vector_name=_require_str(qdrant, "sparse_vector_name"),
            enable_sparse_vectors=_require_bool(qdrant, "enable_sparse_vectors"),
            hnsw_m=_require_int(qdrant, "hnsw_m"),
            hnsw_ef_construct=_require_int(qdrant, "hnsw_ef_construct"),
            hnsw_full_scan_threshold=_require_int(qdrant, "hnsw_full_scan_threshold"),
            dense_on_disk=_require_bool(qdrant, "dense_on_disk"),
            sparse_on_disk=_require_bool(qdrant, "sparse_on_disk"),
            hybrid_prefetch_limit=_require_int(qdrant, "hybrid_prefetch_limit"),
            hybrid_fusion=_require_str(qdrant, "hybrid_fusion").upper(),
        ),
        indexing=IndexingConfig(
            poll_interval_seconds=_require_float(indexing, "poll_interval_seconds"),
            idle_jitter_seconds=_require_float(indexing, "idle_jitter_seconds"),
            run_once=_require_bool(indexing, "run_once"),
            continue_on_error=_require_bool(indexing, "continue_on_error"),
            max_jobs_per_iteration=_require_int(indexing, "max_jobs_per_iteration"),
            max_files_per_job=_require_int(indexing, "max_files_per_job"),
        ),
        markers=MarkerConfig(
            done=_require_str(markers, "done"),
            failed=_require_str(markers, "failed"),
            indexed=_require_str(markers, "indexed"),
            index_failed=_require_str(markers, "index_failed"),
        ),
    )

    config.validate()
    return config


@lru_cache(maxsize=1)
def get_embedding_pipeline_config() -> EmbeddingPipelineConfig:
    return load_embedding_pipeline_config()


def _load_toml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings config not found: {path}")

    data = rtoml.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, Mapping):
        raise ValueError(f"Embeddings config must be a TOML object: {path}")

    return data


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected TOML table: {key!r}")
    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string config value: {key!r}")
    return value.strip()


def _optional_str(
    data: Mapping[str, Any],
    key: str,
    *,
    default: Optional[str],
) -> Optional[str]:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string config value: {key!r}")
    value = value.strip()
    return value or default


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean config value: {key!r}")
    return value


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: int = 1,
) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer config value: {key!r}")
    if value < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {value}")
    return value


def _require_float(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: float = 0.0,
) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric config value: {key!r}")
    numeric = float(value)
    if numeric < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {numeric}")
    return numeric


def _parse_qdrant_mode(value: str) -> QdrantMode:
    try:
        return QdrantMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in QdrantMode)
        raise ValueError(f"Invalid qdrant.mode={value!r}. Valid: {valid}") from exc


def _parse_distance(value: str) -> VectorDistance:
    try:
        return VectorDistance(value.strip())
    except ValueError as exc:
        valid = ", ".join(item.value for item in VectorDistance)
        raise ValueError(f"Invalid qdrant.distance={value!r}. Valid: {valid}") from exc