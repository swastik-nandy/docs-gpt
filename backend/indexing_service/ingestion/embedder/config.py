from __future__ import annotations

import tomllib

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# --------- PATHS ------------------------

CONFIG_PATH = (
    Path(__file__)
    .resolve()
    .parent
    / "config.toml"
)


# --------- MODEL CONFIG ------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    vector_size: int
    max_input_tokens: int


# --------- REQUEST CONFIG ------------------------

@dataclass(frozen=True)
class RequestConfig:
    batch_size: int
    timeout_seconds: float
    max_retries: int


# --------- INPUT CONFIG ------------------------

@dataclass(frozen=True)
class InputConfig:
    include_heading_path: bool
    include_chunk_type: bool
    include_parent_context_for_code: bool


# --------- EMBEDDER CONFIG ------------------------

@dataclass(frozen=True)
class EmbedderConfig:
    enabled: bool
    provider: str
    model: ModelConfig
    request: RequestConfig
    input: InputConfig


# --------- QDRANT CONFIG ------------------------

@dataclass(frozen=True)
class QdrantConfig:
    upsert_batch_size: int


# --------- ARTIFACT CONFIG ------------------------

@dataclass(frozen=True)
class ArtifactConfig:
    chunked_directory: str
    indexed_directory: str
    ready_marker: str


# --------- ROOT CONFIG ------------------------

@dataclass(frozen=True)
class IngestionEmbeddingConfig:
    embedder: EmbedderConfig
    qdrant: QdrantConfig
    artifacts: ArtifactConfig


# --------- LOAD CONFIG ------------------------

@lru_cache(maxsize=1)
def get_embedding_config() -> IngestionEmbeddingConfig:
    with CONFIG_PATH.open(
        "rb"
    ) as handle:
        raw = tomllib.load(
            handle
        )

    embedder_raw = raw[
        "embedder"
    ]

    model_raw = embedder_raw[
        "model"
    ]

    request_raw = embedder_raw[
        "request"
    ]

    input_raw = embedder_raw[
        "input"
    ]

    qdrant_raw = raw[
        "qdrant"
    ]

    artifacts_raw = raw[
        "artifacts"
    ]

    config = IngestionEmbeddingConfig(
        embedder=EmbedderConfig(
            enabled=bool(
                embedder_raw[
                    "enabled"
                ]
            ),
            provider=str(
                embedder_raw[
                    "provider"
                ]
            ),
            model=ModelConfig(
                name=str(
                    model_raw[
                        "name"
                    ]
                ),
                vector_size=int(
                    model_raw[
                        "vector_size"
                    ]
                ),
                max_input_tokens=int(
                    model_raw[
                        "max_input_tokens"
                    ]
                ),
            ),
            request=RequestConfig(
                batch_size=int(
                    request_raw[
                        "batch_size"
                    ]
                ),
                timeout_seconds=float(
                    request_raw[
                        "timeout_seconds"
                    ]
                ),
                max_retries=int(
                    request_raw[
                        "max_retries"
                    ]
                ),
            ),
            input=InputConfig(
                include_heading_path=bool(
                    input_raw[
                        "include_heading_path"
                    ]
                ),
                include_chunk_type=bool(
                    input_raw[
                        "include_chunk_type"
                    ]
                ),
                include_parent_context_for_code=bool(
                    input_raw[
                        "include_parent_context_for_code"
                    ]
                ),
            ),
        ),
        qdrant=QdrantConfig(
            upsert_batch_size=int(
                qdrant_raw[
                    "upsert_batch_size"
                ]
            ),
        ),
        artifacts=ArtifactConfig(
            chunked_directory=str(
                artifacts_raw[
                    "chunked_directory"
                ]
            ),
            indexed_directory=str(
                artifacts_raw[
                    "indexed_directory"
                ]
            ),
            ready_marker=str(
                artifacts_raw[
                    "ready_marker"
                ]
            ),
        ),
    )

    _validate(
        config
    )

    return config


# --------- VALIDATION ------------------------

def _validate(
    config: IngestionEmbeddingConfig,
) -> None:
    if (
        config.embedder.provider
        != "cloudflare"
    ):
        raise ValueError(
            "Only Cloudflare embedding "
            "provider is supported"
        )

    if (
        config.embedder.model.vector_size
        != 1024
    ):
        raise ValueError(
            "Qwen3-Embedding-0.6B requires "
            "1024-dimensional vectors"
        )

    if (
        config.embedder.model.max_input_tokens
        <= 0
    ):
        raise ValueError(
            "max_input_tokens must be positive"
        )

    if (
        config.embedder.request.batch_size
        <= 0
    ):
        raise ValueError(
            "batch_size must be positive"
        )

    if (
        config.embedder.request.timeout_seconds
        <= 0
    ):
        raise ValueError(
            "timeout_seconds must be positive"
        )

    if (
        config.embedder.request.max_retries
        < 0
    ):
        raise ValueError(
            "max_retries cannot be negative"
        )

    if (
        config.qdrant.upsert_batch_size
        <= 0
    ):
        raise ValueError(
            "upsert_batch_size must be positive"
        )
